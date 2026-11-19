"""
mov-to-txt 웹 UI

사용법:
    uv run web.py                      # http://127.0.0.1:8000
    uv run web.py --host 0.0.0.0 --port 8080

브라우저에서 파일 선택/업로드 → 엔진·언어·GPU·요약 옵션 선택 → 작업 큐에 추가.
작업은 GPU 메모리 충돌을 피하기 위해 한 번에 하나씩 순차 실행되며,
각 작업은 별도 프로세스로 실행되어 종료 시 GPU 메모리가 완전히 해제됩니다.
"""

import os

# GPU 번호를 nvidia-smi와 일치시킨다 (torch 기본값은 FASTEST_FIRST라 순서가 다를 수 있음)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import codecs
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

import convert
import gpu_selector
import youtube

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
AUDIO_DIR = BASE_DIR / "audio"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
SUMMARY_DIR = BASE_DIR / "summaries"
STATIC_DIR = BASE_DIR / "static"

FOLDERS = {"input": INPUT_DIR, "audio": AUDIO_DIR}
MAX_LOG_CHARS = 200_000
PROGRESS_RE = re.compile(r"\[진행\] (\d+)/(\d+) 완료")


# ─────────────────────────────────────────────
# 작업 큐
# ─────────────────────────────────────────────

@dataclass
class Job:
    id: str
    folder: str
    filename: str
    engine: str
    language: str
    summarize: bool
    gpu: int
    overwrite: bool
    url: str | None = None      # YouTube 작업이면 영상 URL (folder="youtube", filename=영상 제목)
    name: str | None = None     # YouTube 작업의 결과 파일명(확장자 제외)
    subdir: str | None = None   # YouTube 재생목록 작업이면 결과를 모을 하위 폴더명
    subs: str = "off"           # YouTube 자막 정책 (youtube.SUBS_POLICIES)
    caption: str | None = None  # 완료 후: 자막으로 만들었으면 manual | auto
    videos: list[dict] | None = None  # YouTube 묶음 작업: [{id, url, title}] — 한 프로세스에서 파이프라인으로 처리
    progress: str | None = None       # 묶음 작업 진행 상황 "3/27"
    status: str = "queued"  # queued | running | done | error | cancelled
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    log: str = ""

    def append_log(self, s: str) -> None:
        # '\r' 진행 표시줄은 마지막 줄을 덮어쓴다
        for i, part in enumerate(s.split("\r")):
            if i > 0:
                cut = self.log.rfind("\n")
                self.log = self.log[: cut + 1]
            self.log += part
        if len(self.log) > MAX_LOG_CHARS:
            self.log = self.log[-MAX_LOG_CHARS:]
        if self.videos and "[진행]" in self.log[-(len(s) + 64):]:
            if m := list(PROGRESS_RE.finditer(self.log[-(len(s) + 64):])):
                self.progress = f"{m[-1].group(1)}/{m[-1].group(2)}"

    def video_stems(self) -> list[str]:
        """묶음 작업 각 영상의 결과 경로 (convert.py와 같은 규칙으로 계산)."""
        pre = f"{self.subdir}/" if self.subdir else ""
        return [pre + youtube.make_name(v["title"], v["id"]) for v in self.videos or []]

    @property
    def stem(self) -> str | None:
        """결과 파일 경로(transcripts/ 기준, 확장자 제외). 하위 폴더에 있으면 '폴더/이름'. 묶음 작업은 None."""
        if self.videos:
            return None
        if self.name:
            return f"{self.subdir}/{self.name}" if self.subdir else self.name
        return str(Path(self.filename).with_suffix(""))

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("log")
        d["count"] = len(d.pop("videos") or [])
        d["stem"] = self.stem
        return d


_jobs: dict[str, Job] = {}
_job_order: list[str] = []
_queue: "queue.Queue[str]" = queue.Queue()
_lock = threading.Lock()


def _output_paths(stem: str) -> dict[str, Path]:
    return {
        "txt": TRANSCRIPT_DIR / f"{stem}.txt",
        "json": TRANSCRIPT_DIR / f"{stem}.json",
        "md": SUMMARY_DIR / f"{stem}.md",
    }


_procs: dict[str, subprocess.Popen] = {}


def _run_job(job: Job) -> int:
    """convert.py를 별도 프로세스로 실행한다.

    프로세스가 끝나면 CUDA 컨텍스트가 OS에 의해 해제되므로 GPU 점유가 남지 않는다.
    CUDA_VISIBLE_DEVICES로 선택한 GPU 하나만 보이게 하여 다른 GPU에는 손대지 않는다.
    """
    if job.videos:  # 묶음: 한 프로세스에서 다운로드·전사를 겹쳐 처리하고 모델은 한 번만 로드
        source_args = [a for v in job.videos for a in ("--url", v["url"])] + ["--subs", job.subs]
        if job.subdir:
            source_args += ["--folder", job.subdir]
    elif job.url:
        source_args = ["--url", job.url, "--name", job.name, "--subs", job.subs]
        if job.subdir:
            source_args += ["--folder", job.subdir]
    else:
        input_path = FOLDERS[job.folder] / job.filename
        if not input_path.exists():
            raise FileNotFoundError(f"파일이 없습니다: {input_path}")
        source_args = [str(input_path)]

    if job.overwrite:
        for stem in job.video_stems() if job.videos else [job.stem]:
            outs = _output_paths(stem)
            targets = [outs["txt"], outs["json"]] + ([outs["md"]] if job.summarize else [])
            for p in targets:
                if p.exists():
                    p.unlink()
                    job.append_log(f"[덮어쓰기] 기존 파일 삭제: {p.relative_to(BASE_DIR)}\n")

    cmd = [
        sys.executable, "-u", str(BASE_DIR / "convert.py"), *source_args,
        "--engine", job.engine,
        "--language", job.language,
        "--gpu", "0",  # CUDA_VISIBLE_DEVICES로 GPU 하나만 노출하므로 프로세스 내부 인덱스는 항상 0
    ]
    if job.summarize:
        cmd.append("--summarize")

    env = os.environ.copy()
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(job.gpu),
        "PYTHONUNBUFFERED": "1",
        "PYTHONWARNINGS": "ignore::SyntaxWarning",  # soynlp(한국어 정렬용)의 정규식 경고가 로그를 덮는 것 방지
    })

    if job.videos:
        job.append_log(f"[설정] YouTube 영상 {len(job.videos)}개 묶음 처리 · 자막: {youtube.SUBS_POLICIES[job.subs]}\n")
    elif job.url:
        job.append_log(f"[설정] YouTube {job.url} · 자막: {youtube.SUBS_POLICIES[job.subs]}\n")
    job.append_log(
        f"[설정] 엔진={job.engine}, 언어={job.language}, GPU={job.gpu}번 "
        f"(nvidia-smi 기준, 작업 프로세스 안에서는 cuda:0으로 표시됨), 요약={'예' if job.summarize else '아니오'}\n"
    )

    proc = subprocess.Popen(
        cmd, cwd=BASE_DIR, env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,  # 취소 시 ffmpeg 등 자식 프로세스까지 함께 종료하기 위함
    )
    _procs[job.id] = proc
    if job.status == "cancelled":  # 프로세스 시작 직전에 취소된 경우
        os.killpg(proc.pid, signal.SIGTERM)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while chunk := proc.stdout.read1(4096):
            job.append_log(decoder.decode(chunk))
        job.append_log(decoder.decode(b"", final=True))
        return proc.wait()
    finally:
        _procs.pop(job.id, None)


def _worker() -> None:
    while True:
        job_id = _queue.get()
        job = _jobs.get(job_id)
        if job is None or job.status != "queued":
            continue
        job.status = "running"
        job.started = time.time()
        try:
            code = _run_job(job)
            if job.status == "cancelled":
                job.append_log("\n[취소됨] 사용자가 작업을 중단했습니다.\n")
            elif code == 0:
                job.status = "done"
                if job.stem:
                    job.caption = youtube.read_caption_kind(_output_paths(job.stem)["txt"])
            else:
                job.status = "error"
                job.error = f"작업 프로세스 종료 코드 {code}"
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.append_log(traceback.format_exc())
        finally:
            job.finished = time.time()


# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────

app = FastAPI(title="mov-to-txt")


def _is_safe_rel(rel: str) -> bool:
    """'파일' 또는 '하위폴더/파일' 형태만 허용 (재생목록 폴더 한 단계까지, 경로 조작 방지)."""
    parts = rel.split("/")
    return 1 <= len(parts) <= 2 and all(p and not p.startswith(".") and "\\" not in p for p in parts)


def _iter_rel_files(d: Path, pattern: str = "*"):
    """d 바로 아래와 하위 폴더 한 단계의 파일을 '이름' / '폴더/이름' 형태로 나열."""
    for p in sorted(d.glob(pattern), key=lambda p: p.name):
        if p.is_file() and not p.name.startswith("."):
            yield p, p.name
    for sub in sorted((s for s in d.iterdir() if s.is_dir() and not s.name.startswith(".")), key=lambda p: p.name):
        for p in sorted(sub.glob(pattern), key=lambda p: p.name):
            if p.is_file() and not p.name.startswith("."):
                yield p, f"{sub.name}/{p.name}"


def _safe_file(folder: str, filename: str) -> Path:
    if folder not in FOLDERS:
        raise HTTPException(400, "잘못된 폴더")
    if not _is_safe_rel(filename):
        raise HTTPException(400, "잘못된 파일명")
    path = FOLDERS[folder] / filename
    if not path.is_file():
        raise HTTPException(404, "파일을 찾을 수 없습니다")
    return path


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def get_config():
    return {
        "gpus": gpu_selector.list_gpus(),
        "default_gpu": gpu_selector.index(),
        "env_mode": gpu_selector.ENV_MODE,
        "has_api_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "engines": [{"id": eid, **info} for eid, info in convert.ENGINES.items()],
        "default_engine": convert.DEFAULT_ENGINE,
    }


@app.get("/api/files")
def list_files():
    result = []
    stems = set()
    for folder, d in FOLDERS.items():
        if not d.exists():
            continue
        for f, rel in _iter_rel_files(d):
            stem = rel[: -len(f.suffix)] if f.suffix else rel
            outs = _output_paths(stem)
            st = f.stat()
            stems.add(stem)
            has_txt = outs["txt"].exists()
            result.append({
                "folder": folder,
                "name": rel,
                "stem": stem,
                "size": st.st_size,
                "mtime": st.st_mtime,
                "is_audio": f.suffix.lower() in convert.AUDIO_EXTENSIONS,
                "has_transcript": has_txt,
                "has_summary": outs["md"].exists(),
                "caption": youtube.read_caption_kind(outs["txt"]) if has_txt else None,
            })
    # YouTube 자막으로 만든 전사본은 원본 파일이 없으므로 따로 표시 (변환 대상으로 선택 불가)
    if TRANSCRIPT_DIR.exists():
        for t, rel in _iter_rel_files(TRANSCRIPT_DIR, "*.txt"):
            stem = rel[: -len(".txt")]
            if stem in stems or not (caption := youtube.read_caption_kind(t)):
                continue
            st = t.stat()
            result.append({
                "folder": "youtube",
                "name": stem,
                "stem": stem,
                "size": None,
                "mtime": st.st_mtime,
                "is_audio": False,
                "has_transcript": True,
                "has_summary": _output_paths(stem)["md"].exists(),
                "caption": caption,
                "selectable": False,
            })
    return result


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    saved = []
    for uf in files:
        name = Path(uf.filename or "").name
        if not name or name.startswith("."):
            continue
        is_audio = Path(name).suffix.lower() in convert.AUDIO_EXTENSIONS
        dest_dir = AUDIO_DIR if is_audio else INPUT_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        with dest.open("wb") as out:
            shutil.copyfileobj(uf.file, out, length=1024 * 1024)
        saved.append({"folder": "audio" if is_audio else "input", "name": name})
    return {"saved": saved}


class JobRequest(BaseModel):
    files: list[dict]  # [{folder, name}]
    engine: str = convert.DEFAULT_ENGINE
    language: str = "ko"
    summarize: bool = False
    gpu: int = 0
    overwrite: bool = False


def _validate_options(engine: str, language: str, gpu: int) -> None:
    if engine not in convert.ENGINES:
        raise HTTPException(400, "잘못된 엔진")
    if language not in ("ko", "en"):
        raise HTTPException(400, "잘못된 언어")
    gpus = {g["index"]: g for g in gpu_selector.list_gpus()}
    if gpus and (gpu not in gpus or not gpus[gpu]["supported"]):
        raise HTTPException(400, f"GPU {gpu}번은 사용할 수 없습니다 (Compute Capability 7.0 미만 또는 없는 GPU)")


def _enqueue(job: Job) -> None:
    with _lock:
        _jobs[job.id] = job
        _job_order.append(job.id)
    _queue.put(job.id)


@app.post("/api/jobs")
def create_jobs(req: JobRequest):
    _validate_options(req.engine, req.language, req.gpu)
    created = []
    for f in req.files:
        path = _safe_file(f.get("folder", ""), f.get("name", ""))
        job = Job(
            id=uuid.uuid4().hex[:8],
            folder=f["folder"],
            filename=path.relative_to(FOLDERS[f["folder"]]).as_posix(),  # 하위 폴더 포함
            engine=req.engine,
            language=req.language,
            summarize=req.summarize,
            gpu=req.gpu,
            overwrite=req.overwrite,
        )
        _enqueue(job)
        created.append(job.id)
    return {"created": created}


# ── YouTube ──

@app.get("/api/youtube/info")
def youtube_info(url: str):
    """URL의 영상(또는 재생목록 항목) 정보와 자막 트랙을 조회한다. 다운로드는 하지 않는다."""
    try:
        info = youtube.fetch_info(url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001 — yt-dlp 오류(비공개 영상, 잘못된 ID 등)
        msg = str(e).replace("ERROR: ", "")
        raise HTTPException(400, f"영상 정보를 가져오지 못했습니다: {msg}")

    def entry(vid: str, title: str | None, folder: str | None = None, **extra) -> dict:
        name = youtube.make_name(title, vid)
        stem = f"{folder}/{name}" if folder else name
        return {"id": vid, "url": youtube.video_url(vid), "title": title or vid, "name": name,
                "has_transcript": _output_paths(stem)["txt"].exists(), **extra}

    if info.get("_type") == "playlist":
        videos = youtube.playlist_videos(info)
        if not videos:
            raise HTTPException(400, "재생목록에 영상이 없습니다")
        folder = youtube.make_name(info.get("title"), info["id"])
        return {
            "type": "playlist",
            "id": info["id"],
            "title": info.get("title"),
            "folder": folder,  # 결과를 모을 하위 폴더 (audio/·transcripts/·summaries/ 아래)
            "uploader": info.get("uploader") or info.get("channel"),
            "videos": [entry(v["id"], v["title"], folder, duration=v["duration"]) for v in videos],
        }
    tracks = [t for t in youtube.caption_tracks(info) if t["lang"] in ("ko", "en")]
    return {
        "type": "video",
        "title": info.get("title"),
        "uploader": info.get("uploader") or info.get("channel"),
        "language": info.get("language"),
        "videos": [entry(info["id"], info.get("title"), duration=info.get("duration"),
                         tracks=tracks, choices=youtube.caption_choices(info))],
    }


class YoutubeJobRequest(BaseModel):
    videos: list[dict]  # [{url, title, name}] — /api/youtube/info 응답의 videos 항목
    playlist: dict | None = None  # {id, title} — 지정 시 결과를 재생목록 폴더에 모음
    engine: str = convert.DEFAULT_ENGINE
    subs: str = "off"
    language: str = "ko"
    summarize: bool = False
    gpu: int = 0
    overwrite: bool = False


@app.post("/api/jobs/youtube")
def create_youtube_jobs(req: YoutubeJobRequest):
    _validate_options(req.engine, req.language, req.gpu)
    if req.subs not in youtube.SUBS_POLICIES:
        raise HTTPException(400, "잘못된 자막 옵션")
    subdir = None
    if req.playlist:
        pid = str(req.playlist.get("id", ""))
        if not re.fullmatch(r"[\w-]{2,64}", pid):
            raise HTTPException(400, "잘못된 재생목록 ID")
        # 폴더명도 클라이언트 값 대신 서버에서 계산 (경로 조작 방지)
        subdir = youtube.make_name(str(req.playlist.get("title") or ""), pid)
    videos = []
    for v in req.videos:
        url, title = str(v.get("url", "")), str(v.get("title") or "")
        m = re.fullmatch(r"https://www\.youtube\.com/watch\?v=([\w-]{6,20})", url)
        if not m:
            raise HTTPException(400, f"잘못된 영상 URL: {url}")
        videos.append({"id": m.group(1), "url": url, "title": title})
    if not videos:
        raise HTTPException(400, "선택한 영상이 없습니다")

    if len(videos) > 1:
        # 여러 영상은 작업 하나로 묶는다: 다운로드와 전사를 겹쳐 처리하고 모델은 한 번만 로드
        title = str((req.playlist or {}).get("title") or "YouTube")
        job = Job(
            id=uuid.uuid4().hex[:8],
            folder="youtube",
            filename=f"{title} — 영상 {len(videos)}개",
            videos=videos,
            subdir=subdir,
            subs=req.subs,
            engine=req.engine,
            language=req.language,
            summarize=req.summarize,
            gpu=req.gpu,
            overwrite=req.overwrite,
        )
        _enqueue(job)
        return {"created": [job.id]}

    created = []
    for v in videos:
        url, title = v["url"], v["title"]
        m = re.fullmatch(r"https://www\.youtube\.com/watch\?v=([\w-]{6,20})", url)
        job = Job(
            id=uuid.uuid4().hex[:8],
            folder="youtube",
            filename=title or m.group(1),
            name=youtube.make_name(title, m.group(1)),  # 클라이언트 값 대신 서버에서 다시 계산 (경로 조작 방지)
            subdir=subdir,
            url=url,
            subs=req.subs,
            engine=req.engine,
            language=req.language,
            summarize=req.summarize,
            gpu=req.gpu,
            overwrite=req.overwrite,
        )
        _enqueue(job)
        created.append(job.id)
    return {"created": created}


@app.get("/api/jobs")
def list_jobs():
    with _lock:
        return [_jobs[i].summary() for i in reversed(_job_order)]


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    return job.log


@app.delete("/api/jobs/{job_id}")
def cancel_or_remove(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    if job.status == "running":
        proc = _procs.get(job_id)
        job.status = "cancelled"
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return {"ok": True}
    with _lock:
        if job.status == "queued":
            job.status = "cancelled"
        else:
            _jobs.pop(job_id, None)
            _job_order.remove(job_id)
    return {"ok": True}


@app.get("/api/result/{kind}/{stem:path}")
def get_result(kind: str, stem: str, download: bool = False):
    if kind not in ("txt", "json", "md") or not _is_safe_rel(stem):
        raise HTTPException(400)
    path = _output_paths(stem)[kind]
    if not path.exists():
        raise HTTPException(404, "결과 파일이 없습니다")
    if download:
        return FileResponse(path, filename=path.name)
    return PlainTextResponse(path.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="mov-to-txt 웹 UI")
    parser.add_argument("--host", default="127.0.0.1", help="바인드 주소 (기본: 127.0.0.1, 외부 접속은 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    for d in (INPUT_DIR, AUDIO_DIR, TRANSCRIPT_DIR, SUMMARY_DIR):
        d.mkdir(parents=True, exist_ok=True)

    threading.Thread(target=_worker, daemon=True).start()

    print(f"mov-to-txt 웹 UI: http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
