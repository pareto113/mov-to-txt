"""
YouTube 영상 → 오디오/자막 가져오기 (yt-dlp)

convert.py의 --url 옵션과 web.py의 YouTube 패널에서 사용한다.

- 오디오: 가장 좋은 오디오 스트림만 내려받아 임시 폴더에 저장한다.
  MP3 변환은 convert.extract_audio()가 기존 영상 처리와 동일한 설정으로 수행한다.
- 자막: 영상에 붙은 자막을 json3 형식으로 받아 전사본(TXT/JSON)으로 저장한다.
  업로더가 올린 자막(manual)과 YouTube 자동 생성 자막(auto)을 구분해 표기하며,
  다른 언어에서 기계 번역된 자동 자막은 실제 발화가 아니므로 전사본으로 쓰지 않는다.

YouTube 추출에는 JS 런타임이 필요해 PyPI의 deno 패키지를 사용한다(시스템 설치 불필요).
YouTube 쪽 변경으로 다운로드가 실패하면 `uv lock --upgrade-package yt-dlp && uv sync`로 업데이트한다.
"""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import imageio_ffmpeg
import yt_dlp
from yt_dlp.utils import sanitize_filename

YOUTUBE_HOSTS = ("youtube.com", "youtu.be")
NAME_MAX_TITLE = 80

# 자막 사용 정책 (CLI --subs / 웹 UI)
SUBS_POLICIES = {
    "off": "자막 사용 안 함 (음성 인식)",
    "manual": "업로더 자막 우선 (없으면 음성 인식)",
    "any": "자막 우선, 자동 생성 포함 (없으면 음성 인식)",
}
CAPTION_LABELS = {"manual": "업로더", "auto": "자동 생성"}

# 자동 생성 자막은 몇 단어씩 끊겨 있어 읽기 좋게 이어 붙인다
AUTO_LINE_MAX_SECONDS = 15
AUTO_LINE_GAP_SECONDS = 1.5

# 자막으로 만든 전사본 TXT 두 번째 줄. 웹 UI가 이 줄로 자막 종류를 판별한다.
CAPTION_HEADER_RE = re.compile(r"^# 출처: YouTube 자막 \[(업로더|자동 생성)\]")


# ─────────────────────────────────────────────
# 공통
# ─────────────────────────────────────────────

def is_youtube_url(url: str) -> bool:
    try:
        p = urlparse(url.strip())
    except ValueError:
        return False
    host = (p.hostname or "").lower()
    return p.scheme in ("http", "https") and any(host == h or host.endswith("." + h) for h in YOUTUBE_HOSTS)


def _base_opts() -> dict:
    opts = {
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,  # watch?v=...&list=... 는 영상 하나로 취급, 재생목록 URL만 펼침
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
    }
    try:
        import deno
        opts["js_runtimes"] = {"deno": {"path": deno.find_deno_bin()}}
    except Exception:  # noqa: BLE001 — 없으면 yt-dlp 기본값(시스템 deno)에 맡김
        pass
    return opts


def video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def is_safe_folder(folder: str) -> bool:
    """하위 폴더 이름이 한 단계짜리 안전한 이름인지 (경로 조작 방지)."""
    return bool(folder) and "/" not in folder and "\\" not in folder and not folder.startswith(".") and len(folder) <= 200


def make_name(title: str | None, video_id: str) -> str:
    """결과 파일명(확장자 제외)이나 재생목록 폴더명. 같은 대상은 항상 같은 이름이 되도록 ID를 붙인다."""
    title = sanitize_filename(title or "", restricted=False).strip(" .") or "youtube"
    if len(title) > NAME_MAX_TITLE:
        title = title[:NAME_MAX_TITLE].rstrip() + "…"
    return f"{title} [{video_id}]"


# ─────────────────────────────────────────────
# 정보 조회
# ─────────────────────────────────────────────

def fetch_info(url: str) -> dict:
    """영상이면 전체 정보, 재생목록이면 항목 목록(가벼운 정보)만 가져온다."""
    if not is_youtube_url(url):
        raise ValueError("YouTube 영상 또는 재생목록 URL이 아닙니다")
    opts = _base_opts() | {"skip_download": True, "extract_flat": "in_playlist"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url.strip(), download=False)


def fetch_video(url: str) -> dict:
    opts = _base_opts() | {"skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def playlist_videos(info: dict) -> list[dict]:
    """재생목록 info에서 영상 항목만 골라 [{id, url, title, duration}]로 반환."""
    videos = []
    for e in info.get("entries") or []:
        if not e or not e.get("id") or e.get("ie_key") not in (None, "Youtube"):
            continue  # 채널 탭 같은 하위 재생목록은 제외
        videos.append({
            "id": e["id"],
            "url": video_url(e["id"]),
            "title": e.get("title") or e["id"],
            "duration": e.get("duration"),
        })
    return videos


def caption_tracks(info: dict) -> list[dict]:
    """영상의 자막 트랙 목록.

    kind: manual(업로더가 올린 자막) | auto(YouTube 자동 생성)
    translated: 원래 발화 언어가 아닌 번역 자막인지. 번역 자막은 전사본으로 쓰지 않는다.
    """
    vlang = (info.get("language") or "").split("-")[0] or None
    tracks = []
    for key, fmts in (info.get("subtitles") or {}).items():
        if key == "live_chat" or not any(f.get("ext") == "json3" for f in fmts):
            continue
        lang = key.split("-")[0]
        tracks.append({"key": key, "lang": lang, "kind": "manual", "translated": bool(vlang and lang != vlang),
                       "name": fmts[0].get("name") or key})

    auto = info.get("automatic_captions") or {}
    has_orig = any(k.endswith("-orig") for k in auto)
    for key, fmts in auto.items():
        if not any(f.get("ext") == "json3" for f in fmts):
            continue
        if key.endswith("-orig"):
            lang, translated = key[:-5].split("-")[0], False
        else:
            lang = key.split("-")[0]
            if has_orig and f"{key}-orig" in auto:
                continue  # 같은 언어의 원본(-orig)과 중복
            # -orig 표시가 없으면 영상 언어와 같은 트랙만 원본으로 본다 (언어를 모르면 판단 불가 → 번역 취급)
            translated = has_orig or not vlang or lang != vlang
        tracks.append({"key": key, "lang": lang, "kind": "auto", "translated": translated,
                       "name": fmts[0].get("name") or key})
    return tracks


def pick_caption(info: dict, language: str, subs: str) -> dict | None:
    """자막 정책에 따라 사용할 트랙을 고른다. 없으면 None (→ 음성 인식)."""
    if subs == "off":
        return None
    cands = [t for t in caption_tracks(info) if t["lang"] == language and not t["translated"]]
    manual = sorted((t for t in cands if t["kind"] == "manual"), key=lambda t: t["key"] != language)
    if manual:
        return manual[0]
    if subs == "any":
        auto = [t for t in cands if t["kind"] == "auto"]
        if auto:
            return auto[0]
    return None


def caption_choices(info: dict) -> dict:
    """웹 UI용: 언어별로 정책에 따라 어떤 자막이 쓰일지 미리 계산."""
    out = {}
    for lang in ("ko", "en"):
        m = pick_caption(info, lang, "manual")
        a = pick_caption(info, lang, "any")
        out[lang] = {"manual": m and m["kind"], "any": a and a["kind"]}
    return out


# ─────────────────────────────────────────────
# 오디오 다운로드
# ─────────────────────────────────────────────

def download_audio(info: dict, dest_dir: Path) -> Path:
    """오디오 스트림만 dest_dir에 내려받고 파일 경로를 반환한다 (MP3 변환은 호출 측에서)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    last = {"pct": -1}

    def hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                pct = int(d["downloaded_bytes"] * 100 / total)
                if pct != last["pct"]:
                    last["pct"] = pct
                    print(f"\r[Step 0] 오디오 다운로드 중... {pct}% ({total / 1048576:.1f}MB)", end="", flush=True)
        elif d["status"] == "finished":
            print()

    # 조회 때 받은 스트림 URL은 다른 세션에서 쓰면 403이 나므로 다운로드 시 새로 추출한다.
    # YouTube는 같은 포맷도 간헐적으로 403을 주므로, 실패하면 새로 추출해 포맷을 바꿔가며 재시도한다.
    formats = ["bestaudio/best", "bestaudio[ext=m4a]/bestaudio/best", "bestaudio/best"]
    for attempt, fmt in enumerate(formats, 1):
        opts = _base_opts() | {
            "format": fmt,
            "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
            "progress_hooks": [hook],
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                result = ydl.extract_info(video_url(info["id"]), download=True)
            return Path(result["requested_downloads"][0]["filepath"])
        except yt_dlp.utils.DownloadError as e:
            if "403" not in str(e) or attempt == len(formats):
                raise
            print(f"\n[Step 0] 다운로드 거부(403) — 다시 시도합니다 ({attempt}/{len(formats) - 1})")
            for p in dest_dir.iterdir():  # 받다 만 조각 정리
                p.unlink(missing_ok=True)
    raise AssertionError("unreachable")


# ─────────────────────────────────────────────
# 자막 → 전사본
# ─────────────────────────────────────────────

def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _parse_json3(data: dict, auto: bool) -> list[dict]:
    events = [e for e in data.get("events", []) if e.get("segs") and not e.get("aAppend")]
    lines = []
    for i, e in enumerate(events):
        start = e.get("tStartMs", 0) / 1000
        end = start + e.get("dDurationMs", 0) / 1000
        if auto and i + 1 < len(events):  # 자동 자막은 다음 줄과 화면에 겹쳐 표시되므로 다음 줄 시작에서 자른다
            end = max(start, min(end, events[i + 1].get("tStartMs", 0) / 1000))
        text = re.sub(r"\s+", " ", "".join(s.get("utf8", "") for s in e["segs"])).strip()
        if not text:
            continue
        line = {"start": round(start, 3), "end": round(end, 3), "text": text}
        if auto:  # 자동 자막은 단어별 시작 시각(tOffsetMs)이 있다
            ws = [(start + s.get("tOffsetMs", 0) / 1000, s["utf8"].strip()) for s in e["segs"] if s.get("utf8", "").strip()]
            line["words"] = [
                {"start": round(t, 3), "end": round(max(t, ws[k + 1][0] if k + 1 < len(ws) else end), 3), "text": w}
                for k, (t, w) in enumerate(ws)
            ]
        lines.append(line)

    if not auto:
        return lines
    merged = []
    for line in lines:
        prev = merged[-1] if merged else None
        if prev and line["start"] - prev["end"] <= AUTO_LINE_GAP_SECONDS and line["end"] - prev["start"] <= AUTO_LINE_MAX_SECONDS:
            prev["text"] += " " + line["text"]
            prev["end"] = line["end"]
            prev["words"] += line["words"]
        else:
            merged.append(line)
    return merged


def save_captions(info: dict, track: dict, transcript_dir: Path, name: str, language: str) -> Path:
    fmt = next(f for f in (info["subtitles"] if track["kind"] == "manual" else info["automatic_captions"])[track["key"]]
               if f.get("ext") == "json3")
    with yt_dlp.YoutubeDL(_base_opts()) as ydl:
        data = json.loads(ydl.urlopen(fmt["url"]).read().decode("utf-8"))
    segments = _parse_json3(data, auto=track["kind"] == "auto")
    if not segments:
        raise RuntimeError(f"자막 트랙({track['key']})이 비어 있습니다")

    transcript_dir.mkdir(parents=True, exist_ok=True)
    txt_path = transcript_dir / f"{name}.txt"
    json_path = transcript_dir / f"{name}.json"

    label = CAPTION_LABELS[track["kind"]]
    lines = [
        f"# {name} 전사본",
        f"# 출처: YouTube 자막 [{label}] {track['key']} · {video_url(info['id'])}",
        "",
    ]
    lines += [f"[{_format_timestamp(s['start'])} --> {_format_timestamp(s['end'])}]  {s['text']}" for s in segments]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Step 2] TXT 저장: {txt_path}")

    json_data = {"language": language, "engine": f"youtube-captions-{track['kind']}",
                 "source": source_meta(info, track), "segments": segments}
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Step 2] JSON 저장: {json_path}")
    return txt_path


def source_meta(info: dict, track: dict | None) -> dict:
    return {
        "type": "youtube",
        "id": info["id"],
        "url": video_url(info["id"]),
        "title": info.get("title"),
        "uploader": info.get("uploader") or info.get("channel"),
        "caption": track and track["kind"],      # manual | auto | None(음성 인식)
        "caption_track": track and track["key"],
    }


def tag_transcript(json_path: Path, info: dict) -> None:
    """음성 인식으로 만든 전사본 JSON에 YouTube 출처 정보를 덧붙인다."""
    if not json_path.exists():
        return
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if data.get("source"):
        return
    data["source"] = source_meta(info, None)
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_caption_kind(txt_path: Path) -> str | None:
    """전사본 TXT 머리말에서 자막 종류(manual/auto)를 읽는다. 자막으로 만든 게 아니면 None."""
    try:
        with txt_path.open(encoding="utf-8") as f:
            f.readline()
            m = CAPTION_HEADER_RE.match(f.readline())
    except OSError:
        return None
    if not m:
        return None
    return {v: k for k, v in CAPTION_LABELS.items()}[m.group(1)]
