"""
동영상/음성 -> 텍스트 전사 파이프라인

사용법:
    uv run convert.py <파일명> [--summarize] [--language {ko,en}] [--gpu N]
    uv run convert.py --url <YouTube URL> [--subs {off,manual,any}] [--summarize] ...

    파일은 input/ 폴더에 위치해야 합니다.
    영상 파일이면 오디오 추출 후 전사, 음성 파일이면 바로 전사합니다.
    YouTube URL(영상/재생목록)은 오디오를 내려받아 전사하거나, --subs 지정 시 영상 자막을 사용합니다.

단계:
    0. YouTube 오디오 다운로드 (--url인 경우에만, youtube.py)
    1. 영상 → MP3 오디오 추출 (영상 파일인 경우에만)
    2. 오디오 → TXT/JSON 전사 (faster-whisper large-v3 또는 Qwen3-ASR-1.7B + ForcedAligner)
    3. TXT → MD 요약 (Claude API, --summarize 플래그 시 실행)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import gpu_selector

import imageio_ffmpeg
import numpy as np
import torch

# ─────────────────────────────────────────────
# 설정 상수
# ─────────────────────────────────────────────

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

FFMPEG_AUDIO_OPTS = [
    "-vn",
    "-acodec", "libmp3lame",
    "-ar", "16000",
    "-ac", "1",
    "-b:a", "64k",
]

FASTER_WHISPER_MODEL = "large-v3"

QWEN_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
QWEN_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
QWEN_LANGUAGES = {"ko": "Korean", "en": "English"}
QWEN_CHUNK_MAX_SECONDS = 60     # 120초 이상이면 정렬기 타임스탬프가 끝부분에서 뭉개짐(실측)
QWEN_CHUNK_MIN_SECONDS = 40     # 이 구간~최대 길이 사이에서 가장 조용한 지점을 잘라 단어가 끊기지 않게 함
QWEN_BATCH_SIZE = 8             # 60초 x 8 배치 ≈ VRAM 10GB 내외
QWEN_LINE_MAX_SECONDS = 20      # 문장 부호가 없어도 이 길이를 넘으면 줄을 나눔
QWEN_LINE_GAP_SECONDS = 3.0     # 단어 사이 공백이 이보다 길면 줄을 나눔 (문장 중간 쉼은 보통 이보다 짧음)
QWEN_SECONDS_PER_CHAR = 0.25    # 타임스탬프 보간 시 글자당 발화 시간 추정치

# 음성 인식 엔진 목록. 새 엔진은 여기와 transcribe()에 추가한다.
# 첫 항목이 대화형 모드/웹 UI의 기본 선택이므로 DEFAULT_ENGINE을 맨 앞에 둔다.
ENGINES = {
    "qwen3-asr": {"label": "Qwen3-ASR 1.7B", "desc": "Qwen3-ASR-1.7B + ForcedAligner 단어 단위 타임스탬프"},
    "faster-whisper": {"label": f"faster-whisper {FASTER_WHISPER_MODEL}", "desc": "고정밀, 다국어 지원"},
}
DEFAULT_ENGINE = "qwen3-asr"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}

CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_CHARS = 700_000

SYSTEM_PROMPT = (
    "당신은 해당 동영상이 다루는 분야의 전문가이며 발표 내용을 정리하는 전문 요약가입니다.\n"
    "발표 전사본을 읽고 구조화된 한국어 요약본을 작성해주세요."
)

USER_PROMPT_TEMPLATE = """\
다음은 {name} 세션의 전사본입니다.
전사본에는 타임스탬프와 함께 발표자(주 발표자 + 소수의 진행자/청중 질문)의 발언이 담겨 있습니다.

전사본:
---
{transcript}
---

위 전사본을 바탕으로 다음 형식으로 요약해주세요:

## {name} 세션 요약

### 핵심 주제
(1-2문장으로 이 세션의 핵심을 서술)

### 주요 내용
(발표의 주요 내용을 논리적 흐름에 따라 bullet point로 정리, 5-10개)

### 기술적 세부사항
(언급된 기술 스택, 모델, 데이터, 성능 수치 등 구체적 사실 정리)

### 인상적인 인사이트 / 시사점
(기억할 만한 관점, 흥미로운 주장, 시험에 나올 만한 내용)

### 질의응답 주요 내용
(청중 질문과 답변 중 중요한 내용 정리, 없으면 생략)
"""

# ─────────────────────────────────────────────
# 공통 유틸리티
# ─────────────────────────────────────────────

def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _load_audio_raw(file: str, sr: int = 16000) -> np.ndarray:
    """ffmpeg로 오디오를 float32 numpy 배열로 읽어옴 (imageio-ffmpeg 바이너리 사용)."""
    cmd = [
        FFMPEG, "-nostdin", "-threads", "0",
        "-i", file,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr),
        "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


# ─────────────────────────────────────────────
# Step 1: 오디오 추출
# ─────────────────────────────────────────────

def extract_audio(video_path: Path, audio_dir: Path, name: str | None = None) -> Path:
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_path = audio_dir / ((name or video_path.stem) + ".mp3")

    if output_path.exists():
        print(f"[Step 1] 건너뜀 (이미 존재): {output_path}")
        return output_path

    print(f"[Step 1] 오디오 추출: {video_path} → {output_path}")
    cmd = [FFMPEG, "-y", "-i", str(video_path)] + FFMPEG_AUDIO_OPTS + [str(output_path)]
    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0:
        stderr_tail = result.stderr.decode("utf-8", errors="replace")[-500:]
        print(f"[Step 1] ffmpeg 오류:\n{stderr_tail}", file=sys.stderr)
        sys.exit(1)

    print(f"[Step 1] 완료: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Step 2: 음성 인식
# ─────────────────────────────────────────────

def _transcribe_faster_whisper(audio_path: Path, transcript_dir: Path, name: str, language: str = "ko") -> Path:
    from faster_whisper import WhisperModel

    transcript_dir.mkdir(parents=True, exist_ok=True)
    txt_path = transcript_dir / (name + ".txt")
    json_path = transcript_dir / (name + ".json")

    if txt_path.exists():
        print(f"[Step 2] 건너뜀 (이미 존재): {txt_path}")
        return txt_path

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    device_index = gpu_selector.index()
    compute_type = gpu_selector.compute_type()

    print(f"[Step 2] faster-whisper 모델 로드 중: {FASTER_WHISPER_MODEL} (cuda:{device_index}, {compute_type})" if cuda_available else f"[Step 2] faster-whisper 모델 로드 중: {FASTER_WHISPER_MODEL} (cpu, {compute_type})")
    model = WhisperModel(FASTER_WHISPER_MODEL, device=device, device_index=device_index, compute_type=compute_type)

    print(f"[Step 2] 전사 중: {audio_path}")
    fw_segments, info = model.transcribe(str(audio_path), language=language, beam_size=5)
    print(f"[Step 2] 감지 언어: {info.language} (확률 {info.language_probability:.2f})")

    segments = []
    lines = [f"# {name} 전사본\n"]
    total = 0

    for seg in fw_segments:
        start_fmt = _format_timestamp(seg.start)
        end_fmt = _format_timestamp(seg.end)
        text = seg.text.strip()
        if text:
            lines.append(f"[{start_fmt} --> {end_fmt}]  {text}")
            segments.append({"start": seg.start, "end": seg.end, "text": text})
            total += 1
        print(f"\r[Step 2] 세그먼트 {total}개 처리 중... ({start_fmt})", end="", flush=True)

    print()

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Step 2] TXT 저장: {txt_path}")

    json_data = {"language": language, "segments": segments}
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Step 2] JSON 저장: {json_path}")

    return txt_path


def _split_at_silence(audio: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """오디오를 최대 QWEN_CHUNK_MAX_SECONDS 길이로 나누되, 가장 조용한 지점에서 자른다."""
    frame = sr // 10  # 100ms
    n_frames = len(audio) // frame
    energy = np.sqrt(np.mean(audio[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1)) if n_frames else np.array([])

    bounds = []
    start = 0
    max_len, min_len = QWEN_CHUNK_MAX_SECONDS * sr, QWEN_CHUNK_MIN_SECONDS * sr
    while len(audio) - start > max_len:
        lo, hi = (start + min_len) // frame, (start + max_len) // frame
        cut = (lo + int(np.argmin(energy[lo:hi]))) * frame if hi > lo else start + max_len
        bounds.append((start, cut))
        start = cut
    bounds.append((start, len(audio)))
    return bounds


def _normalize_transcript(text: str) -> str:
    """문장 부호 뒤에 띄어쓰기가 빠진 경우(예: '니다.둘째') 공백을 넣는다. 소수점(3.5)은 유지."""
    import re
    return re.sub(r"([.?!])(?=[^\s\d.?!\"'”’)])", r"\1 ", text).strip()


def _repair_word_times(words: list[dict], chunk_dur: float) -> int:
    """정렬기가 위치를 못 잡은 단어(길이 0, 비정상적으로 긴 구간)의 타임스탬프를 보간한다.

    앞뒤로 정상 정렬된 단어 사이 구간을 글자 수 비례로 나눠 준다. 보정한 단어 수를 반환.
    """
    def bad(w):
        dur = w["end_time"] - w["start_time"]
        return dur <= 0 or dur > max(3.0, len(w["text"]) * 0.6)

    n, i, repaired = len(words), 0, 0
    while i < n:
        if not bad(words[i]):
            i += 1
            continue
        j = i
        while j < n and bad(words[j]):
            j += 1
        chars = [max(1, len(w["text"])) for w in words[i:j]]
        est = sum(chars) * QWEN_SECONDS_PER_CHAR
        left = words[i - 1]["end_time"] if i > 0 else None
        right = words[j]["start_time"] if j < n else None
        if left is None and right is None:
            left, right = 0.0, min(chunk_dur, est)
        elif left is None:
            left = max(0.0, right - est)
        elif right is None:
            right = min(chunk_dur, left + est)
        right = max(right, left)
        t = left
        for w, c in zip(words[i:j], chars):
            span = (right - left) * c / sum(chars)
            w["start_time"], w["end_time"] = round(t, 3), round(t + span, 3)
            t += span
        repaired += j - i
        i = j
    return repaired


def _group_words_into_lines(words: list[dict], text: str) -> list[dict]:
    """정렬기의 단어별 타임스탬프를 문장 단위 줄로 묶는다.

    정렬기 출력에는 문장 부호가 빠져 있으므로, 원문(text)에서 각 단어의 위치를 찾아
    문장 부호·원래 띄어쓰기를 살린 문자열로 줄을 만든다.
    """
    lines = []
    cursor = 0
    cur = None  # {"start", "end", "pos_start", "pos_end", "words"}

    def flush():
        nonlocal cur
        if cur:
            line_text = text[cur["pos_start"]:cur["pos_end"]].strip() or " ".join(w["text"] for w in cur["words"])
            lines.append({"start": cur["start"], "end": cur["end"], "text": line_text, "words": cur["words"]})
        cur = None

    for w in words:
        pos = text.find(w["text"], cursor)
        if pos < 0:  # 원문에서 못 찾으면 위치 추적 없이 이어붙임
            pos, end_pos = cursor, cursor
        else:
            end_pos = pos + len(w["text"])
            cursor = end_pos
        # 단어 바로 뒤의 문장 부호까지 줄에 포함
        while cursor < len(text) and text[cursor] in ".,?!。…\"'”’)":
            cursor += 1
        sentence_end = any(c in ".?!。…" for c in text[end_pos:cursor])

        if cur and (w["start_time"] - cur["end"] > QWEN_LINE_GAP_SECONDS
                    or w["end_time"] - cur["start"] > QWEN_LINE_MAX_SECONDS):
            flush()
        if cur is None:
            cur = {"start": w["start_time"], "end": w["end_time"], "pos_start": pos, "pos_end": cursor, "words": []}
        cur["end"] = max(cur["end"], w["end_time"])
        cur["pos_end"] = max(cur["pos_end"], cursor)
        cur["words"].append({"start": w["start_time"], "end": w["end_time"], "text": w["text"]})
        if sentence_end:
            flush()
    flush()
    return lines


def _transcribe_qwen(audio_path: Path, transcript_dir: Path, name: str, language: str = "ko") -> Path:
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration, Qwen3ASRForTokenClassification

    transcript_dir.mkdir(parents=True, exist_ok=True)
    txt_path = transcript_dir / (name + ".txt")
    json_path = transcript_dir / (name + ".json")

    if txt_path.exists():
        print(f"[Step 2] 건너뜀 (이미 존재): {txt_path}")
        return txt_path

    device = gpu_selector.device()
    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    lang_name = QWEN_LANGUAGES[language]

    print(f"[Step 2] Qwen3-ASR 모델 로드 중: {QWEN_ASR_MODEL} + {QWEN_ALIGNER_MODEL} ({device}, {str(dtype).replace('torch.', '')})")
    asr_proc = AutoProcessor.from_pretrained(QWEN_ASR_MODEL)
    asr = Qwen3ASRForConditionalGeneration.from_pretrained(QWEN_ASR_MODEL, dtype=dtype, device_map=device)
    al_proc = AutoProcessor.from_pretrained(QWEN_ALIGNER_MODEL)
    aligner = Qwen3ASRForTokenClassification.from_pretrained(QWEN_ALIGNER_MODEL, dtype=dtype, device_map=device)

    sr = asr_proc.feature_extractor.sampling_rate  # 16000
    print(f"[Step 2] 오디오 로드 중: {audio_path}")
    audio = _load_audio_raw(str(audio_path), sr=sr)
    bounds = _split_at_silence(audio, sr)
    n_batches = (len(bounds) + QWEN_BATCH_SIZE - 1) // QWEN_BATCH_SIZE
    print(f"[Step 2] 전사 중 (총 {_format_timestamp(len(audio) / sr)}, {len(bounds)}청크 / {n_batches}배치)...")

    segments = []
    total_words = total_repaired = 0
    for b in range(n_batches):
        batch = bounds[b * QWEN_BATCH_SIZE:(b + 1) * QWEN_BATCH_SIZE]
        chunks = [audio[s:e] for s, e in batch]
        print(f"\r[Step 2] 배치 {b + 1}/{n_batches} ({_format_timestamp(batch[0][0] / sr)})", end="", flush=True)

        # 1) 음성 인식 (배치)
        inputs = asr_proc.apply_transcription_request(audio=chunks, language=[lang_name] * len(chunks))
        inputs = inputs.to(asr.device, asr.dtype)
        with torch.inference_mode():
            out = asr.generate(**inputs, max_new_tokens=2048)
        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = [_normalize_transcript(p["transcription"]) for p in asr_proc.decode(gen, return_format="parsed")]

        # 2) 강제 정렬로 단어별 타임스탬프 (청크별, 실패 시 청크 단위 타임스탬프로 대체)
        for (s, e), chunk, text in zip(batch, chunks, texts):
            if not text:
                continue
            offset = s / sr
            try:
                ai, wl = al_proc.prepare_forced_aligner_inputs(audio=chunk, transcript=text, language=lang_name)
                ai = ai.to(aligner.device, aligner.dtype)
                with torch.inference_mode():
                    logits = aligner(**ai).logits
                words = al_proc.decode_forced_alignment(
                    logits=logits, input_ids=ai["input_ids"], word_lists=wl,
                    timestamp_token_id=aligner.config.timestamp_token_id,
                )[0]
                total_repaired += _repair_word_times(words, (e - s) / sr)
                total_words += len(words)
                lines = _group_words_into_lines(words, text)
            except Exception as ex:  # noqa: BLE001
                print(f"\n[Step 2] 경고: {_format_timestamp(offset)} 청크 정렬 실패 ({type(ex).__name__}: {ex}) — 청크 단위 타임스탬프 사용")
                lines = [{"start": 0.0, "end": (e - s) / sr, "text": text, "words": []}]

            for k, line in enumerate(lines):
                words_abs = [
                    {"start": round(offset + w["start"], 3), "end": round(offset + w["end"], 3), "text": w["text"]}
                    for w in line["words"]
                ]
                prev = segments[-1] if segments else None
                # 청크 경계에서 끊긴 문장은 앞 줄에 이어 붙인다
                if (k == 0 and prev and not prev["text"].rstrip().endswith((".", "?", "!", "。", "…"))
                        and offset + line["start"] - prev["end"] <= QWEN_LINE_GAP_SECONDS
                        and offset + line["end"] - prev["start"] <= QWEN_LINE_MAX_SECONDS * 1.5):
                    prev["text"] = f"{prev['text']} {line['text']}"
                    prev["end"] = round(offset + line["end"], 3)
                    prev["words"].extend(words_abs)
                    continue
                segments.append({
                    "start": round(offset + line["start"], 3),
                    "end": round(offset + line["end"], 3),
                    "text": line["text"],
                    "words": words_abs,
                })

    print()
    if total_repaired:
        print(f"[Step 2] 정렬기가 위치를 못 잡은 단어 {total_repaired}/{total_words}개는 앞뒤 단어 기준으로 타임스탬프를 보간했습니다.")

    lines_out = [f"# {name} 전사본\n"]
    for seg in segments:
        lines_out.append(f"[{_format_timestamp(seg['start'])} --> {_format_timestamp(seg['end'])}]  {seg['text']}")
    txt_path.write_text("\n".join(lines_out), encoding="utf-8")
    print(f"[Step 2] TXT 저장: {txt_path}")

    json_data = {"language": language, "engine": "qwen3-asr", "segments": segments}
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Step 2] JSON 저장: {json_path}")

    return txt_path


def transcribe(audio_path: Path, transcript_dir: Path, name: str, language: str = "ko", engine: str = DEFAULT_ENGINE) -> Path:
    import time
    t0 = time.perf_counter()
    if engine == "faster-whisper":
        result = _transcribe_faster_whisper(audio_path, transcript_dir, name, language)
    elif engine == "qwen3-asr":
        result = _transcribe_qwen(audio_path, transcript_dir, name, language)
    else:
        raise ValueError(f"알 수 없는 엔진: {engine}")
    elapsed = time.perf_counter() - t0
    m, s = divmod(int(elapsed), 60)
    print(f"[Step 2] 소요시간: {m}분 {s}초 ({elapsed:.1f}초)")
    return result


# ─────────────────────────────────────────────
# Step 3 (optional): Claude API 요약
# ─────────────────────────────────────────────

def summarize(txt_path: Path, summary_dir: Path, name: str) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    md_path = summary_dir / (name + ".md")

    if md_path.exists():
        print(f"[Step 3] 건너뜀 (이미 존재): {md_path}")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[Step 3] 오류: ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.", file=sys.stderr)
        sys.exit(1)

    import anthropic

    transcript = txt_path.read_text(encoding="utf-8")
    if len(transcript) > MAX_CHARS:
        print(f"[Step 3] 전사본이 {MAX_CHARS}자를 초과하여 잘라냅니다.")
        transcript = transcript[:MAX_CHARS]

    prompt = USER_PROMPT_TEMPLATE.format(name=name, transcript=transcript)

    print(f"[Step 3] Claude API 호출 중 (모델: {CLAUDE_MODEL})")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    summary_text = message.content[0].text

    md_path.write_text(summary_text, encoding="utf-8")
    print(f"[Step 3] 요약 저장: {md_path}")


# ─────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────

def _prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)


def _pick_input_file(input_dir: Path, audio_dir: Path) -> Path:
    """input/ 및 audio/ 폴더의 파일 목록을 출력하고 사용자가 선택하게 함."""
    input_files = sorted(f for f in input_dir.iterdir() if f.is_file()) if input_dir.exists() else []
    audio_files = sorted(f for f in audio_dir.iterdir() if f.is_file()) if audio_dir.exists() else []
    all_files: list[Path] = []

    if input_files:
        print("\ninput/ 폴더의 파일 목록:")
        for f in input_files:
            print(f"  [{len(all_files) + 1}] {f.name}")
            all_files.append(f)

    if audio_files:
        print("\naudio/ 폴더의 파일 목록:")
        for f in audio_files:
            print(f"  [{len(all_files) + 1}] {f.name}")
            all_files.append(f)

    if not all_files:
        print("오류: input/ 또는 audio/ 폴더에 파일이 없습니다.", file=sys.stderr)
        sys.exit(1)

    print()
    while True:
        choice = _prompt("번호 또는 파일명 입력: ")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(all_files):
                return all_files[idx]
            print(f"  1 ~ {len(all_files)} 사이의 번호를 입력하세요.")
        else:
            matched = [f for f in all_files if f.name == choice]
            if matched:
                return matched[0]
            print(f"  '{choice}' 파일을 찾을 수 없습니다.")


def _pick_engine() -> str:
    ids = list(ENGINES)
    if len(ids) == 1:
        print(f"\n모델: {ENGINES[ids[0]]['label']} — {ENGINES[ids[0]]['desc']}")
        return ids[0]
    print("\n모델 선택:")
    for i, eid in enumerate(ids, 1):
        print(f"  [{i}] {ENGINES[eid]['label']} — {ENGINES[eid]['desc']}")
    while True:
        choice = _prompt("번호 입력 (Enter = 1): ")
        if choice == "":
            return ids[0]
        if choice.isdigit() and 1 <= int(choice) <= len(ids):
            return ids[int(choice) - 1]
        if choice in ENGINES:
            return choice
        print(f"  1 ~ {len(ids)} 사이의 번호를 입력하세요.")


def _pick_language() -> str:
    print("\n언어 선택:")
    print("  [1] ko — 한국어 (기본값)")
    print("  [2] en — 영어")
    while True:
        choice = _prompt("번호 또는 언어 코드 입력 (Enter = ko): ")
        if choice in ("", "1", "ko"):
            return "ko"
        if choice in ("2", "en"):
            return "en"
        print("  1 또는 2를 입력하세요.")


def _pick_summarize() -> bool:
    choice = _prompt("\nClaude API로 요약본 생성? [y/N]: ")
    return choice.lower() in ("y", "yes")


def main() -> None:
    parser = argparse.ArgumentParser(description="동영상/음성 → 텍스트 전사 파이프라인")
    parser.add_argument("file", nargs="?", help="변환할 파일명 (생략 시 input/audio/ 목록에서 선택)")
    parser.add_argument(
        "--all-audio",
        action="store_true",
        default=False,
        help="audio/ 폴더의 모든 오디오 파일을 일괄 전사",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        default=False,
        help="Claude API를 사용하여 요약본 생성 (ANTHROPIC_API_KEY 필요)",
    )
    parser.add_argument(
        "--language",
        choices=["ko", "en"],
        default="ko",
        help="전사 언어 (ko: 한국어, en: 영어, 기본값: ko)",
    )
    parser.add_argument(
        "--engine",
        choices=list(ENGINES),
        default=DEFAULT_ENGINE,
        help=f"음성 인식 엔진 (기본값: {DEFAULT_ENGINE})",
    )
    parser.add_argument("--url", help="YouTube 영상 또는 재생목록 URL")
    parser.add_argument(
        "--subs",
        choices=["off", "manual", "any"],
        default="off",
        help="--url에서 영상 자막 사용: off=음성 인식(기본), manual=업로더 자막 우선, any=자동 생성 자막도 허용. "
             "해당 언어 자막이 없으면 음성 인식으로 대체",
    )
    parser.add_argument("--name", help="--url 결과 파일명(확장자 제외). 생략 시 '영상 제목 [영상ID]'")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="사용할 GPU 인덱스 (지정 시 ENV_MODE=server의 대화형 GPU 선택을 건너뜀)",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        gpu_selector.set_index(args.gpu)
    else:
        gpu_selector.setup()

    script_dir = Path(__file__).parent
    input_dir = script_dir / "input"
    audio_dir = script_dir / "audio"
    transcript_dir = script_dir / "transcripts"
    summary_dir = script_dir / "summaries"

    # --url: YouTube 영상/재생목록
    if args.url:
        run_youtube(args.url, language=args.language, engine=args.engine, subs=args.subs,
                    summarize_flag=args.summarize, name=args.name)
        print("\n완료.")
        return

    # --all-audio: audio/ 폴더 전체 배치 처리
    if args.all_audio:
        batch_files = sorted(
            f for f in audio_dir.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        ) if audio_dir.exists() else []
        if not batch_files:
            print("오류: audio/ 폴더에 오디오 파일이 없습니다.", file=sys.stderr)
            sys.exit(1)
        print(f"[배치] audio/ 폴더에서 {len(batch_files)}개 파일을 전사합니다.")
        for audio_file in batch_files:
            print(f"\n[배치] 처리 중: {audio_file.name}")
            txt_path = transcribe(audio_file, transcript_dir, audio_file.stem, language=args.language, engine=args.engine)
            if args.summarize:
                summarize(txt_path, summary_dir, audio_file.stem)
        print("\n모든 파일 완료.")
        return

    # 파일 결정
    if args.file is None:
        input_path = _pick_input_file(input_dir, audio_dir)
        args.engine = _pick_engine()
        args.language = _pick_language()
        args.summarize = _pick_summarize()
    else:
        input_path = Path(args.file)
        if not input_path.exists():
            # input/ 폴더에서 찾기
            candidate = input_dir / args.file
            if candidate.exists():
                input_path = candidate
            else:
                # audio/ 폴더에서 찾기
                candidate = audio_dir / args.file
                if candidate.exists():
                    input_path = candidate
                else:
                    print(f"오류: 파일을 찾을 수 없습니다: {args.file}", file=sys.stderr)
                    sys.exit(1)

    run_pipeline(input_path, language=args.language, engine=args.engine, summarize_flag=args.summarize)
    print("\n완료.")


def run_pipeline(
    input_path: Path,
    language: str = "ko",
    engine: str = DEFAULT_ENGINE,
    summarize_flag: bool = False,
    base_dir: Path | None = None,
) -> Path:
    """파일 1개에 대해 Step 1~3을 실행하고 전사본 경로를 반환한다 (CLI/웹 공용)."""
    base_dir = base_dir or Path(__file__).parent
    audio_dir = base_dir / "audio"
    transcript_dir = base_dir / "transcripts"
    summary_dir = base_dir / "summaries"

    name = input_path.stem
    is_audio = input_path.suffix.lower() in AUDIO_EXTENSIONS

    if is_audio:
        print(f"[Step 1] 음성 파일 감지 — 오디오 추출 건너뜀: {input_path.name}")
        audio_path = input_path
    else:
        print(f"[Step 1] 영상 파일 감지: {input_path.name}")
        audio_path = extract_audio(input_path, audio_dir)

    # Step 2
    txt_path = transcribe(audio_path, transcript_dir, name, language=language, engine=engine)

    # Step 3 (optional)
    if summarize_flag:
        summarize(txt_path, summary_dir, name)

    return txt_path


def run_youtube(
    url: str,
    language: str = "ko",
    engine: str = DEFAULT_ENGINE,
    subs: str = "off",
    summarize_flag: bool = False,
    name: str | None = None,
    base_dir: Path | None = None,
) -> list[Path]:
    """YouTube 영상(또는 재생목록의 각 영상)을 전사하고 전사본 경로 목록을 반환한다."""
    import shutil
    import tempfile
    import youtube

    base_dir = base_dir or Path(__file__).parent
    audio_dir = base_dir / "audio"
    transcript_dir = base_dir / "transcripts"
    summary_dir = base_dir / "summaries"

    print(f"[Step 0] YouTube 정보 조회: {url}")
    info = youtube.fetch_info(url)
    if info.get("_type") == "playlist":
        videos = youtube.playlist_videos(info)
        if not videos:
            print("[Step 0] 오류: 재생목록에 영상이 없습니다.", file=sys.stderr)
            sys.exit(1)
        print(f"[Step 0] 재생목록 '{info.get('title')}' — {len(videos)}개 영상")
        name = None  # 재생목록은 영상별 이름 사용
    else:
        videos = [{"id": info["id"], "url": youtube.video_url(info["id"]), "title": info.get("title")}]

    results, failed = [], []
    for i, v in enumerate(videos, 1):
        if len(videos) > 1:
            print(f"\n[재생목록 {i}/{len(videos)}] {v['title']}")
        try:
            vinfo = info if len(videos) == 1 and info.get("_type") != "playlist" else youtube.fetch_video(v["url"])
            vname = name or youtube.make_name(vinfo.get("title"), vinfo["id"])
            txt_path = transcript_dir / f"{vname}.txt"
            dur = vinfo.get("duration")
            print(f"[Step 0] 영상: {vinfo.get('title')} ({vinfo.get('uploader') or '-'}"
                  f"{', ' + _format_timestamp(dur) if dur else ''}) → {vname}")

            if txt_path.exists():
                print(f"[Step 2] 건너뜀 (이미 존재): {txt_path}")
            else:
                track = youtube.pick_caption(vinfo, language, subs)
                if track:
                    print(f"[Step 2] YouTube 자막 사용: [{youtube.CAPTION_LABELS[track['kind']]}] {track['key']} ({track['name']})")
                    txt_path = youtube.save_captions(vinfo, track, transcript_dir, vname, language)
                else:
                    if subs != "off":
                        need = "업로더 자막" if subs == "manual" else "자막"
                        print(f"[Step 2] '{language}' {need}이 없어 음성 인식({ENGINES[engine]['label']})으로 전사합니다.")
                    audio_path = audio_dir / f"{vname}.mp3"
                    if audio_path.exists():
                        print(f"[Step 0] 건너뜀 (오디오 이미 존재): {audio_path}")
                    else:
                        tmp = Path(tempfile.mkdtemp(prefix=".yt-", dir=audio_dir))
                        try:
                            raw = youtube.download_audio(vinfo, tmp)
                            audio_path = extract_audio(raw, audio_dir, name=vname)
                        finally:
                            shutil.rmtree(tmp, ignore_errors=True)
                    txt_path = transcribe(audio_path, transcript_dir, vname, language=language, engine=engine)
                    youtube.tag_transcript(transcript_dir / f"{vname}.json", vinfo)

            if summarize_flag:
                summarize(txt_path, summary_dir, vname)
            results.append(txt_path)
        except Exception as e:  # noqa: BLE001 — 재생목록에서는 한 영상이 실패해도 계속 진행
            if len(videos) == 1:
                raise
            print(f"\n[오류] {v['title']}: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(v["title"])

    if failed:
        print(f"\n[재생목록] {len(failed)}개 실패: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)
    return results


if __name__ == "__main__":
    main()
