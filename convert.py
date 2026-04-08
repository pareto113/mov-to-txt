"""
동영상/음성 -> 텍스트 전사 파이프라인

사용법:
    uv run convert.py <파일명> [--summarize] [--language {ko,en}]

    파일은 input/ 폴더에 위치해야 합니다.
    영상 파일이면 오디오 추출 후 전사, 음성 파일이면 바로 전사합니다.

단계:
    1. 영상 → MP3 오디오 추출 (영상 파일인 경우에만)
    2. 오디오 → TXT/JSON 전사 (ko: Moonshine Tiny Korean, en: Moonshine Tiny)
    3. TXT → MD 요약 (Claude API, --summarize 플래그 시 실행)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from transformers import AutoProcessor, MoonshineForConditionalGeneration

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

MOONSHINE_MODELS = {
    "ko": "UsefulSensors/moonshine-tiny-ko",
    "en": "UsefulSensors/moonshine-tiny",
}
MOONSHINE_CHUNK_SECONDS = 30

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
(기억할 만한 관점, 흥미로운 주장, 업계 트렌드 관련 내용)

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

def extract_audio(video_path: Path, audio_dir: Path) -> Path:
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_path = audio_dir / (video_path.stem + ".mp3")

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
# Step 2: 음성 인식 — Moonshine Tiny Korean
# ─────────────────────────────────────────────

def transcribe(audio_path: Path, transcript_dir: Path, name: str, language: str = "ko") -> Path:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    txt_path = transcript_dir / (name + ".txt")
    json_path = transcript_dir / (name + ".json")

    if txt_path.exists():
        print(f"[Step 2] 건너뜀 (이미 존재): {txt_path}")
        return txt_path

    model_id = MOONSHINE_MODELS[language]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[Step 2] Moonshine 모델 로드 중: {model_id}")
    model = MoonshineForConditionalGeneration.from_pretrained(model_id).to(device).to(torch_dtype)
    processor = AutoProcessor.from_pretrained(model_id)

    sr = processor.feature_extractor.sampling_rate  # 16000
    print(f"[Step 2] 오디오 로드 중: {audio_path}")
    audio = _load_audio_raw(str(audio_path), sr=sr)

    chunk_size = sr * MOONSHINE_CHUNK_SECONDS
    total_chunks = (len(audio) + chunk_size - 1) // chunk_size
    segments = []
    lines = [f"# {name} 전사본\n"]

    print(f"[Step 2] 전사 중 (총 {total_chunks}청크, {MOONSHINE_CHUNK_SECONDS}초 단위)...")
    for idx, offset in enumerate(range(0, len(audio), chunk_size)):
        chunk = audio[offset : offset + chunk_size]
        chunk_start = offset / sr
        chunk_end = min((offset + chunk_size) / sr, len(audio) / sr)

        print(f"\r[Step 2] 청크 {idx + 1}/{total_chunks} ({_format_timestamp(chunk_start)})", end="", flush=True)

        inputs = processor(chunk, return_tensors="pt", sampling_rate=sr)
        inputs = {
            k: v.to(device=device, dtype=torch_dtype) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = model.generate(**inputs)

        text = processor.decode(generated_ids[0], skip_special_tokens=True).strip()

        if text:
            start_fmt = _format_timestamp(chunk_start)
            end_fmt = _format_timestamp(chunk_end)
            lines.append(f"[{start_fmt} --> {end_fmt}]  {text}")
            segments.append({"start": chunk_start, "end": chunk_end, "text": text})

    print()

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Step 2] TXT 저장: {txt_path}")

    json_data = {"language": language, "segments": segments}
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Step 2] JSON 저장: {json_path}")

    return txt_path


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
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_dir = script_dir / "input"
    audio_dir = script_dir / "audio"
    transcript_dir = script_dir / "transcripts"
    summary_dir = script_dir / "summaries"

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
            txt_path = transcribe(audio_file, transcript_dir, audio_file.stem, language=args.language)
            if args.summarize:
                summarize(txt_path, summary_dir, audio_file.stem)
        print("\n모든 파일 완료.")
        return

    # 파일 결정
    if args.file is None:
        input_path = _pick_input_file(input_dir, audio_dir)
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

    name = input_path.stem
    is_audio = input_path.suffix.lower() in AUDIO_EXTENSIONS

    if is_audio:
        print(f"[Step 1] 음성 파일 감지 — 오디오 추출 건너뜀: {input_path.name}")
        audio_path = input_path
    else:
        print(f"[Step 1] 영상 파일 감지: {input_path.name}")
        audio_path = extract_audio(input_path, audio_dir)

    # Step 2
    txt_path = transcribe(audio_path, transcript_dir, name, language=args.language)

    # Step 3 (optional)
    if args.summarize:
        summarize(txt_path, summary_dir, name)

    print("\n완료.")


if __name__ == "__main__":
    main()
