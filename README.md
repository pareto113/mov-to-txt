# mov-to-txt

동영상/음성 파일을 텍스트 전사본 및 요약본으로 변환하는 로컬 파이프라인.

```
[영상/음성] → [MP3] → [TXT/JSON 전사본] → [MD 요약본 (선택)]
```

## 특징

- **오프라인 전사**: [Moonshine Tiny Korean](https://huggingface.co/UsefulSensors/moonshine-tiny-ko) 모델로 로컬에서 음성 인식 (GPU 가속)
- **다국어 지원**: 한국어(`ko`) / 영어(`en`) 선택 가능
- **일괄 처리**: `--all-audio` 플래그로 `audio/` 폴더 전체 배치 전사
- **AI 요약**: `--summarize` 플래그로 Claude API를 통한 구조화된 한국어 요약 생성
- **스킵 로직**: 이미 생성된 파일은 건너뛰어 중복 작업 방지

## 요구사항

| 항목 | 값 |
|------|-----|
| Python | 3.10 ~ 3.13 |
| 패키지 관리자 | [uv](https://docs.astral.sh/uv/) |
| GPU | CUDA 12.4 권장 (없으면 CPU로 fallback, 느림) |

## 설치

```bash
# uv 설치 (없는 경우)
pip install uv

# 의존성 설치
uv sync
```

## 사용법

### 대화형 모드 (파일/언어/요약 여부를 묻는 프롬프트)

```bash
uv run convert.py
```

### 파일 지정 실행

```bash
# input/ 폴더의 파일 전사 (한국어)
uv run convert.py lecture.mp4

# 영어 전사
uv run convert.py lecture.mp4 --language en

# 전사 + Claude 요약
uv run convert.py lecture.mp4 --summarize
```

### 배치 처리 (`audio/` 폴더 전체)

```bash
uv run convert.py --all-audio
uv run convert.py --all-audio --summarize --language ko
```

## 디렉토리 구조

```
mov-to-txt/
├── convert.py          # 메인 스크립트
├── pyproject.toml      # 의존성 정의
├── uv.lock             # 잠금 파일 (재현 가능한 환경)
├── input/              # 원본 영상/음성 파일을 여기에 넣으세요
├── audio/              # Step 1 출력: 추출된 MP3
├── transcripts/        # Step 2 출력: TXT + JSON 전사본
└── summaries/          # Step 3 출력: MD 요약본
```

> `input/`, `audio/`, `transcripts/`, `summaries/` 폴더의 내용물은 `.gitignore`에 의해 추적되지 않습니다.

## Claude API 요약 설정

`--summarize` 사용 시 `ANTHROPIC_API_KEY` 환경변수가 필요합니다.

```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-...

# macOS/Linux
export ANTHROPIC_API_KEY=sk-ant-...
```

## 출력 형식

### 전사본 (`.txt`)

```
# lecture 전사본

[00:00 --> 00:30]  안녕하세요, 오늘 발표할 내용은...
[00:30 --> 01:00]  첫 번째 주제는...
```

### 전사본 (`.json`)

```json
{
  "language": "ko",
  "segments": [
    { "start": 0.0, "end": 30.0, "text": "안녕하세요, 오늘 발표할 내용은..." }
  ]
}
```

### 요약본 (`.md`)

Claude API가 전사본을 바탕으로 핵심 주제, 주요 내용, 기술적 세부사항, 인사이트, Q&A를 구조화된 마크다운으로 작성합니다.
