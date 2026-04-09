# mov-to-txt

동영상/음성 파일을 텍스트 전사본 및 요약본으로 변환하는 로컬 파이프라인.

```
[영상/음성] → [MP3] → [TXT/JSON 전사본] → [MD 요약본 (선택)]
```

## 특징

- **오프라인 전사**: `moonshine-tiny` / `faster-whisper` 2가지 로컬 음성 인식 엔진 중 선택 (GPU 가속)
- **다국어 지원**: 한국어(`ko`) / 영어(`en`) 선택 가능
- **일괄 처리**: `--all-audio` 플래그로 `audio/` 폴더 전체 배치 전사
- **AI 요약**: `--summarize` 플래그로 Claude API를 통한 구조화된 한국어 요약 생성
- **스킵 로직**: 이미 생성된 파일은 건너뛰어 중복 작업 방지

## 음성 인식 엔진

`--engine` 플래그 또는 대화형 모드의 프롬프트로 선택합니다.

| 엔진 | 모델 | 특징 | 언어 |
|------|------|------|------|
| `moonshine` (기본값) | [Moonshine Tiny Korean](https://huggingface.co/UsefulSensors/moonshine-tiny-ko) / [Moonshine Tiny](https://huggingface.co/UsefulSensors/moonshine-tiny) | 경량, 빠른 속도 | ko / en 별도 모델 |
| `faster-whisper` | [Whisper large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | 고정밀, 다국어 지원 | ko / en 동일 모델 |

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
# input/ 폴더의 파일 전사 (한국어, moonshine 기본)
uv run convert.py lecture.mp4

# 영어 전사
uv run convert.py lecture.mp4 --language en

# faster-whisper 엔진 사용 (고정밀)
uv run convert.py lecture.mp4 --engine faster-whisper

# 전사 + Claude 요약
uv run convert.py lecture.mp4 --summarize
```

### 배치 처리 (`audio/` 폴더 전체)

```bash
uv run convert.py --all-audio
uv run convert.py --all-audio --engine faster-whisper --language ko
uv run convert.py --all-audio --summarize --language ko
```

## 환경 설정 (`.env`)

프로젝트 루트의 `.env` 파일로 실행 환경을 전환합니다.

### A환경 — 로컬 그래픽카드

```dotenv
ENV_MODE=local
ANTHROPIC_API_KEY=sk-ant-...   # --summarize 사용 시 필요
```

### B환경 — 서버 그래픽카드

```dotenv
ENV_MODE=server
ANTHROPIC_API_KEY=sk-ant-...   # --summarize 사용 시 필요
```

`ENV_MODE=server`로 설정하면 실행 시 **Step 0** 에서 사용할 GPU 인덱스(0~7)를 입력하는 프롬프트가 나타납니다.  
입력한 값은 `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=<인덱스>`로 자동 적용됩니다.

## 디렉토리 구조

```
mov-to-txt/
├── convert.py          # 메인 스크립트
├── config.py           # 환경 감지 및 GPU 설정
├── .env                # 환경 변수 (ENV_MODE, ANTHROPIC_API_KEY)
├── pyproject.toml      # 의존성 정의
├── uv.lock             # 잠금 파일 (재현 가능한 환경)
├── input/              # 원본 영상/음성 파일을 여기에 넣으세요
├── audio/              # Step 1 출력: 추출된 MP3
├── transcripts/        # Step 2 출력: TXT + JSON 전사본
└── summaries/          # Step 3 출력: MD 요약본
```

> `input/`, `audio/`, `transcripts/`, `summaries/` 폴더의 내용물은 `.gitignore`에 의해 추적되지 않습니다.

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
