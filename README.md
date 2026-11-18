# mov-to-txt

동영상/음성 파일을 텍스트 전사본 및 요약본으로 변환하는 로컬 파이프라인.

```
[영상/음성] → [MP3] → [TXT/JSON 전사본] → [MD 요약본 (선택)]
```

## 특징

- **오프라인 전사**: `Qwen3-ASR`(기본) / `faster-whisper` 로컬 음성 인식 엔진 (GPU 가속)
- **다국어 지원**: 한국어(`ko`) / 영어(`en`) 선택 가능
- **일괄 처리**: `--all-audio` 플래그로 `audio/` 폴더 전체 배치 전사
- **YouTube 지원**: 영상/재생목록 URL로 바로 전사, 업로더 자막·자동 생성 자막을 구분해 선택적으로 활용
- **AI 요약**: `--summarize` 플래그로 Claude API를 통한 구조화된 한국어 요약 생성
- **스킵 로직**: 이미 생성된 파일은 건너뛰어 중복 작업 방지

## 음성 인식 엔진

`--engine` 플래그 또는 대화형 모드의 프롬프트로 선택합니다.

| 엔진 | 모델 | 특징 | 언어 |
|------|------|------|------|
| `qwen3-asr` (기본값) | [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) + [ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf) | 단어 단위 타임스탬프 → 문장 단위 줄 | ko / en |
| `faster-whisper` | [Whisper large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | 고정밀, 다국어 지원 | ko / en 동일 모델 |

### Qwen3-ASR 참고사항

- 오디오를 최대 60초 청크(가장 조용한 지점에서 분할)로 나눠 8개씩 배치 처리합니다. 정렬기는 청크가 길면(120초 이상) 끝부분 타임스탬프가 뭉개져 60초로 제한했습니다.
- ForcedAligner가 단어별 시간을 찾고, 이를 문장 부호 기준으로 묶어 `[시작 --> 끝] 문장` 형태로 저장합니다. JSON에는 단어별 타임스탬프(`words`)도 포함됩니다.
- 정렬기가 위치를 못 잡은 단어는 앞뒤 단어 기준으로 보간하며, 그 개수를 로그에 표시합니다.
- bfloat16을 사용하므로 Compute Capability 7.0 미만 GPU(TITAN Xp 등)는 웹 UI에서 선택할 수 없습니다.
- 첫 실행 시 모델(약 5GB)을 내려받습니다.

## 요구사항

| 항목 | 값 |
|------|-----|
| Python | 3.11 ~ 3.13 |
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

### 웹 UI (권장)

```bash
uv run web.py                       # http://127.0.0.1:8000
uv run web.py --host 0.0.0.0 --port 8080   # 외부 접속 허용
```

브라우저에서 파일을 업로드하거나 `input/`·`audio/` 목록에서 여러 개를 선택한 뒤 엔진·언어·GPU·요약 옵션을 고르고 **변환 시작**을 누르면 됩니다.
작업은 큐에 쌓여 하나씩 순차 실행되며, 진행 로그와 전사본/요약본을 페이지에서 바로 확인·다운로드할 수 있습니다.
GPU 번호는 `nvidia-smi`와 동일한 순서(`PCI_BUS_ID`)로 표시됩니다.

> 원격 서버에서 실행할 경우 VS Code 포트 포워딩이나 `ssh -L 8000:localhost:8000 서버` 로 접속하세요.

### 대화형 모드 (파일/언어/요약 여부를 묻는 프롬프트)

```bash
uv run convert.py
```

### 파일 지정 실행

```bash
# input/ 폴더의 파일 전사 (한국어, Qwen3-ASR)
uv run convert.py lecture.mp4

# 영어 전사
uv run convert.py lecture.mp4 --language en

# faster-whisper 엔진 사용 (고정밀)
uv run convert.py lecture.mp4 --engine faster-whisper

# 전사 + Claude 요약
uv run convert.py lecture.mp4 --summarize
```

### YouTube 영상 / 재생목록

웹 UI의 **파일** 패널 맨 위에 URL을 붙여넣고 **불러오기**를 누르면 영상 정보와 자막 트랙이 표시됩니다(재생목록이면 영상을 골라 일괄 추가).
CLI에서는 `--url`을 씁니다.

```bash
uv run convert.py --url "https://youtu.be/VIDEO_ID"                 # 오디오 다운로드 → Qwen3-ASR 전사
uv run convert.py --url "https://youtu.be/VIDEO_ID" --subs manual   # 업로더 자막 우선, 없으면 음성 인식
uv run convert.py --url "https://youtu.be/VIDEO_ID" --subs any      # 자동 생성 자막까지 허용
uv run convert.py --url "https://www.youtube.com/playlist?list=..." --language en --summarize
```

| `--subs` | 동작 |
|---|---|
| `off` (기본값) | 자막을 쓰지 않고 오디오를 내려받아 음성 인식 |
| `manual` | **업로더가 직접 올린 자막**이 있으면 사용, 없으면 음성 인식 |
| `any` | 업로더 자막 → 없으면 **YouTube 자동 생성 자막** → 없으면 음성 인식 |

- 결과 파일명은 `영상 제목 [영상ID]`이며, 이미 전사된 영상은 건너뜁니다. 오디오는 `audio/`에 MP3로 저장됩니다.
- 자막으로 만든 전사본은 TXT 두 번째 줄에 `# 출처: YouTube 자막 [업로더]` 또는 `[자동 생성]`이 표기되고, 웹 UI 파일 목록과 작업 목록에도 `자막·업로더` / `자막·자동 생성` 칩으로 구분됩니다. JSON에는 `source` 필드로 영상 정보가 기록됩니다.
- 다른 언어에서 **번역된 자막**(업로더 번역본, 자동 번역)은 실제 발화가 아니므로 사용하지 않습니다. 웹 UI에는 "번역본 · 사용 안 함"으로만 표시됩니다.
- 자동 생성 자막은 문장 부호가 없어 최대 15초 단위로 이어 붙여 저장합니다.
- `watch?v=...&list=...` 형태의 URL은 해당 영상 하나만 처리합니다. 재생목록 전체는 `playlist?list=...` URL을 쓰세요.
- YouTube 추출에는 JS 런타임이 필요해 `deno` 패키지를 함께 설치합니다(시스템 설치 불필요). YouTube 변경으로 다운로드가 실패하면 `uv lock --upgrade-package yt-dlp && uv sync`로 업데이트하세요.

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
├── convert.py          # 메인 스크립트 (CLI)
├── web.py              # 웹 UI 서버
├── static/index.html   # 웹 UI 페이지
├── youtube.py          # YouTube 오디오/자막 가져오기 (yt-dlp)
├── gpu_selector.py     # GPU 선택
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
