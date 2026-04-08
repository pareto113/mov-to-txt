# 동영상 -> 텍스트 전사 파이프라인 — 상세 명세

## 개요

영상 파일(.ts 등)에서 텍스트 전사본 및 요약본을 생성하는 3단계 로컬 파이프라인.

```
[영상 파일] →  [MP3] → [TXT/JSON] → [MD 요약](옵
```

---

## 환경

| 항목      | 값                             |
| ------- | ----------------------------- |
| Python  | 3.13 (pinned: `>=3.10,<3.14`) |
| 패키지 관리자 | `uv` (`uv run`, `uv add`)     |
| GPU     | RTX 3070, CUDA 12.4 (cu124)   |
| 실행 방법   | `uv run <script>.py`          |

### 의존성 패키지

| 패키지              | 버전 조건                   | 용도             |
| ---------------- | ----------------------- | -------------- |
| `openai-whisper` | `>=20250625`            | 음성 → 텍스트       |
| `anthropic`      | `>=0.84.0`              | 텍스트 → 요약       |
| `imageio-ffmpeg` | `>=0.6.0`               | ffmpeg 바이너리 번들 |
| `torch`          | `>=2.5.0` (cu124 index) | Whisper GPU 가속 |

### PyTorch 커스텀 인덱스

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu124" }

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

---

## 디렉토리 구조

```
1일차/
├── conver.py
├── pyproject.toml
├── movie.ts or movie.mp4
├── audio/               ← Step 1 출력
│   └── movie.mp3
├── transcripts/         ← Step 2 출력
│   ├── movie.txt
│   └── movie.json
└── summaries/           ← Step 3 출력
    └── movie.md
```

---

## Step 1:  — 오디오 추출

### 역할

영상 파일에서 오디오를 추출해 Whisper 최적화 MP3로 변환.

### 

### ffmpeg 바이너리

```python
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
# 실제 바이너리명: ffmpeg-win-x86_64-v7.1.exe (Windows)
# whisper의 "ffmpeg" 호출과 이름이 달라 별도 monkey-patch 필요 (Step 2 참고)
```

### 출력 오디오 파라미터

```python
FFMPEG_AUDIO_OPTS = [
    "-vn",                  # 비디오 스트림 제거
    "-acodec", "libmp3lame",# 코덱: MP3
    "-ar", "16000",         # 샘플레이트: 16kHz (Whisper 요구사항)
    "-ac", "1",             # 채널: mono
    "-b:a", "64k",          # 비트레이트: 64kbps
]
```

### 단일 파일 추출 명령

```
ffmpeg -y -i <input> -vn -acodec libmp3lame -ar 16000 -ac 1 -b:a 64k <output.mp3>
```

### ### 스킵 조건

출력 파일(`audio/<name>.mp3`)이 이미 존재하면 건너뜀.

### 에러 처리

입력 파일 없음 → `sys.exit(1)`
ffmpeg returncode != 0 → stderr 마지막 500자 출력 후 `sys.exit(1)`

---

## Step 2:  — 음성 인식 (Whisper)

### 역할

MP3 → 타임스탬프가 포함된 전사본 (.txt, .json) 생성.

### Whisper 설정

```python
WHISPER_MODEL = "large-v3"   # 권장. GPU 메모리 <6GB이면 "medium"
LANGUAGE = "ko"              # 한국어 고정
```

#### 지원 모델 목록 (Whisper)

| 모델           | 특징                      |
| ------------ | ----------------------- |
| `"base"`     | 빠름, 정확도 낮음              |
| `"small"`    | 균형                      |
| `"medium"`   | 높은 정확도                  |
| `"large-v2"` | 최고 정확도                  |
| `"large-v3"` | 최고 정확도, 한국어 권장 ← 현재 기본값 |

### Monkey-patch (Windows ffmpeg 호환)

```python
# whisper.audio.load_audio는 "ffmpeg"라는 이름을 직접 호출하지만,
# imageio-ffmpeg 번들 바이너리는 다른 이름(ffmpeg-win-x86_64-v7.1.exe)이므로
# load_audio 함수를 전체 교체.

_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

def _load_audio(file: str, sr: int = whisper.audio.SAMPLE_RATE) -> np.ndarray:
    cmd = [
        _FFMPEG_EXE, "-nostdin", "-threads", "0",
        "-i", file,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr),
        "-",            # stdout으로 출력
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

whisper.audio.load_audio = _load_audio
```

### `model.transcribe()` 파라미터

```python
result = model.transcribe(
    str(audio_path),
    language="ko",                          # 언어 코드
    verbose=False,                          # 진행 출력 억제
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # fallback 온도 튜플 (hallucination 방지)
    condition_on_previous_text=False,       # 이전 텍스트 조건화 비활성 (반복 루프 방지)
    compression_ratio_threshold=2.4,        # 이 값 초과 시 해당 세그먼트 재시도
    no_speech_threshold=0.6,               # 무음 판별 임계값
    word_timestamps=True,                   # 단어 단위 타임스탬프 (hallucination_silence_threshold 필수 조건)
    hallucination_silence_threshold=1.0,    # 1초 이상 무음 구간에서 hallucination 차단
)
```

#### hallucination 방지 파라미터 설명

| 파라미터                              | 값                                | 역할                                     |
| --------------------------------- | -------------------------------- | -------------------------------------- |
| `temperature`                     | `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` | 실패 시 온도를 높여 재시도 (단일값 `0.0`이면 반복 루프 발생) |
| `condition_on_previous_text`      | `False`                          | 이전 세그먼트 텍스트를 컨텍스트로 넣지 않음               |
| `hallucination_silence_threshold` | `1.0` (초)                        | 무음 구간에서 환각 텍스트 생성 차단                   |
| `compression_ratio_threshold`     | `2.4`                            | 텍스트 압축률이 높으면(반복 의심) 재시도                |
| `no_speech_threshold`             | `0.6`                            | 무음 확률 0.6 이상이면 무음 처리                   |

### 출력 형식

**`.txt` (human-readable)**

```
# NC 세션 전사본

[00:00 --> 00:05]  안녕하세요, 반갑습니다.
[00:05 --> 00:12]  오늘 발표할 내용은...
```

**`.json` (구조화)**

```json
{
  "language": "ko",
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "안녕하세요, 반갑습니다."
    }
  ]
}
```

> 주의: JSON에서 `segments`의 원본 word-level 토큰 데이터는 제거하고 저장.

### 스킵 조건

`transcripts/<name>.txt`가 이미 존재하면 건너뜀.

### 알려진 이슈

- Windows에서 Triton 커널 불가 → DTW가 CPU fallback (느리지만 작동)
- 모델 최초 실행 시 large-v3 다운로드 (~3GB), 이후 캐시

---

## Step 3 (optional) :  — Claude API 요약

### 역할

전사본 텍스트 → Claude API로 구조화된 한국어 요약 마크다운 생성.

### 환경변수

```
ANTHROPIC_API_KEY=sk-ant-...
```

Windows 설정:

```cmd
set ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Claude API 설정

```python
CLAUDE_MODEL = "claude-sonnet-4-6"  # 대안: "claude-opus-4-6" (고품질)
MAX_CHARS = 700_000                 # 전사본 최대 글자수 (초과 시 truncate)
                                    # large-v3 기준 ~1자/초, 2시간 ≈ 50,000자
                                    # Claude 컨텍스트 한도 ~800,000자
max_tokens = 4096                   # 응답 최대 토큰
```

### 시스템 프롬프트

```
당신은 해당 동영상이 다루는 분야의 전문가이며 발표 내용을 정리하는 전문 요약가입니다.
발표 전사본을 읽고 구조화된 한국어 요약본을 작성해주세요.
```

### 유저 프롬프트 템플릿

```
다음은 {company} 세션의 전사본입니다.
전사본에는 타임스탬프와 함께 발표자(주 발표자 + 소수의 진행자/청중 질문)의 발언이 담겨 있습니다.

전사본:
---
{transcript}
---

위 전사본을 바탕으로 다음 형식으로 요약해주세요:

## {company} 세션 요약

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
```

### API 호출 구조

```python
message = client.messages.create(
    model=CLAUDE_MODEL,
    max_tokens=4096,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": prompt}],
)
summary = message.content[0].text
```

### 스킵 조건

`summaries/<name>.md`가 이미 존재하면 건너뜀.

---

## 전체 파라미터 요약표

| 파라미터                            | 현재 값                             | 위치     |
| ------------------------------- | -------------------------------- | ------ |
| ffmpeg 코덱                       | `libmp3lame`                     | Step 1 |
| ffmpeg 샘플레이트                    | `16000` Hz                       | Step 1 |
| ffmpeg 채널                       | `1` (mono)                       | Step 1 |
| ffmpeg 비트레이트                    | `64k`                            | Step 1 |
| Whisper 모델                      | `large-v3`                       | Step 2 |
| Whisper 언어                      | `ko`                             | Step 2 |
| Whisper temperature             | `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` | Step 2 |
| condition_on_previous_text      | `False`                          | Step 2 |
| compression_ratio_threshold     | `2.4`                            | Step 2 |
| no_speech_threshold             | `0.6`                            | Step 2 |
| word_timestamps                 | `True`                           | Step 2 |
| hallucination_silence_threshold | `1.0`                            | Step 2 |
| Claude 모델                       | `claude-sonnet-4-6`              | Step 3 |
| Claude max_tokens               | `4096`                           | Step 3 |
| 전사본 최대 글자수                      | `700,000`                        | Step 3 |

---

## 알려진 이슈 및 해결책

| 이슈                  | 원인                                     | 해결                                                                                          |
| ------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------- |
| ffmpeg 실행 안됨        | imageio-ffmpeg 바이너리명이 `ffmpeg.exe`가 아님 | `whisper.audio.load_audio` monkey-patch                                                     |
| Triton 경고           | Windows에서 Triton 미지원                   | DTW가 CPU로 자동 fallback (느리지만 정상 작동)                                                          |
| hallucination/반복 루프 | Whisper가 무음 구간에서 텍스트 반복 생성             | temperature 튜플 + `condition_on_previous_text=False` + `hallucination_silence_threshold=1.0` |
| 화자 분리 없음            | 미구현                                    | 의도적 생략 (pyannote 등 필요 시 추가)                                                                 |

## 프로그램 전체 args

1. 변환할 동영상 파일명

2. 요약본 생성 여부 (default=False)
