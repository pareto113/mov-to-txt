"""
GPU 선택 보일러플레이트 — 다른 프로젝트에 이 파일만 복사해서 사용하세요.

사용법:
    import gpu_selector

    gpu_selector.setup()           # main() 진입점 최상단에서 한 번 호출
    dev   = gpu_selector.device()  # "cuda:1" 또는 "cpu"
    idx   = gpu_selector.index()   # 1  (CUDA 없으면 0)
    ctype = gpu_selector.compute_type()  # faster-whisper용: "float16" / "int8_float32" / "int8"

환경변수:
    ENV_MODE=server  → 실행 시 GPU 인덱스 입력 프롬프트 표시
    ENV_MODE=local   → GPU_INDEX=0 기본값 사용 (기본값)
    GPU_INDEX=2      → 프롬프트 없이 인덱스 고정 (server 모드에서도 우선 적용)
"""

import os
import sys

ENV_MODE: str = os.getenv("ENV_MODE", "local")
_GPU_INDEX: int = int(os.getenv("GPU_INDEX", "0"))


def setup(max_index: int = 7) -> None:
    """서버 환경에서 GPU 인덱스를 대화형으로 선택한다.

    GPU_INDEX 환경변수가 이미 지정된 경우에는 프롬프트를 건너뛴다.
    로컬 환경(ENV_MODE != 'server')에서는 아무 동작도 하지 않는다.
    """
    global _GPU_INDEX

    if ENV_MODE != "server":
        return

    if "GPU_INDEX" in os.environ:
        print(f"[GPU] GPU_INDEX={_GPU_INDEX} (환경변수 고정)\n")
        return

    print(f"[GPU] 서버 환경(ENV_MODE=server) 감지됨.")
    print(f"[GPU] 사용할 GPU 인덱스를 입력하세요 (0~{max_index}):")

    while True:
        try:
            raw = input("  GPU 인덱스: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(0)

        if raw.isdigit() and 0 <= int(raw) <= max_index:
            _GPU_INDEX = int(raw)
            break
        print(f"  0~{max_index} 사이의 정수를 입력하세요.")

    print(f"[GPU] 설정 완료: cuda:{_GPU_INDEX}\n")


def index() -> int:
    """선택된 GPU 인덱스를 반환한다. CUDA가 없어도 설정값 그대로 반환."""
    return _GPU_INDEX


def device() -> str:
    """torch device 문자열을 반환한다. CUDA를 사용할 수 없으면 'cpu'."""
    try:
        import torch
        return f"cuda:{_GPU_INDEX}" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def compute_type() -> str:
    """faster-whisper의 compute_type을 GPU 아키텍처에 맞게 반환한다.

    - Compute Capability 7.0 이상 (Volta+): float16
    - Compute Capability 7.0 미만          : int8_float32
    - CPU 또는 torch 없음                  : int8
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "int8"
        cap = torch.cuda.get_device_capability(_GPU_INDEX)
        return "float16" if cap[0] >= 7 else "int8_float32"
    except ImportError:
        return "int8"
