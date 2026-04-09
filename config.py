import os
import sys
from dotenv import load_dotenv

load_dotenv()

ENV_MODE = os.getenv("ENV_MODE", "local")


def setup_gpu() -> None:
    """B환경(server)에서만 호출. GPU 인덱스를 사용자로부터 입력받아 환경변수를 설정한다."""
    if ENV_MODE != "server":
        return

    print("[Step 0] 서버 환경(ENV_MODE=server)이 감지되었습니다.")
    print("[Step 0] 사용할 GPU 인덱스를 입력하세요 (0~7):")

    while True:
        try:
            raw = input("  GPU 인덱스: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(0)

        if raw.isdigit() and 0 <= int(raw) <= 7:
            gpu_index = raw
            break
        print("  0~7 사이의 정수를 입력하세요.")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    print(f"[Step 0] GPU 설정 완료: CUDA_VISIBLE_DEVICES={gpu_index}\n")
