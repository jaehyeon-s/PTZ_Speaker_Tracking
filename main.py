"""
PTZ Speaker Tracking System - Main Orchestrator

전체 시스템을 multiprocessing 기반으로 실행합니다.
Vision, Audio, Control 프로세스를 OS 레벨에서 분리하여
Python GIL 제약을 우회하고 Pi 5의 4개 코어를 모두 활용합니다.
"""

import argparse
import logging
import multiprocessing as mp
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    """메인 엔트리포인트."""
    parser = argparse.ArgumentParser(description="PTZ Speaker Tracking System")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="설정 파일 경로",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("PTZ Speaker Tracking System starting...")
    logger.info(f"Config: {args.config}")

    # TODO: 구현 예정
    # 1. Config 로드 (configs/default.yaml)
    # 2. shared_memory 초기화
    # 3. Vision Process 시작 (src.vision.process)
    # 4. Audio Process 시작 (src.audio.process)
    # 5. Control Process 시작 (src.ptz + src.api)
    # 6. Signal handler 등록 (graceful shutdown)

    logger.info("System initialized. Waiting for implementation...")


if __name__ == "__main__":
    main()
