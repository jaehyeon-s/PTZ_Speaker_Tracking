from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO model to NCNN.")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = PROJECT_ROOT / ".cache"
    for path in (cache_root / "matplotlib", cache_root / "ultralytics"):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_root / "ultralytics"))
    from ultralytics import YOLO

    model = YOLO(args.model, task="detect")
    output = model.export(format="ncnn", imgsz=args.imgsz)
    print(output)


if __name__ == "__main__":
    main()
