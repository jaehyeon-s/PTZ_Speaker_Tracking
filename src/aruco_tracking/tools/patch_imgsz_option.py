#!/usr/bin/env python3
"""Patch older marker_tracker/app.py files to add the --imgsz runtime option."""

from __future__ import annotations

import argparse
from pathlib import Path


MODEL_OPTION = '    parser.add_argument("--model", default="yolo26n.pt")\n'
IMGSZ_OPTION = '''    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size. Lower this, e.g. 576 or 512, to improve FPS.",
    )
'''
PERSON_CONFIDENCE_BLOCK = '''        self._person_confidence = (
            args.confidence if args.confidence is not None else args.person_confidence
        )
'''
IMGSZ_ASSIGNMENT = "        self._imgsz = args.imgsz\n"
TRACK_CONF = "            conf=inference_confidence,\n"
TRACK_IMGSZ = "            imgsz=self._imgsz,\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add the --imgsz YOLO inference option to marker_tracker/app.py."
    )
    parser.add_argument(
        "app_py",
        nargs="?",
        default="marker_tracker/app.py",
        help="Path to app.py. Defaults to marker_tracker/app.py in the current repo.",
    )
    args = parser.parse_args()

    path = Path(args.app_py)
    text = path.read_text()
    original = text

    if "--imgsz" not in text:
        text = text.replace(MODEL_OPTION, MODEL_OPTION + IMGSZ_OPTION)

    if IMGSZ_ASSIGNMENT not in text:
        text = text.replace(
            PERSON_CONFIDENCE_BLOCK,
            PERSON_CONFIDENCE_BLOCK + IMGSZ_ASSIGNMENT,
        )

    if TRACK_IMGSZ not in text:
        text = text.replace(TRACK_CONF, TRACK_CONF + TRACK_IMGSZ, 1)

    if text == original:
        print(f"No changes needed: {path}")
        return

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(original)
    path.write_text(text)
    print(f"Patched {path}")
    print(f"Backup written to {backup}")


if __name__ == "__main__":
    main()
