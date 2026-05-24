"""Generate printable ArUco marker images for the PTZ tracking demo."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an ArUco marker PNG")
    parser.add_argument("--id", type=int, required=True, help="Marker id")
    parser.add_argument("--size", type=int, default=800, help="Output image size in pixels")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--dictionary", default="DICT_4X4_50", help="OpenCV ArUco dictionary name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import cv2
    except ModuleNotFoundError:
        print("OpenCV is required. Install with: pip install opencv-contrib-python")
        return 2

    if not hasattr(cv2, "aruco"):
        print("OpenCV ArUco support is required. Install with: pip install opencv-contrib-python")
        return 2

    dictionary_id = getattr(cv2.aruco, args.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, args.id, args.size)
    else:
        marker = cv2.aruco.drawMarker(dictionary, args.id, args.size)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), marker):
        print(f"Failed to write marker image: {output}")
        return 1

    print(f"Wrote ArUco marker id={args.id} size={args.size} dictionary={args.dictionary} to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
