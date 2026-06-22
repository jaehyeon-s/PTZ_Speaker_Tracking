from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic ArUco marker-transfer fixture video."
    )
    parser.add_argument(
        "--output",
        default="test_assets/marker_transfer_fixture.mp4",
        help="Output video path.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--duration-seconds", type=float, default=8.0)
    parser.add_argument("--marker-size", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "Install runtime dependencies first: pip install -r requirements.txt"
        ) from exc

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer: {output_path}")

    marker = _create_marker(cv2, args.marker_size)
    total_frames = max(1, int(round(args.duration_seconds * args.fps)))
    left_box = _scale_box(args.width, args.height, (0.08, 0.18, 0.42, 0.92))
    right_box = _scale_box(args.width, args.height, (0.58, 0.18, 0.92, 0.92))
    left_marker = _box_anchor(left_box, args.marker_size)
    right_marker = _box_anchor(right_box, args.marker_size)

    for frame_index in range(total_frames):
        progress = frame_index / max(total_frames - 1, 1)
        frame = _draw_scene(np, cv2, args.width, args.height, left_box, right_box)
        marker_xy = _marker_position(progress, left_marker, right_marker)
        _overlay_marker(frame, marker, marker_xy)
        writer.write(frame)

    writer.release()
    print(output_path)


def _create_marker(cv2, marker_size: int):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_bits = cv2.aruco.generateImageMarker(dictionary, 0, marker_size)
    border = max(12, marker_size // 5)
    return cv2.copyMakeBorder(
        marker_bits,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def _scale_box(width: int, height: int, values: tuple[float, float, float, float]):
    x1, y1, x2, y2 = values
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def _box_anchor(box: tuple[int, int, int, int], marker_size: int):
    x1, y1, x2, y2 = box
    marker_extent = marker_size + 2 * max(12, marker_size // 5)
    center_x = (x1 + x2) // 2
    center_y = y1 + int(0.42 * (y2 - y1))
    return (center_x - marker_extent // 2, center_y - marker_extent // 2)


def _marker_position(progress: float, left_xy, right_xy):
    if progress < 0.35:
        return left_xy
    if progress > 0.65:
        return right_xy
    local = (progress - 0.35) / 0.30
    eased = local * local * (3 - 2 * local)
    return (
        int(round(left_xy[0] + (right_xy[0] - left_xy[0]) * eased)),
        int(round(left_xy[1] + (right_xy[1] - left_xy[1]) * eased)),
    )


def _draw_scene(np, cv2, width: int, height: int, left_box, right_box):
    frame = np.full((height, width, 3), (235, 238, 241), dtype=np.uint8)
    _draw_person(cv2, frame, left_box, (76, 121, 168), "fixture id 1")
    _draw_person(cv2, frame, right_box, (168, 102, 76), "fixture id 2")
    cv2.putText(
        frame,
        "ArUco marker transfer fixture",
        (30, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (40, 45, 52),
        2,
    )
    return frame


def _draw_person(cv2, frame, box, color, label: str) -> None:
    x1, y1, x2, y2 = box
    width = x2 - x1
    head_center = ((x1 + x2) // 2, y1 + int(0.13 * (y2 - y1)))
    head_radius = max(22, width // 9)
    body_top = y1 + int(0.24 * (y2 - y1))
    cv2.rectangle(frame, (x1, body_top), (x2, y2), color, -1)
    cv2.circle(frame, head_center, head_radius, color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (45, 52, 60), 2)
    cv2.putText(
        frame,
        label,
        (x1 + 16, max(28, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (45, 52, 60),
        2,
    )


def _overlay_marker(frame, marker, xy) -> None:
    x, y = xy
    marker_bgr = marker[:, :, None].repeat(3, axis=2)
    height, width = marker_bgr.shape[:2]
    frame[y : y + height, x : x + width] = marker_bgr


if __name__ == "__main__":
    main()
