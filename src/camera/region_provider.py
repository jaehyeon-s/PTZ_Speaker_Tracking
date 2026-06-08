"""Obtain the subject region followed by the camera's internal tracker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from src.vision.models import BBox


class TrackedRegionProvider(Protocol):
    def region_for_frame(self, frame_index: int, frame) -> tuple[BBox | None, str]:
        """Return the camera-tracked subject bbox and the region source."""


class CenterCropRegionProvider:
    """Fallback when the camera follows a subject but exposes no bbox API.

    Qon Auto Tracking is expected to keep the followed presenter near frame
    center. This crop is usable for early validation, but it is not equivalent
    to obtaining the internal tracking bbox.
    """

    def __init__(self, width_ratio: float = 0.42, height_ratio: float = 0.86) -> None:
        if not 0 < width_ratio <= 1 or not 0 < height_ratio <= 1:
            raise ValueError("center crop ratios must be within (0, 1]")
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio

    def region_for_frame(self, frame_index: int, frame) -> tuple[BBox, str]:
        del frame_index
        frame_h, frame_w = frame.shape[:2]
        width = int(frame_w * self.width_ratio)
        height = int(frame_h * self.height_ratio)
        x1 = (frame_w - width) // 2
        y1 = (frame_h - height) // 2
        return (x1, y1, x1 + width, y1 + height), "center_crop"


class JsonlRegionProvider:
    """Development adapter for bbox samples captured from camera metadata.

    Each line must contain `{"frame": 1, "bbox": [x1, y1, x2, y2]}`.
    Once a Qon bbox endpoint or metadata stream is identified, its adapter can
    implement the same `region_for_frame` interface.
    """

    def __init__(self, path: str) -> None:
        self.regions: dict[int, BBox] = {}
        with Path(path).open(encoding="utf-8") as input_file:
            for line in input_file:
                record = json.loads(line)
                bbox = record["bbox"]
                if len(bbox) != 4:
                    raise ValueError("metadata bbox must contain four coordinates")
                self.regions[int(record["frame"])] = tuple(int(value) for value in bbox)

    def region_for_frame(self, frame_index: int, frame) -> tuple[BBox | None, str]:
        del frame
        return self.regions.get(frame_index), "metadata_jsonl"
