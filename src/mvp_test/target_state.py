"""State and data models for the marker-based PTZ tracking MVP."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


BBox = Tuple[int, int, int, int]
Point = Tuple[float, float]


class TrackingState(Enum):
    IDLE = auto()
    ACTIVE = auto()
    SUSPENDED = auto()
    ENDED = auto()


@dataclass(frozen=True)
class PersonDetection:
    bbox: BBox
    confidence: float = 1.0
    class_name: str = "person"

    @property
    def center(self) -> Point:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0


@dataclass(frozen=True)
class MarkerDetection:
    marker_id: int
    center: Point
    corners: Tuple[Point, Point, Point, Point]


@dataclass
class ActiveTarget:
    target_id: int
    bbox: BBox
    marker_id: Optional[int] = None
    source: str = "marker"
    missed_frames: int = 0

    @property
    def center(self) -> Point:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0


@dataclass(frozen=True)
class DemoConfig:
    dead_zone_ratio: float = 0.12
    max_suspended_frames: int = 45
    tracker_iou_threshold: float = 0.15
    tracker_center_threshold_ratio: float = 0.35
    gesture_hold_seconds: float = 1.0
    marker_only: bool = False
    gesture_enabled: bool = False
    target_marker_id: Optional[int] = None
