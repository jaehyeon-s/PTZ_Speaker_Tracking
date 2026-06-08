"""Shared data models for camera-managed verification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
BBox = tuple[int, int, int, int]  # x1, y1, x2, y2


class TrackingState(str, Enum):
    UNREGISTERED = "UNREGISTERED"
    VERIFIED = "VERIFIED"
    SUSPECT = "SUSPECT"
    MISMATCH = "MISMATCH"
    NO_REGION = "NO_REGION"
    RECOVERY = "RECOVERY"
    LOST = "LOST"


@dataclass(frozen=True)
class Candidate:
    bbox: BBox
    confidence: float = 1.0
    candidate_id: int | None = None


@dataclass(frozen=True)
class VerificationResult:
    state: TrackingState
    bbox: BBox | None
    score: float = 0.0
    event: str = ""
    source: str = ""
