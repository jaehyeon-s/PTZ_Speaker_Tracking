"""ArUco marker based person selection for PTZ camera tracking."""

from .core import BoundingBox, CandidateScore, Detection, HandoffDecision, MarkerTracker

__all__ = [
    "BoundingBox",
    "CandidateScore",
    "Detection",
    "HandoffDecision",
    "MarkerTracker",
]
