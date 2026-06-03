from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def contains(self, point: tuple[float, float]) -> bool:
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: BoundingBox


@dataclass(frozen=True)
class HandoffDecision:
    state: str
    target_box: Optional[BoundingBox]
    marker_box: Optional[BoundingBox]
    emit_handoff: bool = False
    emit_release: bool = False


class MarkerTracker:
    """Select a person associated with a phone marker.

    A camera API adapter can consume emit_handoff and emit_release later.
    Until then the runtime visualizes and logs these events.
    """

    def __init__(
        self,
        confirmation_seconds: float = 3.0,
        release_grace_seconds: float = 3.0,
        marker_dropout_seconds: float = 0.5,
    ) -> None:
        if (
            confirmation_seconds < 0
            or release_grace_seconds < 0
            or marker_dropout_seconds < 0
        ):
            raise ValueError("durations must not be negative")
        self.confirmation_seconds = confirmation_seconds
        self.release_grace_seconds = release_grace_seconds
        self.marker_dropout_seconds = marker_dropout_seconds
        self._candidate_since: Optional[float] = None
        self._candidate_target: Optional[BoundingBox] = None
        self._last_candidate_match_at: Optional[float] = None
        self._active_target: Optional[BoundingBox] = None
        self._last_match_at: Optional[float] = None

    def update(
        self, detections: Iterable[Detection], now: float
    ) -> HandoffDecision:
        detections = list(detections)
        phones = [d for d in detections if d.label == "cell phone"]
        people = [d for d in detections if d.label == "person"]
        marker = phones[0].box if len(phones) == 1 else None
        matched_person = self._match_person(marker, people)

        if matched_person is not None:
            self._last_match_at = now
            if self._active_target is not None:
                self._active_target = matched_person.box
                return HandoffDecision("tracking", matched_person.box, marker)

            if self._candidate_since is None:
                self._candidate_since = now
            self._candidate_target = matched_person.box
            self._last_candidate_match_at = now
            if now - self._candidate_since >= self.confirmation_seconds:
                self._active_target = matched_person.box
                self._reset_candidate()
                return HandoffDecision(
                    "tracking", matched_person.box, marker, emit_handoff=True
                )
            return HandoffDecision("confirming", matched_person.box, marker)

        if (
            self._candidate_since is not None
            and self._last_candidate_match_at is not None
            and now - self._last_candidate_match_at <= self.marker_dropout_seconds
        ):
            return HandoffDecision("confirming", self._candidate_target, marker)

        self._reset_candidate()
        if self._active_target is None:
            return HandoffDecision("default", None, marker)

        assert self._last_match_at is not None
        if now - self._last_match_at <= self.release_grace_seconds:
            return HandoffDecision("grace", self._active_target, marker)

        previous_target = self._active_target
        self._active_target = None
        self._last_match_at = None
        return HandoffDecision(
            "default", previous_target, marker, emit_release=True
        )

    def _reset_candidate(self) -> None:
        self._candidate_since = None
        self._candidate_target = None
        self._last_candidate_match_at = None

    @staticmethod
    def _match_person(
        marker: Optional[BoundingBox], people: list[Detection]
    ) -> Optional[Detection]:
        if marker is None:
            return None
        containing = [person for person in people if person.box.contains(marker.center)]
        if not containing:
            return None
        # Prefer the smallest containing box when overlapping people are detected.
        return min(containing, key=lambda person: person.box.area)
