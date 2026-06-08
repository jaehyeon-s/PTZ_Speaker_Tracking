"""Obtain the subject region followed by the camera's internal tracker."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.tracking.target_state import PersonDetection
from src.vision.models import BBox, Candidate, TrackingState


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


@dataclass(frozen=True)
class IdentityObservation:
    bbox: BBox | None
    source: str
    state: TrackingState
    event: str
    best_score: float = 0.0
    center_score: float = 0.0
    second_score: float = 0.0
    identity_margin: float = 0.0
    center_margin: float = 0.0
    candidate_count: int = 0
    hold_count: int = 0
    mismatch_count: int = 0


class IdentityMatchedRegionProvider:
    """Select the observed presenter bbox by identity, not by camera center.

    The Qon camera does not expose its internal tracking bbox on the tested
    firmware. This provider uses all person detections in the frame and chooses
    the candidate closest to the registered Re-ID identity. Center proximity is
    only used to decide whether the camera is aligned with that identity.
    """

    def __init__(
        self,
        detector,
        matcher,
        fallback_provider: TrackedRegionProvider | None = None,
        weak_min_score: float = 0.25,
        identity_margin: float = 0.08,
        center_margin: float = 0.12,
        center_dead_zone_ratio: float = 0.18,
        mismatch_limit: int = 10,
        hold_limit: int = 30,
    ) -> None:
        self.detector = detector
        self.matcher = matcher
        self.fallback_provider = fallback_provider or CenterCropRegionProvider()
        self.weak_min_score = weak_min_score
        self.identity_margin = identity_margin
        self.center_margin = center_margin
        self.center_dead_zone_ratio = center_dead_zone_ratio
        self.mismatch_limit = max(1, mismatch_limit)
        self.hold_limit = max(1, hold_limit)
        self.last_trusted_bbox: BBox | None = None
        self.hold_count = 0
        self.mismatch_count = 0
        self.last_observation = IdentityObservation(
            None,
            "identity_uninitialized",
            TrackingState.UNREGISTERED,
            "REGISTER_REQUIRED",
        )

    def region_for_frame(self, frame_index: int, frame) -> tuple[BBox | None, str]:
        if not self._has_reference():
            bbox, source = self.fallback_provider.region_for_frame(frame_index, frame)
            self.last_observation = IdentityObservation(
                bbox,
                f"identity_bootstrap_{source}",
                TrackingState.UNREGISTERED,
                "REGISTER_REQUIRED",
            )
            return bbox, self.last_observation.source

        people = self.detector.detect(frame)
        candidates = [_candidate_from_person(index, person) for index, person in enumerate(people)]
        if not candidates:
            return self._hold_or_lost("NO_PERSON_CANDIDATES", 0.0, 0.0, 0.0, 0)

        scored = [(candidate, self.matcher.score(frame, candidate.bbox)) for candidate in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        best, best_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        center_candidate, center_score = self._center_candidate(scored, frame.shape)
        identity_delta = best_score - second_score
        center_delta = best_score - center_score
        reliable = best_score >= self.weak_min_score and identity_delta >= self.identity_margin

        if not reliable:
            return self._hold_or_lost(
                "OBSERVATION_HOLD",
                best_score,
                center_score,
                second_score,
                len(candidates),
                best.bbox,
            )

        self.hold_count = 0
        self.last_trusted_bbox = best.bbox
        if center_candidate is not None and best.bbox != center_candidate.bbox:
            away = _is_away_from_center(best.bbox, frame.shape, self.center_dead_zone_ratio)
            if away and center_delta >= self.center_margin:
                self.mismatch_count += 1
                event = "MISTRACK_SUSPECTED"
                state = TrackingState.SUSPECT
                if self.mismatch_count >= self.mismatch_limit:
                    event = "PRESENTER_MISTRACK_CONFIRMED"
                    state = TrackingState.MISMATCH
                self.last_observation = IdentityObservation(
                    best.bbox,
                    "identity_mistrack",
                    state,
                    event,
                    best_score,
                    center_score,
                    second_score,
                    identity_delta,
                    center_delta,
                    len(candidates),
                    self.hold_count,
                    self.mismatch_count,
                )
                return best.bbox, self.last_observation.source

        self.mismatch_count = 0
        self.last_observation = IdentityObservation(
            best.bbox,
            "identity_aligned",
            TrackingState.CAMERA_ALIGNED,
            "CAMERA_ALIGNED",
            best_score,
            center_score,
            second_score,
            identity_delta,
            center_delta,
            len(candidates),
            self.hold_count,
            self.mismatch_count,
        )
        return best.bbox, self.last_observation.source

    def _hold_or_lost(
        self,
        event: str,
        best_score: float,
        center_score: float,
        second_score: float,
        candidate_count: int,
        candidate_bbox: BBox | None = None,
    ) -> tuple[BBox | None, str]:
        self.hold_count += 1
        if self.hold_count >= self.hold_limit:
            bbox = None
            state = TrackingState.LOST
            source = "identity_lost"
            event = "TARGET_LOST"
            self.mismatch_count = 0
        else:
            bbox = self.last_trusted_bbox or candidate_bbox
            state = TrackingState.HOLD
            source = "identity_hold"
        self.last_observation = IdentityObservation(
            bbox,
            source,
            state,
            event,
            best_score,
            center_score,
            second_score,
            best_score - second_score,
            best_score - center_score,
            candidate_count,
            self.hold_count,
            self.mismatch_count,
        )
        return bbox, source

    def _center_candidate(self, scored, frame_shape) -> tuple[Candidate | None, float]:
        frame_h, frame_w = frame_shape[:2]
        center = (frame_w / 2.0, frame_h / 2.0)
        best = None
        for candidate, score in scored:
            distance = _distance_sq(_bbox_center(candidate.bbox), center)
            if best is None or distance < best[0]:
                best = (distance, candidate, score)
        if best is None:
            return None, 0.0
        return best[1], best[2]

    def _has_reference(self) -> bool:
        return getattr(self.matcher, "reference", None) is not None


def _candidate_from_person(index: int, person: PersonDetection) -> Candidate:
    x, y, w, h = person.bbox
    return Candidate((x, y, x + w, y + h), person.confidence, index)


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _distance_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_away_from_center(bbox: BBox, frame_shape, dead_zone_ratio: float) -> bool:
    frame_h, frame_w = frame_shape[:2]
    cx, cy = _bbox_center(bbox)
    error_x = abs((cx - frame_w / 2.0) / max(frame_w / 2.0, 1.0))
    error_y = abs((cy - frame_h / 2.0) / max(frame_h / 2.0, 1.0))
    return error_x > dead_zone_ratio or error_y > dead_zone_ratio
