from __future__ import annotations

import math
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
    track_id: Optional[int] = None


@dataclass(frozen=True)
class HandoffDecision:
    state: str
    target_box: Optional[BoundingBox]
    marker_box: Optional[BoundingBox]
    target_track_id: Optional[int] = None
    reason: str = ""
    emit_handoff: bool = False
    emit_release: bool = False
    candidate_scores: tuple["CandidateScore", ...] = ()


@dataclass(frozen=True)
class CandidateScore:
    track_id: Optional[int]
    score: float
    box: BoundingBox
    is_active: bool = False
    area_growth_ratio: Optional[float] = None


class MarkerTracker:
    """Select a person associated with a visual marker.

    A camera API adapter can consume emit_handoff and emit_release later.
    Until then the runtime visualizes and logs these events.
    """

    _ACTIVE_TRACK_BONUS = 30.0
    _CONFIDENCE_WEIGHT = 20.0
    _AREA_PENALTY_WEIGHT = 120.0
    _AREA_GROWTH_PENALTY_WEIGHT = 90.0
    _CENTER_DISTANCE_PENALTY_WEIGHT = 15.0

    def __init__(
        self,
        confirmation_seconds: float = 3.0,
        release_grace_seconds: float = 3.0,
        marker_dropout_seconds: float = 0.5,
        switch_score_margin: float = 12.0,
        switch_cooldown_seconds: float = 1.0,
        area_growth_penalty_threshold: float = 1.8,
    ) -> None:
        if (
            confirmation_seconds < 0
            or release_grace_seconds < 0
            or marker_dropout_seconds < 0
            or switch_score_margin < 0
            or switch_cooldown_seconds < 0
            or area_growth_penalty_threshold < 1.0
        ):
            raise ValueError("invalid tracker scoring parameters")
        self.confirmation_seconds = confirmation_seconds
        self.release_grace_seconds = release_grace_seconds
        self.marker_dropout_seconds = marker_dropout_seconds
        self.switch_score_margin = switch_score_margin
        self.switch_cooldown_seconds = switch_cooldown_seconds
        self.area_growth_penalty_threshold = area_growth_penalty_threshold
        self._candidate_since: Optional[float] = None
        self._candidate_target: Optional[BoundingBox] = None
        self._candidate_track_id: Optional[int] = None
        self._last_candidate_match_at: Optional[float] = None
        self._active_target: Optional[BoundingBox] = None
        self._active_track_id: Optional[int] = None
        self._last_match_at: Optional[float] = None
        self._last_handoff_at: Optional[float] = None

    def update(
        self,
        detections: Iterable[Detection],
        now: float,
        requested_track_id: Optional[int] = None,
    ) -> HandoffDecision:
        detections = list(detections)
        markers = [d for d in detections if d.label == "marker"]
        people = [d for d in detections if d.label == "person"]
        marker = markers[0].box if len(markers) == 1 else None
        scored_candidates = self._score_marker_candidates(marker, people)
        candidate_scores = tuple(score for _, score in scored_candidates)
        matched_person = scored_candidates[0][0] if scored_candidates else None

        if requested_track_id is not None:
            requested_person = self._find_track(people, requested_track_id)
            if requested_person is not None:
                handoff = self._active_track_id != requested_track_id
                self._active_track_id = requested_track_id
                self._active_target = requested_person.box
                self._last_match_at = now
                self._reset_candidate()
                if handoff:
                    self._last_handoff_at = now
                return HandoffDecision(
                    "tracking",
                    requested_person.box,
                    marker,
                    requested_track_id,
                    reason="manual_target_selected",
                    emit_handoff=handoff,
                    candidate_scores=candidate_scores,
                )

        active_person = self._find_track(people, self._active_track_id)
        if active_person is not None:
            previous_active_target = self._active_target
            self._active_target = active_person.box
            self._last_match_at = now
            if matched_person is not None and not self._is_active_match(
                matched_person
            ) and self._should_switch(
                matched_person,
                active_person,
                marker,
                scored_candidates,
                previous_active_target,
                now,
            ):
                return self._update_candidate(
                    matched_person,
                    marker,
                    now,
                    pending_state="switching",
                    pending_target=active_person.box,
                    pending_track_id=self._active_track_id,
                    candidate_scores=candidate_scores,
                )
            self._reset_candidate()
            return HandoffDecision(
                "tracking",
                active_person.box,
                marker,
                self._active_track_id,
                reason="active_track_visible",
                candidate_scores=candidate_scores,
            )

        if matched_person is not None:
            self._last_match_at = now
            if self._active_target is not None:
                self._active_track_id = matched_person.track_id
                self._active_target = matched_person.box
                self._last_handoff_at = now
                self._reset_candidate()
                return HandoffDecision(
                    "tracking",
                    matched_person.box,
                    marker,
                    matched_person.track_id,
                    reason="marker_reacquired_target",
                    emit_handoff=True,
                    candidate_scores=candidate_scores,
                )

            return self._update_candidate(
                matched_person,
                marker,
                now,
                pending_state="confirming",
                pending_target=matched_person.box,
                pending_track_id=matched_person.track_id,
                candidate_scores=candidate_scores,
            )

        if (
            self._candidate_since is not None
            and self._last_candidate_match_at is not None
            and now - self._last_candidate_match_at <= self.marker_dropout_seconds
        ):
            return HandoffDecision(
                "confirming",
                self._candidate_target,
                marker,
                reason="marker_dropout_hold",
                candidate_scores=candidate_scores,
            )

        self._reset_candidate()
        if self._active_target is None:
            return HandoffDecision(
                "default",
                None,
                marker,
                reason="no_confirmed_target",
                candidate_scores=candidate_scores,
            )

        assert self._last_match_at is not None
        if now - self._last_match_at <= self.release_grace_seconds:
            return HandoffDecision(
                "grace",
                self._active_target,
                marker,
                self._active_track_id,
                reason="target_lost_grace",
                candidate_scores=candidate_scores,
            )

        previous_target = self._active_target
        previous_track_id = self._active_track_id
        self._active_target = None
        self._active_track_id = None
        self._last_match_at = None
        return HandoffDecision(
            "default",
            previous_target,
            marker,
            previous_track_id,
            reason="target_released_after_grace",
            emit_release=True,
            candidate_scores=candidate_scores,
        )

    def _reset_candidate(self) -> None:
        self._candidate_since = None
        self._candidate_target = None
        self._candidate_track_id = None
        self._last_candidate_match_at = None

    def _update_candidate(
        self,
        matched_person: Detection,
        marker: Optional[BoundingBox],
        now: float,
        pending_state: str,
        pending_target: BoundingBox,
        pending_track_id: Optional[int],
        candidate_scores: tuple[CandidateScore, ...],
    ) -> HandoffDecision:
        if not self._is_candidate_match(matched_person):
            self._candidate_since = now
        self._candidate_target = matched_person.box
        self._candidate_track_id = matched_person.track_id
        self._last_candidate_match_at = now
        assert self._candidate_since is not None
        if now - self._candidate_since >= self.confirmation_seconds:
            self._active_target = matched_person.box
            self._active_track_id = matched_person.track_id
            self._last_match_at = now
            self._last_handoff_at = now
            self._reset_candidate()
            return HandoffDecision(
                "tracking",
                matched_person.box,
                marker,
                matched_person.track_id,
                reason="confirmed_marker_handoff",
                emit_handoff=True,
                candidate_scores=candidate_scores,
            )
        return HandoffDecision(
            pending_state,
            pending_target,
            marker,
            pending_track_id,
            reason=(
                "switch_candidate_confirming"
                if pending_state == "switching"
                else "marker_candidate_confirming"
            ),
            candidate_scores=candidate_scores,
        )

    def _should_switch(
        self,
        matched_person: Detection,
        active_person: Detection,
        marker: Optional[BoundingBox],
        scored_candidates: list[tuple[Detection, CandidateScore]],
        previous_active_target: Optional[BoundingBox],
        now: float,
    ) -> bool:
        if (
            self._last_handoff_at is not None
            and now - self._last_handoff_at < self.switch_cooldown_seconds
        ):
            return False
        if marker is None:
            return False

        matched_score = self._score_for(scored_candidates, matched_person)
        active_score = self._score_for(scored_candidates, active_person)
        if active_score is None:
            return True

        if previous_active_target is not None:
            growth_ratio = self._area_growth_ratio(active_person.box, previous_active_target)
            if growth_ratio >= self.area_growth_penalty_threshold:
                return matched_score is not None

        if matched_score is None:
            return False
        return matched_score >= active_score + self.switch_score_margin

    def _is_candidate_match(self, matched_person: Detection) -> bool:
        if self._candidate_since is None:
            return False
        if (
            self._candidate_track_id is not None
            and matched_person.track_id is not None
        ):
            return self._candidate_track_id == matched_person.track_id
        return self._candidate_target == matched_person.box

    def _is_active_match(self, matched_person: Detection) -> bool:
        if self._active_track_id is not None and matched_person.track_id is not None:
            return self._active_track_id == matched_person.track_id
        return self._active_target == matched_person.box

    def _score_marker_candidates(
        self,
        marker: Optional[BoundingBox],
        people: list[Detection],
    ) -> list[tuple[Detection, CandidateScore]]:
        if marker is None:
            return []
        containing = [person for person in people if person.box.contains(marker.center)]
        if not containing:
            return []
        min_area = min(person.box.area for person in containing)
        scored = [
            (
                person,
                CandidateScore(
                    track_id=person.track_id,
                    score=self._marker_match_score(marker, person, min_area),
                    box=person.box,
                    is_active=self._is_active_match(person),
                    area_growth_ratio=self._active_area_growth_ratio(person),
                ),
            )
            for person in containing
        ]
        return sorted(scored, key=lambda item: item[1].score, reverse=True)

    @staticmethod
    def _score_for(
        scored_candidates: list[tuple[Detection, CandidateScore]],
        person: Detection,
    ) -> Optional[float]:
        for candidate, score in scored_candidates:
            if (
                candidate.track_id is not None
                and person.track_id is not None
                and candidate.track_id == person.track_id
            ):
                return score.score
            if candidate.box == person.box:
                return score.score
        return None

    def _active_area_growth_ratio(self, person: Detection) -> Optional[float]:
        if not self._is_active_match(person) or self._active_target is None:
            return None
        return self._area_growth_ratio(person.box, self._active_target)

    @staticmethod
    def _area_growth_ratio(current: BoundingBox, previous: BoundingBox) -> float:
        return current.area / max(previous.area, 1.0)

    def _area_growth_penalty(self, person: Detection) -> float:
        growth_ratio = self._active_area_growth_ratio(person)
        if growth_ratio is None or growth_ratio <= self.area_growth_penalty_threshold:
            return 0.0
        return self._AREA_GROWTH_PENALTY_WEIGHT * math.log(
            growth_ratio / self.area_growth_penalty_threshold
        )

    def _marker_match_score(
        self,
        marker: BoundingBox,
        person: Detection,
        min_area: float,
    ) -> float:
        score = self._CONFIDENCE_WEIGHT * person.confidence
        if (
            self._active_track_id is not None
            and person.track_id == self._active_track_id
        ):
            score += self._ACTIVE_TRACK_BONUS

        if min_area > 0 and person.box.area > 0:
            area_ratio = person.box.area / min_area
            score -= self._AREA_PENALTY_WEIGHT * math.log(area_ratio)
        score -= self._area_growth_penalty(person)

        score -= self._CENTER_DISTANCE_PENALTY_WEIGHT * self._normalized_center_distance(
            marker,
            person.box,
        )
        return score

    @staticmethod
    def _normalized_center_distance(
        marker: BoundingBox, person_box: BoundingBox
    ) -> float:
        marker_x, marker_y = marker.center
        person_x, person_y = person_box.center
        width = max(1.0, person_box.x2 - person_box.x1)
        height = max(1.0, person_box.y2 - person_box.y1)
        return math.hypot(
            (marker_x - person_x) / width,
            (marker_y - person_y) / height,
        )

    @staticmethod
    def _find_track(
        people: list[Detection], track_id: Optional[int]
    ) -> Optional[Detection]:
        if track_id is None:
            return None
        for person in people:
            if person.track_id == track_id:
                return person
        return None
