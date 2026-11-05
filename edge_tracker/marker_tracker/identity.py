from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, replace
from statistics import median
from typing import Protocol, Sequence

import numpy as np

from .core import (
    BoundingBox,
    Detection,
    HandoffDecision,
    IdentityResolutionContext,
)


@dataclass
class _ResolutionHypothesis:
    path_id: int
    last_box: BoundingBox
    current_track_id: int | None
    scores: deque[float]
    observations: int = 0
    wins: int = 0
    last_score: float = 0.0
    last_at: float | None = None
    velocity_x: float = 0.0
    velocity_y: float = 0.0


class AppearanceEncoder(Protocol):
    def encode(
        self,
        frame,
        people: Sequence[Detection],
    ) -> list[np.ndarray]: ...


class NoopSpeakerIdentityManager:
    """Preserve the legacy behavior when speaker appearance identity is disabled."""

    def score_detections(
        self,
        frame,
        detections: Sequence[Detection],
        now: float,
    ) -> list[Detection]:
        del frame, now
        return list(detections)

    def observe(
        self,
        decision: HandoffDecision,
        detections: Sequence[Detection],
        frame,
        now: float,
    ) -> HandoffDecision:
        del detections, frame, now
        return decision

    def resolution_context(self) -> IdentityResolutionContext:
        return IdentityResolutionContext()


class SpeakerIdentityManager:
    """Maintain one logical speaker across mutable MOT track IDs.

    Registration evidence (currently ArUco) establishes the
    logical speaker and supplies trusted crops for a multi-shot appearance
    gallery. Appearance inference is selective: it runs only when the active
    tracker ID is absent, another person overlaps it, or a trusted gallery
    sample is due.
    """

    def __init__(
        self,
        encoder: AppearanceEncoder,
        gallery_size: int = 12,
        gallery_interval_seconds: float = 0.35,
        gallery_novelty_similarity: float = 0.98,
        gallery_max_overlap: float = 0.25,
        trigger_overlap: float = 0.20,
        score_burst_frames: int = 4,
        min_similarity: float = 0.55,
        top_k_gallery: int = 3,
        min_crop_area: float = 400.0,
        resolution_min_frames: int = 6,
        resolution_min_seconds: float = 0.50,
        resolution_min_similarity: float = 0.35,
        resolution_min_margin: float = 0.02,
        resolution_winner_ratio: float = 0.75,
        crossing_recent_seconds: float = 1.0,
        negative_gallery_weight: float = 0.5,
        resolution_close_distance: float = 1.75,
        resolution_exit_distance: float = 1.9,
        resolution_exit_frames: int = 3,
        resolution_cooldown_seconds: float = 0.75,
        coast_max_seconds: float = 0.8,
        single_resolution_min_similarity: float = 0.45,
        single_resolution_min_frames: int = 4,
        single_resolution_min_seconds: float = 0.25,
        single_path_max_distance: float = 1.25,
        single_path_max_gap_seconds: float = 0.5,
        resolution_search_after_seconds: float = 2.0,
        resolution_search_interval_frames: int = 3,
        anchor_gallery_size: int = 6,
        gallery_freeze_seconds: float = 1.0,
        # [room_01 수정] 신규 인자 2개.
        resolution_strong_margin: float = 0.15,
        resolution_timeout_seconds: float = 0.0,
        # Linear map from the encoder's cosine similarity onto the scale the
        # thresholds were tuned on (yolo26n-reid). 1 and 0 leave it unchanged.
        similarity_scale: float = 1.0,
        similarity_offset: float = 0.0,
    ) -> None:
        if (
            gallery_size <= 0
            or top_k_gallery <= 0
            or score_burst_frames <= 0
            or resolution_min_frames <= 0
            or resolution_exit_frames < 0
            or single_resolution_min_frames <= 0
            or resolution_search_interval_frames <= 0
            or anchor_gallery_size <= 0
        ):
            raise ValueError("gallery sizes must be positive")
        if (
            gallery_interval_seconds < 0
            or min_crop_area < 0
            or resolution_min_seconds < 0
            or resolution_min_margin < 0
            or crossing_recent_seconds < 0
            or negative_gallery_weight < 0
            or resolution_close_distance < 0
            or resolution_exit_distance < resolution_close_distance
            or resolution_cooldown_seconds < 0
            or coast_max_seconds < 0
            or single_resolution_min_seconds < 0
            or single_path_max_distance < 0
            or single_path_max_gap_seconds < 0
            or resolution_search_after_seconds < 0
            or gallery_freeze_seconds < 0
            or resolution_strong_margin < 0
            or resolution_timeout_seconds < 0
        ):
            raise ValueError("identity timing and area values must be non-negative")
        for value in (
            gallery_novelty_similarity,
            gallery_max_overlap,
            trigger_overlap,
            min_similarity,
            resolution_min_similarity,
            single_resolution_min_similarity,
            resolution_winner_ratio,
        ):
            if not 0 <= value <= 1:
                raise ValueError("identity similarity and overlap values must be in [0, 1]")
        if similarity_scale <= 0:
            raise ValueError("similarity_scale must be positive")
        self._similarity_scale = similarity_scale
        self._similarity_offset = similarity_offset
        self._encoder = encoder
        anchor_gallery_size = min(anchor_gallery_size, gallery_size)
        adaptive_gallery_size = gallery_size - anchor_gallery_size
        self._anchor_gallery: deque[np.ndarray] = deque(maxlen=anchor_gallery_size)
        self._adaptive_gallery: deque[np.ndarray] = deque(maxlen=adaptive_gallery_size)
        self._negative_anchor_gallery: deque[np.ndarray] = deque(
            maxlen=anchor_gallery_size
        )
        self._negative_adaptive_gallery: deque[np.ndarray] = deque(
            maxlen=adaptive_gallery_size
        )
        self._gallery_interval_seconds = gallery_interval_seconds
        self._gallery_novelty_similarity = gallery_novelty_similarity
        self._gallery_max_overlap = gallery_max_overlap
        self._trigger_overlap = trigger_overlap
        self._score_burst_frames = score_burst_frames
        self._min_similarity = min_similarity
        self._top_k_gallery = top_k_gallery
        self._min_crop_area = min_crop_area
        self._speaker_id: int | None = None
        self._current_track_id: int | None = None
        self._current_box: BoundingBox | None = None
        self._last_gallery_at: float | None = None
        # Retain the actual ndarray so its object ID cannot be recycled while
        # cached features are still considered current.
        self._cache_frame = None
        self._cache: dict[tuple[float, float, float, float], np.ndarray] = {}
        self._score_after_separation = False
        self._hold_ambiguous_frame = False
        self._remaining_score_frames = 0
        self._active_was_missing = False
        self._identity_active = False
        self._resolution_min_frames = resolution_min_frames
        self._resolution_min_seconds = resolution_min_seconds
        self._resolution_min_similarity = resolution_min_similarity
        self._resolution_min_margin = resolution_min_margin
        self._resolution_winner_ratio = resolution_winner_ratio
        self._crossing_recent_seconds = crossing_recent_seconds
        self._negative_gallery_weight = negative_gallery_weight
        self._resolution_close_distance = resolution_close_distance
        self._resolution_exit_distance = resolution_exit_distance
        self._resolution_exit_frames = resolution_exit_frames
        self._resolution_cooldown_seconds = resolution_cooldown_seconds
        self._coast_max_seconds = coast_max_seconds
        self._single_resolution_min_similarity = single_resolution_min_similarity
        self._single_resolution_min_frames = single_resolution_min_frames
        self._single_resolution_min_seconds = single_resolution_min_seconds
        self._single_path_max_distance = single_path_max_distance
        self._single_path_max_gap_seconds = single_path_max_gap_seconds
        self._resolution_search_after_seconds = resolution_search_after_seconds
        self._resolution_search_interval_frames = resolution_search_interval_frames
        self._gallery_freeze_seconds = gallery_freeze_seconds
        # [room_01 수정] 신규 상태: 뚜렷한 점수 차 기준, 가설 유지 시간, 탐색 시간 초과.
        self._resolution_strong_margin = resolution_strong_margin
        self._hypothesis_max_gap_seconds = max(1.0, coast_max_seconds)
        # 0 disables the timeout: the search then lasts until evidence arrives.
        self._resolution_timeout_seconds = resolution_timeout_seconds
        self._resolution_last_evidence_at: float | None = None
        self._last_multi_person_at: float | None = None
        self._last_overlap_at: float | None = None
        self._resolution_active = False
        self._resolution_started_at: float | None = None
        self._resolution_reason = ""
        self._resolution_hypotheses: list[_ResolutionHypothesis] = []
        self._resolution_next_path_id = 1
        self._resolution_clean_frames = 0
        self._resolution_evidence_started_at: float | None = None
        self._resolution_verified_track_id: int | None = None
        self._resolution_single_track_id: int | None = None
        self._resolution_single_since: float | None = None
        self._resolution_single_scores: deque[float] = deque(maxlen=12)
        self._resolution_single_last_box: BoundingBox | None = None
        self._resolution_single_last_at: float | None = None
        self._resolution_pair_min_distance: float | None = None
        self._resolution_clear_frames = 0
        self._resolution_saw_close_pair = False
        self._resolution_coast_box: BoundingBox | None = None
        self._resolution_coast_at: float | None = None
        self._resolution_cooldown_until = 0.0
        self._resolution_search_mode = False
        self._resolution_search_frame = 0
        self._resolution_expired = False
        self._resolution_all_clean = True
        self._gallery_frozen_until = 0.0

    @property
    def gallery_size(self) -> int:
        return len(self._anchor_gallery) + len(self._adaptive_gallery)

    @property
    def anchor_gallery_size(self) -> int:
        return len(self._anchor_gallery)

    @property
    def negative_gallery_size(self) -> int:
        return len(self._negative_anchor_gallery) + len(
            self._negative_adaptive_gallery
        )

    @property
    def speaker_id(self) -> int | None:
        return self._speaker_id

    def resolution_context(self) -> IdentityResolutionContext:
        return IdentityResolutionContext(
            active=self._resolution_active,
            verified_track_id=self._resolution_verified_track_id,
            reason=(
                "identity_candidate_verified"
                if self._resolution_verified_track_id is not None
                else self._resolution_reason or "identity_evidence_collecting"
            ),
            coast_box=self._resolution_coast_box if self._resolution_active else None,
            expired=self._resolution_expired,
        )

    def score_detections(
        self,
        frame,
        detections: Sequence[Detection],
        now: float,
    ) -> list[Detection]:
        detections = list(detections)
        if self.gallery_size == 0:
            return detections
        people = [item for item in detections if item.label == "person"]
        identity_people = self._deduplicate_people(people)
        previous_multi_person_at = self._last_multi_person_at
        if len(identity_people) >= 2:
            self._last_multi_person_at = now
        if not people:
            if self._identity_active:
                self._active_was_missing = True
            if self._resolution_active:
                self._update_resolution_coast(now)
                self._update_resolution_expiry(now, active_visible=False)
            return detections
        active = next(
            (person for person in people if person.track_id == self._current_track_id),
            None,
        )
        # [room_01 수정] 기존: 화면의 아무 두 사람이나 겹치면 resolution 시작 + 전원 점수 -1.
        # 변경: 발화자가 관여한 겹침만 resolution을 시작함.
        # 이유: room_01에서 주변인 둘이 25초간 붙어 서 있어 발화자가 떨어져 있어도
        # 아무도 점수를 받지 못했음.
        occluded = self._occluded_people(identity_people)
        # Only an overlap that involves the speaker makes the identity
        # ambiguous. While the speaker's track is visible and clear, other
        # people standing together are irrelevant.
        active_representative = (
            None
            if active is None
            else next(
                (
                    person
                    for person in identity_people
                    if person is active
                    or self._same_physical_person_boxes(person.box, active.box)
                ),
                None,
            )
        )
        overlapping = (
            bool(occluded)
            if active_representative is None
            else id(active_representative) in occluded
        )
        if overlapping:
            self._last_overlap_at = now
        recent_multi_person = (
            previous_multi_person_at is not None
            and now - previous_multi_person_at <= self._crossing_recent_seconds
        )
        if self._identity_active and (
            overlapping or (active is None and recent_multi_person)
        ):
            self._start_resolution(
                now,
                "identity_overlap_hold" if overlapping else "identity_recent_crossing_hold",
            )
        if self._resolution_active:
            self._enter_resolution_search_if_due(now)
            self._update_resolution_expiry(now, active_visible=active is not None)
            # [room_01 수정] 겹친 사람만 제외하고 나머지는 계속 점수를 매김.
            # People inside an overlap have mixed crops and carry no identity
            # evidence. Everyone else can still be compared, so two bystanders
            # standing together cannot blind the search for the speaker.
            clean_people = [
                person for person in identity_people if id(person) not in occluded
            ]
            # [room_01 수정] 겹친 사람이 있는 프레임 표시. 이때는 "뚜렷한 차이" 규칙을 쓰지 않음
            # (발화자가 겹친 사람 중 하나일 수 있음).
            # With someone unreadable in view, a clean candidate cannot be
            # shown to beat everyone; only the absolute threshold counts then.
            self._resolution_all_clean = not occluded
            if overlapping:
                self._resolution_saw_close_pair = True
                self._resolution_clear_frames = 0
            if not clean_people:
                self._update_resolution_coast(now)
                return [
                    replace(item, appearance_similarity=-1.0)
                    if item.label == "person"
                    else item
                    for item in detections
                ]
            if self._resolution_search_mode:
                self._resolution_search_frame += 1
                if (
                    (self._resolution_search_frame - 1)
                    % self._resolution_search_interval_frames
                    != 0
                ):
                    self._update_resolution_coast(now)
                    return detections
            features = self._features(frame, clean_people)
            similarities = [self._gallery_similarity(feature) for feature in features]
            scored_by_identity = {
                id(person): similarity
                for person, similarity in zip(clean_people, similarities)
            }
            scored_by_identity.update((key, -1.0) for key in occluded)
            scored = [
                replace(item, appearance_similarity=scored_by_identity[id(item)])
                if item.label == "person" and id(item) in scored_by_identity
                else item
                for item in detections
            ]
            tracked_identity_people = [
                person for person in clean_people if person.track_id is not None
            ]
            if not overlapping and len(identity_people) >= 2:
                # [room_01 수정] 기존: 면적이 큰 두 사람의 거리(발화자와 무관할 수 있음).
                # 변경: 최고 점수 후보와 가장 가까운 사람의 거리.
                leader = clean_people[
                    max(range(len(similarities)), key=similarities.__getitem__)
                ]
                self._update_resolution_pair_state(
                    leader.box,
                    [person.box for person in identity_people if person is not leader],
                )
            if (
                len(clean_people) >= 2
                and len(tracked_identity_people) == len(clean_people)
            ):
                self._reset_single_resolution_candidate()
                self._update_resolution_hypotheses(
                    tracked_identity_people,
                    similarities,
                    now,
                )
            elif len(clean_people) == 1:
                self._update_single_resolution_candidate(
                    clean_people[0],
                    similarities[0],
                    now,
                )
            else:
                # Trackless crops around the overlap boundary are useful for
                # diagnostics but are not stable identity evidence. They must
                # not seed paths later inherited by a different person.
                self._reset_single_resolution_candidate()
            self._log_scores(clean_people, scored_by_identity)
            return scored
        needs_appearance = self._needs_appearance(identity_people)
        if self._hold_ambiguous_frame:
            self._hold_ambiguous_frame = False
            return [
                replace(item, appearance_similarity=-1.0)
                if item.label == "person"
                else item
                for item in detections
            ]
        if not needs_appearance:
            return detections

        features = self._features(frame, people)
        scored_by_identity: dict[int, float] = {}
        for person, feature in zip(people, features):
            scored_by_identity[id(person)] = self._gallery_similarity(feature)
        scored = [
            replace(
                item,
                appearance_similarity=scored_by_identity[id(item)],
            )
            if item.label == "person" and id(item) in scored_by_identity
            else item
            for item in detections
        ]
        self._log_scores(people, scored_by_identity)
        return scored

    def observe(
        self,
        decision: HandoffDecision,
        detections: Sequence[Detection],
        frame,
        now: float,
    ) -> HandoffDecision:
        people = [item for item in detections if item.label == "person"]
        markers = [item for item in detections if item.label == "marker"]
        if decision.speaker_changed:
            previous_speaker_id = self._speaker_id
            previous_track_id = self._current_track_id
            self._speaker_id = 1 if self._speaker_id is None else self._speaker_id + 1
            self._reset_appearance_identity()
            logging.info(
                "Speaker changed: previous_speaker_id=%s current_speaker_id=%s "
                "previous_track_id=%s current_track_id=%s reason=%s",
                previous_speaker_id,
                self._speaker_id,
                previous_track_id,
                decision.target_track_id,
                decision.reason,
            )
        elif decision.emit_handoff and self._speaker_id is None:
            self._speaker_id = 1
            logging.info(
                "Speaker identity registered: speaker_id=1 track_id=%s reason=%s",
                decision.target_track_id,
                decision.reason,
            )

        if decision.reason == "identity_verified_target":
            # A verified candidate can retain the same raw track ID. That is
            # still a completed identity transaction even though no camera
            # handoff event is emitted.
            self._finish_resolution(now)

        if decision.emit_handoff:
            self._identity_active = True
            if decision.reason in {
                "confirmed_marker_handoff",
                "confirmed_marker_transfer",
                "marker_reacquired_target",
            }:
                self._finish_resolution(now)

        if decision.emit_release:
            self._identity_active = False
            self._active_was_missing = False
            self._remaining_score_frames = 0
            self._current_track_id = None
            self._current_box = decision.target_box
            self._finish_resolution(now)
        elif decision.target_track_id is not None and decision.state in {
            "tracking",
            "recovering",
            "grace",
            "switching",
        }:
            self._current_track_id = decision.target_track_id
            if decision.target_box is not None:
                self._current_box = decision.target_box

        trusted_target = self._trusted_gallery_target(decision, people, markers)
        if trusted_target is not None and self._gallery_sample_due(now):
            negative_candidates = [
                person
                for person in self._deduplicate_people(people)
                if person is not trusted_target
                and self._box_iou(trusted_target.box, person.box)
                <= self._gallery_max_overlap
            ]
            gallery_people = [trusted_target, *negative_candidates]
            features = self._features(frame, gallery_people)
            target_feature = self._normalize(features[0])
            self._add_gallery_feature(
                target_feature,
                now,
                trusted_target.track_id,
            )
            for candidate, feature in zip(negative_candidates, features[1:]):
                self._add_negative_gallery_feature(
                    feature,
                    target_feature,
                    candidate.track_id,
                    now,
                )

        return replace(decision, speaker_id=self._speaker_id)

    def _start_resolution(self, now: float, reason: str) -> None:
        if self._resolution_active or now < self._resolution_cooldown_until:
            return
        self._resolution_active = True
        self._resolution_started_at = now
        self._resolution_reason = reason
        self._resolution_hypotheses = []
        self._resolution_clean_frames = 0
        self._resolution_evidence_started_at = None
        self._resolution_verified_track_id = None
        self._resolution_pair_min_distance = None
        self._resolution_clear_frames = 0
        self._resolution_saw_close_pair = (
            reason == "identity_overlap_hold"
            or (
                self._last_overlap_at is not None
                and now - self._last_overlap_at <= self._crossing_recent_seconds
            )
        )
        self._resolution_coast_box = self._current_box
        self._resolution_coast_at = now if self._current_box is not None else None
        self._resolution_search_mode = False
        self._resolution_search_frame = 0
        self._resolution_last_evidence_at = now
        self._resolution_expired = False
        self._reset_single_resolution_candidate()
        logging.info(
            "Speaker identity resolution started: speaker_id=%s active_track_id=%s reason=%s",
            self._speaker_id,
            self._current_track_id,
            reason,
        )

    def _finish_resolution(self, now: float | None = None) -> None:
        was_active = self._resolution_active
        self._resolution_active = False
        self._resolution_started_at = None
        self._resolution_reason = ""
        self._resolution_hypotheses = []
        self._resolution_clean_frames = 0
        self._resolution_evidence_started_at = None
        self._resolution_verified_track_id = None
        self._resolution_pair_min_distance = None
        self._resolution_clear_frames = 0
        self._resolution_saw_close_pair = False
        self._resolution_coast_box = None
        self._resolution_coast_at = None
        self._resolution_search_mode = False
        self._resolution_search_frame = 0
        self._resolution_last_evidence_at = None
        self._resolution_expired = False
        if now is not None and was_active:
            self._resolution_cooldown_until = max(
                self._resolution_cooldown_until,
                now + self._resolution_cooldown_seconds,
            )
            self._gallery_frozen_until = max(
                self._gallery_frozen_until,
                now + self._gallery_freeze_seconds,
            )
        self._reset_single_resolution_candidate()

    def _reset_single_resolution_candidate(self) -> None:
        self._resolution_single_track_id = None
        self._resolution_single_since = None
        self._resolution_single_scores.clear()
        self._resolution_single_last_box = None
        self._resolution_single_last_at = None

    def _update_single_resolution_candidate(
        self,
        person: Detection,
        similarity: float,
        now: float,
    ) -> None:
        if self._resolution_verified_track_id is not None:
            return
        if person.track_id is None:
            self._reset_single_resolution_candidate()
            return
        continuing_path = self._single_candidate_continues_path(person, now)
        if not continuing_path:
            reference = self._resolution_coast_box or self._current_box
            geometry_match = (
                reference is not None
                and self._normalized_box_center_distance(reference, person.box)
                <= self._single_path_max_distance
            )
            if not geometry_match and similarity < self._min_similarity:
                self._reset_single_resolution_candidate()
                self._update_resolution_coast(now)
                return
            self._resolution_single_track_id = person.track_id
            self._resolution_single_since = now
            self._resolution_single_scores.clear()
        else:
            # MOT IDs are transport-level labels. Preserve evidence when the
            # observed box continues the same physical path under a new ID.
            self._resolution_single_track_id = person.track_id
            geometry_match = False
        self._resolution_single_last_box = person.box
        self._resolution_single_last_at = now
        self._resolution_single_scores.append(float(similarity))
        self._update_single_resolution_coast(
            person,
            similarity,
            now,
            follows_path=continuing_path or geometry_match,
        )
        if self._resolution_single_since is None:
            self._resolution_single_since = now
        # [room_01 수정] 겹친 사람이 있으면 단일 후보 검증에도 절대 임계값 적용.
        required_similarity = self._single_resolution_min_similarity
        if not self._resolution_all_clean:
            required_similarity = max(required_similarity, self._min_similarity)
        if (
            len(self._resolution_single_scores) < self._single_resolution_min_frames
            or now - self._resolution_single_since
            < self._single_resolution_min_seconds
            or median(self._resolution_single_scores) < required_similarity
        ):
            return
        self._resolution_verified_track_id = person.track_id
        self._resolution_reason = "identity_single_candidate_verified"
        logging.info(
            "Speaker identity single candidate verified: speaker_id=%s track_id=%s "
            "frames=%s median=%.3f",
            self._speaker_id,
            person.track_id,
            len(self._resolution_single_scores),
            median(self._resolution_single_scores),
        )

    def _update_single_resolution_coast(
        self,
        person: Detection,
        similarity: float,
        now: float,
        *,
        follows_path: bool,
    ) -> None:
        if follows_path and similarity >= self._resolution_min_similarity:
            # Rear or distant crops can fall below the verification threshold.
            # Keep following a coherent single path, but do not commit identity
            # until the stricter single-candidate evidence gate passes.
            self._resolution_coast_box = person.box
            self._resolution_coast_at = now
            return
        self._update_resolution_coast(now)

    def _single_candidate_continues_path(
        self,
        person: Detection,
        now: float,
    ) -> bool:
        if (
            self._resolution_single_last_box is None
            or self._resolution_single_last_at is None
            or now - self._resolution_single_last_at
            > self._single_path_max_gap_seconds
        ):
            return False
        return (
            self._normalized_box_center_distance(
                self._resolution_single_last_box,
                person.box,
            )
            <= self._single_path_max_distance
        )

    def _enter_resolution_search_if_due(self, now: float) -> None:
        if (
            self._resolution_search_mode
            or self._resolution_started_at is None
            or now - self._resolution_started_at
            < self._resolution_search_after_seconds
        ):
            return
        self._resolution_search_mode = True
        self._resolution_search_frame = 0
        self._resolution_reason = "identity_searching"
        self._resolution_coast_box = None
        self._resolution_coast_at = None
        logging.info(
            "Speaker identity search mode entered: speaker_id=%s active_track_id=%s "
            "elapsed=%.3f interval_frames=%s",
            self._speaker_id,
            self._current_track_id,
            now - self._resolution_started_at,
            self._resolution_search_interval_frames,
        )

    def _update_resolution_hypotheses(
        self,
        people: Sequence[Detection],
        similarities: Sequence[float],
        now: float,
    ) -> None:
        if self._resolution_evidence_started_at is None:
            self._resolution_evidence_started_at = now
        assignments = self._assign_resolution_paths(people, now)
        self._resolution_clean_frames += 1
        winner_index = max(range(len(similarities)), key=similarities.__getitem__)
        for index, (person, similarity, hypothesis) in enumerate(
            zip(people, similarities, assignments)
        ):
            previous_box = hypothesis.last_box
            previous_at = hypothesis.last_at
            if previous_at is not None and now > previous_at:
                previous_x, previous_y = previous_box.center
                current_x, current_y = person.box.center
                elapsed = now - previous_at
                measured_x = (current_x - previous_x) / elapsed
                measured_y = (current_y - previous_y) / elapsed
                hypothesis.velocity_x = 0.55 * hypothesis.velocity_x + 0.45 * measured_x
                hypothesis.velocity_y = 0.55 * hypothesis.velocity_y + 0.45 * measured_y
            hypothesis.last_box = person.box
            hypothesis.last_at = now
            hypothesis.current_track_id = person.track_id
            hypothesis.scores.append(float(similarity))
            hypothesis.observations += 1
            hypothesis.last_score = float(similarity)
            if index == winner_index:
                hypothesis.wins += 1
        self._update_resolution_coast(now)
        self._try_verify_resolution(now)

    def _update_resolution_pair_state(
        self,
        leader: BoundingBox,
        others: Sequence[BoundingBox],
    ) -> None:
        # Separation is measured around the leading candidate: that is the
        # crop whose cleanliness decides whether the evidence can be trusted.
        pair_distance = min(
            self._resolution_pair_distance(leader, other) for other in others
        )
        self._resolution_pair_min_distance = (
            pair_distance
            if self._resolution_pair_min_distance is None
            else min(self._resolution_pair_min_distance, pair_distance)
        )
        if pair_distance <= self._resolution_close_distance:
            self._resolution_saw_close_pair = True
            self._resolution_clear_frames = 0
        elif (
            self._resolution_saw_close_pair
            and pair_distance >= self._resolution_exit_distance
        ):
            self._resolution_clear_frames += 1
        else:
            self._resolution_clear_frames = 0

    def _assign_resolution_paths(
        self,
        people: Sequence[Detection],
        now: float,
    ) -> list[_ResolutionHypothesis]:
        # [room_01 수정] 오래된 가설 제거. 기존에는 사라진 사람의 가설이 남아
        # 1,2위 비교에 끼어들어 검증을 막을 수 있었음.
        # Forget paths that stopped being observed; a stale path must neither
        # win nor block verification on behalf of someone no longer visible.
        self._resolution_hypotheses = [
            hypothesis
            for hypothesis in self._resolution_hypotheses
            if hypothesis.last_at is None
            or now - hypothesis.last_at <= self._hypothesis_max_gap_seconds
        ]
        assignments: list[_ResolutionHypothesis | None] = [None] * len(people)
        available = set(range(len(self._resolution_hypotheses)))
        pairs: list[tuple[float, int, int]] = []
        for person_index, person in enumerate(people):
            for hypothesis_index, hypothesis in enumerate(self._resolution_hypotheses):
                predicted = self._predict_hypothesis_box(hypothesis, now)
                pairs.append(
                    (
                        self._normalized_box_center_distance(predicted, person.box),
                        person_index,
                        hypothesis_index,
                    )
                )
        assigned_people: set[int] = set()
        for distance, person_index, hypothesis_index in sorted(pairs):
            if (
                distance > 3.0
                or person_index in assigned_people
                or hypothesis_index not in available
            ):
                continue
            assignments[person_index] = self._resolution_hypotheses[hypothesis_index]
            assigned_people.add(person_index)
            available.remove(hypothesis_index)

        for index, person in enumerate(people):
            if assignments[index] is None:
                hypothesis = _ResolutionHypothesis(
                    path_id=self._resolution_next_path_id,
                    last_box=person.box,
                    current_track_id=person.track_id,
                    scores=deque(maxlen=12),
                    last_at=now,
                )
                self._resolution_next_path_id += 1
                self._resolution_hypotheses.append(hypothesis)
                assignments[index] = hypothesis
        return [item for item in assignments if item is not None]

    def _try_verify_resolution(self, now: float) -> None:
        if self._resolution_verified_track_id is not None:
            return
        if self._resolution_evidence_started_at is None:
            return
        if (
            self._resolution_clean_frames < self._resolution_min_frames
            or now - self._resolution_evidence_started_at
            < self._resolution_min_seconds
        ):
            return
        eligible = [
            hypothesis
            for hypothesis in self._resolution_hypotheses
            if hypothesis.current_track_id is not None
            and hypothesis.observations >= self._resolution_min_frames
            and hypothesis.scores
        ]
        if len(eligible) < 2:
            return
        ranked = sorted(
            eligible,
            key=lambda hypothesis: median(hypothesis.scores),
            reverse=True,
        )
        best, second = ranked[:2]
        best_similarity = float(median(best.scores))
        second_similarity = float(median(second.scores))
        # [room_01 수정] 승률 분모를 전체 프레임 수에서 해당 가설의 관측 수로 변경.
        # Paths come and go as raw track IDs churn, so the ratio is taken over
        # the frames in which the leading path was actually observed.
        winner_ratio = best.wins / max(1, best.observations)
        # [room_01 수정] 신규 "뚜렷한 차이" 판정.
        # room_01: 교차 후 발화자(옆/뒷모습) 0.47~0.56, 주변인 0.15~0.30.
        # 차이가 큰데도 절대 임계값 0.55 때문에 3분간 검증되지 않았음.
        decisive = (
            self._resolution_all_clean
            and best_similarity
            >= second_similarity + self._resolution_strong_margin
        )
        if (
            # [room_01 수정] 분리 조건(exit distance)을 점수가 비슷할 때만 요구.
            # 기존에는 항상 요구했는데, 옆에 서 있는 강의실 환경에서는 충족되지 않았고
            # 가까운 쌍을 본 적이 없으면 영원히 충족될 수 없었음.
            # Separation frames are owed only after an actual close pair, and
            # only while the scores are close. Overlapping crops are already
            # excluded from the evidence, and people who simply stand next to
            # each other never reach the exit distance.
            not decisive
            and self._resolution_saw_close_pair
            and self._resolution_exit_frames > 0
            and self._resolution_clear_frames < self._resolution_exit_frames
        ):
            return
        required_similarity = self._resolution_min_similarity
        if best.current_track_id != self._current_track_id and not decisive:
            # A different raw ID needs the absolute threshold unless it leads
            # every alternative decisively: rear and side views of the speaker
            # score below frontal gallery crops but still far above bystanders.
            required_similarity = max(required_similarity, self._min_similarity)
        if (
            best_similarity < required_similarity
            or best_similarity < second_similarity + self._resolution_min_margin
            or best.last_score
            < second.last_score + self._resolution_min_margin
            or winner_ratio < self._resolution_winner_ratio
        ):
            return
        self._resolution_verified_track_id = best.current_track_id
        self._resolution_reason = "identity_candidate_verified"
        logging.info(
            "Speaker identity candidate verified: speaker_id=%s track_id=%s "
            "frames=%s median=%.3f runner_up=%.3f winner_ratio=%.3f",
            self._speaker_id,
            best.current_track_id,
            self._resolution_clean_frames,
            best_similarity,
            second_similarity,
            winner_ratio,
        )

    def _update_resolution_coast(self, now: float) -> None:
        single_box = self._single_resolution_coast_box(now)
        if single_box is not None:
            self._resolution_coast_box = single_box
            self._resolution_coast_at = now
            return
        candidates = [
            hypothesis
            for hypothesis in self._resolution_hypotheses
            if hypothesis.scores and hypothesis.last_at is not None
        ]
        if candidates:
            ranked = sorted(
                candidates,
                key=lambda item: median(item.scores),
                reverse=True,
            )
            best = ranked[0]
            best_score = median(best.scores)
            runner_up_score = median(ranked[1].scores) if len(ranked) > 1 else None
            # [room_01 수정] 기존: 점수 1위 가설을 무조건 따라감 -> 주변인 사이를 오가며 카메라가 배회.
            # 변경: 최소 유사도와 2위와의 차이를 만족할 때만 따라감.
            # The camera may only follow a path that is plausibly the speaker.
            # Following whichever bystander happens to score highest makes the
            # camera wander between people.
            credible = best_score >= self._resolution_min_similarity and (
                runner_up_score is None
                or best_score >= runner_up_score + self._resolution_min_margin
            )
            if credible and now - best.last_at <= self._coast_max_seconds:
                self._resolution_coast_box = self._predict_hypothesis_box(best, now)
                self._resolution_coast_at = now
                return
        if (
            self._resolution_coast_at is None
            or now - self._resolution_coast_at > self._coast_max_seconds
        ):
            self._resolution_coast_box = None

    def _single_resolution_coast_box(self, now: float) -> BoundingBox | None:
        if (
            self._resolution_single_last_box is None
            or self._resolution_single_last_at is None
            or not self._resolution_single_scores
            or now - self._resolution_single_last_at > self._coast_max_seconds
            or median(self._resolution_single_scores) < self._resolution_min_similarity
        ):
            return None
        return self._resolution_single_last_box

    def _predict_hypothesis_box(
        self,
        hypothesis: _ResolutionHypothesis,
        now: float,
    ) -> BoundingBox:
        if hypothesis.last_at is None:
            return hypothesis.last_box
        elapsed = min(self._coast_max_seconds, max(0.0, now - hypothesis.last_at))
        width = max(1.0, hypothesis.last_box.x2 - hypothesis.last_box.x1)
        height = max(1.0, hypothesis.last_box.y2 - hypothesis.last_box.y1)
        shift_x = max(-2.0 * width, min(2.0 * width, hypothesis.velocity_x * elapsed))
        shift_y = max(-1.0 * height, min(1.0 * height, hypothesis.velocity_y * elapsed))
        return BoundingBox(
            hypothesis.last_box.x1 + shift_x,
            hypothesis.last_box.y1 + shift_y,
            hypothesis.last_box.x2 + shift_x,
            hypothesis.last_box.y2 + shift_y,
        )

    @staticmethod
    def _resolution_pair_distance(first: BoundingBox, second: BoundingBox) -> float:
        first_x, first_y = first.center
        second_x, second_y = second.center
        width = max(
            1.0,
            ((first.x2 - first.x1) + (second.x2 - second.x1)) / 2,
        )
        height = max(
            1.0,
            ((first.y2 - first.y1) + (second.y2 - second.y1)) / 2,
        )
        return float(np.hypot((second_x - first_x) / width, (second_y - first_y) / height))

    def _log_scores(
        self,
        people: Sequence[Detection],
        scored_by_identity: dict[int, float],
    ) -> None:
        summary = ",".join(
            f"track={person.track_id}:sim={scored_by_identity[id(person)]:.3f}"
            for person in people
        )
        logging.info(
            "Speaker ReID candidates: speaker_id=%s active_track_id=%s gallery=%s %s",
            self._speaker_id,
            self._current_track_id,
            self.gallery_size,
            summary,
        )

    def _reset_appearance_identity(self) -> None:
        previous_samples = self.gallery_size
        self._anchor_gallery.clear()
        self._adaptive_gallery.clear()
        self._negative_anchor_gallery.clear()
        self._negative_adaptive_gallery.clear()
        self._last_gallery_at = None
        self._cache_frame = None
        self._cache = {}
        self._score_after_separation = False
        self._hold_ambiguous_frame = False
        self._remaining_score_frames = 0
        self._active_was_missing = False
        self._current_track_id = None
        self._current_box = None
        self._finish_resolution()
        logging.info(
            "Speaker ReID gallery reset: speaker_id=%s previous_samples=%s",
            self._speaker_id,
            previous_samples,
        )

    def _needs_appearance(self, people: Sequence[Detection]) -> bool:
        if not self._identity_active:
            return False
        active = next(
            (person for person in people if person.track_id == self._current_track_id),
            None,
        )
        if active is None:
            self._active_was_missing = True
        overlapping = any(
            self._box_iou(first.box, second.box) >= self._trigger_overlap
            for index, first in enumerate(people)
            for second in people[index + 1 :]
        )
        if overlapping:
            # Crops are mixed at maximum overlap. Remember the ambiguity and
            # compare clean crops on the first separated frame instead.
            self._score_after_separation = True
            self._hold_ambiguous_frame = True
            return False
        if active is None:
            return True
        if self._active_was_missing:
            # A raw ID can reappear on the wrong physical person after a
            # crossing. Verify it, and all alternatives, for several clean
            # frames instead of trusting the numeric ID.
            self._active_was_missing = False
            self._remaining_score_frames = max(
                self._remaining_score_frames,
                self._score_burst_frames,
            )
        if self._score_after_separation:
            self._score_after_separation = False
            self._remaining_score_frames = self._score_burst_frames
        if self._remaining_score_frames > 0:
            self._remaining_score_frames -= 1
            return True
        return False

    # [room_01 수정] 신규 메서드(기존 _people_overlap 대체): 겹침에 관여한 사람 집합.
    def _occluded_people(self, people: Sequence[Detection]) -> set[int]:
        """Return ``id()`` of every person whose box overlaps another person."""
        occluded: set[int] = set()
        for index, first in enumerate(people):
            for second in people[index + 1 :]:
                if self._box_iou(first.box, second.box) >= self._trigger_overlap:
                    occluded.add(id(first))
                    occluded.add(id(second))
        return occluded

    # [room_01 수정] 신규 메서드: 대상/검증 후보/유효한 coast가 없는 시간이
    # timeout을 넘으면 expired 표시 -> core.py가 release.
    def _update_resolution_expiry(self, now: float, *, active_visible: bool) -> None:
        if (
            active_visible
            or self._resolution_verified_track_id is not None
            or self._resolution_last_evidence_at is None
        ):
            self._resolution_last_evidence_at = now
            return
        if self._resolution_coast_box is not None and self._resolution_coast_at is not None:
            # A coast estimate counts from when it was last refreshed, not
            # for as long as it lingers.
            self._resolution_last_evidence_at = max(
                self._resolution_last_evidence_at,
                self._resolution_coast_at,
            )
        if (
            self._resolution_timeout_seconds > 0
            and not self._resolution_expired
            and now - self._resolution_last_evidence_at
            >= self._resolution_timeout_seconds
        ):
            self._resolution_expired = True
            logging.info(
                "Speaker identity resolution timed out: speaker_id=%s "
                "active_track_id=%s without_evidence=%.1fs",
                self._speaker_id,
                self._current_track_id,
                now - self._resolution_last_evidence_at,
            )

    def _deduplicate_people(
        self,
        people: Sequence[Detection],
    ) -> list[Detection]:
        """Collapse nested full/partial detections of one physical person.

        The detector can emit a full body box and a top-aligned upper-body box
        with different raw track IDs. Treating that pair as two people creates
        a false crossing and can permanently hold PTZ motion.
        """

        representatives: list[Detection] = []
        for person in sorted(people, key=lambda item: item.box.area, reverse=True):
            if any(
                self._same_physical_person_boxes(person.box, existing.box)
                for existing in representatives
            ):
                continue
            representatives.append(person)
        return representatives

    @classmethod
    def _same_physical_person_boxes(
        cls,
        first: BoundingBox,
        second: BoundingBox,
    ) -> bool:
        intersection = cls._box_intersection(first, second)
        smaller_area = min(first.area, second.area)
        if smaller_area <= 0 or intersection / smaller_area < 0.90:
            return False
        max_height = max(first.y2 - first.y1, second.y2 - second.y1, 1.0)
        max_width = max(first.x2 - first.x1, second.x2 - second.x1, 1.0)
        first_x, _ = first.center
        second_x, _ = second.center
        return (
            abs(first.y1 - second.y1) / max_height <= 0.10
            and abs(first_x - second_x) / max_width <= 0.25
        )

    @staticmethod
    def _box_intersection(first: BoundingBox, second: BoundingBox) -> float:
        x1 = max(first.x1, second.x1)
        y1 = max(first.y1, second.y1)
        x2 = min(first.x2, second.x2)
        y2 = min(first.y2, second.y2)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _trusted_gallery_target(
        self,
        decision: HandoffDecision,
        people: Sequence[Detection],
        markers: Sequence[Detection],
    ) -> Detection | None:
        if (
            self._speaker_id is None
            or decision.state != "tracking"
            or decision.target_track_id is None
            or len(markers) != 1
        ):
            return None
        target = next(
            (person for person in people if person.track_id == decision.target_track_id),
            None,
        )
        if (
            target is None
            or target.box.area < self._min_crop_area
            or not target.box.contains(markers[0].box.center)
            or (
                not decision.speaker_changed
                and target.appearance_similarity is not None
                and target.appearance_similarity < self._min_similarity
            )
        ):
            return None
        if any(
            person is not target
            and self._box_iou(target.box, person.box) > self._gallery_max_overlap
            for person in people
        ):
            return None
        return target

    def _gallery_sample_due(self, now: float) -> bool:
        if (
            len(self._anchor_gallery) >= self._anchor_gallery.maxlen
            and now < self._gallery_frozen_until
        ):
            return False
        # [room_01 수정] 갤러리가 가득 차면 샘플 간격 3배.
        # 이유: 0.35초마다 ReID 추론이 돌아 추적 중 fps가 7~9로 떨어졌음.
        interval = self._gallery_interval_seconds
        if self.gallery_size >= (
            self._anchor_gallery.maxlen + self._adaptive_gallery.maxlen
        ):
            # A full gallery only needs occasional refreshes; every sample
            # costs appearance inference that stalls the tracking loop.
            interval *= 3
        return self._last_gallery_at is None or now - self._last_gallery_at >= interval

    def _add_gallery_feature(
        self,
        feature: np.ndarray,
        now: float,
        track_id: int | None,
    ) -> None:
        feature = self._normalize(feature)
        gallery = self._positive_gallery()
        if gallery:
            nearest = max(float(feature @ item) for item in gallery)
            if nearest >= self._gallery_novelty_similarity:
                self._last_gallery_at = now
                return
        if len(self._anchor_gallery) < self._anchor_gallery.maxlen:
            self._anchor_gallery.append(feature)
            gallery_kind = "anchor"
        elif now >= self._gallery_frozen_until:
            self._adaptive_gallery.append(feature)
            gallery_kind = "adaptive"
        else:
            return
        self._last_gallery_at = now
        logging.info(
            "Speaker ReID gallery updated: speaker_id=%s track_id=%s samples=%s "
            "anchors=%s adaptive=%s kind=%s",
            self._speaker_id,
            track_id,
            self.gallery_size,
            len(self._anchor_gallery),
            len(self._adaptive_gallery),
            gallery_kind,
        )

    def _add_negative_gallery_feature(
        self,
        feature: np.ndarray,
        target_feature: np.ndarray,
        track_id: int | None,
        now: float,
    ) -> None:
        feature = self._normalize(feature)
        # Do not create an exclusion identity from a likely duplicate crop of
        # the speaker. This also protects against transient detector splits.
        if float(feature @ target_feature) >= 0.90:
            return
        negative_gallery = self._negative_gallery_features()
        if negative_gallery:
            nearest = max(float(feature @ item) for item in negative_gallery)
            if nearest >= self._gallery_novelty_similarity:
                return
        if len(self._negative_anchor_gallery) < self._negative_anchor_gallery.maxlen:
            self._negative_anchor_gallery.append(feature)
        elif now >= self._gallery_frozen_until:
            self._negative_adaptive_gallery.append(feature)
        else:
            return
        logging.info(
            "Speaker ReID exclusion gallery updated: speaker_id=%s "
            "track_id=%s samples=%s",
            self._speaker_id,
            track_id,
            self.negative_gallery_size,
        )

    def _gallery_similarity(self, feature: np.ndarray) -> float:
        feature = self._normalize(feature)
        anchor_similarity = self._top_k_similarity(feature, self._anchor_gallery)
        adaptive_similarity = self._top_k_similarity(feature, self._adaptive_gallery)
        if anchor_similarity is None:
            if adaptive_similarity is None:
                return 0.0
            positive_similarity = adaptive_similarity
        elif adaptive_similarity is None:
            positive_similarity = anchor_similarity
        else:
            # Marker-confirmed startup crops remain the identity reference;
            # adaptive samples only cover pose/view changes and cannot replace it.
            positive_similarity = 0.8 * anchor_similarity + 0.2 * adaptive_similarity
        negative_gallery = self._negative_gallery_features()
        if not negative_gallery or self._negative_gallery_weight <= 0:
            return positive_similarity
        negative_similarity = max(
            self._similarity(feature, item) for item in negative_gallery
        )
        score = (
            positive_similarity
            + self._negative_gallery_weight * (1.0 - negative_similarity)
        ) / (1.0 + self._negative_gallery_weight)
        return min(1.0, max(0.0, score))

    def _positive_gallery(self) -> list[np.ndarray]:
        return [*self._anchor_gallery, *self._adaptive_gallery]

    def _negative_gallery_features(self) -> list[np.ndarray]:
        return [*self._negative_anchor_gallery, *self._negative_adaptive_gallery]

    def _similarity(self, first: np.ndarray, second: np.ndarray) -> float:
        """Calibrated similarity for identity scores.

        Near-duplicate checks (gallery novelty, duplicate speaker crop) compare
        raw cosines instead: "almost the same crop" does not depend on the model.
        """
        return self._similarity_scale * float(first @ second) + self._similarity_offset

    def _top_k_similarity(
        self,
        feature: np.ndarray,
        gallery: Sequence[np.ndarray],
    ) -> float | None:
        if not gallery:
            return None
        similarities = sorted(
            (self._similarity(feature, item) for item in gallery),
            reverse=True,
        )
        count = min(self._top_k_gallery, len(similarities))
        return sum(similarities[:count]) / count

    def _features(
        self,
        frame,
        people: Sequence[Detection],
    ) -> list[np.ndarray]:
        if frame is not self._cache_frame:
            self._cache_frame = frame
            self._cache = {}
        missing = [person for person in people if self._box_key(person.box) not in self._cache]
        if missing:
            encoded = self._encoder.encode(frame, missing)
            if len(encoded) != len(missing):
                raise RuntimeError("appearance encoder returned an unexpected feature count")
            for person, feature in zip(missing, encoded):
                self._cache[self._box_key(person.box)] = self._normalize(feature)
        return [self._cache[self._box_key(person.box)] for person in people]

    @staticmethod
    def _normalize(feature: np.ndarray) -> np.ndarray:
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("appearance encoder returned an invalid feature")
        return vector / norm

    @staticmethod
    def _box_key(box: BoundingBox) -> tuple[float, float, float, float]:
        return (box.x1, box.y1, box.x2, box.y2)

    @staticmethod
    def _box_iou(first: BoundingBox, second: BoundingBox) -> float:
        x1 = max(first.x1, second.x1)
        y1 = max(first.y1, second.y1)
        x2 = min(first.x2, second.x2)
        y2 = min(first.y2, second.y2)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = first.area + second.area - intersection
        return 0.0 if union <= 0 else intersection / union

    @staticmethod
    def _normalized_box_center_distance(
        reference: BoundingBox,
        candidate: BoundingBox,
    ) -> float:
        reference_x, reference_y = reference.center
        candidate_x, candidate_y = candidate.center
        width = max(1.0, reference.x2 - reference.x1)
        height = max(1.0, reference.y2 - reference.y1)
        return float(
            np.hypot(
                (candidate_x - reference_x) / width,
                (candidate_y - reference_y) / height,
            )
        )
