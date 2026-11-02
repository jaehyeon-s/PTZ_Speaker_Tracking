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
    appearance_similarity: Optional[float] = None


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
    provisional_target: bool = False
    speaker_id: Optional[int] = None
    speaker_changed: bool = False
    # The marker holder being confirmed while the previous target is not in
    # view; the camera may follow it meanwhile (see --switch-follow-candidate).
    candidate_box: Optional[BoundingBox] = None


@dataclass(frozen=True)
class CandidateScore:
    track_id: Optional[int]
    score: float
    box: BoundingBox
    is_active: bool = False
    area_growth_ratio: Optional[float] = None


@dataclass(frozen=True)
class IdentityResolutionContext:
    """Identity evidence supplied independently from mutable MOT track IDs."""

    active: bool = False
    verified_track_id: Optional[int] = None
    reason: str = ""
    coast_box: Optional[BoundingBox] = None
    # [room_01 수정] 신규 필드. identity.py가 탐색 시간 초과를 알리는 신호.
    # The search ran out of evidence; the target must be released.
    expired: bool = False


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
    # holder_geometry only: points per box-size fraction the marker lies
    # outside the chest area, and per unit of |ln(distance ratio)|.
    _HOLDER_CHEST_BAND = (0.25, 0.5)
    _HOLDER_WIDTH_BAND = (0.25, 0.75)
    _HOLDER_POSITION_PENALTY_WEIGHT = 200.0
    _HOLDER_DISTANCE_PENALTY_WEIGHT = 60.0
    _HOLDER_SAME_PERSON_IOU = 0.5
    # Recovery must tolerate partial-person and full-person boxes alternating
    # during real-world occlusion. These bounds are only used together with
    # temporal position, top-edge and width continuity; area alone never
    # establishes identity.
    _RECOVERY_MIN_AREA_RATIO = 0.2
    _RECOVERY_MAX_AREA_RATIO = 4.5
    _RECOVERY_MIN_WIDTH_RATIO = 0.4
    _RECOVERY_MAX_WIDTH_RATIO = 2.5
    _RECOVERY_MAX_TOP_DISTANCE = 0.35
    _RECOVERY_MAX_HORIZONTAL_DISTANCE = 1.25
    _RECOVERY_DROPOUT_SECONDS = 0.35

    def __init__(
        self,
        confirmation_seconds: float = 3.0,
        release_grace_seconds: float = 3.0,
        marker_dropout_seconds: float = 0.5,
        switch_score_margin: float = 12.0,
        switch_cooldown_seconds: float = 1.0,
        area_growth_penalty_threshold: float = 1.8,
        recovery_confirmation_seconds: float = 0.12,
        recovery_min_iou: float = 0.2,
        recovery_max_center_distance: float = 0.75,
        recovery_score_margin: float = 0.25,
        recovery_appearance_weight: float = 2.0,
        recovery_min_appearance: float = 0.55,
        recovery_appearance_margin: float = 0.08,
        recovery_min_track_observations: int = 2,
        # [room_01 수정] 신규 인자 2개: 놓친 대상을 마커 N회 관측(창 W초)으로 재등록.
        reacquire_marker_observations: int = 3,
        reacquire_marker_window_seconds: float = 3.0,
        # Shortest time the marker must stay with one new holder before a lost
        # speaker is re-registered; one passing glimpse must not be enough.
        reacquire_marker_min_seconds: float = 0.5,
        recovery_min_area_ratio: Optional[float] = None,
        # False only to compare against the behaviour before the bystander
        # rules (pi_unity_04/05): a bystander's marker glimpse re-registers
        # quickly and ID recovery may pick a bystander or a non-holder.
        bystander_rules: bool = True,
        # When the marker centre lies in several people's boxes, judge the
        # holder by where the marker sits on each body and whether the marker
        # size fits each person's distance, instead of preferring the smaller
        # box. Close calls are not attributed until the boxes separate.
        holder_geometry: bool = False,
        holder_marker_meters: float = 0.093,
        holder_person_meters: float = 1.7,
        holder_ambiguity_margin: float = 20.0,
    ) -> None:
        if (
            confirmation_seconds < 0
            or release_grace_seconds < 0
            or marker_dropout_seconds < 0
            or switch_score_margin < 0
            or switch_cooldown_seconds < 0
            or area_growth_penalty_threshold < 1.0
            or recovery_confirmation_seconds < 0
            or not 0 <= recovery_min_iou <= 1
            or recovery_max_center_distance < 0
            or recovery_score_margin < 0
            or recovery_appearance_weight < 0
            or not -1 <= recovery_min_appearance <= 1
            or recovery_appearance_margin < 0
            or recovery_min_track_observations < 1
            or reacquire_marker_observations < 1
            or reacquire_marker_window_seconds < 0
            or reacquire_marker_min_seconds < 0
            or holder_marker_meters <= 0
            or holder_person_meters <= 0
            or holder_ambiguity_margin < 0
        ):
            raise ValueError("invalid tracker scoring parameters")
        self.confirmation_seconds = confirmation_seconds
        self.release_grace_seconds = release_grace_seconds
        self.marker_dropout_seconds = marker_dropout_seconds
        self.switch_score_margin = switch_score_margin
        self.switch_cooldown_seconds = switch_cooldown_seconds
        self.area_growth_penalty_threshold = area_growth_penalty_threshold
        self.recovery_confirmation_seconds = recovery_confirmation_seconds
        self.recovery_min_iou = recovery_min_iou
        # Lower this to accept a head-only box as the same person when the
        # body becomes hidden (desk, lectern); None keeps the built-in limit.
        self.recovery_min_area_ratio = (
            self._RECOVERY_MIN_AREA_RATIO
            if recovery_min_area_ratio is None
            else recovery_min_area_ratio
        )
        self.recovery_max_center_distance = recovery_max_center_distance
        self.recovery_score_margin = recovery_score_margin
        self.recovery_appearance_weight = recovery_appearance_weight
        self.recovery_min_appearance = recovery_min_appearance
        self.recovery_appearance_margin = recovery_appearance_margin
        self.recovery_min_track_observations = recovery_min_track_observations
        self.reacquire_marker_observations = reacquire_marker_observations
        self.reacquire_marker_window_seconds = reacquire_marker_window_seconds
        self.reacquire_marker_min_seconds = reacquire_marker_min_seconds
        # Tracks seen in the same frame as the confirmed speaker. Such a track
        # is another person, so it cannot be the speaker reappearing.
        self._bystander_track_ids: set[int] = set()
        self.bystander_rules = bystander_rules
        self.holder_geometry = holder_geometry
        self.holder_marker_meters = holder_marker_meters
        self.holder_person_meters = holder_person_meters
        self.holder_ambiguity_margin = holder_ambiguity_margin
        self._holder_ambiguous = False
        self._candidate_since: Optional[float] = None
        self._candidate_target: Optional[BoundingBox] = None
        self._candidate_track_id: Optional[int] = None
        self._last_candidate_match_at: Optional[float] = None
        self._active_target: Optional[BoundingBox] = None
        self._active_track_id: Optional[int] = None
        self._last_match_at: Optional[float] = None
        self._view_scale_at_target: Optional[float] = None
        self._grace_extension = 0.0
        self._size_scale = 1.0
        self._last_handoff_at: Optional[float] = None
        self._recovery_candidate_since: Optional[float] = None
        self._recovery_candidate_track_id: Optional[int] = None
        self._recovery_candidate_track_observations = 0
        self._recovery_candidate_target: Optional[BoundingBox] = None
        self._recovery_candidate_last_at: Optional[float] = None
        # [room_01 수정] 놓친 대상 마커 재등록용 상태.
        # _released_after_loss: 놓쳐서 해제된 뒤에도 같은 빠른 재등록 규칙을 쓰기 위한 표시.
        self._lost_marker_since: Optional[float] = None
        self._lost_marker_track_id: Optional[int] = None
        self._lost_marker_target: Optional[BoundingBox] = None
        self._lost_marker_observations = 0
        self._lost_marker_new_appearance = False
        self._released_after_loss = False

    def update(
        self,
        detections: Iterable[Detection],
        now: float,
        requested_track_id: Optional[int] = None,
        identity_resolution: Optional[IdentityResolutionContext] = None,
        view_scale: Optional[float] = None,
        grace_extension: float = 0.0,
    ) -> HandoffDecision:
        # Extra seconds of grace for this update, e.g. while the target is
        # predicted to be still inside the picture behind an obstacle.
        self._grace_extension = max(0.0, grace_extension)
        # view_scale is proportional to image magnification (focal length). When
        # it is given, size checks against the remembered target box allow for
        # the zoom change since that box was seen.
        if view_scale is not None and self._view_scale_at_target is not None and view_scale > 0:
            self._size_scale = view_scale / self._view_scale_at_target
        else:
            self._size_scale = 1.0
        decision = self._update(detections, now, requested_track_id, identity_resolution)
        if view_scale is not None and decision.state == "tracking":
            self._view_scale_at_target = view_scale
        return decision

    def _update(
        self,
        detections: Iterable[Detection],
        now: float,
        requested_track_id: Optional[int] = None,
        identity_resolution: Optional[IdentityResolutionContext] = None,
    ) -> HandoffDecision:
        detections = list(detections)
        markers = [d for d in detections if d.label == "marker"]
        people = [d for d in detections if d.label == "person"]
        marker = markers[0].box if len(markers) == 1 else None
        scored_candidates = self._score_marker_candidates(marker, people)
        candidate_scores = tuple(score for _, score in scored_candidates)
        matched_person = scored_candidates[0][0] if scored_candidates else None
        self._holder_ambiguous = self._is_holder_ambiguous(scored_candidates)
        if self._holder_ambiguous:
            # Nobody is credited with the marker until the boxes separate.
            matched_person = None

        if requested_track_id is not None:
            requested_person = self._find_track(people, requested_track_id)
            if requested_person is not None:
                handoff = self._active_track_id != requested_track_id
                self._released_after_loss = False
                self._active_track_id = requested_track_id
                self._active_target = requested_person.box
                self._last_match_at = now
                self._reset_candidate()
                self._reset_recovery_candidate()
                self._reset_lost_marker_candidate()
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

        if identity_resolution is not None and identity_resolution.active and marker is None:
            verified_person = self._find_track(
                people,
                identity_resolution.verified_track_id,
            )
            if verified_person is None:
                active_person = self._find_track(people, self._active_track_id)
                if active_person is not None:
                    self._active_target = active_person.box
                    self._last_match_at = now
                    target_box = active_person.box
                    provisional_target = False
                    self._reset_recovery_candidate()
                    self._reset_lost_marker_candidate()
                # [room_01 수정] 탐색 시간 초과 시 release.
                # 기존: identity_resolving에서 빠져나올 방법이 없어 room_01은 3분간 유지됨.
                elif identity_resolution.expired:
                    self._reset_candidate()
                    return self._release(
                        marker,
                        candidate_scores,
                        reason="identity_resolution_timeout",
                    )
                else:
                    recovery_candidate = self._select_recovery_candidate(
                        people,
                        now,
                        require_verified_appearance=True,
                    )
                    if recovery_candidate is not None:
                        self._last_match_at = now
                        self._recovery_candidate_target = recovery_candidate.box
                        self._recovery_candidate_last_at = now
                        self._reset_candidate()
                        return HandoffDecision(
                            "identity_resolving",
                            recovery_candidate.box,
                            marker,
                            self._active_track_id,
                            reason=identity_resolution.reason
                            or "identity_evidence_collecting",
                            candidate_scores=candidate_scores,
                            provisional_target=True,
                        )
                    self._expire_recovery_candidate(now)
                    # [room_01 수정] 기존: target_box = coast_box or self._active_target
                    # 변경: coast_box만 사용(없으면 None -> 카메라 정지).
                    # 이유: _active_target은 놓치기 전 화면 좌표라 카메라가 돌면 의미가 없는데,
                    # 그 좌표를 향해 계속 회전해 사람이 없는 벽까지 돌아갔음.
                    # Without a fresh coast estimate there is nothing to aim
                    # at. The last confirmed box is a stale pixel position:
                    # steering toward it pans the camera away indefinitely.
                    target_box = identity_resolution.coast_box
                    provisional_target = target_box is not None
                self._reset_candidate()
                return HandoffDecision(
                    "identity_resolving",
                    target_box,
                    marker,
                    self._active_track_id,
                    reason=identity_resolution.reason or "identity_evidence_collecting",
                    candidate_scores=candidate_scores,
                    provisional_target=provisional_target,
                )
            previous_track_id = self._active_track_id
            self._active_target = verified_person.box
            self._active_track_id = verified_person.track_id
            self._last_match_at = now
            self._last_handoff_at = now
            self._reset_candidate()
            self._reset_recovery_candidate()
            self._reset_lost_marker_candidate()
            return HandoffDecision(
                "tracking",
                verified_person.box,
                marker,
                verified_person.track_id,
                reason="identity_verified_target",
                emit_handoff=previous_track_id != verified_person.track_id,
                candidate_scores=candidate_scores,
            )

        active_person = self._find_track(people, self._active_track_id)
        if active_person is not None:
            appearance_candidate = self._appearance_switch_candidate(
                active_person,
                people,
            )
            if marker is None and appearance_candidate is not None:
                return self._update_recovery_candidate(
                    appearance_candidate,
                    marker,
                    now,
                    candidate_scores,
                    success_reason="appearance_reidentified_target",
                )
            self._reset_recovery_candidate()
            self._reset_lost_marker_candidate()
            self._bystander_track_ids.discard(self._active_track_id)
            self._bystander_track_ids.update(
                person.track_id
                for person in people
                if person.track_id is not None
                and person.track_id != self._active_track_id
                # A duplicate or partial box of the speaker overlaps the
                # speaker's box; only a clearly separate box is someone else.
                and self._box_iou(active_person.box, person.box)
                < self._BYSTANDER_MAX_IOU
            )
            previous_active_target = self._active_target
            self._active_target = active_person.box
            self._last_match_at = now
            if matched_person is not None and not self._is_active_match(
                matched_person
            ) and (
                self._appearance_allows_reassignment(matched_person)
                or self._appearance_indicates_new_speaker(matched_person)
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
                    speaker_change=self._is_distinct_speaker_candidate(
                        active_person,
                        matched_person,
                    ),
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

        # [room_01 수정] 대상이 안 보일 때의 마커 처리를 _update_lost_target_marker로 통합.
        # 기존: ReID 유사도 < 0.55면 "새 발화자"로 보고 2초 연속 확인(switching),
        # 아니면 0.12초 복구 경로. 마커가 간헐적으로만 보이면 둘 다 매번 리셋되어
        # room_01에서 마커를 다시 보여도 재등록되지 않았음.
        if matched_person is not None and (
            self._active_target is not None or self._released_after_loss
        ):
            if self.bystander_rules and matched_person.track_id in self._bystander_track_ids:
                # The holder was in view together with the speaker, so this is
                # another person: either the marker really changed hands or it
                # only crossed their box while the speaker's own box was
                # missing (pi_unity_04: a seated neighbour for 0.14 s). Both
                # need the full confirmation time, not the quick re-registration.
                self._reset_lost_marker_candidate()
                self._last_match_at = now
                has_target = self._active_target is not None
                return self._update_candidate(
                    matched_person,
                    marker,
                    now,
                    pending_state="switching" if has_target else "confirming",
                    pending_target=self._active_target if has_target else matched_person.box,
                    pending_track_id=(
                        self._active_track_id if has_target else matched_person.track_id
                    ),
                    candidate_scores=candidate_scores,
                    speaker_change=True,
                    # --switch-follow-candidate may steer to this holder while
                    # the confirmed target is out of view.
                    target_missing=has_target,
                )
            # The confirmed target is not visible (or was given up as lost),
            # so the marker is the only ground truth left. It decides who the
            # speaker is, whatever the appearance model thinks of the holder.
            return self._update_lost_target_marker(
                matched_person,
                marker,
                now,
                candidate_scores,
            )

        if matched_person is not None:
            # [room_01 수정] 기존에는 유사도가 낮으면 등록 자체가 막혔음(_appearance_allows_reassignment).
            # 대상이 없을 때는 항상 등록하고, 유사도가 낮으면 갤러리를 새로 만듦.
            # No target is held, so the marker always registers its holder. A
            # holder who does not match the previous gallery starts a new one.
            self._last_match_at = now
            return self._update_candidate(
                matched_person,
                marker,
                now,
                pending_state="confirming",
                pending_target=matched_person.box,
                pending_track_id=matched_person.track_id,
                candidate_scores=candidate_scores,
                speaker_change=self._appearance_indicates_new_speaker(matched_person),
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
                candidate_box=self._candidate_target if self._active_target is not None else None,
            )

        self._reset_candidate()
        if self._active_target is None:
            self._reset_recovery_candidate()
            return HandoffDecision(
                "default",
                None,
                marker,
                reason=(
                    "marker_holder_ambiguous"
                    if self._holder_ambiguous
                    else "no_confirmed_target"
                ),
                candidate_scores=candidate_scores,
            )

        assert self._last_match_at is not None
        if now - self._last_match_at <= self.release_grace_seconds + self._grace_extension:
            recovery_candidate = self._select_recovery_candidate(
                people, now, marker=marker
            )
            if recovery_candidate is not None:
                return self._update_recovery_candidate(
                    recovery_candidate,
                    marker,
                    now,
                    candidate_scores,
                )
            self._expire_recovery_candidate(now)
            return HandoffDecision(
                "grace",
                self._active_target,
                marker,
                self._active_track_id,
                reason="target_lost_grace",
                candidate_scores=candidate_scores,
            )

        return self._release(
            marker,
            candidate_scores,
            reason="target_released_after_grace",
        )

    # [room_01 수정] 기존 update() 끝의 release 코드를 메서드로 분리
    # (grace 만료와 탐색 시간 초과가 함께 사용).
    def _release(
        self,
        marker: Optional[BoundingBox],
        candidate_scores: tuple[CandidateScore, ...],
        reason: str,
    ) -> HandoffDecision:
        previous_target = self._active_target
        previous_track_id = self._active_track_id
        self._active_target = None
        self._active_track_id = None
        self._last_match_at = None
        self._reset_recovery_candidate()
        self._reset_lost_marker_candidate()
        # The speaker was lost rather than deliberately dropped: let the
        # marker bring them back as quickly as while the search was running.
        self._released_after_loss = True
        return HandoffDecision(
            "default",
            previous_target,
            marker,
            previous_track_id,
            reason=reason,
            emit_release=True,
            candidate_scores=candidate_scores,
        )

    def _reset_lost_marker_candidate(self) -> None:
        self._lost_marker_since = None
        self._lost_marker_track_id = None
        self._lost_marker_target = None
        self._lost_marker_observations = 0
        self._lost_marker_new_appearance = False

    # [room_01 수정] 신규 메서드. 같은 사람에게서 마커가 window 안에 N번 보이면 재등록.
    # 연속 유지 시간 대신 횟수를 세므로 간헐 검출에도 동작함.
    def _update_lost_target_marker(
        self,
        person: Detection,
        marker: Optional[BoundingBox],
        now: float,
        candidate_scores: tuple[CandidateScore, ...],
    ) -> HandoffDecision:
        """Re-register the speaker from marker sightings while the target is lost.

        A small or moving marker is detected only intermittently, so sightings
        are counted inside a time window instead of requiring an unbroken dwell.
        """
        self._reset_candidate()
        self._last_match_at = now
        same_holder = (
            self._lost_marker_since is not None
            and self._lost_marker_target is not None
            and now - self._lost_marker_since <= self.reacquire_marker_window_seconds
            and (
                (
                    person.track_id is not None
                    and person.track_id == self._lost_marker_track_id
                )
                or self._box_iou(self._lost_marker_target, person.box)
                >= self.recovery_min_iou
                or self._normalized_box_center_distance(
                    self._lost_marker_target,
                    person.box,
                )
                <= self.recovery_max_center_distance
            )
        )
        if not same_holder:
            self._reset_lost_marker_candidate()
            self._lost_marker_since = now
        self._lost_marker_track_id = person.track_id
        self._lost_marker_target = person.box
        self._lost_marker_observations += 1
        if self._appearance_indicates_new_speaker(person):
            self._lost_marker_new_appearance = True
        if (
            person.track_id is not None
            and self._lost_marker_observations >= self.reacquire_marker_observations
            and now - self._lost_marker_since >= self.reacquire_marker_min_seconds
        ):
            # Low similarity means another person took the marker, or the same
            # person no longer looks like the gallery (for example after
            # putting on a jacket). Both need a fresh appearance gallery.
            speaker_changed = self._lost_marker_new_appearance
            self._active_target = person.box
            self._active_track_id = person.track_id
            self._released_after_loss = False
            self._last_handoff_at = now
            self._reset_recovery_candidate()
            self._reset_lost_marker_candidate()
            return HandoffDecision(
                "tracking",
                person.box,
                marker,
                person.track_id,
                reason=(
                    "confirmed_marker_transfer"
                    if speaker_changed
                    else "marker_reacquired_target"
                ),
                emit_handoff=True,
                candidate_scores=candidate_scores,
                speaker_changed=speaker_changed,
            )
        return HandoffDecision(
            "recovering",
            person.box,
            marker,
            self._active_track_id,
            reason="marker_reacquire_confirming",
            candidate_scores=candidate_scores,
        )

    def _reset_candidate(self) -> None:
        self._candidate_since = None
        self._candidate_target = None
        self._candidate_track_id = None
        self._last_candidate_match_at = None

    def _reset_recovery_candidate(self) -> None:
        self._recovery_candidate_since = None
        self._recovery_candidate_track_id = None
        self._recovery_candidate_track_observations = 0
        self._recovery_candidate_target = None
        self._recovery_candidate_last_at = None

    def _expire_recovery_candidate(self, now: float) -> None:
        if (
            self._recovery_candidate_last_at is None
            or now - self._recovery_candidate_last_at
            > self._RECOVERY_DROPOUT_SECONDS
        ):
            self._reset_recovery_candidate()

    def _update_recovery_candidate(
        self,
        person: Detection,
        marker: Optional[BoundingBox],
        now: float,
        candidate_scores: tuple[CandidateScore, ...],
        success_reason: str = "target_recovered_new_track_id",
    ) -> HandoffDecision:
        continuing_path = (
            self._recovery_candidate_target is not None
            and self._recovery_candidate_last_at is not None
            and now - self._recovery_candidate_last_at
            <= self._RECOVERY_DROPOUT_SECONDS
        )
        same_track = (
            person.track_id is not None
            and person.track_id == self._recovery_candidate_track_id
        )
        # [room_01 수정] (room_01 이전 미커밋 수정) 확인 타이머가 None인 채로
        # 복구 경로에 들어와 assert로 죽던 문제 수정.
        # The identity-resolving branch of update() records a candidate box
        # without starting the confirmation timer; start it here in that case.
        if not continuing_path or self._recovery_candidate_since is None:
            self._recovery_candidate_since = now
            self._recovery_candidate_track_observations = 0
        if same_track:
            self._recovery_candidate_track_observations += 1
        elif person.track_id is not None:
            self._recovery_candidate_track_observations = 1
        else:
            self._recovery_candidate_track_observations = 0
        self._recovery_candidate_track_id = person.track_id
        self._recovery_candidate_target = person.box
        self._recovery_candidate_last_at = now
        # A coherent physical observation is evidence that the speaker has not
        # disappeared, even if the underlying MOT ID is temporarily absent.
        self._last_match_at = now
        assert self._recovery_candidate_since is not None
        if (
            person.track_id is not None
            and now - self._recovery_candidate_since
            >= self.recovery_confirmation_seconds
            and self._recovery_candidate_track_observations
            >= self.recovery_min_track_observations
        ):
            self._active_target = person.box
            self._active_track_id = person.track_id
            self._last_match_at = now
            self._last_handoff_at = now
            self._reset_recovery_candidate()
            return HandoffDecision(
                "tracking",
                person.box,
                marker,
                person.track_id,
                reason=success_reason,
                emit_handoff=True,
                candidate_scores=candidate_scores,
            )
        return HandoffDecision(
            "recovering",
            person.box,
            marker,
            self._active_track_id,
            reason="id_recovery_confirming",
            candidate_scores=candidate_scores,
            provisional_target=True,
        )

    def _appearance_switch_candidate(
        self,
        active_person: Detection,
        people: list[Detection],
    ) -> Optional[Detection]:
        active_appearance = active_person.appearance_similarity
        if active_appearance is None:
            return None
        candidates = [
            person
            for person in people
            if person.track_id != self._active_track_id
            and person.appearance_similarity is not None
            and person.appearance_similarity >= self.recovery_min_appearance
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda person: person.appearance_similarity)
        assert best.appearance_similarity is not None
        if (
            best.appearance_similarity
            < active_appearance + self.recovery_appearance_margin
        ):
            return None
        return best

    def _appearance_allows_reassignment(self, person: Detection) -> bool:
        appearance = person.appearance_similarity
        return appearance is None or appearance >= self.recovery_min_appearance

    def _appearance_indicates_new_speaker(self, person: Detection) -> bool:
        appearance = person.appearance_similarity
        return (
            appearance is not None
            and appearance > -1.0
            and appearance < self.recovery_min_appearance
        )

    def _is_distinct_speaker_candidate(
        self,
        active_person: Detection,
        candidate: Detection,
    ) -> bool:
        if self._appearance_indicates_new_speaker(candidate):
            return True
        return (
            self._box_iou(active_person.box, candidate.box) < 0.2
            and self._normalized_box_center_distance(
                active_person.box,
                candidate.box,
            )
            > self.recovery_max_center_distance
        )

    _BYSTANDER_MAX_IOU = 0.1

    def _select_recovery_candidate(
        self,
        people: list[Detection],
        now: float,
        require_verified_appearance: bool = False,
        marker: Optional[BoundingBox] = None,
    ) -> Optional[Detection]:
        if self._active_target is None:
            return None
        reference = self._active_target
        if (
            self._recovery_candidate_target is not None
            and self._recovery_candidate_last_at is not None
            and now - self._recovery_candidate_last_at
            <= self._RECOVERY_DROPOUT_SECONDS
        ):
            reference = self._recovery_candidate_target
        ranked = []
        for person in people:
            if person.track_id == self._active_track_id:
                continue
            # pi_unity_05: with the speaker's box missing for a few frames,
            # the nearest box was a seated neighbour and became the target
            # while the marker was in plain view next to it.
            if self.bystander_rules and person.track_id in self._bystander_track_ids:
                # Seen together with the speaker, so not the speaker under a new ID.
                continue
            if (
                self.bystander_rules
                and marker is not None
                and not person.box.contains(marker.center)
            ):
                # The marker is visible and this person does not hold it.
                continue
            iou = self._box_iou(reference, person.box)
            center_distance = self._normalized_box_center_distance(
                reference, person.box
            )
            # Undo the zoom change before comparing sizes with the remembered box.
            area_ratio = person.box.area / max(reference.area, 1.0) / self._size_scale**2
            width_ratio = (
                self._box_width(person.box) / self._box_width(reference) / self._size_scale
            )
            top_distance = abs(person.box.y1 - reference.y1) / max(
                self._box_height(reference), self._box_height(person.box)
            )
            horizontal_distance = abs(
                person.box.center[0] - reference.center[0]
            ) / max(self._box_width(reference), self._box_width(person.box))
            conventional_match = (
                0.4 <= area_ratio <= 2.5
                and (
                    iou >= self.recovery_min_iou
                    or center_distance <= self.recovery_max_center_distance
                )
            )
            occlusion_transition = (
                self.recovery_min_area_ratio
                <= area_ratio
                <= self._RECOVERY_MAX_AREA_RATIO
                and self._RECOVERY_MIN_WIDTH_RATIO
                <= width_ratio
                <= self._RECOVERY_MAX_WIDTH_RATIO
                and top_distance <= self._RECOVERY_MAX_TOP_DISTANCE
                and horizontal_distance
                <= self._RECOVERY_MAX_HORIZONTAL_DISTANCE
            )
            appearance = person.appearance_similarity
            appearance_match = (
                appearance is not None
                and appearance >= self.recovery_min_appearance
            )
            if require_verified_appearance and not appearance_match:
                continue
            if not conventional_match and not occlusion_transition and not appearance_match:
                continue
            # A low appearance score must not veto a strong short-range path.
            # Partial boxes and overlap-contaminated crops often reduce ReID
            # similarity precisely when geometry is most useful. The absolute
            # appearance threshold is required only when appearance is the
            # sole evidence for a distant reassignment.
            if not (
                iou < self.recovery_min_iou
                and center_distance > self.recovery_max_center_distance
            ):
                center_penalty = center_distance
            else:
                # Full/partial-person transitions move the box center because
                # the lower edge changes. Top and horizontal continuity are
                # more stable signals in that case.
                center_penalty = horizontal_distance + top_distance
            score = (
                2.0 * iou
                + person.confidence
                - center_penalty
                - 0.5 * abs(math.log(max(width_ratio, 1e-6)))
                - 0.25 * abs(math.log(max(area_ratio, 1e-6)))
            )
            if appearance is not None:
                score += self.recovery_appearance_weight * appearance
            ranked.append((score, person))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            return None
        if (
            len(ranked) > 1
            and ranked[0][0] < ranked[1][0] + self.recovery_score_margin
        ):
            return None
        return ranked[0][1]

    @staticmethod
    def _box_width(box: BoundingBox) -> float:
        return max(1.0, box.x2 - box.x1)

    @staticmethod
    def _box_height(box: BoundingBox) -> float:
        return max(1.0, box.y2 - box.y1)

    def _update_candidate(
        self,
        matched_person: Detection,
        marker: Optional[BoundingBox],
        now: float,
        pending_state: str,
        pending_target: BoundingBox,
        pending_track_id: Optional[int],
        candidate_scores: tuple[CandidateScore, ...],
        speaker_change: bool = False,
        target_missing: bool = False,
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
            self._released_after_loss = False
            self._last_match_at = now
            self._last_handoff_at = now
            self._reset_candidate()
            return HandoffDecision(
                "tracking",
                matched_person.box,
                marker,
                matched_person.track_id,
                reason=(
                    "confirmed_marker_transfer"
                    if speaker_change
                    else "confirmed_marker_reassociation"
                    if pending_state == "switching"
                    else "confirmed_marker_handoff"
                ),
                emit_handoff=True,
                candidate_scores=candidate_scores,
                speaker_changed=speaker_change,
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
            candidate_box=matched_person.box if target_missing else None,
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
            if self._candidate_track_id == matched_person.track_id:
                return True
        if self._candidate_target is None:
            return False
        # Initial ArUco confirmation belongs to a physical marker-holder, not
        # to an unstable MOT number. Preserve the dwell timer across duplicate
        # partial/full detections when their boxes follow the same path.
        return (
            self._box_iou(self._candidate_target, matched_person.box) >= 0.4
            or self._normalized_box_center_distance(
                self._candidate_target,
                matched_person.box,
            )
            <= 0.35
        )

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
        geometry = self.holder_geometry and len(containing) > 1
        scored = [
            (
                person,
                CandidateScore(
                    track_id=person.track_id,
                    score=self._marker_match_score(
                        marker, person, min_area, geometry=geometry
                    ),
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
        geometry: bool = False,
    ) -> float:
        score = self._CONFIDENCE_WEIGHT * person.confidence
        if (
            self._active_track_id is not None
            and person.track_id == self._active_track_id
        ):
            score += self._ACTIVE_TRACK_BONUS

        if geometry:
            score -= self._holder_geometry_penalty(marker, person.box)
        elif min_area > 0 and person.box.area > 0:
            area_ratio = person.box.area / min_area
            score -= self._AREA_PENALTY_WEIGHT * math.log(area_ratio)
        score -= self._area_growth_penalty(person)

        score -= self._CENTER_DISTANCE_PENALTY_WEIGHT * self._normalized_center_distance(
            marker,
            person.box,
        )
        return score

    def _holder_geometry_penalty(self, marker: BoundingBox, box: BoundingBox) -> float:
        """Penalty for a marker that does not look like it sits on this body."""
        marker_x, marker_y = marker.center
        width = self._box_width(box)
        height = self._box_height(box)
        # A card held in front of the chest sits about 25-50 % down the box
        # and in the middle half of its width.
        rel_x = (marker_x - box.x1) / width
        rel_y = (marker_y - box.y1) / height
        low_y, high_y = self._HOLDER_CHEST_BAND
        low_x, high_x = self._HOLDER_WIDTH_BAND
        outside = max(0.0, low_y - rel_y, rel_y - high_y) + max(
            0.0, low_x - rel_x, rel_x - high_x
        )
        # Pixels per metre from the box height and from the marker size are
        # equal for the holder; someone nearer or farther gives another ratio.
        marker_pixels = max(marker.x2 - marker.x1, marker.y2 - marker.y1, 1.0)
        ratio = (height / self.holder_person_meters) / (
            marker_pixels / self.holder_marker_meters
        )
        return (
            self._HOLDER_POSITION_PENALTY_WEIGHT * outside
            + self._HOLDER_DISTANCE_PENALTY_WEIGHT * abs(math.log(ratio))
        )

    def _is_holder_ambiguous(
        self, scored_candidates: list[tuple[Detection, CandidateScore]]
    ) -> bool:
        if not self.holder_geometry or len(scored_candidates) < 2:
            return False
        best, best_score = scored_candidates[0]
        for person, score in scored_candidates[1:]:
            # A second box mostly on top of the best one is the same person
            # (a partial and a full box), not a rival holder.
            if self._box_iou(best.box, person.box) >= self._HOLDER_SAME_PERSON_IOU:
                continue
            return best_score.score - score.score < self.holder_ambiguity_margin
        return False

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
        reference: BoundingBox, candidate: BoundingBox
    ) -> float:
        reference_x, reference_y = reference.center
        candidate_x, candidate_y = candidate.center
        width = max(1.0, reference.x2 - reference.x1)
        height = max(1.0, reference.y2 - reference.y1)
        return math.hypot(
            (candidate_x - reference_x) / width,
            (candidate_y - reference_y) / height,
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
