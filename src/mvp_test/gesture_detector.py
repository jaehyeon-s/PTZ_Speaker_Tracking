"""Optional raised-hand fallback acquisition."""

from __future__ import annotations

import time
from typing import Iterable, Optional

from target_state import PersonDetection


class GestureFallbackDetector:
    """Detects a single raised-hand candidate for at least hold_seconds.

    This intentionally stays small for the MVP. If MediaPipe is available, it can
    be added here later. The current heuristic uses upper-body motion/skin-free
    geometry only when callers provide a hand-like point via a detector extension.
    Without a pose model, it returns no target instead of guessing aggressively.
    """

    def __init__(self, hold_seconds: float = 1.0) -> None:
        self.hold_seconds = hold_seconds
        self._candidate_index: Optional[int] = None
        self._candidate_since: Optional[float] = None
        self._pose = None
        self._mp_pose = None
        self._pose_init_attempted = False

    def update(
        self,
        people: Iterable[PersonDetection],
        raised_hand_indices: Optional[set[int]] = None,
    ) -> Optional[PersonDetection]:
        people_list = list(people)
        raised = raised_hand_indices or set()

        if len(raised) != 1:
            self._candidate_index = None
            self._candidate_since = None
            return None

        index = next(iter(raised))
        if index < 0 or index >= len(people_list):
            return None

        now = time.monotonic()
        if self._candidate_index != index:
            self._candidate_index = index
            self._candidate_since = now
            return None

        if self._candidate_since is not None and now - self._candidate_since >= self.hold_seconds:
            return people_list[index]
        return None

    def detect_raised_hand_indices(self, frame, people: Iterable[PersonDetection]) -> set[int]:
        """Return person indices whose wrist is above shoulder level.

        Requires MediaPipe. If it is not installed, gesture fallback remains
        disabled without breaking the main marker-based demo.
        """
        if not self._pose_init_attempted:
            self._init_pose()
        if self._pose is None:
            return set()

        import cv2

        people_list = list(people)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return set()

        landmarks = result.pose_landmarks.landmark
        wrist_ids = [
            self._mp_pose.PoseLandmark.LEFT_WRIST,
            self._mp_pose.PoseLandmark.RIGHT_WRIST,
        ]
        shoulder_ids = [
            self._mp_pose.PoseLandmark.LEFT_SHOULDER,
            self._mp_pose.PoseLandmark.RIGHT_SHOULDER,
        ]
        frame_h, frame_w = frame.shape[:2]
        raised_points = []
        shoulder_y = min(landmarks[item].y for item in shoulder_ids) * frame_h
        for item in wrist_ids:
            wrist = landmarks[item]
            if wrist.visibility > 0.5 and wrist.y * frame_h < shoulder_y:
                raised_points.append((wrist.x * frame_w, wrist.y * frame_h))

        raised_indices = set()
        for point in raised_points:
            for index, person in enumerate(people_list):
                x, y, w, h = person.bbox
                expanded = (x - w * 0.25, y - h * 0.35, w * 1.5, h * 1.35)
                ex, ey, ew, eh = expanded
                if ex <= point[0] <= ex + ew and ey <= point[1] <= ey + eh:
                    raised_indices.add(index)
        return raised_indices

    def _init_pose(self) -> None:
        self._pose_init_attempted = True
        try:
            import os

            os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
            import mediapipe as mp

            solutions = getattr(mp, "solutions", None)
            if solutions is None or not hasattr(solutions, "pose"):
                return
            self._mp_pose = solutions.pose
            self._pose = self._mp_pose.Pose(static_image_mode=False, model_complexity=0)
        except Exception:
            self._mp_pose = None
            self._pose = None
