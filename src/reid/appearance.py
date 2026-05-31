"""Appearance features for presenter verification and recovery."""

from __future__ import annotations

import numpy as np

from src.vision.models import BBox, Candidate


class AppearanceReIdentifier:
    def __init__(self, threshold: float = 0.68) -> None:
        self.threshold = threshold
        self.reference: dict[str, np.ndarray | float] | None = None

    def register(self, frame, bbox: BBox) -> bool:
        self.reference = self._features(frame, bbox)
        return self.reference is not None

    def score(self, frame, bbox: BBox) -> float:
        if self.reference is None:
            return 0.0
        return self._compare(self.reference, self._features(frame, bbox))

    def best_match(self, frame, candidates: list[Candidate]) -> tuple[Candidate | None, float]:
        if self.reference is None:
            return None, 0.0
        best_candidate = None
        best_score = 0.0
        for candidate in candidates:
            score = self.score(frame, candidate.bbox)
            if score > best_score:
                best_candidate, best_score = candidate, score
        if best_score < self.threshold:
            return None, best_score
        return best_candidate, best_score

    def reset(self) -> None:
        self.reference = None

    @staticmethod
    def _features(frame, bbox: BBox):
        import cv2

        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        upper = crop[: max(1, crop.shape[0] // 2)]
        hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
        hue = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
        saturation = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        hue /= hue.sum() + 1e-6
        saturation /= saturation.sum() + 1e-6
        return {
            "hue": hue,
            "saturation": saturation,
            "aspect": float((x2 - x1) / max(y2 - y1, 1)),
        }

    @staticmethod
    def _compare(reference, candidate) -> float:
        import cv2

        if candidate is None:
            return 0.0
        hue = cv2.compareHist(reference["hue"], candidate["hue"], cv2.HISTCMP_INTERSECT)
        saturation = cv2.compareHist(
            reference["saturation"], candidate["saturation"], cv2.HISTCMP_INTERSECT
        )
        aspect = max(0.0, 1.0 - abs(reference["aspect"] - candidate["aspect"]) * 2.0)
        return float(np.clip(hue * 0.5 + saturation * 0.3 + aspect * 0.2, 0.0, 1.0))
