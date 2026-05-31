"""Validate the presenter selected by a camera-managed auto tracker."""

from __future__ import annotations

from dataclasses import dataclass

from src.vision.models import BBox, Candidate, TrackingState, VerificationResult


@dataclass(frozen=True)
class VerifierConfig:
    mismatch_limit: int = 2


class PresenterVerifier:
    def __init__(self, matcher, config: VerifierConfig | None = None) -> None:
        self.matcher = matcher
        self.config = config or VerifierConfig()
        self.registered = False
        self.mismatch_count = 0

    def register(self, frame, bbox: BBox) -> bool:
        self.registered = self.matcher.register(frame, bbox)
        self.mismatch_count = 0
        return self.registered

    def verify(self, frame, bbox: BBox | None, source: str = "") -> VerificationResult:
        if not self.registered:
            return VerificationResult(TrackingState.UNREGISTERED, bbox, event="REGISTER_REQUIRED", source=source)
        if bbox is None:
            return VerificationResult(TrackingState.NO_REGION, None, event="NO_TRACKED_REGION", source=source)

        score = self.matcher.score(frame, bbox)
        if score >= self.matcher.threshold:
            self.mismatch_count = 0
            return VerificationResult(TrackingState.VERIFIED, bbox, score, "PRESENTER_VERIFIED", source)

        self.mismatch_count += 1
        event = "MISMATCH_SUSPECTED"
        if self.mismatch_count >= self.config.mismatch_limit:
            event = "PRESENTER_MISMATCH"
        return VerificationResult(TrackingState.MISMATCH, bbox, score, event, source)

    def find_presenter(self, frame, candidates: list[Candidate]) -> tuple[Candidate | None, float]:
        return self.matcher.best_match(frame, candidates)
