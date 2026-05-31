import unittest

from src.reid.verifier import PresenterVerifier, VerifierConfig
from src.vision.models import Candidate, TrackingState


class FakeMatcher:
    threshold = 0.7

    def __init__(self, scores=None):
        self.scores = iter(scores or [])

    def register(self, frame, bbox):
        return True

    def score(self, frame, bbox):
        return next(self.scores)

    def best_match(self, frame, candidates):
        return candidates[0], 0.85


class PresenterVerifierTests(unittest.TestCase):
    def test_requires_registration_before_verification(self):
        result = PresenterVerifier(FakeMatcher()).verify(None, (1, 2, 3, 4))

        self.assertEqual(result.state, TrackingState.UNREGISTERED)

    def test_reports_verified_subject(self):
        verifier = PresenterVerifier(FakeMatcher([0.91]))
        verifier.register(None, (1, 2, 3, 4))

        result = verifier.verify(None, (1, 2, 3, 4), "camera_bbox")

        self.assertEqual(result.state, TrackingState.VERIFIED)
        self.assertEqual(result.source, "camera_bbox")

    def test_confirms_mismatch_only_after_configured_count(self):
        verifier = PresenterVerifier(FakeMatcher([0.2, 0.3]), VerifierConfig(mismatch_limit=2))
        verifier.register(None, (1, 2, 3, 4))

        suspected = verifier.verify(None, (1, 2, 3, 4))
        confirmed = verifier.verify(None, (1, 2, 3, 4))

        self.assertEqual(suspected.event, "MISMATCH_SUSPECTED")
        self.assertEqual(confirmed.event, "PRESENTER_MISMATCH")

    def test_delegates_recovery_candidate_matching(self):
        verifier = PresenterVerifier(FakeMatcher())
        candidate = Candidate((1, 2, 3, 4), candidate_id=5)

        match, score = verifier.find_presenter(None, [candidate])

        self.assertEqual(match, candidate)
        self.assertEqual(score, 0.85)


if __name__ == "__main__":
    unittest.main()
