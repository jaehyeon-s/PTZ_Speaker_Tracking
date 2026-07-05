import unittest

import numpy as np

from src.reid.appearance import AppearanceGuard, AppearanceReIdentifier
from src.vision.models import Candidate


class AppearanceReIdentifierTests(unittest.TestCase):
    def test_selects_candidate_matching_registered_appearance(self):
        frame = np.zeros((120, 300, 3), dtype=np.uint8)
        frame[10:110, 10:70] = (0, 255, 0)
        frame[10:110, 100:160] = (255, 0, 0)
        frame[10:110, 200:260] = (0, 255, 0)
        reid = AppearanceReIdentifier(threshold=0.68)
        reid.register(frame, (10, 10, 70, 110))

        match, score = reid.best_match(
            frame,
            [
                Candidate((100, 10, 160, 110), candidate_id=2),
                Candidate((200, 10, 260, 110), candidate_id=3),
            ],
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.candidate_id, 3)
        self.assertGreaterEqual(score, 0.68)


class FakeAppearanceMatcher:
    """Scriptable appearance matcher for AppearanceGuard tests."""

    def __init__(self, scores, threshold=0.5):
        self.scores = iter(scores)
        self.threshold = threshold
        self.registered_bbox = None
        self.reset_called = False

    def register(self, frame, bbox):
        del frame
        self.registered_bbox = bbox

    def score(self, frame, bbox):
        del frame, bbox
        return next(self.scores)

    def reset(self):
        self.reset_called = True


class AppearanceGuardTests(unittest.TestCase):
    def test_passes_through_between_periodic_checks(self):
        matcher = FakeAppearanceMatcher(scores=[0.0])  # only consumed on a check frame
        guard = AppearanceGuard(matcher, verify_every_frames=3, mismatch_limit=1)

        # Frames 1-2 fall between checks (verify_every_frames=3): must pass
        # without even calling matcher.score (an exhausted iterator would raise).
        self.assertTrue(guard.check(None, (0, 0, 10, 10)))
        self.assertTrue(guard.check(None, (0, 0, 10, 10)))

    def test_confirms_drift_after_consecutive_failed_checks(self):
        # verify_every_frames=1 -> every call is a check; mismatch_limit=2 ->
        # drift confirmed on the 2nd consecutive low score.
        matcher = FakeAppearanceMatcher(scores=[0.1, 0.1], threshold=0.5)
        guard = AppearanceGuard(matcher, verify_every_frames=1, mismatch_limit=2)

        self.assertTrue(guard.check(None, (0, 0, 10, 10)))  # 1st failure: not yet confirmed
        self.assertFalse(guard.check(None, (0, 0, 10, 10)))  # 2nd failure: confirmed

    def test_a_passing_check_resets_the_mismatch_streak(self):
        matcher = FakeAppearanceMatcher(scores=[0.1, 0.9, 0.1], threshold=0.5)
        guard = AppearanceGuard(matcher, verify_every_frames=1, mismatch_limit=2)

        self.assertTrue(guard.check(None, (0, 0, 10, 10)))  # failure #1
        self.assertTrue(guard.check(None, (0, 0, 10, 10)))  # passes, resets streak
        self.assertTrue(guard.check(None, (0, 0, 10, 10)))  # failure #1 again, not confirmed

    def test_register_captures_baseline_and_clears_counters(self):
        matcher = FakeAppearanceMatcher(scores=[0.1, 0.1, 0.1], threshold=0.5)
        guard = AppearanceGuard(matcher, verify_every_frames=1, mismatch_limit=2)

        self.assertTrue(guard.check(None, (0, 0, 10, 10)))  # 1st failure, not yet confirmed
        self.assertFalse(guard.check(None, (0, 0, 10, 10)))  # 2nd failure, confirmed

        guard.register(None, (5, 5, 15, 15))

        self.assertEqual(matcher.registered_bbox, (5, 5, 15, 15))
        # register() must clear the mismatch streak: a single failing check
        # afterward should not immediately re-confirm drift (it would if the
        # streak from before registration had carried over).
        self.assertTrue(guard.check(None, (5, 5, 15, 15)))

    def test_reset_clears_matcher_and_counters(self):
        matcher = FakeAppearanceMatcher(scores=[0.1])
        guard = AppearanceGuard(matcher, verify_every_frames=1, mismatch_limit=1)
        guard.check(None, (0, 0, 10, 10))

        guard.reset()

        self.assertTrue(matcher.reset_called)


if __name__ == "__main__":
    unittest.main()
