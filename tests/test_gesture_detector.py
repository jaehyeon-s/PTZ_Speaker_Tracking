import unittest

from src.tracking.gesture_detector import GestureFallbackDetector
from src.tracking.target_state import PersonDetection


class GestureFallbackDetectorTests(unittest.TestCase):
    def test_registers_after_hold_seconds_when_same_person_keeps_raising_hand(self):
        detector = GestureFallbackDetector(hold_seconds=0.0)
        people = [PersonDetection((10, 10, 20, 40))]

        first = detector.update(people, {0})
        second = detector.update(people, {0})

        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(second.bbox, (10, 10, 20, 40))

    def test_reset_when_raised_hand_count_is_not_exactly_one(self):
        detector = GestureFallbackDetector(hold_seconds=0.0)
        people = [PersonDetection((10, 10, 20, 40)), PersonDetection((100, 10, 20, 40))]

        detector.update(people, {0})
        result = detector.update(people, set())
        result_again = detector.update(people, {0})

        self.assertIsNone(result)
        self.assertIsNone(result_again)  # candidate was cleared, must restart the hold

    def test_index_reused_by_a_different_person_does_not_inherit_the_hold_timer(self):
        """A detector's output order can change frame to frame. If person A (at
        index 0) starts a hold and then a frame later index 0 belongs to an
        unrelated person B who also raises a hand, B must not inherit A's
        elapsed hold time just because they share the list index."""
        detector = GestureFallbackDetector(hold_seconds=0.5)
        person_a = PersonDetection((10, 10, 20, 40))
        person_b = PersonDetection((500, 10, 20, 40))  # far away: a different person

        detector.update([person_a], {0})
        import time

        time.sleep(0.05)
        # Same index (0), but now it is a different, distant person.
        result = detector.update([person_b], {0})

        self.assertIsNone(result)

    def test_same_person_slightly_shifted_position_keeps_the_hold_timer(self):
        """Small frame-to-frame jitter in the same person's bbox must not
        reset the hold timer."""
        detector = GestureFallbackDetector(hold_seconds=0.05)
        person_first = PersonDetection((10, 10, 20, 40))
        person_shifted = PersonDetection((13, 12, 20, 40))  # a few pixels of jitter

        detector.update([person_first], {0})
        import time

        time.sleep(0.06)
        result = detector.update([person_shifted], {0})

        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
