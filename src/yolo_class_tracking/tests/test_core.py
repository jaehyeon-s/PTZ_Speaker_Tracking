import unittest

from marker_tracker import BoundingBox, Detection, MarkerTracker


PERSON = Detection("person", 0.9, BoundingBox(0, 0, 200, 300))
PHONE = Detection("cell phone", 0.8, BoundingBox(80, 100, 100, 140))


class MarkerTrackerTest(unittest.TestCase):
    def test_handoff_after_three_seconds_of_continuous_detection(self):
        tracker = MarkerTracker()

        self.assertEqual(tracker.update([PERSON, PHONE], 10.0).state, "confirming")
        self.assertFalse(tracker.update([PERSON, PHONE], 12.9).emit_handoff)
        decision = tracker.update([PERSON, PHONE], 13.0)

        self.assertEqual(decision.state, "tracking")
        self.assertTrue(decision.emit_handoff)
        self.assertEqual(decision.target_box, PERSON.box)

    def test_interruption_resets_confirmation_timer(self):
        tracker = MarkerTracker(marker_dropout_seconds=0)

        tracker.update([PERSON, PHONE], 10.0)
        self.assertEqual(tracker.update([PERSON], 12.0).state, "default")
        self.assertFalse(tracker.update([PERSON, PHONE], 14.0).emit_handoff)
        self.assertTrue(tracker.update([PERSON, PHONE], 17.0).emit_handoff)

    def test_short_marker_dropout_does_not_reset_confirmation_timer(self):
        tracker = MarkerTracker(marker_dropout_seconds=0.5)

        tracker.update([PERSON, PHONE], 10.0)
        self.assertEqual(tracker.update([PERSON], 10.4).state, "confirming")
        self.assertTrue(tracker.update([PERSON, PHONE], 13.0).emit_handoff)

    def test_long_marker_dropout_resets_confirmation_timer(self):
        tracker = MarkerTracker(marker_dropout_seconds=0.5)

        tracker.update([PERSON, PHONE], 10.0)
        self.assertEqual(tracker.update([PERSON], 10.6).state, "default")
        self.assertFalse(tracker.update([PERSON, PHONE], 13.0).emit_handoff)

    def test_tracking_is_released_after_three_second_grace_period(self):
        tracker = MarkerTracker()
        tracker.update([PERSON, PHONE], 10.0)
        tracker.update([PERSON, PHONE], 13.0)

        self.assertEqual(tracker.update([], 15.9).state, "grace")
        decision = tracker.update([], 16.1)

        self.assertTrue(decision.emit_release)
        self.assertEqual(decision.state, "default")

    def test_phone_outside_person_does_not_confirm(self):
        tracker = MarkerTracker()
        outside_phone = Detection("cell phone", 0.8, BoundingBox(300, 300, 320, 340))

        self.assertEqual(tracker.update([PERSON, outside_phone], 10.0).state, "default")
        self.assertEqual(tracker.update([PERSON, outside_phone], 14.0).state, "default")

    def test_multiple_phones_are_ignored_for_initial_prototype(self):
        tracker = MarkerTracker()
        second_phone = Detection("cell phone", 0.7, BoundingBox(20, 20, 40, 60))

        decision = tracker.update([PERSON, PHONE, second_phone], 10.0)

        self.assertEqual(decision.state, "default")
        self.assertIsNone(decision.marker_box)

    def test_smallest_containing_person_is_selected(self):
        tracker = MarkerTracker(confirmation_seconds=0)
        larger_person = Detection("person", 0.95, BoundingBox(0, 0, 400, 400))

        decision = tracker.update([larger_person, PERSON, PHONE], 10.0)

        self.assertTrue(decision.emit_handoff)
        self.assertEqual(decision.target_box, PERSON.box)


if __name__ == "__main__":
    unittest.main()
