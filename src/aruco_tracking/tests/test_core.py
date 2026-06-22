import unittest

from marker_tracker import BoundingBox, Detection, MarkerTracker


PERSON = Detection("person", 0.9, BoundingBox(0, 0, 200, 300))
MARKER = Detection("marker", 1.0, BoundingBox(80, 100, 100, 140))
TRACKED_PERSON = Detection("person", 0.9, BoundingBox(0, 0, 200, 300), track_id=7)
OTHER_TRACKED_PERSON = Detection(
    "person", 0.9, BoundingBox(220, 0, 420, 300), track_id=8
)
OTHER_MARKER = Detection("marker", 1.0, BoundingBox(300, 100, 320, 140))


class MarkerTrackerTest(unittest.TestCase):
    def test_handoff_after_three_seconds_of_continuous_detection(self):
        tracker = MarkerTracker()

        self.assertEqual(tracker.update([PERSON, MARKER], 10.0).state, "confirming")
        self.assertFalse(tracker.update([PERSON, MARKER], 12.9).emit_handoff)
        decision = tracker.update([PERSON, MARKER], 13.0)

        self.assertEqual(decision.state, "tracking")
        self.assertEqual(decision.reason, "confirmed_marker_handoff")
        self.assertTrue(decision.emit_handoff)
        self.assertEqual(decision.target_box, PERSON.box)

    def test_interruption_resets_confirmation_timer(self):
        tracker = MarkerTracker(marker_dropout_seconds=0)

        tracker.update([PERSON, MARKER], 10.0)
        self.assertEqual(tracker.update([PERSON], 12.0).state, "default")
        self.assertFalse(tracker.update([PERSON, MARKER], 14.0).emit_handoff)
        self.assertTrue(tracker.update([PERSON, MARKER], 17.0).emit_handoff)

    def test_short_marker_dropout_does_not_reset_confirmation_timer(self):
        tracker = MarkerTracker(marker_dropout_seconds=0.5)

        tracker.update([PERSON, MARKER], 10.0)
        self.assertEqual(tracker.update([PERSON], 10.4).state, "confirming")
        self.assertTrue(tracker.update([PERSON, MARKER], 13.0).emit_handoff)

    def test_long_marker_dropout_resets_confirmation_timer(self):
        tracker = MarkerTracker(marker_dropout_seconds=0.5)

        tracker.update([PERSON, MARKER], 10.0)
        self.assertEqual(tracker.update([PERSON], 10.6).state, "default")
        self.assertFalse(tracker.update([PERSON, MARKER], 13.0).emit_handoff)

    def test_tracking_is_released_after_three_second_grace_period(self):
        tracker = MarkerTracker()
        tracker.update([PERSON, MARKER], 10.0)
        tracker.update([PERSON, MARKER], 13.0)

        self.assertEqual(tracker.update([], 15.9).state, "grace")
        decision = tracker.update([], 16.1)

        self.assertTrue(decision.emit_release)
        self.assertEqual(decision.state, "default")

    def test_marker_outside_person_does_not_confirm(self):
        tracker = MarkerTracker()
        outside_marker = Detection("marker", 1.0, BoundingBox(300, 300, 320, 340))

        self.assertEqual(tracker.update([PERSON, outside_marker], 10.0).state, "default")
        self.assertEqual(tracker.update([PERSON, outside_marker], 14.0).state, "default")

    def test_multiple_markers_are_ignored_for_initial_prototype(self):
        tracker = MarkerTracker()
        second_marker = Detection("marker", 1.0, BoundingBox(20, 20, 40, 60))

        decision = tracker.update([PERSON, MARKER, second_marker], 10.0)

        self.assertEqual(decision.state, "default")
        self.assertIsNone(decision.marker_box)

    def test_smallest_containing_person_is_selected(self):
        tracker = MarkerTracker(confirmation_seconds=0)
        larger_person = Detection("person", 0.95, BoundingBox(0, 0, 400, 400))

        decision = tracker.update([larger_person, PERSON, MARKER], 10.0)

        self.assertTrue(decision.emit_handoff)
        self.assertEqual(decision.target_box, PERSON.box)

    def test_active_track_bonus_avoids_equal_box_jitter(self):
        tracker = MarkerTracker(confirmation_seconds=0)
        active = Detection("person", 0.9, BoundingBox(0, 0, 200, 300), track_id=7)
        duplicate = Detection("person", 0.9, BoundingBox(0, 0, 200, 300), track_id=8)
        tracker.update([active, MARKER], 10.0)

        decision = tracker.update([duplicate, active, MARKER], 11.0)

        self.assertEqual(decision.state, "tracking")
        self.assertEqual(decision.target_track_id, 7)
        self.assertFalse(decision.emit_handoff)

    def test_oversized_active_box_loses_to_smaller_marker_holder(self):
        tracker = MarkerTracker(confirmation_seconds=0)
        active = Detection("person", 0.9, BoundingBox(0, 0, 200, 300), track_id=7)
        oversized_active = Detection(
            "person", 0.9, BoundingBox(0, 0, 440, 320), track_id=7
        )
        smaller_holder = Detection(
            "person", 0.9, BoundingBox(220, 0, 420, 300), track_id=8
        )
        marker_on_smaller = Detection("marker", 1.0, BoundingBox(300, 100, 320, 140))
        tracker.update([active, MARKER], 10.0)

        decision = tracker.update(
            [oversized_active, smaller_holder, marker_on_smaller], 11.0
        )

        self.assertTrue(decision.emit_handoff)
        self.assertEqual(decision.target_track_id, 8)

    def test_tracked_person_continues_after_marker_disappears(self):
        tracker = MarkerTracker(confirmation_seconds=0)
        updated_person = Detection(
            "person", 0.9, BoundingBox(10, 0, 210, 300), track_id=7
        )

        handoff = tracker.update([TRACKED_PERSON, MARKER], 10.0)
        decision = tracker.update([updated_person], 11.0)

        self.assertTrue(handoff.emit_handoff)
        self.assertEqual(handoff.target_track_id, 7)
        self.assertEqual(decision.state, "tracking")
        self.assertEqual(decision.reason, "active_track_visible")
        self.assertEqual(decision.target_track_id, 7)
        self.assertEqual(decision.target_box, updated_person.box)

    def test_marker_transfer_switches_after_confirmation(self):
        tracker = MarkerTracker()
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        tracker.update([TRACKED_PERSON, MARKER], 13.0)

        switching = tracker.update(
            [TRACKED_PERSON, OTHER_TRACKED_PERSON, OTHER_MARKER], 14.0
        )
        self.assertEqual(switching.state, "switching")
        self.assertEqual(switching.reason, "switch_candidate_confirming")
        self.assertFalse(switching.emit_handoff)
        self.assertEqual(switching.target_track_id, 7)
        self.assertEqual(switching.target_box, TRACKED_PERSON.box)

        decision = tracker.update(
            [TRACKED_PERSON, OTHER_TRACKED_PERSON, OTHER_MARKER], 17.0
        )
        self.assertEqual(decision.state, "tracking")
        self.assertEqual(decision.reason, "confirmed_marker_handoff")
        self.assertTrue(decision.emit_handoff)
        self.assertEqual(decision.target_track_id, 8)
        self.assertEqual(decision.target_box, OTHER_TRACKED_PERSON.box)

    def test_switch_cooldown_blocks_immediate_transfer(self):
        tracker = MarkerTracker(
            confirmation_seconds=0,
            switch_cooldown_seconds=1.0,
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)

        blocked = tracker.update(
            [TRACKED_PERSON, OTHER_TRACKED_PERSON, OTHER_MARKER], 10.5
        )
        self.assertEqual(blocked.state, "tracking")
        self.assertEqual(blocked.target_track_id, 7)

        switched = tracker.update(
            [TRACKED_PERSON, OTHER_TRACKED_PERSON, OTHER_MARKER], 11.1
        )
        self.assertTrue(switched.emit_handoff)
        self.assertEqual(switched.target_track_id, 8)

    def test_switch_score_margin_blocks_ambiguous_containing_candidate(self):
        tracker = MarkerTracker(
            confirmation_seconds=0,
            switch_score_margin=100.0,
            switch_cooldown_seconds=0,
        )
        active = Detection("person", 0.9, BoundingBox(0, 0, 220, 300), track_id=7)
        other = Detection("person", 0.9, BoundingBox(40, 0, 260, 300), track_id=8)
        shared_marker = Detection("marker", 1.0, BoundingBox(120, 100, 140, 140))
        tracker.update([active, shared_marker], 10.0)

        decision = tracker.update([active, other, shared_marker], 11.0)

        self.assertEqual(decision.state, "tracking")
        self.assertEqual(decision.target_track_id, 7)
        self.assertFalse(decision.emit_handoff)

    def test_marker_transfer_resets_if_candidate_changes(self):
        tracker = MarkerTracker()
        third_person = Detection("person", 0.9, BoundingBox(440, 0, 640, 300), track_id=9)
        third_marker = Detection("marker", 1.0, BoundingBox(520, 100, 540, 140))
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        tracker.update([TRACKED_PERSON, MARKER], 13.0)

        tracker.update([TRACKED_PERSON, OTHER_TRACKED_PERSON, OTHER_MARKER], 14.0)
        self.assertFalse(
            tracker.update([TRACKED_PERSON, third_person, third_marker], 16.9).emit_handoff
        )
        decision = tracker.update([TRACKED_PERSON, third_person, third_marker], 19.9)

        self.assertTrue(decision.emit_handoff)
        self.assertEqual(decision.target_track_id, 9)

    def test_requested_track_id_selects_manual_target(self):
        tracker = MarkerTracker()

        decision = tracker.update([TRACKED_PERSON], 10.0, requested_track_id=7)

        self.assertEqual(decision.state, "tracking")
        self.assertEqual(decision.reason, "manual_target_selected")
        self.assertTrue(decision.emit_handoff)
        self.assertEqual(decision.target_track_id, 7)
        self.assertEqual(decision.target_box, TRACKED_PERSON.box)

    def test_missing_tracked_person_uses_release_grace_period(self):
        tracker = MarkerTracker(confirmation_seconds=0, release_grace_seconds=1.0)
        tracker.update([TRACKED_PERSON, MARKER], 10.0)

        self.assertEqual(tracker.update([], 10.5).state, "grace")
        decision = tracker.update([], 11.1)

        self.assertTrue(decision.emit_release)
        self.assertEqual(decision.reason, "target_released_after_grace")
        self.assertEqual(decision.target_track_id, 7)


if __name__ == "__main__":
    unittest.main()
