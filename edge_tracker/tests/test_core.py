import unittest

from marker_tracker import BoundingBox, Detection, IdentityResolutionContext, MarkerTracker


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
        self.assertEqual(decision.reason, "confirmed_marker_reassociation")
        self.assertFalse(decision.speaker_changed)

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
        self.assertEqual(decision.reason, "confirmed_marker_transfer")
        self.assertTrue(decision.emit_handoff)
        self.assertTrue(decision.speaker_changed)
        self.assertEqual(decision.target_track_id, 8)
        self.assertEqual(decision.target_box, OTHER_TRACKED_PERSON.box)

    def test_low_similarity_marker_holder_confirms_new_speaker_when_old_is_missing(self):
        tracker = MarkerTracker(
            confirmation_seconds=0.5,
            switch_cooldown_seconds=0,
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        tracker.update([TRACKED_PERSON, MARKER], 10.5)
        new_speaker = Detection(
            "person",
            0.9,
            OTHER_TRACKED_PERSON.box,
            track_id=8,
            appearance_similarity=0.20,
        )

        pending = tracker.update([new_speaker, OTHER_MARKER], 11.0)
        tracker.update([new_speaker, OTHER_MARKER], 11.2)
        changed = tracker.update([new_speaker, OTHER_MARKER], 11.5)

        self.assertEqual(pending.state, "recovering")
        self.assertEqual(pending.reason, "marker_reacquire_confirming")
        self.assertFalse(pending.provisional_target)
        self.assertFalse(pending.emit_handoff)
        self.assertEqual(changed.reason, "confirmed_marker_transfer")
        self.assertTrue(changed.emit_handoff)
        self.assertTrue(changed.speaker_changed)
        self.assertEqual(changed.target_track_id, 8)

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

    def test_identity_resolution_visible_active_track_keeps_release_grace_alive(self):
        tracker = MarkerTracker(
            confirmation_seconds=0,
            release_grace_seconds=1.0,
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        resolving = IdentityResolutionContext(
            active=True,
            reason="identity_searching",
        )

        held = tracker.update(
            [TRACKED_PERSON],
            10.8,
            identity_resolution=resolving,
        )
        still_held = tracker.update(
            [TRACKED_PERSON],
            11.6,
            identity_resolution=resolving,
        )
        missing = tracker.update([], 12.0)

        self.assertEqual(held.state, "identity_resolving")
        self.assertEqual(held.target_track_id, 7)
        self.assertEqual(held.target_box, TRACKED_PERSON.box)
        self.assertEqual(still_held.state, "identity_resolving")
        self.assertFalse(still_held.emit_release)
        self.assertEqual(missing.state, "grace")
        self.assertFalse(missing.emit_release)

    def test_identity_resolution_follows_reid_candidate_without_committing_track_id(self):
        tracker = MarkerTracker(
            confirmation_seconds=0,
            release_grace_seconds=1.0,
            recovery_confirmation_seconds=0.12,
        )
        replacement = Detection(
            "person",
            0.8,
            BoundingBox(10, 0, 210, 300),
            track_id=12,
            appearance_similarity=0.56,
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        resolving = IdentityResolutionContext(
            active=True,
            reason="identity_searching",
        )

        confirming = tracker.update(
            [replacement],
            10.10,
            identity_resolution=resolving,
        )
        still_resolving = tracker.update(
            [replacement],
            10.23,
            identity_resolution=resolving,
        )

        self.assertEqual(confirming.state, "identity_resolving")
        self.assertEqual(confirming.reason, "identity_searching")
        self.assertEqual(confirming.target_track_id, 7)
        self.assertEqual(confirming.target_box, replacement.box)
        self.assertTrue(confirming.provisional_target)
        self.assertFalse(confirming.emit_handoff)
        self.assertEqual(still_resolving.state, "identity_resolving")
        self.assertEqual(still_resolving.target_track_id, 7)
        self.assertEqual(still_resolving.target_box, replacement.box)
        self.assertTrue(still_resolving.provisional_target)
        self.assertFalse(still_resolving.emit_handoff)

    def test_identity_resolution_rejects_low_appearance_recovery_candidate(self):
        tracker = MarkerTracker(
            confirmation_seconds=0,
            release_grace_seconds=1.0,
            recovery_confirmation_seconds=0.12,
        )
        weak_candidate = Detection(
            "person",
            0.8,
            BoundingBox(10, 0, 210, 300),
            track_id=12,
            appearance_similarity=0.40,
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        resolving = IdentityResolutionContext(
            active=True,
            reason="identity_searching",
        )

        decision = tracker.update(
            [weak_candidate],
            10.10,
            identity_resolution=resolving,
        )

        self.assertEqual(decision.state, "identity_resolving")
        self.assertEqual(decision.target_track_id, 7)
        self.assertIsNone(decision.target_box)
        self.assertFalse(decision.provisional_target)
        self.assertFalse(decision.emit_handoff)

    def test_identity_resolution_requires_appearance_before_recovery_candidate(self):
        tracker = MarkerTracker(
            confirmation_seconds=0,
            release_grace_seconds=1.0,
            recovery_confirmation_seconds=0.12,
        )
        unscored_candidate = Detection(
            "person",
            0.8,
            BoundingBox(10, 0, 210, 300),
            track_id=12,
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        resolving = IdentityResolutionContext(
            active=True,
            reason="identity_searching",
        )

        decision = tracker.update(
            [unscored_candidate],
            10.10,
            identity_resolution=resolving,
        )

        self.assertEqual(decision.state, "identity_resolving")
        self.assertEqual(decision.target_track_id, 7)
        self.assertIsNone(decision.target_box)
        self.assertFalse(decision.provisional_target)
        self.assertFalse(decision.emit_handoff)

    def test_recovery_after_identity_resolving_candidate_does_not_crash(self):
        tracker = MarkerTracker(
            confirmation_seconds=0,
            release_grace_seconds=1.0,
            recovery_confirmation_seconds=0.12,
        )
        strong_candidate = Detection(
            "person",
            0.8,
            BoundingBox(10, 0, 210, 300),
            track_id=12,
            appearance_similarity=0.90,
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        resolving = IdentityResolutionContext(active=True, reason="identity_searching")

        # Identity resolution notes a recovery candidate box while the marker is hidden...
        decision = tracker.update([strong_candidate], 10.10, identity_resolution=resolving)
        self.assertEqual(decision.state, "identity_resolving")
        self.assertTrue(decision.provisional_target)

        # ...and on the next frame resolution has ended, so the ordinary
        # recovery path continues from that candidate.
        decision = tracker.update([strong_candidate], 10.20)
        self.assertEqual(decision.state, "recovering")

        decision = tracker.update([strong_candidate], 10.40)
        self.assertEqual(decision.state, "tracking")
        self.assertEqual(decision.target_track_id, 12)

    def test_identity_resolution_marks_missing_active_target_as_provisional(self):
        tracker = MarkerTracker(
            confirmation_seconds=0,
            release_grace_seconds=1.0,
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        resolving = IdentityResolutionContext(
            active=True,
            reason="identity_searching",
        )

        decision = tracker.update([], 10.5, identity_resolution=resolving)

        # The last confirmed box is a stale pixel position, not a target.
        self.assertEqual(decision.state, "identity_resolving")
        self.assertEqual(decision.target_track_id, 7)
        self.assertIsNone(decision.target_box)
        self.assertFalse(decision.provisional_target)

    def test_identity_resolution_follows_only_a_fresh_coast_estimate(self):
        tracker = MarkerTracker(confirmation_seconds=0, release_grace_seconds=1.0)
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        coast_box = BoundingBox(40, 0, 240, 300)
        resolving = IdentityResolutionContext(
            active=True,
            reason="identity_searching",
            coast_box=coast_box,
        )

        decision = tracker.update([], 10.5, identity_resolution=resolving)

        self.assertEqual(decision.target_box, coast_box)
        self.assertTrue(decision.provisional_target)

    def test_expired_identity_resolution_releases_the_target(self):
        tracker = MarkerTracker(confirmation_seconds=2.0, release_grace_seconds=1.0)
        tracker.update([TRACKED_PERSON, MARKER], 8.0)
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        bystander = Detection(
            "person", 0.9, OTHER_TRACKED_PERSON.box, track_id=8, appearance_similarity=0.3
        )

        searching = tracker.update(
            [bystander],
            15.0,
            identity_resolution=IdentityResolutionContext(active=True),
        )
        released = tracker.update(
            [bystander],
            20.0,
            identity_resolution=IdentityResolutionContext(active=True, expired=True),
        )
        # After a loss the marker re-registers from a few sparse sightings,
        # without the full initial dwell.
        first = tracker.update([bystander, OTHER_MARKER], 30.0)
        tracker.update([bystander], 30.5)
        tracker.update([bystander, OTHER_MARKER], 31.0)
        tracker.update([bystander], 31.5)
        registered = tracker.update([bystander, OTHER_MARKER], 32.0)

        self.assertEqual(searching.state, "identity_resolving")
        self.assertTrue(released.emit_release)
        self.assertEqual(released.state, "default")
        self.assertEqual(released.reason, "identity_resolution_timeout")
        self.assertEqual(first.reason, "marker_reacquire_confirming")
        self.assertFalse(first.emit_handoff)
        self.assertTrue(registered.emit_handoff)
        self.assertTrue(registered.speaker_changed)
        self.assertEqual(registered.target_track_id, 8)

    def test_expired_identity_resolution_keeps_a_visible_target(self):
        tracker = MarkerTracker(confirmation_seconds=0)
        tracker.update([TRACKED_PERSON, MARKER], 10.0)

        decision = tracker.update(
            [TRACKED_PERSON],
            20.0,
            identity_resolution=IdentityResolutionContext(active=True, expired=True),
        )

        self.assertFalse(decision.emit_release)
        self.assertEqual(decision.target_box, TRACKED_PERSON.box)

    def test_lost_target_is_reregistered_from_intermittent_marker_sightings(self):
        tracker = MarkerTracker(confirmation_seconds=2.0, marker_dropout_seconds=0.75)
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        tracker.update([TRACKED_PERSON, MARKER], 12.0)
        resolving = IdentityResolutionContext(active=True, reason="identity_searching")
        # Same person after an appearance change: new raw ID, low similarity.
        holder = Detection(
            "person", 0.9, OTHER_TRACKED_PERSON.box, track_id=30, appearance_similarity=0.4
        )
        unscored_holder = Detection("person", 0.9, OTHER_TRACKED_PERSON.box, track_id=30)

        # A small marker is detected only now and then (gaps of about 1 s),
        # with identity resolution running on the frames in between.
        first = tracker.update([unscored_holder, OTHER_MARKER], 50.0, identity_resolution=resolving)
        tracker.update([holder], 50.5, identity_resolution=resolving)
        second = tracker.update([holder, OTHER_MARKER], 50.9, identity_resolution=resolving)
        tracker.update([holder], 51.5, identity_resolution=resolving)
        third = tracker.update([holder, OTHER_MARKER], 52.1, identity_resolution=resolving)

        self.assertEqual(first.reason, "marker_reacquire_confirming")
        self.assertFalse(second.emit_handoff)
        self.assertTrue(third.emit_handoff)
        self.assertEqual(third.state, "tracking")
        self.assertEqual(third.target_track_id, 30)
        self.assertEqual(third.reason, "confirmed_marker_transfer")
        self.assertTrue(third.speaker_changed)

    def test_marker_crossing_a_bystander_box_does_not_reregister_the_bystander(self):
        # pi_unity_04: the speaker walked past a seated neighbour while their
        # own box was missing, and the marker lay inside the neighbour's box
        # for three frames (0.14 s).
        tracker = MarkerTracker(confirmation_seconds=2.0, marker_dropout_seconds=1.5)
        neighbour = Detection(
            "person", 0.9, OTHER_TRACKED_PERSON.box, track_id=104, appearance_similarity=0.45
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        tracker.update([TRACKED_PERSON, neighbour, MARKER], 12.0)
        tracker.update([TRACKED_PERSON, neighbour, MARKER], 20.0)

        decisions = [
            tracker.update([neighbour, OTHER_MARKER], at) for at in (20.5, 20.57, 20.64)
        ]
        after = tracker.update([neighbour], 20.71)

        self.assertFalse(any(decision.emit_handoff for decision in decisions))
        self.assertTrue(all(decision.target_track_id == 7 for decision in decisions))
        self.assertEqual(decisions[-1].state, "switching")
        self.assertNotEqual(after.target_track_id, 104)
        # Only --switch-follow-candidate would steer to the holder meanwhile;
        # the decision itself still holds the speaker.
        self.assertEqual(decisions[-1].candidate_box, neighbour.box)
        self.assertEqual(decisions[-1].target_box, TRACKED_PERSON.box)

    def test_marker_staying_with_a_bystander_is_a_transfer_after_full_confirmation(self):
        tracker = MarkerTracker(confirmation_seconds=2.0, marker_dropout_seconds=1.5)
        neighbour = Detection(
            "person", 0.9, OTHER_TRACKED_PERSON.box, track_id=104, appearance_similarity=0.2
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        tracker.update([TRACKED_PERSON, neighbour, MARKER], 12.0)

        early = tracker.update([neighbour, OTHER_MARKER], 12.5)
        still_early = tracker.update([neighbour, OTHER_MARKER], 13.5)
        transferred = tracker.update([neighbour, OTHER_MARKER], 14.6)

        self.assertFalse(early.emit_handoff)
        self.assertFalse(still_early.emit_handoff)
        self.assertTrue(transferred.emit_handoff)
        self.assertEqual(transferred.target_track_id, 104)
        self.assertTrue(transferred.speaker_changed)

    def test_bystander_rules_can_be_turned_off_for_comparison(self):
        def run(**options):
            tracker = MarkerTracker(confirmation_seconds=2.0, marker_dropout_seconds=1.5, **options)
            neighbour = Detection(
                "person", 0.9, OTHER_TRACKED_PERSON.box, track_id=104, appearance_similarity=0.45
            )
            tracker.update([TRACKED_PERSON, MARKER], 10.0)
            tracker.update([TRACKED_PERSON, neighbour, MARKER], 12.0)
            tracker.update([TRACKED_PERSON, neighbour, MARKER], 20.0)
            return [tracker.update([neighbour, OTHER_MARKER], at) for at in (20.5, 20.57, 20.64)]

        self.assertEqual(run()[-1].state, "switching")
        off = run(bystander_rules=False, reacquire_marker_min_seconds=0.0)
        self.assertTrue(off[-1].emit_handoff)
        self.assertEqual(off[-1].target_track_id, 104)

    def test_id_recovery_does_not_pick_a_known_bystander(self):
        # pi_unity_05: the speaker's box went missing for a few frames and the
        # nearest box, a seated neighbour, was taken as the speaker's new ID.
        tracker = MarkerTracker(confirmation_seconds=0, recovery_confirmation_seconds=0.12)
        neighbour = Detection("person", 0.8, BoundingBox(215, 0, 415, 300), track_id=12)
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        tracker.update([TRACKED_PERSON, neighbour], 10.5)
        # The speaker's box is gone and the neighbour's box has drifted over it.
        drifted = Detection("person", 0.8, BoundingBox(40, 0, 240, 300), track_id=12)

        decisions = [tracker.update([drifted], at) for at in (10.6, 10.7, 10.8, 11.0)]

        self.assertFalse(any(decision.emit_handoff for decision in decisions))
        self.assertTrue(all(decision.target_track_id == 7 for decision in decisions))

    def test_id_recovery_does_not_pick_a_person_while_the_marker_is_elsewhere(self):
        tracker = MarkerTracker(confirmation_seconds=0, recovery_confirmation_seconds=0.12)
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        replacement = Detection("person", 0.8, BoundingBox(10, 0, 210, 300), track_id=12)
        # The marker is detected, but outside every person box.
        marker_elsewhere = Detection("marker", 1.0, BoundingBox(400, 100, 420, 140))

        decisions = [
            tracker.update([replacement, marker_elsewhere], at) for at in (10.1, 10.2, 10.3)
        ]
        # Without a visible marker the same nearby box is a valid recovery.
        recovered = [tracker.update([replacement], at) for at in (10.4, 10.55)]

        self.assertFalse(any(decision.emit_handoff for decision in decisions))
        self.assertTrue(recovered[-1].emit_handoff)
        self.assertEqual(recovered[-1].target_track_id, 12)

    def test_marker_sightings_outside_the_window_do_not_reregister(self):
        tracker = MarkerTracker(confirmation_seconds=0, reacquire_marker_window_seconds=3.0)
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        resolving = IdentityResolutionContext(active=True)
        holder = Detection("person", 0.9, OTHER_TRACKED_PERSON.box, track_id=30)

        decisions = [
            tracker.update([holder, OTHER_MARKER], at, identity_resolution=resolving)
            for at in (50.0, 52.0, 54.0, 56.0)
        ]

        self.assertFalse(any(decision.emit_handoff for decision in decisions))

    def test_recovers_target_when_bytetrack_assigns_a_new_nearby_id(self):
        tracker = MarkerTracker(
            confirmation_seconds=0,
            recovery_confirmation_seconds=0.12,
        )
        replacement = Detection(
            "person", 0.8, BoundingBox(10, 0, 210, 300), track_id=12
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)

        confirming = tracker.update([replacement], 10.10)
        recovered = tracker.update([replacement], 10.23)

        self.assertEqual(confirming.state, "recovering")
        self.assertEqual(confirming.reason, "id_recovery_confirming")
        self.assertEqual(confirming.target_track_id, 7)
        self.assertEqual(confirming.target_box, replacement.box)
        self.assertTrue(confirming.provisional_target)
        self.assertTrue(recovered.emit_handoff)
        self.assertEqual(recovered.state, "tracking")
        self.assertEqual(recovered.reason, "target_recovered_new_track_id")
        self.assertEqual(recovered.target_track_id, 12)
        self.assertEqual(recovered.target_box, replacement.box)

    def test_does_not_recover_to_a_distant_person_without_marker(self):
        tracker = MarkerTracker(confirmation_seconds=0)
        distant = Detection(
            "person", 0.9, BoundingBox(350, 0, 550, 300), track_id=12
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)

        decision = tracker.update([distant], 10.1)

        self.assertEqual(decision.state, "grace")
        self.assertEqual(decision.target_track_id, 7)
        self.assertFalse(decision.emit_handoff)

    def test_grace_extension_delays_release(self):
        for extension, expected in ((0.0, "default"), (2.0, "grace")):
            tracker = MarkerTracker(confirmation_seconds=0, release_grace_seconds=3.0)
            tracker.update([TRACKED_PERSON, MARKER], 10.0)
            self.assertEqual(tracker.update([], 12.0, grace_extension=extension).state, "grace")
            self.assertEqual(tracker.update([], 14.0, grace_extension=extension).state, expected)

    def test_head_only_box_recovers_only_with_lower_area_ratio(self):
        head_only = Detection("person", 0.4, BoundingBox(60, 5, 140, 80), track_id=12)
        for ratio, expected in ((None, "grace"), (0.05, "recovering")):
            tracker = MarkerTracker(
                confirmation_seconds=0,
                recovery_confirmation_seconds=0.12,
                recovery_min_area_ratio=ratio,
            )
            tracker.update([TRACKED_PERSON, MARKER], 10.0)
            self.assertEqual(tracker.update([head_only], 10.10).state, expected)

    def test_recovery_confirmation_follows_a_physical_path_across_new_ids(self):
        tracker = MarkerTracker(
            confirmation_seconds=0,
            recovery_confirmation_seconds=0.12,
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        first = Detection(
            "person", 0.8, BoundingBox(10, 0, 210, 300), track_id=12
        )
        second = Detection(
            "person", 0.8, BoundingBox(20, 0, 220, 300), track_id=13
        )
        third = Detection(
            "person", 0.8, BoundingBox(30, 0, 230, 300), track_id=14
        )
        fourth = Detection(
            "person", 0.8, BoundingBox(35, 0, 235, 300), track_id=14
        )

        self.assertEqual(tracker.update([first], 10.10).state, "recovering")
        self.assertEqual(tracker.update([second], 10.16).state, "recovering")
        self.assertEqual(tracker.update([third], 10.23).state, "recovering")
        recovered = tracker.update([fourth], 10.29)

        self.assertTrue(recovered.emit_handoff)
        self.assertEqual(recovered.target_track_id, 14)
        self.assertEqual(recovered.target_box, fourth.box)

    def test_recovery_accepts_top_aligned_partial_to_full_box_transition(self):
        tracker = MarkerTracker(
            confirmation_seconds=0,
            recovery_confirmation_seconds=0.12,
        )
        partial = Detection(
            "person", 0.8, BoundingBox(100, 100, 120, 135), track_id=7
        )
        marker = Detection("marker", 1.0, BoundingBox(105, 105, 110, 110))
        full = Detection(
            "person", 0.8, BoundingBox(95, 100, 125, 184), track_id=12
        )
        tracker.update([partial, marker], 10.0)

        confirming = tracker.update([full], 10.10)
        recovered = tracker.update([full], 10.23)

        self.assertAlmostEqual(full.box.area / partial.box.area, 3.6)
        self.assertEqual(confirming.state, "recovering")
        self.assertTrue(recovered.emit_handoff)
        self.assertEqual(recovered.target_track_id, 12)

    def test_trackless_observation_can_bridge_recovery_until_id_appears(self):
        tracker = MarkerTracker(
            confirmation_seconds=0,
            recovery_confirmation_seconds=0.12,
        )
        trackless = Detection(
            "person", 0.7, BoundingBox(10, 0, 210, 300), track_id=None
        )
        assigned = Detection(
            "person", 0.8, BoundingBox(20, 0, 220, 300), track_id=12
        )
        assigned_again = Detection(
            "person", 0.8, BoundingBox(25, 0, 225, 300), track_id=12
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)

        confirming = tracker.update([trackless], 10.10)
        assigned_confirming = tracker.update([assigned], 10.23)
        recovered = tracker.update([assigned_again], 10.30)

        self.assertEqual(confirming.state, "recovering")
        self.assertFalse(confirming.emit_handoff)
        self.assertEqual(assigned_confirming.state, "recovering")
        self.assertTrue(recovered.emit_handoff)
        self.assertEqual(recovered.target_track_id, 12)

    def test_ambiguous_recovery_candidates_remain_in_grace(self):
        tracker = MarkerTracker(confirmation_seconds=0)
        first = Detection("person", 0.8, BoundingBox(5, 0, 205, 300), track_id=12)
        second = Detection("person", 0.8, BoundingBox(5, 0, 205, 300), track_id=13)
        tracker.update([TRACKED_PERSON, MARKER], 10.0)

        decision = tracker.update([first, second], 10.1)

        self.assertEqual(decision.state, "grace")
        self.assertFalse(decision.emit_handoff)

    def test_appearance_recovers_distant_target_when_geometry_cannot(self):
        tracker = MarkerTracker(
            confirmation_seconds=0,
            recovery_confirmation_seconds=0,
            recovery_min_track_observations=1,
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        same_person = Detection(
            "person",
            0.8,
            BoundingBox(350, 0, 550, 300),
            track_id=12,
            appearance_similarity=0.90,
        )

        recovered = tracker.update([same_person], 10.1)

        self.assertTrue(recovered.emit_handoff)
        self.assertEqual(recovered.target_track_id, 12)

    def test_low_appearance_does_not_veto_strong_geometric_recovery(self):
        tracker = MarkerTracker(confirmation_seconds=0)
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        weak_match = Detection(
            "person",
            0.8,
            BoundingBox(10, 0, 210, 300),
            track_id=12,
            appearance_similarity=0.40,
        )

        decision = tracker.update([weak_match], 10.1)

        self.assertEqual(decision.state, "recovering")

    def test_low_appearance_cannot_recover_a_distant_candidate(self):
        tracker = MarkerTracker(confirmation_seconds=0)
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        weak_match = Detection(
            "person",
            0.8,
            BoundingBox(350, 0, 550, 300),
            track_id=12,
            appearance_similarity=0.40,
        )

        decision = tracker.update([weak_match], 10.1)

        self.assertEqual(decision.state, "grace")

    def test_initial_marker_confirmation_survives_raw_id_churn_on_same_path(self):
        tracker = MarkerTracker(confirmation_seconds=2.0)
        first = Detection(
            "person", 0.8, BoundingBox(100, 0, 200, 300), track_id=2
        )
        second = Detection(
            "person", 0.8, BoundingBox(102, 0, 202, 300), track_id=3
        )
        marker = Detection("marker", 1.0, BoundingBox(130, 100, 150, 140))

        tracker.update([first, marker], 10.0)
        tracker.update([second, marker], 11.0)
        decision = tracker.update([first, marker], 12.0)

        self.assertTrue(decision.emit_handoff)
        self.assertEqual(decision.target_track_id, 2)

    def test_appearance_switches_raw_id_after_margin_and_confirmation(self):
        tracker = MarkerTracker(
            confirmation_seconds=0,
            recovery_confirmation_seconds=0.12,
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        wrong_active = Detection(
            "person",
            0.8,
            BoundingBox(0, 0, 200, 300),
            track_id=7,
            appearance_similarity=0.45,
        )
        speaker = Detection(
            "person",
            0.8,
            BoundingBox(220, 0, 420, 300),
            track_id=8,
            appearance_similarity=0.90,
        )

        confirming = tracker.update([wrong_active, speaker], 10.10)
        recovered = tracker.update([wrong_active, speaker], 10.23)

        self.assertEqual(confirming.state, "recovering")
        self.assertTrue(recovered.emit_handoff)
        self.assertEqual(recovered.reason, "appearance_reidentified_target")
        self.assertEqual(recovered.target_track_id, 8)

    def test_marker_reacquisition_rejects_low_similarity_candidate(self):
        tracker = MarkerTracker(confirmation_seconds=0.5)
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        tracker.update([TRACKED_PERSON, MARKER], 10.5)
        candidate = Detection(
            "person",
            0.8,
            BoundingBox(10, 0, 210, 300),
            track_id=12,
            appearance_similarity=0.40,
        )

        decision = tracker.update([candidate, MARKER], 11.0)

        # One sighting is not enough to hand the target to a different-looking holder.
        self.assertEqual(decision.state, "recovering")
        self.assertEqual(decision.reason, "marker_reacquire_confirming")
        self.assertEqual(decision.target_track_id, 7)
        self.assertFalse(decision.emit_handoff)

    def test_marker_reacquisition_accepts_verified_appearance(self):
        tracker = MarkerTracker(confirmation_seconds=0)
        tracker.update([TRACKED_PERSON, MARKER], 10.0)
        candidate = Detection(
            "person",
            0.8,
            BoundingBox(10, 0, 210, 300),
            track_id=12,
            appearance_similarity=0.80,
        )

        decision = tracker.update([candidate, MARKER], 10.1)
        tracker.update([candidate, MARKER], 10.17)
        glimpse = tracker.update([candidate, MARKER], 10.23)
        recovered = tracker.update([candidate, MARKER], 10.65)

        # Three sightings within 0.13 s are a glimpse, not a re-registration.
        self.assertFalse(glimpse.emit_handoff)
        self.assertEqual(glimpse.reason, "marker_reacquire_confirming")
        self.assertEqual(decision.state, "recovering")
        self.assertEqual(decision.reason, "marker_reacquire_confirming")
        self.assertEqual(decision.target_track_id, 7)
        self.assertFalse(decision.emit_handoff)
        self.assertFalse(recovered.speaker_changed)
        self.assertEqual(recovered.reason, "marker_reacquired_target")
        self.assertEqual(recovered.target_track_id, 12)
        self.assertTrue(recovered.emit_handoff)

    def test_does_not_recover_after_release_grace_has_expired(self):
        tracker = MarkerTracker(
            confirmation_seconds=0,
            release_grace_seconds=1.0,
            recovery_confirmation_seconds=0,
        )
        replacement = Detection(
            "person", 0.8, BoundingBox(10, 0, 210, 300), track_id=12
        )
        tracker.update([TRACKED_PERSON, MARKER], 10.0)

        decision = tracker.update([replacement], 11.1)

        self.assertTrue(decision.emit_release)
        self.assertEqual(decision.reason, "target_released_after_grace")


if __name__ == "__main__":
    unittest.main()


class HolderGeometryTest(unittest.TestCase):
    # full1_walk_alongside_back frame 150: the holder (36) and the person
    # walking just behind them (15), both about as far from the camera.
    BEHIND = Detection("person", 0.91, BoundingBox(116, 92, 198, 358), track_id=15)
    HOLDER = Detection("person", 0.90, BoundingBox(166, 103, 268, 359), track_id=36)
    CARD = Detection("marker", 1.0, BoundingBox(183, 163, 199, 179))

    def test_default_rule_prefers_the_smaller_box(self):
        tracker = MarkerTracker()

        decision = tracker.update([self.BEHIND, self.HOLDER, self.CARD], 10.0)

        self.assertEqual(decision.target_track_id, 15)

    def test_geometry_credits_the_person_the_card_sits_on(self):
        tracker = MarkerTracker(holder_geometry=True)

        tracker.update([self.BEHIND, self.HOLDER, self.CARD], 10.0)
        decision = tracker.update([self.BEHIND, self.HOLDER, self.CARD], 13.0)

        self.assertEqual(decision.state, "tracking")
        self.assertEqual(decision.target_track_id, 36)

    def test_geometry_rejects_a_person_at_another_distance(self):
        far = Detection("person", 0.9, BoundingBox(140, 120, 200, 250), track_id=1)
        near = Detection("person", 0.9, BoundingBox(110, 40, 250, 340), track_id=2)
        # 16 px at 0.093 m is about 172 px per metre: a 1.7 m person 292 px tall.
        card = Detection("marker", 1.0, BoundingBox(172, 150, 188, 166))
        tracker = MarkerTracker(holder_geometry=True)

        tracker.update([far, near, card], 10.0)
        decision = tracker.update([far, near, card], 13.0)

        self.assertEqual(decision.target_track_id, 2)

    def test_close_call_waits_until_the_boxes_separate(self):
        left = Detection("person", 0.9, BoundingBox(100, 50, 200, 340), track_id=1)
        right = Detection("person", 0.9, BoundingBox(140, 50, 240, 340), track_id=2)
        card = Detection("marker", 1.0, BoundingBox(162, 150, 178, 166))
        tracker = MarkerTracker(holder_geometry=True, marker_dropout_seconds=0)

        decision = tracker.update([left, right, card], 10.0)
        self.assertEqual(decision.reason, "marker_holder_ambiguous")
        self.assertIsNone(tracker.update([left, right, card], 13.5).target_track_id)

        apart = Detection("person", 0.9, BoundingBox(250, 50, 350, 340), track_id=2)
        tracker.update([left, apart, card], 14.0)
        decision = tracker.update([left, apart, card], 17.0)

        self.assertEqual(decision.state, "tracking")
        self.assertEqual(decision.target_track_id, 1)

    def test_overlapping_boxes_of_one_person_are_not_a_close_call(self):
        full = Detection("person", 0.9, BoundingBox(100, 50, 200, 340), track_id=1)
        upper = Detection("person", 0.8, BoundingBox(102, 50, 198, 250), track_id=2)
        card = Detection("marker", 1.0, BoundingBox(142, 150, 158, 166))
        tracker = MarkerTracker(holder_geometry=True)

        decision = tracker.update([full, upper, card], 10.0)

        self.assertEqual(decision.state, "confirming")
        self.assertNotEqual(decision.reason, "marker_holder_ambiguous")
