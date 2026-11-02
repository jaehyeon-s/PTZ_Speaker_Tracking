import unittest

import numpy as np

from marker_tracker.core import BoundingBox, Detection, HandoffDecision, MarkerTracker
from marker_tracker.identity import SpeakerIdentityManager


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, frame, people):
        del frame
        self.calls.append(tuple(person.track_id for person in people))
        return [
            np.asarray([0.0, 1.0], dtype=np.float32)
            if person.track_id in {3, 4}
            else np.asarray([1.0, 0.0], dtype=np.float32)
            for person in people
        ]


class ScriptedSimilarityEncoder:
    def __init__(self):
        self.similarities = {}
        self.calls = []

    def encode(self, frame, people):
        del frame
        self.calls.append(tuple(person.track_id for person in people))
        features = []
        for person in people:
            similarity = self.similarities.get(person.track_id, 1.0)
            features.append(
                np.asarray(
                    [similarity, max(0.0, 1.0 - similarity**2) ** 0.5],
                    dtype=np.float32,
                )
            )
        return features


TARGET = Detection("person", 0.9, BoundingBox(0, 0, 100, 200), track_id=1)
MARKER = Detection("marker", 1.0, BoundingBox(30, 50, 50, 80))
FRAME = np.zeros((240, 640, 3), dtype=np.uint8)


class SpeakerIdentityManagerTest(unittest.TestCase):
    def _registered_manager(self):
        encoder = FakeEncoder()
        manager = SpeakerIdentityManager(encoder, gallery_interval_seconds=0)
        decision = HandoffDecision(
            "tracking",
            TARGET.box,
            MARKER.box,
            target_track_id=1,
            reason="confirmed_marker_handoff",
            emit_handoff=True,
        )
        observed = manager.observe(decision, [TARGET, MARKER], FRAME, 1.0)
        return encoder, manager, observed

    def test_marker_registration_creates_gallery_and_logical_speaker(self):
        encoder, manager, observed = self._registered_manager()

        self.assertEqual(observed.speaker_id, 1)
        self.assertEqual(manager.gallery_size, 1)
        self.assertEqual(encoder.calls, [(1,)])

    def test_anchor_gallery_is_not_evicted_by_adaptive_samples(self):
        encoder = ScriptedSimilarityEncoder()
        manager = SpeakerIdentityManager(
            encoder,
            gallery_size=3,
            anchor_gallery_size=2,
            gallery_interval_seconds=0,
            gallery_novelty_similarity=0.999,
        )
        decision = HandoffDecision(
            "tracking",
            TARGET.box,
            MARKER.box,
            target_track_id=1,
            reason="confirmed_marker_handoff",
            emit_handoff=True,
        )
        for index, similarity in enumerate((1.0, 0.8, 0.55, 0.25)):
            encoder.similarities = {1: similarity}
            manager.observe(
                decision,
                [TARGET, MARKER],
                FRAME.copy(),
                1.0 + index,
            )
            decision = HandoffDecision(
                "tracking",
                TARGET.box,
                MARKER.box,
                target_track_id=1,
                reason="active_track_visible",
            )

        self.assertEqual(manager.anchor_gallery_size, 2)
        self.assertEqual(manager.gallery_size, 3)

    def test_missing_raw_id_scores_replacement_candidates(self):
        _, manager, _ = self._registered_manager()
        same_person = Detection(
            "person", 0.8, BoundingBox(120, 0, 220, 200), track_id=2
        )
        other_person = Detection(
            "person", 0.8, BoundingBox(300, 0, 400, 200), track_id=3
        )

        scored = manager.score_detections(
            FRAME.copy(), [same_person, other_person], 2.0
        )

        self.assertAlmostEqual(scored[0].appearance_similarity, 1.0)
        self.assertAlmostEqual(scored[1].appearance_similarity, 0.0)

    def test_feature_cache_is_scoped_to_the_actual_frame_object(self):
        encoder, manager, _ = self._registered_manager()
        replacement = Detection(
            "person", 0.8, BoundingBox(120, 0, 220, 200), track_id=2
        )

        manager.score_detections(FRAME.copy(), [replacement], 2.0)
        manager.score_detections(FRAME.copy(), [replacement], 2.1)

        self.assertEqual(encoder.calls[-2:], [(2,), (2,)])

    def test_exclusion_gallery_penalizes_a_known_non_speaker(self):
        encoder = FakeEncoder()
        manager = SpeakerIdentityManager(
            encoder,
            gallery_interval_seconds=0,
            negative_gallery_weight=0.5,
        )
        non_speaker = Detection(
            "person", 0.9, BoundingBox(300, 0, 400, 200), track_id=3
        )
        manager.observe(
            HandoffDecision(
                "tracking",
                TARGET.box,
                MARKER.box,
                target_track_id=1,
                reason="confirmed_marker_handoff",
                emit_handoff=True,
            ),
            [TARGET, non_speaker, MARKER],
            FRAME.copy(),
            1.0,
        )
        same_speaker = Detection(
            "person", 0.8, BoundingBox(120, 0, 220, 200), track_id=2
        )
        known_non_speaker = Detection(
            "person", 0.8, BoundingBox(320, 0, 420, 200), track_id=4
        )

        scored = manager.score_detections(
            FRAME.copy(),
            [same_speaker, known_non_speaker],
            2.0,
        )

        self.assertEqual(manager.negative_gallery_size, 1)
        self.assertGreater(
            scored[0].appearance_similarity,
            scored[1].appearance_similarity,
        )

    def test_overlap_is_held_and_scored_after_boxes_separate(self):
        encoder, manager, _ = self._registered_manager()
        encoder.calls.clear()
        active = Detection("person", 0.8, BoundingBox(0, 0, 100, 200), track_id=1)
        overlapping = Detection(
            "person", 0.8, BoundingBox(40, 0, 140, 200), track_id=3
        )

        held = manager.score_detections(FRAME.copy(), [active, overlapping], 2.0)

        self.assertEqual(encoder.calls, [])
        self.assertEqual(held[0].appearance_similarity, -1.0)
        separated = Detection(
            "person", 0.8, BoundingBox(200, 0, 300, 200), track_id=3
        )
        scored = manager.score_detections(FRAME.copy(), [active, separated], 2.1)
        self.assertEqual(encoder.calls, [(1, 3)])
        self.assertAlmostEqual(scored[0].appearance_similarity, 1.0)
        self.assertAlmostEqual(scored[1].appearance_similarity, 0.0)

    def test_gallery_is_not_updated_from_overlapping_marker_crop(self):
        _, manager, _ = self._registered_manager()
        overlapping = Detection(
            "person", 0.8, BoundingBox(40, 0, 140, 200), track_id=3
        )
        decision = HandoffDecision(
            "tracking", TARGET.box, MARKER.box, target_track_id=1
        )

        manager.observe(decision, [TARGET, overlapping, MARKER], FRAME.copy(), 2.0)

        self.assertEqual(manager.gallery_size, 1)

    def test_speaker_id_survives_raw_track_id_change(self):
        _, manager, _ = self._registered_manager()
        replacement = Detection(
            "person", 0.9, BoundingBox(120, 0, 220, 200), track_id=2
        )
        decision = HandoffDecision(
            "tracking",
            replacement.box,
            None,
            target_track_id=2,
            reason="target_recovered_new_track_id",
            emit_handoff=True,
        )

        observed = manager.observe(decision, [replacement], FRAME.copy(), 2.0)

        self.assertEqual(observed.speaker_id, 1)
        self.assertEqual(observed.target_track_id, 2)

    def test_reappeared_raw_id_triggers_identity_verification_burst(self):
        encoder, manager, _ = self._registered_manager()
        encoder.calls.clear()

        manager.score_detections(FRAME.copy(), [], 2.0)
        active = Detection("person", 0.8, TARGET.box, track_id=1)
        alternative = Detection(
            "person", 0.8, BoundingBox(200, 0, 300, 200), track_id=3
        )
        scored = manager.score_detections(
            FRAME.copy(), [active, alternative], 2.1
        )

        self.assertEqual(encoder.calls, [(1, 3)])
        self.assertAlmostEqual(scored[0].appearance_similarity, 1.0)
        self.assertAlmostEqual(scored[1].appearance_similarity, 0.0)

    def test_release_stops_selective_reid_until_next_handoff(self):
        encoder, manager, _ = self._registered_manager()
        encoder.calls.clear()
        release = HandoffDecision(
            "default",
            TARGET.box,
            None,
            target_track_id=1,
            reason="target_released_after_grace",
            emit_release=True,
        )
        manager.observe(release, [], FRAME.copy(), 2.0)
        candidate = Detection(
            "person", 0.8, BoundingBox(200, 0, 300, 200), track_id=3
        )

        scored = manager.score_detections(FRAME.copy(), [candidate], 2.1)

        self.assertEqual(encoder.calls, [])
        self.assertIsNone(scored[0].appearance_similarity)

    def test_low_similarity_marker_candidate_does_not_pollute_gallery(self):
        _, manager, _ = self._registered_manager()
        candidate = Detection(
            "person",
            0.8,
            BoundingBox(0, 0, 100, 200),
            track_id=3,
            appearance_similarity=0.40,
        )
        decision = HandoffDecision(
            "tracking", candidate.box, MARKER.box, target_track_id=3
        )

        manager.observe(decision, [candidate, MARKER], FRAME.copy(), 2.0)

        self.assertEqual(manager.gallery_size, 1)

    def test_confirmed_marker_transfer_increments_speaker_and_replaces_gallery(self):
        encoder, manager, _ = self._registered_manager()
        new_speaker = Detection(
            "person", 0.9, BoundingBox(200, 0, 300, 200), track_id=3
        )
        new_marker = Detection("marker", 1.0, BoundingBox(230, 50, 250, 80))
        transfer = HandoffDecision(
            "tracking",
            new_speaker.box,
            new_marker.box,
            target_track_id=3,
            reason="confirmed_marker_transfer",
            emit_handoff=True,
            speaker_changed=True,
        )

        observed = manager.observe(
            transfer,
            [new_speaker, new_marker],
            FRAME.copy(),
            2.0,
        )

        self.assertEqual(observed.speaker_id, 2)
        self.assertEqual(manager.gallery_size, 1)
        manager.score_detections(FRAME.copy(), [], 2.1)
        scored = manager.score_detections(
            FRAME.copy(),
            [TARGET, new_speaker],
            2.2,
        )
        self.assertAlmostEqual(scored[0].appearance_similarity, 0.0)
        self.assertAlmostEqual(scored[1].appearance_similarity, 1.0)

    def test_new_speaker_remains_identity_owner_across_later_raw_id_change(self):
        encoder = FakeEncoder()
        manager = SpeakerIdentityManager(encoder, gallery_interval_seconds=0)
        tracker = MarkerTracker(
            confirmation_seconds=0,
            recovery_confirmation_seconds=0,
            switch_cooldown_seconds=0,
        )
        initial = tracker.update([TARGET, MARKER], 1.0)
        initial = manager.observe(initial, [TARGET, MARKER], FRAME.copy(), 1.0)
        new_speaker = Detection(
            "person", 0.9, BoundingBox(200, 0, 300, 200), track_id=3
        )
        new_marker = Detection("marker", 1.0, BoundingBox(230, 50, 250, 80))

        transferred = tracker.update(
            [TARGET, new_speaker, new_marker],
            2.0,
        )
        transferred = manager.observe(
            transferred,
            [TARGET, new_speaker, new_marker],
            FRAME.copy(),
            2.0,
        )
        replacement = Detection(
            "person", 0.8, BoundingBox(210, 0, 310, 200), track_id=4
        )
        scored = manager.score_detections(
            FRAME.copy(),
            [TARGET, replacement],
            2.1,
        )
        recovered = tracker.update(scored, 2.1)
        recovered = manager.observe(
            recovered,
            scored,
            FRAME.copy(),
            2.1,
        )
        scored = manager.score_detections(
            FRAME.copy(),
            [TARGET, replacement],
            2.2,
        )
        recovered = tracker.update(scored, 2.2)
        recovered = manager.observe(
            recovered,
            scored,
            FRAME.copy(),
            2.2,
        )

        self.assertEqual(initial.speaker_id, 1)
        self.assertEqual(transferred.speaker_id, 2)
        self.assertEqual(transferred.reason, "confirmed_marker_transfer")
        self.assertEqual(recovered.speaker_id, 2)
        self.assertEqual(recovered.target_track_id, 4)
        self.assertEqual(recovered.reason, "target_recovered_new_track_id")

    def test_crossing_id_swap_waits_for_temporal_identity_evidence(self):
        encoder = ScriptedSimilarityEncoder()
        manager = SpeakerIdentityManager(
            encoder,
            gallery_interval_seconds=10,
            resolution_min_frames=4,
            resolution_min_seconds=0.35,
            resolution_min_similarity=0.35,
            resolution_min_margin=0.02,
            resolution_winner_ratio=0.75,
            resolution_exit_frames=0,
        )
        tracker = MarkerTracker(recovery_confirmation_seconds=0.12)
        speaker = Detection(
            "person", 0.8, BoundingBox(100, 0, 200, 200), track_id=2
        )
        other = Detection(
            "person", 0.8, BoundingBox(300, 0, 400, 200), track_id=1
        )
        marker = Detection("marker", 1.0, BoundingBox(130, 50, 150, 80))
        registered = HandoffDecision(
            "tracking",
            speaker.box,
            marker.box,
            target_track_id=2,
            reason="confirmed_marker_handoff",
            emit_handoff=True,
        )
        manager.observe(registered, [speaker, other, marker], FRAME.copy(), 1.0)
        manager.score_detections(FRAME.copy(), [speaker, other], 1.1)

        # This reproduces the failed run: the original speaker ID disappears
        # and the non-speaker is the only detection. Geometry must not commit it.
        encoder.similarities = {1: 0.404}
        scored = manager.score_detections(FRAME.copy(), [other], 2.0)
        held = tracker.update(
            scored,
            2.0,
            identity_resolution=manager.resolution_context(),
        )
        held = manager.observe(held, scored, FRAME.copy(), 2.0)
        self.assertEqual(held.state, "identity_resolving")
        self.assertFalse(held.emit_handoff)

        # The raw ID 2 then appears on the wrong physical person. It must not
        # regain ownership merely because its numeric ID matches the old one.
        wrong_id_two = Detection(
            "person", 0.7, BoundingBox(220, 0, 320, 200), track_id=2
        )
        encoder.similarities = {2: 0.435}
        scored = manager.score_detections(FRAME.copy(), [wrong_id_two], 2.1)
        held = tracker.update(
            scored,
            2.1,
            identity_resolution=manager.resolution_context(),
        )
        self.assertEqual(held.state, "identity_resolving")
        self.assertFalse(held.emit_handoff)

        correct_id_thirteen = Detection(
            "person", 0.7, BoundingBox(500, 0, 600, 200), track_id=13
        )
        score_pairs = [
            (0.425, 0.566),
            (0.467, 0.612),
            (0.464, 0.581),
            (0.414, 0.586),
            (0.430, 0.600),
        ]
        decision = held
        frames = []
        for index, (wrong_score, correct_score) in enumerate(score_pairs):
            encoder.similarities = {2: wrong_score, 13: correct_score}
            frame = FRAME.copy()
            frames.append(frame)
            detections = manager.score_detections(
                frame,
                [wrong_id_two, correct_id_thirteen],
                2.2 + index * 0.1,
            )
            decision = tracker.update(
                detections,
                2.2 + index * 0.1,
                identity_resolution=manager.resolution_context(),
            )
            decision = manager.observe(
                decision,
                detections,
                frame,
                2.2 + index * 0.1,
            )

        self.assertEqual(decision.state, "tracking")
        self.assertTrue(decision.emit_handoff)
        self.assertEqual(decision.reason, "identity_verified_target")
        self.assertEqual(decision.target_track_id, 13)
        self.assertEqual(decision.speaker_id, 1)

    def test_ambiguous_crossing_holds_instead_of_committing_wrong_target(self):
        encoder = ScriptedSimilarityEncoder()
        manager = SpeakerIdentityManager(
            encoder,
            gallery_interval_seconds=10,
            resolution_min_frames=4,
            resolution_min_seconds=0.2,
            resolution_min_similarity=0.35,
            resolution_min_margin=0.03,
            resolution_winner_ratio=0.75,
        )
        tracker = MarkerTracker(recovery_confirmation_seconds=0)
        speaker = Detection(
            "person", 0.8, BoundingBox(100, 0, 200, 200), track_id=2
        )
        other = Detection(
            "person", 0.8, BoundingBox(300, 0, 400, 200), track_id=1
        )
        marker = Detection("marker", 1.0, BoundingBox(130, 50, 150, 80))
        manager.observe(
            HandoffDecision(
                "tracking",
                speaker.box,
                marker.box,
                target_track_id=2,
                reason="confirmed_marker_handoff",
                emit_handoff=True,
            ),
            [speaker, other, marker],
            FRAME.copy(),
            1.0,
        )
        manager.score_detections(FRAME.copy(), [speaker, other], 1.1)
        encoder.similarities = {1: 0.42}
        manager.score_detections(FRAME.copy(), [other], 1.2)

        left = Detection("person", 0.8, BoundingBox(100, 0, 200, 200), track_id=8)
        right = Detection("person", 0.8, BoundingBox(300, 0, 400, 200), track_id=9)
        decision = None
        frames = []
        for index in range(8):
            # Neither physical path has a stable or meaningful advantage.
            encoder.similarities = (
                {8: 0.45, 9: 0.44}
                if index % 2 == 0
                else {8: 0.44, 9: 0.45}
            )
            frame = FRAME.copy()
            frames.append(frame)
            detections = manager.score_detections(
                frame,
                [left, right],
                1.3 + index * 0.1,
            )
            decision = tracker.update(
                detections,
                1.3 + index * 0.1,
                identity_resolution=manager.resolution_context(),
            )
            manager.observe(
                decision,
                detections,
                frame,
                1.3 + index * 0.1,
            )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.state, "identity_resolving")
        self.assertFalse(decision.emit_handoff)

    def test_approaching_people_are_not_committed_until_paths_separate(self):
        """Regression for crossing_03: raw IDs swap at the physical crossing."""

        encoder = ScriptedSimilarityEncoder()
        manager = SpeakerIdentityManager(
            encoder,
            gallery_interval_seconds=10,
            negative_gallery_weight=0,
            resolution_min_frames=6,
            resolution_min_seconds=0.3,
            resolution_min_margin=0.05,
            resolution_winner_ratio=0.60,
            resolution_close_distance=1.75,
            resolution_exit_distance=1.9,
            resolution_exit_frames=3,
        )
        speaker = Detection(
            "person", 0.8, BoundingBox(300, 0, 350, 180), track_id=67
        )
        marker = Detection("marker", 1.0, BoundingBox(315, 50, 325, 70))
        manager.observe(
            HandoffDecision(
                "tracking",
                speaker.box,
                marker.box,
                target_track_id=67,
                reason="confirmed_marker_handoff",
                emit_handoff=True,
            ),
            [speaker, marker],
            FRAME.copy(),
            1.0,
        )
        other = Detection(
            "person", 0.8, BoundingBox(550, 0, 600, 180), track_id=69
        )
        manager.score_detections(FRAME.copy(), [speaker, other], 1.1)

        # Both candidates are still approaching. The old implementation
        # committed ID 70 here and then silently followed that raw ID onto the
        # other physical person.
        approaching = [
            (340, 590),
            (365, 565),
            (390, 540),
            (415, 515),
            (440, 490),
            (455, 475),
        ]
        for index, (speaker_x, other_x) in enumerate(approaching):
            black = Detection(
                "person",
                0.8,
                BoundingBox(speaker_x, 0, speaker_x + 50, 180),
                track_id=70,
            )
            gray = Detection(
                "person",
                0.8,
                BoundingBox(other_x, 0, other_x + 50, 180),
                track_id=69,
            )
            encoder.similarities = {70: 0.60, 69: 0.42}
            manager.score_detections(
                FRAME.copy(), [black, gray], 1.2 + index * 0.1
            )

        self.assertTrue(manager.resolution_context().active)
        self.assertIsNone(manager.resolution_context().verified_track_id)

        # After the physical paths cross, the tracker assigns the old ID 70
        # to gray while black receives ID 81. Velocity-based path association
        # must carry speaker evidence onto ID 81.
        separated = [
            (500, 430),
            (535, 395),
            (570, 360),
            (605, 325),
        ]
        for index, (black_x, gray_x) in enumerate(separated):
            black = Detection(
                "person",
                0.8,
                BoundingBox(black_x, 0, black_x + 50, 180),
                track_id=81,
            )
            gray = Detection(
                "person",
                0.8,
                BoundingBox(gray_x, 0, gray_x + 50, 180),
                track_id=70,
            )
            encoder.similarities = {81: 0.60, 70: 0.42}
            manager.score_detections(
                FRAME.copy(), [black, gray], 1.8 + index * 0.1
            )

        self.assertEqual(manager.resolution_context().verified_track_id, 81)

    def test_soft_coast_expires_without_multi_path_evidence(self):
        encoder = ScriptedSimilarityEncoder()
        manager = SpeakerIdentityManager(
            encoder,
            gallery_interval_seconds=10,
            coast_max_seconds=0.5,
        )
        speaker = Detection(
            "person", 0.8, BoundingBox(100, 0, 200, 200), track_id=2
        )
        other = Detection(
            "person", 0.8, BoundingBox(400, 0, 500, 200), track_id=1
        )
        marker = Detection("marker", 1.0, BoundingBox(130, 50, 150, 80))
        manager.observe(
            HandoffDecision(
                "tracking",
                speaker.box,
                marker.box,
                target_track_id=2,
                reason="confirmed_marker_handoff",
                emit_handoff=True,
            ),
            [speaker, marker],
            FRAME.copy(),
            1.0,
        )
        manager.score_detections(FRAME.copy(), [speaker, other], 1.1)
        manager.score_detections(FRAME.copy(), [other], 1.2)

        self.assertIsNotNone(manager.resolution_context().coast_box)
        manager.score_detections(FRAME.copy(), [], 1.8)
        self.assertIsNone(manager.resolution_context().coast_box)

    def test_trackless_crossing_crops_do_not_seed_identity_paths(self):
        encoder = ScriptedSimilarityEncoder()
        manager = SpeakerIdentityManager(
            encoder,
            gallery_interval_seconds=10,
            resolution_min_frames=3,
            resolution_min_seconds=0.15,
            resolution_min_similarity=0.35,
            resolution_min_margin=0.02,
            resolution_winner_ratio=0.66,
        )
        speaker = Detection(
            "person", 0.8, BoundingBox(100, 0, 200, 200), track_id=2
        )
        other = Detection(
            "person", 0.8, BoundingBox(400, 0, 500, 200), track_id=1
        )
        marker = Detection("marker", 1.0, BoundingBox(130, 50, 150, 80))
        manager.observe(
            HandoffDecision(
                "tracking",
                speaker.box,
                marker.box,
                target_track_id=2,
                reason="confirmed_marker_handoff",
                emit_handoff=True,
            ),
            [speaker, other, marker],
            FRAME.copy(),
            1.0,
        )
        manager.score_detections(FRAME.copy(), [speaker, other], 1.1)

        trackless_left = Detection(
            "person", 0.6, BoundingBox(200, 0, 260, 180), track_id=None
        )
        trackless_right = Detection(
            "person", 0.6, BoundingBox(300, 0, 360, 180), track_id=None
        )
        manager.score_detections(
            FRAME.copy(),
            [trackless_left, trackless_right],
            1.2,
        )
        self.assertIsNone(manager.resolution_context().verified_track_id)

        correct = Detection(
            "person", 0.8, BoundingBox(300, 0, 400, 200), track_id=8
        )
        wrong = Detection(
            "person", 0.8, BoundingBox(100, 0, 200, 200), track_id=9
        )
        frames = []
        for index in range(3):
            encoder.similarities = {8: 0.70, 9: 0.30}
            frame = FRAME.copy()
            frames.append(frame)
            manager.score_detections(
                frame,
                [correct, wrong],
                1.3 + index * 0.1,
            )

        self.assertEqual(manager.resolution_context().verified_track_id, 8)

    def test_single_high_confidence_identity_can_exit_resolution(self):
        encoder = ScriptedSimilarityEncoder()
        manager = SpeakerIdentityManager(
            encoder,
            gallery_interval_seconds=10,
            resolution_min_frames=4,
            resolution_min_seconds=0.3,
        )
        tracker = MarkerTracker(recovery_confirmation_seconds=0)
        speaker = Detection(
            "person", 0.8, BoundingBox(100, 0, 200, 200), track_id=2
        )
        other = Detection(
            "person", 0.8, BoundingBox(300, 0, 400, 200), track_id=1
        )
        marker = Detection("marker", 1.0, BoundingBox(130, 50, 150, 80))
        manager.observe(
            HandoffDecision(
                "tracking",
                speaker.box,
                marker.box,
                target_track_id=2,
                reason="confirmed_marker_handoff",
                emit_handoff=True,
            ),
            [speaker, other, marker],
            FRAME.copy(),
            1.0,
        )
        manager.score_detections(FRAME.copy(), [speaker, other], 1.1)
        replacement = Detection(
            "person", 0.8, BoundingBox(120, 0, 220, 200), track_id=7
        )
        encoder.similarities = {7: 0.75}

        decision = None
        frames = []
        for index in range(4):
            frame = FRAME.copy()
            frames.append(frame)
            detections = manager.score_detections(
                frame,
                [replacement],
                1.2 + index * 0.1,
            )
            decision = tracker.update(
                detections,
                1.2 + index * 0.1,
                identity_resolution=manager.resolution_context(),
            )
            decision = manager.observe(
                decision,
                detections,
                frame,
                1.2 + index * 0.1,
            )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.state, "tracking")
        self.assertEqual(decision.target_track_id, 7)
        self.assertEqual(decision.reason, "identity_verified_target")

    def test_single_resolution_preserves_evidence_across_raw_id_churn(self):
        encoder = ScriptedSimilarityEncoder()
        manager = SpeakerIdentityManager(
            encoder,
            gallery_interval_seconds=10,
            single_resolution_min_similarity=0.45,
            single_resolution_min_frames=4,
            single_resolution_min_seconds=0.25,
        )
        speaker = Detection(
            "person", 0.8, BoundingBox(100, 0, 200, 200), track_id=2
        )
        other = Detection(
            "person", 0.8, BoundingBox(300, 0, 400, 200), track_id=1
        )
        marker = Detection("marker", 1.0, BoundingBox(130, 50, 150, 80))
        manager.observe(
            HandoffDecision(
                "tracking",
                speaker.box,
                marker.box,
                target_track_id=2,
                reason="confirmed_marker_handoff",
                emit_handoff=True,
            ),
            [speaker, other, marker],
            FRAME.copy(),
            1.0,
        )
        crossing = Detection(
            "person", 0.8, BoundingBox(150, 0, 250, 200), track_id=1
        )
        manager.score_detections(FRAME.copy(), [speaker, crossing], 1.1)

        for index, track_id in enumerate((7, 8, 9, 10)):
            encoder.similarities[track_id] = 0.48
            candidate = Detection(
                "person",
                0.8,
                BoundingBox(110 + index * 5, 0, 210 + index * 5, 200),
                track_id=track_id,
            )
            manager.score_detections(
                FRAME.copy(),
                [candidate],
                1.2 + index * 0.1,
            )

        self.assertEqual(manager.resolution_context().verified_track_id, 10)

    def test_single_low_similarity_path_is_coasted_without_verification(self):
        encoder = ScriptedSimilarityEncoder()
        manager = SpeakerIdentityManager(
            encoder,
            gallery_interval_seconds=10,
            single_resolution_min_similarity=0.50,
            single_resolution_min_frames=6,
            single_resolution_min_seconds=0.45,
            resolution_min_similarity=0.35,
        )
        speaker = Detection(
            "person", 0.8, BoundingBox(100, 0, 200, 200), track_id=2
        )
        other = Detection(
            "person", 0.8, BoundingBox(300, 0, 400, 200), track_id=1
        )
        marker = Detection("marker", 1.0, BoundingBox(130, 50, 150, 80))
        manager.observe(
            HandoffDecision(
                "tracking",
                speaker.box,
                marker.box,
                target_track_id=2,
                reason="confirmed_marker_handoff",
                emit_handoff=True,
            ),
            [speaker, other, marker],
            FRAME.copy(),
            1.0,
        )
        crossing = Detection(
            "person", 0.8, BoundingBox(150, 0, 250, 200), track_id=1
        )
        manager.score_detections(FRAME.copy(), [speaker, crossing], 1.1)

        encoder.similarities[7] = 0.40
        candidate = Detection(
            "person", 0.8, BoundingBox(115, 0, 215, 200), track_id=7
        )
        manager.score_detections(FRAME.copy(), [candidate], 1.2)

        context = manager.resolution_context()
        self.assertTrue(context.active)
        self.assertIsNone(context.verified_track_id)
        self.assertEqual(context.coast_box, candidate.box)

    def test_distant_medium_similarity_single_candidate_is_not_fast_tracked(self):
        encoder = ScriptedSimilarityEncoder()
        manager = SpeakerIdentityManager(
            encoder,
            gallery_interval_seconds=10,
            single_resolution_min_similarity=0.45,
            single_resolution_min_frames=3,
            single_resolution_min_seconds=0.2,
        )
        speaker = Detection(
            "person", 0.8, BoundingBox(100, 0, 200, 200), track_id=2
        )
        other = Detection(
            "person", 0.8, BoundingBox(300, 0, 400, 200), track_id=1
        )
        marker = Detection("marker", 1.0, BoundingBox(130, 50, 150, 80))
        manager.observe(
            HandoffDecision(
                "tracking",
                speaker.box,
                marker.box,
                target_track_id=2,
                reason="confirmed_marker_handoff",
                emit_handoff=True,
            ),
            [speaker, other, marker],
            FRAME.copy(),
            1.0,
        )
        crossing = Detection(
            "person", 0.8, BoundingBox(150, 0, 250, 200), track_id=1
        )
        manager.score_detections(FRAME.copy(), [speaker, crossing], 1.1)
        encoder.similarities[7] = 0.48

        for index in range(5):
            distant = Detection(
                "person", 0.8, BoundingBox(450, 0, 550, 200), track_id=7
            )
            manager.score_detections(
                FRAME.copy(), [distant], 1.2 + index * 0.1
            )

        self.assertIsNone(manager.resolution_context().verified_track_id)

    def test_long_resolution_enters_lower_rate_search_mode(self):
        encoder = ScriptedSimilarityEncoder()
        manager = SpeakerIdentityManager(
            encoder,
            gallery_interval_seconds=10,
            resolution_search_after_seconds=0.5,
            resolution_search_interval_frames=3,
            single_resolution_min_frames=10,
        )
        speaker = Detection(
            "person", 0.8, BoundingBox(100, 0, 200, 200), track_id=2
        )
        other = Detection(
            "person", 0.8, BoundingBox(300, 0, 400, 200), track_id=1
        )
        marker = Detection("marker", 1.0, BoundingBox(130, 50, 150, 80))
        manager.observe(
            HandoffDecision(
                "tracking",
                speaker.box,
                marker.box,
                target_track_id=2,
                reason="confirmed_marker_handoff",
                emit_handoff=True,
            ),
            [speaker, other, marker],
            FRAME.copy(),
            1.0,
        )
        crossing = Detection(
            "person", 0.8, BoundingBox(150, 0, 250, 200), track_id=1
        )
        manager.score_detections(FRAME.copy(), [speaker, crossing], 1.1)
        encoder.similarities[7] = 0.48
        calls_before = len(encoder.calls)

        for index in range(4):
            candidate = Detection(
                "person", 0.8, BoundingBox(110, 0, 210, 200), track_id=7
            )
            manager.score_detections(FRAME.copy(), [candidate], 1.7 + index * 0.1)

        context = manager.resolution_context()
        self.assertTrue(context.active)
        self.assertEqual(context.reason, "identity_searching")
        self.assertIsNotNone(context.coast_box)
        self.assertEqual(len(encoder.calls) - calls_before, 2)

    def test_same_raw_id_verification_completes_resolution_without_handoff(self):
        encoder = ScriptedSimilarityEncoder()
        manager = SpeakerIdentityManager(
            encoder,
            gallery_interval_seconds=10,
            resolution_min_frames=4,
            resolution_min_seconds=0.3,
            resolution_min_similarity=0.35,
            resolution_min_margin=0.02,
            resolution_winner_ratio=0.75,
        )
        tracker = MarkerTracker(
            confirmation_seconds=0,
            recovery_confirmation_seconds=0,
        )
        speaker = Detection(
            "person", 0.8, BoundingBox(100, 0, 200, 200), track_id=2
        )
        other = Detection(
            "person", 0.8, BoundingBox(300, 0, 400, 200), track_id=1
        )
        marker = Detection("marker", 1.0, BoundingBox(130, 50, 150, 80))
        registered = tracker.update([speaker, other, marker], 1.0)
        manager.observe(
            registered,
            [speaker, other, marker],
            FRAME.copy(),
            1.0,
        )

        # Two distinct boxes overlap enough to enter crossing resolution.
        crossing_other = Detection(
            "person", 0.8, BoundingBox(150, 0, 250, 200), track_id=1
        )
        manager.score_detections(FRAME.copy(), [speaker, crossing_other], 1.1)
        self.assertTrue(manager.resolution_context().active)

        encoder.similarities = {2: 0.75, 1: 0.25}
        decision = None
        reasons = []
        for index in range(5):
            detections = manager.score_detections(
                FRAME.copy(),
                [speaker, other],
                1.2 + index * 0.1,
            )
            decision = tracker.update(
                detections,
                1.2 + index * 0.1,
                identity_resolution=manager.resolution_context(),
            )
            decision = manager.observe(
                decision,
                detections,
                FRAME.copy(),
                1.2 + index * 0.1,
            )
            reasons.append(decision.reason)

        self.assertIsNotNone(decision)
        self.assertIn("identity_verified_target", reasons)
        self.assertEqual(decision.target_track_id, 2)
        self.assertFalse(decision.emit_handoff)
        self.assertFalse(manager.resolution_context().active)

    def test_different_raw_id_verification_requires_reid_similarity_floor(self):
        encoder = ScriptedSimilarityEncoder()
        manager = SpeakerIdentityManager(
            encoder,
            gallery_interval_seconds=10,
            min_similarity=0.55,
            resolution_min_frames=4,
            resolution_min_seconds=0.3,
            resolution_min_similarity=0.35,
            resolution_min_margin=0.02,
            resolution_winner_ratio=0.75,
        )
        tracker = MarkerTracker(
            confirmation_seconds=0,
            recovery_confirmation_seconds=0,
        )
        speaker = Detection(
            "person", 0.8, BoundingBox(100, 0, 200, 200), track_id=2
        )
        other = Detection(
            "person", 0.8, BoundingBox(300, 0, 400, 200), track_id=1
        )
        marker = Detection("marker", 1.0, BoundingBox(130, 50, 150, 80))
        registered = tracker.update([speaker, other, marker], 1.0)
        manager.observe(
            registered,
            [speaker, other, marker],
            FRAME.copy(),
            1.0,
        )

        crossing_other = Detection(
            "person", 0.8, BoundingBox(150, 0, 250, 200), track_id=1
        )
        manager.score_detections(FRAME.copy(), [speaker, crossing_other], 1.1)
        self.assertTrue(manager.resolution_context().active)

        replacement = Detection(
            "person", 0.8, BoundingBox(100, 0, 200, 200), track_id=50
        )
        distant = Detection(
            "person", 0.8, BoundingBox(400, 0, 500, 200), track_id=43
        )
        decision = None
        for index in range(6):
            encoder.similarities = {50: 0.52, 43: 0.42}
            detections = manager.score_detections(
                FRAME.copy(),
                [replacement, distant],
                1.2 + index * 0.1,
            )
            decision = tracker.update(
                detections,
                1.2 + index * 0.1,
                identity_resolution=manager.resolution_context(),
            )
            decision = manager.observe(
                decision,
                detections,
                FRAME.copy(),
                1.2 + index * 0.1,
            )

        self.assertIsNotNone(decision)
        self.assertNotEqual(decision.reason, "identity_verified_target")
        self.assertNotEqual(decision.target_track_id, 50)
        self.assertTrue(manager.resolution_context().active)

    def test_nested_upper_body_box_does_not_start_false_crossing(self):
        encoder = ScriptedSimilarityEncoder()
        manager = SpeakerIdentityManager(encoder, gallery_interval_seconds=10)
        speaker = Detection(
            "person", 0.8, BoundingBox(100, 0, 200, 200), track_id=2
        )
        upper_body_duplicate = Detection(
            "person", 0.5, BoundingBox(103, 1, 198, 125), track_id=8
        )
        other = Detection(
            "person", 0.8, BoundingBox(400, 0, 500, 200), track_id=1
        )
        marker = Detection("marker", 1.0, BoundingBox(130, 50, 150, 80))
        manager.observe(
            HandoffDecision(
                "tracking",
                speaker.box,
                marker.box,
                target_track_id=2,
                reason="confirmed_marker_handoff",
                emit_handoff=True,
            ),
            [speaker, other, marker],
            FRAME.copy(),
            1.0,
        )

        scored = manager.score_detections(
            FRAME.copy(),
            [speaker, upper_body_duplicate, other],
            1.1,
        )

        self.assertFalse(manager.resolution_context().active)
        self.assertNotEqual(scored[0].appearance_similarity, -1.0)
        self.assertNotEqual(scored[1].appearance_similarity, -1.0)

    def _resolving_manager(self, encoder, **kwargs):
        manager = SpeakerIdentityManager(
            encoder,
            gallery_interval_seconds=0,
            resolution_min_frames=3,
            resolution_min_seconds=0.1,
            resolution_search_after_seconds=100.0,
            **kwargs,
        )
        manager.observe(
            HandoffDecision(
                "tracking",
                TARGET.box,
                MARKER.box,
                target_track_id=1,
                reason="confirmed_marker_handoff",
                emit_handoff=True,
            ),
            [TARGET, MARKER],
            FRAME,
            1.0,
        )
        return manager

    def test_decisive_lead_verifies_side_view_candidate_standing_next_to_others(self):
        # After a crossing the speaker reappears under a new raw ID showing a
        # side view (0.50, below the 0.55 absolute threshold) and stays within
        # two box widths of a bystander who scores 0.20.
        encoder = ScriptedSimilarityEncoder()
        encoder.similarities = {7: 0.50, 2: 0.20}
        manager = self._resolving_manager(encoder)
        bystander = Detection("person", 0.9, BoundingBox(150, 0, 250, 200), track_id=2)
        speaker = Detection("person", 0.9, BoundingBox(300, 0, 400, 200), track_id=7)
        crossing = [
            TARGET,
            Detection("person", 0.9, BoundingBox(40, 0, 140, 200), track_id=2),
        ]

        manager.score_detections(FRAME, crossing, 2.0)
        self.assertTrue(manager.resolution_context().active)
        for index in range(6):
            manager.score_detections(FRAME, [bystander, speaker], 2.1 + 0.1 * index)

        self.assertEqual(manager.resolution_context().verified_track_id, 7)

    def test_close_scores_still_need_the_absolute_threshold(self):
        encoder = ScriptedSimilarityEncoder()
        encoder.similarities = {7: 0.50, 2: 0.42}
        manager = self._resolving_manager(encoder)
        bystander = Detection("person", 0.9, BoundingBox(150, 0, 250, 200), track_id=2)
        candidate = Detection("person", 0.9, BoundingBox(300, 0, 400, 200), track_id=7)

        manager.score_detections(FRAME, [bystander], 2.0)
        for index in range(8):
            manager.score_detections(FRAME, [bystander, candidate], 2.1 + 0.1 * index)

        self.assertTrue(manager.resolution_context().active)
        self.assertIsNone(manager.resolution_context().verified_track_id)

    def test_resolution_started_by_track_dropout_can_verify_without_a_close_pair(self):
        # The speaker's raw ID drops out while two people are far apart: no
        # close pair is ever seen, so no separation frames can be owed.
        encoder = ScriptedSimilarityEncoder()
        encoder.similarities = {7: 0.80, 2: 0.20}
        manager = self._resolving_manager(encoder, resolution_strong_margin=1.0)
        bystander = Detection("person", 0.9, BoundingBox(500, 0, 600, 200), track_id=2)
        speaker = Detection("person", 0.9, BoundingBox(10, 0, 110, 200), track_id=7)

        manager.score_detections(FRAME, [TARGET, bystander], 2.0)
        for index in range(6):
            manager.score_detections(FRAME, [speaker, bystander], 2.1 + 0.1 * index)

        self.assertEqual(manager.resolution_context().verified_track_id, 7)

    def test_bystanders_overlapping_each_other_do_not_start_resolution(self):
        encoder = ScriptedSimilarityEncoder()
        manager = self._resolving_manager(encoder)
        first = Detection("person", 0.9, BoundingBox(400, 0, 500, 200), track_id=2)
        second = Detection("person", 0.9, BoundingBox(430, 0, 530, 200), track_id=3)

        manager.score_detections(FRAME, [TARGET, first, second], 2.0)

        self.assertFalse(manager.resolution_context().active)

    def test_overlapping_bystanders_do_not_blind_the_search(self):
        encoder = ScriptedSimilarityEncoder()
        encoder.similarities = {7: 0.70, 2: 0.20, 3: 0.20}
        manager = self._resolving_manager(encoder)
        first = Detection("person", 0.9, BoundingBox(400, 0, 500, 200), track_id=2)
        second = Detection("person", 0.9, BoundingBox(430, 0, 530, 200), track_id=3)
        speaker = Detection("person", 0.9, BoundingBox(10, 0, 110, 200), track_id=7)

        manager.score_detections(FRAME, [first, second], 2.0)
        self.assertTrue(manager.resolution_context().active)
        scored = manager.score_detections(FRAME, [speaker, first, second], 2.1)
        for index in range(6):
            manager.score_detections(FRAME, [speaker, first, second], 2.2 + 0.1 * index)

        similarities = {item.track_id: item.appearance_similarity for item in scored}
        self.assertGreater(similarities[7], 0.5)
        self.assertEqual(similarities[2], -1.0)
        self.assertEqual(similarities[3], -1.0)
        self.assertEqual(manager.resolution_context().verified_track_id, 7)

    def test_clean_candidate_needs_the_absolute_threshold_while_others_are_occluded(self):
        # The speaker may be one of the two overlapping people, so a clean
        # third person at 0.50 has not been shown to beat everyone.
        encoder = ScriptedSimilarityEncoder()
        encoder.similarities = {7: 0.50, 8: 0.20, 2: 0.20, 3: 0.20}
        manager = self._resolving_manager(
            encoder,
            single_resolution_min_similarity=0.45,
        )
        first = Detection("person", 0.9, BoundingBox(400, 0, 500, 200), track_id=2)
        second = Detection("person", 0.9, BoundingBox(430, 0, 530, 200), track_id=3)
        candidate = Detection("person", 0.9, BoundingBox(10, 0, 110, 200), track_id=7)
        other = Detection("person", 0.9, BoundingBox(200, 0, 300, 200), track_id=8)

        manager.score_detections(FRAME, [first, second], 2.0)
        for index in range(8):
            manager.score_detections(FRAME, [candidate, first, second], 2.1 + 0.1 * index)
        self.assertIsNone(manager.resolution_context().verified_track_id)
        for index in range(8):
            manager.score_detections(
                FRAME, [candidate, other, first, second], 3.0 + 0.1 * index
            )

        self.assertIsNone(manager.resolution_context().verified_track_id)

    def test_resolution_expires_after_timeout_without_evidence(self):
        encoder = ScriptedSimilarityEncoder()
        encoder.similarities = {2: 0.20}
        manager = self._resolving_manager(
            encoder,
            resolution_timeout_seconds=5.0,
            coast_max_seconds=0.5,
        )
        bystander = Detection("person", 0.9, BoundingBox(500, 0, 600, 200), track_id=2)

        manager.score_detections(FRAME, [TARGET, bystander], 2.0)
        manager.score_detections(FRAME, [bystander], 2.1)
        self.assertTrue(manager.resolution_context().active)
        manager.score_detections(FRAME, [bystander], 6.0)
        self.assertFalse(manager.resolution_context().expired)
        manager.score_detections(FRAME, [], 9.0)
        self.assertTrue(manager.resolution_context().expired)

        tracker = MarkerTracker(confirmation_seconds=0)
        tracker.update([TARGET, MARKER], 1.0)
        decision = tracker.update(
            [bystander],
            9.0,
            identity_resolution=manager.resolution_context(),
        )
        manager.observe(decision, [bystander], FRAME, 9.0)

        self.assertTrue(decision.emit_release)
        self.assertFalse(manager.resolution_context().active)
        self.assertFalse(manager.resolution_context().expired)

    def test_resolution_does_not_expire_while_the_target_is_visible(self):
        encoder = ScriptedSimilarityEncoder()
        manager = self._resolving_manager(encoder, resolution_timeout_seconds=5.0)
        overlapping = Detection("person", 0.9, BoundingBox(40, 0, 140, 200), track_id=2)

        for index in range(10):
            manager.score_detections(FRAME, [TARGET, overlapping], 2.0 + index)

        self.assertTrue(manager.resolution_context().active)
        self.assertFalse(manager.resolution_context().expired)

    def test_coast_does_not_follow_an_implausible_candidate(self):
        encoder = ScriptedSimilarityEncoder()
        encoder.similarities = {2: 0.20, 3: 0.25}
        manager = self._resolving_manager(encoder, coast_max_seconds=0.3)
        first = Detection("person", 0.9, BoundingBox(300, 0, 400, 200), track_id=2)
        second = Detection("person", 0.9, BoundingBox(500, 0, 600, 200), track_id=3)

        manager.score_detections(FRAME, [first], 2.0)
        for index in range(8):
            manager.score_detections(FRAME, [first, second], 2.1 + 0.1 * index)

        self.assertIsNone(manager.resolution_context().coast_box)


class SimilarityCalibrationTest(unittest.TestCase):
    def test_similarity_is_mapped_linearly_onto_the_threshold_scale(self):
        first = np.asarray([1.0, 0.0], dtype=np.float32)
        second = np.asarray([0.6, 0.8], dtype=np.float32)
        plain = SpeakerIdentityManager(FakeEncoder())
        calibrated = SpeakerIdentityManager(
            FakeEncoder(), similarity_scale=1.31, similarity_offset=-0.537
        )
        self.assertAlmostEqual(plain._similarity(first, second), 0.6, places=5)
        self.assertAlmostEqual(calibrated._similarity(first, second), 0.249, places=5)

    def test_scale_must_be_positive(self):
        with self.assertRaises(ValueError):
            SpeakerIdentityManager(FakeEncoder(), similarity_scale=0.0)


if __name__ == "__main__":
    unittest.main()
