import json
import tempfile
import unittest
from pathlib import Path

from src.camera.region_provider import CenterCropRegionProvider, IdentityMatchedRegionProvider, JsonlRegionProvider
from src.reid.appearance import AppearanceGuard
from src.tracking.target_state import PersonDetection
from src.vision.models import TrackingState


class Frame:
    shape = (100, 200, 3)


class RegionProviderTests(unittest.TestCase):
    def test_center_crop_is_available_without_camera_metadata_api(self):
        provider = CenterCropRegionProvider(width_ratio=0.5, height_ratio=0.8)

        bbox, source = provider.region_for_frame(1, Frame())

        self.assertEqual(bbox, (50, 10, 150, 90))
        self.assertEqual(source, "center_crop")

    def test_jsonl_provider_supplies_captured_camera_bbox(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bbox.jsonl"
            path.write_text(json.dumps({"frame": 3, "bbox": [10, 20, 80, 95]}) + "\n", encoding="utf-8")
            provider = JsonlRegionProvider(str(path))

            bbox, source = provider.region_for_frame(3, Frame())
            missing, _ = provider.region_for_frame(4, Frame())

        self.assertEqual(bbox, (10, 20, 80, 95))
        self.assertEqual(source, "metadata_jsonl")
        self.assertIsNone(missing)

    def test_identity_provider_confirms_mistrack_with_relative_center_margin(self):
        frame = Frame()
        target = PersonDetection((10, 20, 20, 40))
        center = PersonDetection((90, 20, 20, 40))
        detector = FakeDetector([target, center])
        matcher = FakeMatcher({(10, 20, 30, 60): 0.70, (90, 20, 110, 60): 0.35})
        provider = IdentityMatchedRegionProvider(
            detector,
            matcher,
            weak_min_score=0.20,
            identity_margin=0.05,
            center_margin=0.20,
            center_dead_zone_ratio=0.10,
            mismatch_limit=2,
        )

        bbox, source = provider.region_for_frame(1, frame)
        self.assertEqual(bbox, (10, 20, 30, 60))
        self.assertEqual(source, "identity_mistrack")
        self.assertEqual(provider.last_observation.state, TrackingState.SUSPECT)

        provider.region_for_frame(2, frame)
        self.assertEqual(provider.last_observation.state, TrackingState.MISMATCH)
        self.assertEqual(provider.last_observation.event, "PRESENTER_MISTRACK_CONFIRMED")

    def test_identity_provider_holds_unreliable_frames_before_lost(self):
        frame = Frame()
        people = [PersonDetection((10, 20, 20, 40)), PersonDetection((90, 20, 20, 40))]
        detector = FakeDetector(people)
        matcher = FakeMatcher({(10, 20, 30, 60): 0.24, (90, 20, 110, 60): 0.23})
        provider = IdentityMatchedRegionProvider(
            detector,
            matcher,
            weak_min_score=0.25,
            identity_margin=0.05,
            hold_limit=2,
        )

        bbox, source = provider.region_for_frame(1, frame)
        self.assertEqual(bbox, (10, 20, 30, 60))
        self.assertEqual(source, "identity_hold")
        self.assertEqual(provider.last_observation.state, TrackingState.HOLD)

        bbox, source = provider.region_for_frame(2, frame)
        self.assertIsNone(bbox)
        self.assertEqual(source, "identity_lost")
        self.assertEqual(provider.last_observation.state, TrackingState.LOST)

    def test_identity_provider_reset_clears_lost_hold_state_after_reregistration(self):
        frame = Frame()
        people = [PersonDetection((10, 20, 20, 40))]
        detector = FakeDetector(people)
        matcher = FakeMatcher({(10, 20, 30, 60): 0.20})
        provider = IdentityMatchedRegionProvider(
            detector,
            matcher,
            weak_min_score=0.25,
            identity_margin=0.05,
            hold_limit=2,
        )

        provider.region_for_frame(1, frame)
        provider.region_for_frame(2, frame)
        self.assertEqual(provider.last_observation.state, TrackingState.LOST)

        provider.reset_identity_state((10, 20, 30, 60))
        bbox, source = provider.region_for_frame(3, frame)

        self.assertEqual(bbox, (10, 20, 30, 60))
        self.assertEqual(source, "identity_hold")
        self.assertEqual(provider.last_observation.state, TrackingState.HOLD)
        self.assertEqual(provider.last_observation.hold_count, 1)

    def test_identity_provider_resists_target_steal_within_switch_margin(self):
        frame = Frame()
        target = PersonDetection((10, 20, 20, 40))
        stealer = PersonDetection((40, 20, 20, 40))
        detector = FakeDetector([target, stealer])
        matcher = FakeMatcher({(10, 20, 30, 60): 0.50, (40, 20, 60, 60): 0.60})
        provider = IdentityMatchedRegionProvider(
            detector, matcher, weak_min_score=0.25, identity_margin=0.05, switch_margin=0.15
        )
        provider.last_trusted_bbox = (10, 20, 30, 60)

        bbox, _ = provider.region_for_frame(1, frame)

        self.assertEqual(bbox, (10, 20, 30, 60))

    def test_identity_provider_switches_when_other_beats_switch_margin(self):
        frame = Frame()
        target = PersonDetection((10, 20, 20, 40))
        stealer = PersonDetection((40, 20, 20, 40))
        detector = FakeDetector([target, stealer])
        matcher = FakeMatcher({(10, 20, 30, 60): 0.40, (40, 20, 60, 60): 0.70})
        provider = IdentityMatchedRegionProvider(
            detector, matcher, weak_min_score=0.25, identity_margin=0.05, switch_margin=0.15
        )
        provider.last_trusted_bbox = (10, 20, 30, 60)

        bbox, _ = provider.region_for_frame(1, frame)

        self.assertEqual(bbox, (40, 20, 60, 60))

    def test_identity_provider_clears_last_people_before_registration(self):
        frame = Frame()
        detector = FakeDetector([PersonDetection((10, 20, 20, 40))])
        matcher = FakeMatcher({})
        matcher.reference = None
        provider = IdentityMatchedRegionProvider(detector, matcher)
        provider.last_people = [PersonDetection((10, 20, 20, 40))]

        provider.region_for_frame(1, frame)

        self.assertEqual(provider.last_people, [])

    def test_identity_provider_forces_lost_when_appearance_drift_confirmed_despite_id_score(self):
        """A trackid-style matcher that always scores 1.0 cannot notice an ID
        switch on its own; the appearance guard must force LOST once the
        tracked crop's appearance drifts for enough consecutive periodic checks."""
        frame = Frame()
        target = PersonDetection((10, 20, 20, 40))
        detector = FakeDetector([target])
        matcher = FakeMatcher({(10, 20, 30, 60): 1.0})
        appearance_guard = AppearanceGuard(
            AlwaysLowAppearanceMatcher(threshold=0.5),
            verify_every_frames=1,
            mismatch_limit=2,
        )
        provider = IdentityMatchedRegionProvider(
            detector,
            matcher,
            weak_min_score=0.25,
            identity_margin=0.05,
            enable_center_mistrack=False,
            appearance_guard=appearance_guard,
        )

        bbox, _ = provider.region_for_frame(1, frame)
        self.assertEqual(bbox, (10, 20, 30, 60))
        self.assertEqual(provider.last_observation.state, TrackingState.CAMERA_ALIGNED)

        bbox, source = provider.region_for_frame(2, frame)
        self.assertIsNone(bbox)
        self.assertEqual(source, "identity_lost")
        self.assertEqual(provider.last_observation.state, TrackingState.LOST)
        self.assertEqual(provider.last_observation.event, "APPEARANCE_DRIFT_CONFIRMED")

    def test_identity_provider_stays_aligned_when_appearance_guard_agrees(self):
        frame = Frame()
        target = PersonDetection((10, 20, 20, 40))
        detector = FakeDetector([target])
        matcher = FakeMatcher({(10, 20, 30, 60): 1.0})
        appearance_guard = AppearanceGuard(
            AlwaysHighAppearanceMatcher(threshold=0.5),
            verify_every_frames=1,
            mismatch_limit=2,
        )
        provider = IdentityMatchedRegionProvider(
            detector,
            matcher,
            weak_min_score=0.25,
            identity_margin=0.05,
            enable_center_mistrack=False,
            appearance_guard=appearance_guard,
        )

        bbox, _ = provider.region_for_frame(1, frame)

        self.assertEqual(bbox, (10, 20, 30, 60))
        self.assertEqual(provider.last_observation.state, TrackingState.CAMERA_ALIGNED)

    def test_reset_identity_state_captures_appearance_baseline_on_reregistration(self):
        frame = Frame()
        detector = FakeDetector([])
        matcher = FakeMatcher({})
        appearance_guard = AppearanceGuard(RecordingAppearanceMatcher(), verify_every_frames=1, mismatch_limit=1)
        provider = IdentityMatchedRegionProvider(detector, matcher, appearance_guard=appearance_guard)

        provider.reset_identity_state((1, 2, 3, 4), frame)

        self.assertEqual(appearance_guard.appearance_matcher.registered_bbox, (1, 2, 3, 4))

    def test_reset_identity_state_clears_appearance_guard_on_full_reset(self):
        detector = FakeDetector([])
        matcher = FakeMatcher({})
        recording = RecordingAppearanceMatcher()
        appearance_guard = AppearanceGuard(recording, verify_every_frames=1, mismatch_limit=1)
        provider = IdentityMatchedRegionProvider(detector, matcher, appearance_guard=appearance_guard)

        provider.reset_identity_state(None)

        self.assertTrue(recording.reset_called)


class FakeDetector:
    def __init__(self, people):
        self.people = people

    def detect(self, frame):
        del frame
        return self.people


class FakeMatcher:
    reference = object()

    def __init__(self, scores):
        self.scores = scores

    def score(self, frame, bbox):
        del frame
        return self.scores[bbox]


class AlwaysLowAppearanceMatcher:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def register(self, frame, bbox):
        del frame, bbox

    def score(self, frame, bbox):
        del frame, bbox
        return 0.0

    def reset(self):
        pass


class AlwaysHighAppearanceMatcher:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def register(self, frame, bbox):
        del frame, bbox

    def score(self, frame, bbox):
        del frame, bbox
        return 1.0

    def reset(self):
        pass


class RecordingAppearanceMatcher:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.registered_bbox = None
        self.reset_called = False

    def register(self, frame, bbox):
        del frame
        self.registered_bbox = bbox

    def score(self, frame, bbox):
        del frame, bbox
        return 1.0

    def reset(self):
        self.reset_called = True


if __name__ == "__main__":
    unittest.main()
