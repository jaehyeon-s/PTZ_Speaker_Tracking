import json
import tempfile
import unittest
from pathlib import Path

from src.camera.region_provider import CenterCropRegionProvider, IdentityMatchedRegionProvider, JsonlRegionProvider
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


if __name__ == "__main__":
    unittest.main()
