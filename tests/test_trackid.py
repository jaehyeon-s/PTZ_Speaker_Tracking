import unittest

import numpy as np

from src.camera.region_provider import CenterCropRegionProvider, IdentityMatchedRegionProvider
from src.reid.appearance import TrackIdReIdentifier
from src.tracking.target_state import PersonDetection
from src.vision.models import TrackingState


class FakeTracker:
    """Stand-in for UltralyticsYoloTracker with controllable detections."""

    def __init__(self):
        self.people: list[PersonDetection] = []

    def set_people(self, people):
        self.people = people

    def detect(self, frame):
        return list(self.people)

    @property
    def last_detections(self):
        return self.people

    def track_id_for_bbox(self, bbox_xyxy, min_iou=0.3):
        x1, y1, x2, y2 = bbox_xyxy
        query = (x1, y1, x2 - x1, y2 - y1)
        for detection in self.people:
            if detection.bbox == query and detection.track_id is not None:
                return detection.track_id
        return None


def build_stack(tracker, hold_limit=3):
    matcher = TrackIdReIdentifier(tracker)
    provider = IdentityMatchedRegionProvider(
        tracker,
        matcher,
        CenterCropRegionProvider(),
        weak_min_score=0.5,
        identity_margin=0.1,
        hold_limit=hold_limit,
        detect_during_bootstrap=True,
        enable_center_mistrack=False,
    )
    return matcher, provider


class TrackIdReIdentifierTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)
        self.tracker = FakeTracker()

    def test_registers_track_id_from_bbox(self):
        self.tracker.set_people([PersonDetection((10, 10, 20, 40), 0.9, track_id=7)])
        matcher = TrackIdReIdentifier(self.tracker)
        self.assertTrue(matcher.register(self.frame, (10, 10, 30, 50)))
        self.assertEqual(matcher.reference, 7)
        self.assertEqual(matcher.score(self.frame, (10, 10, 30, 50)), 1.0)

    def test_register_fails_when_no_track_overlaps(self):
        self.tracker.set_people([PersonDetection((10, 10, 20, 40), 0.9, track_id=None)])
        matcher = TrackIdReIdentifier(self.tracker)
        self.assertFalse(matcher.register(self.frame, (10, 10, 30, 50)))
        self.assertIsNone(matcher.reference)

    def test_provider_follows_registered_id_and_ignores_passerby(self):
        target = PersonDetection((10, 10, 20, 40), 0.9, track_id=7)
        passerby = PersonDetection((150, 10, 20, 40), 0.9, track_id=9)
        self.tracker.set_people([target, passerby])
        matcher, provider = build_stack(self.tracker)

        # bootstrap (no reference) returns the center crop, not a person box.
        bbox, _ = provider.region_for_frame(1, self.frame)
        self.assertEqual(provider.last_observation.state, TrackingState.UNREGISTERED)

        self.assertTrue(matcher.register(self.frame, (10, 10, 30, 50)))

        # Passer-by now stands dead center; target stays off to the side.
        passerby_center = PersonDetection((90, 30, 20, 40), 0.9, track_id=9)
        self.tracker.set_people([target, passerby_center])
        bbox, _ = provider.region_for_frame(2, self.frame)
        self.assertEqual(provider.last_observation.state, TrackingState.CAMERA_ALIGNED)
        self.assertEqual(bbox, (10, 10, 30, 50))  # the registered id, not the centered passer-by

    def test_provider_holds_then_loses_when_id_disappears(self):
        target = PersonDetection((10, 10, 20, 40), 0.9, track_id=7)
        self.tracker.set_people([target])
        matcher, provider = build_stack(self.tracker, hold_limit=3)
        matcher.register(self.frame, (10, 10, 30, 50))
        provider.region_for_frame(1, self.frame)

        # Target id gone, only a stranger remains.
        self.tracker.set_people([PersonDetection((150, 10, 20, 40), 0.9, track_id=9)])
        provider.region_for_frame(2, self.frame)
        self.assertEqual(provider.last_observation.state, TrackingState.HOLD)
        provider.region_for_frame(3, self.frame)
        bbox, _ = provider.region_for_frame(4, self.frame)
        self.assertEqual(provider.last_observation.state, TrackingState.LOST)
        self.assertIsNone(bbox)


if __name__ == "__main__":
    unittest.main()
