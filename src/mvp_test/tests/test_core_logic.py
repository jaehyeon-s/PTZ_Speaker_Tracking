import contextlib
import csv
import io
import time
import unittest

from gesture_detector import GestureFallbackDetector
from main_marker_tracking_demo import update_active_target_for_frame, write_csv_row
from person_detector import NCNNPersonDetector, ExistingProjectPersonDetectorAdapter, detections_to_person_detections
from ptz_controller import PTZCommand, PTZSimulator
from target_selector import TargetSelector, bbox_iou, point_in_bbox
from target_state import ActiveTarget, DemoConfig, MarkerDetection, PersonDetection, TrackingState


class TargetSelectorTests(unittest.TestCase):
    def test_marker_selects_person_containing_marker_center(self):
        selector = TargetSelector(DemoConfig())
        people = [
            PersonDetection((0, 0, 80, 120)),
            PersonDetection((100, 0, 80, 120)),
        ]
        markers = [
            MarkerDetection(
                marker_id=7,
                center=(130, 50),
                corners=((125, 45), (135, 45), (135, 55), (125, 55)),
            )
        ]

        target = selector.acquire_from_marker(people, markers)

        self.assertIsNotNone(target)
        self.assertEqual(target.bbox, people[1].bbox)
        self.assertEqual(target.marker_id, 7)

    def test_target_marker_id_selects_only_matching_marker(self):
        selector = TargetSelector(DemoConfig(target_marker_id=7))
        people = [PersonDetection((0, 0, 80, 120)), PersonDetection((100, 0, 80, 120))]
        markers = [
            MarkerDetection(3, (30, 50), ((25, 45), (35, 45), (35, 55), (25, 55))),
            MarkerDetection(7, (130, 50), ((125, 45), (135, 45), (135, 55), (125, 55))),
        ]

        target = selector.acquire_from_marker(people, markers)

        self.assertIsNotNone(target)
        self.assertEqual(target.bbox, people[1].bbox)
        self.assertEqual(target.marker_id, 7)

    def test_target_marker_id_ignores_other_marker_ids(self):
        selector = TargetSelector(DemoConfig(target_marker_id=7))
        people = [PersonDetection((0, 0, 80, 120))]
        markers = [MarkerDetection(3, (30, 50), ((25, 45), (35, 45), (35, 55), (25, 55)))]

        target = selector.acquire_from_marker(people, markers)

        self.assertIsNone(target)

    def test_existing_target_uses_best_iou_and_does_not_switch_to_far_person(self):
        selector = TargetSelector(DemoConfig(tracker_iou_threshold=0.1, tracker_center_threshold_ratio=0.1))
        target = selector.acquire_from_gesture(PersonDetection((10, 10, 100, 160)))
        people = [
            PersonDetection((400, 10, 100, 160)),
            PersonDetection((18, 14, 100, 160)),
        ]

        updated = selector.update_existing(target, people, (480, 640, 3))

        self.assertIsNotNone(updated)
        self.assertEqual(updated.bbox, people[1].bbox)
        self.assertEqual(updated.target_id, target.target_id)

    def test_no_match_returns_none_for_suspended_state(self):
        selector = TargetSelector(DemoConfig(tracker_iou_threshold=0.4, tracker_center_threshold_ratio=0.05))
        target = selector.acquire_from_gesture(PersonDetection((10, 10, 100, 160)))

        updated = selector.update_existing(target, [PersonDetection((500, 300, 80, 120))], (480, 640, 3))

        self.assertIsNone(updated)

    def test_geometry_helpers(self):
        self.assertTrue(point_in_bbox((20, 20), (10, 10, 40, 40)))
        self.assertFalse(point_in_bbox((60, 20), (10, 10, 40, 40)))
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_marker_only_missing_marker_does_not_fallback_to_other_person(self):
        config = DemoConfig(marker_only=True, max_suspended_frames=3)
        selector = TargetSelector(config)
        target = ActiveTarget(target_id=1, bbox=(10, 10, 100, 160), marker_id=7)
        people = [PersonDetection((12, 12, 100, 160)), PersonDetection((300, 10, 100, 160))]

        updated, state = update_active_target_for_frame(
            selector,
            config,
            target,
            TrackingState.ACTIVE,
            people,
            [],
            (480, 640, 3),
            marker_only=True,
        )

        self.assertEqual(state, TrackingState.SUSPENDED)
        self.assertEqual(updated.bbox, (10, 10, 100, 160))
        self.assertEqual(updated.missed_frames, 1)

    def test_target_resets_after_max_suspended_frames(self):
        config = DemoConfig(max_suspended_frames=1, tracker_iou_threshold=0.8, tracker_center_threshold_ratio=0.01)
        selector = TargetSelector(config)
        target = ActiveTarget(target_id=1, bbox=(10, 10, 100, 160), marker_id=7, missed_frames=1)

        updated, state = update_active_target_for_frame(
            selector,
            config,
            target,
            TrackingState.SUSPENDED,
            [PersonDetection((500, 300, 80, 120))],
            [],
            (480, 640, 3),
            marker_only=False,
        )

        self.assertIsNone(updated)
        self.assertEqual(state, TrackingState.IDLE)


class PTZSimulatorTests(unittest.TestCase):
    def test_ptz_command_uses_bbox_center_and_dead_zone(self):
        ptz = PTZSimulator(dead_zone_ratio=0.1)

        self.assertEqual(ptz.command_for_bbox((300, 220, 40, 40), (480, 640, 3)).as_text(), "STOP")
        self.assertEqual(ptz.command_for_bbox((10, 220, 40, 40), (480, 640, 3)).as_text(), "LEFT")
        self.assertEqual(ptz.command_for_bbox((590, 220, 40, 40), (480, 640, 3)).as_text(), "RIGHT")
        self.assertEqual(ptz.command_for_bbox((300, 10, 40, 40), (480, 640, 3)).as_text(), "UP")
        self.assertEqual(ptz.command_for_bbox((300, 430, 40, 40), (480, 640, 3)).as_text(), "DOWN")


class GestureFallbackTests(unittest.TestCase):
    def test_requires_exactly_one_raised_hand_for_hold_time(self):
        detector = GestureFallbackDetector(hold_seconds=0.01)
        people = [PersonDetection((0, 0, 100, 100)), PersonDetection((120, 0, 100, 100))]

        self.assertIsNone(detector.update(people, {0, 1}))
        self.assertIsNone(detector.update(people, {1}))
        time.sleep(0.02)
        self.assertEqual(detector.update(people, {1}), people[1])


class ExistingDetectorAdapterTests(unittest.TestCase):
    def test_converts_bytetrack_tlwh_objects(self):
        class Track:
            def __init__(self, tlwh, score):
                self.tlwh = tlwh
                self.score = score

        people = detections_to_person_detections([Track((10.2, 20.4, 30.0, 40.0), 0.87)])

        self.assertEqual(people, [PersonDetection((10, 20, 30, 40), 0.87)])

    def test_converts_yolo_xyxy_rows_and_filters_non_person_classes(self):
        rows = [
            [10, 20, 50, 80, 0.91, 0],
            [100, 120, 160, 200, 0.99, 2],
        ]

        people = detections_to_person_detections(rows, bbox_format="xyxy")

        self.assertEqual(people, [PersonDetection((10, 20, 40, 60), 0.91)])

    def test_adapter_wraps_existing_detector_without_rewriting_it(self):
        class ExistingDetector:
            def detect(self, frame):
                return [{"xyxy": (1, 2, 11, 22), "confidence": 0.75, "class_id": 0}]

        adapter = ExistingProjectPersonDetectorAdapter(ExistingDetector(), bbox_format="xyxy")

        self.assertEqual(adapter.detect(None), [PersonDetection((1, 2, 10, 20), 0.75)])


class NCNNDetectorDebugTests(unittest.TestCase):
    def test_clamps_out_of_frame_detections(self):
        detector = object.__new__(NCNNPersonDetector)
        detector.debug_detector = False

        detections = detector._clamp_detections(
            [PersonDetection((-10, -5, 80, 90), 0.9)],
            (50, 60, 3),
        )

        self.assertEqual(detections, [PersonDetection((0, 0, 60, 50), 0.9)])

    def test_debug_frame_prints_detection_summary(self):
        detector = object.__new__(NCNNPersonDetector)
        detector.debug_detector = True
        detector.debug_frames = 1
        detector._debug_frame_count = 0
        detector.input_name = "images"
        detector.output_names = ("out0",)
        detector._last_output_shapes = {"out0": (1, 6)}

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            detector._debug_frame([[1, 2, 3, 4, 0.9, 0]], [PersonDetection((1, 2, 3, 4), 0.9)])

        self.assertEqual(detector._debug_frame_count, 1)
        self.assertIn("detections_after_parser=1", output.getvalue())


class DebugLoggingTests(unittest.TestCase):
    def test_write_csv_row_records_target_and_ptz_fields(self):
        output = io.StringIO()
        writer = csv.writer(output)
        target = ActiveTarget(target_id=3, bbox=(10, 20, 30, 40), marker_id=7)
        command = PTZCommand(pan="RIGHT", tilt="UP", error_x=0.25, error_y=-0.3)

        write_csv_row(writer, TrackingState.ACTIVE, target, command)

        row = next(csv.reader(io.StringIO(output.getvalue())))
        self.assertEqual(row[1:], ["ACTIVE", "3", "7", "10,20,30,40", "RIGHT+UP", "0.2500", "-0.3000"])


if __name__ == "__main__":
    unittest.main()
