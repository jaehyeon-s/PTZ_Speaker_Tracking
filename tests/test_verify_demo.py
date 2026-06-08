import contextlib
import csv
import io
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.app.verify_demo import (
    CameraTrackingSupervisor,
    build_ptz_controller,
    config_value,
    load_config,
    main,
    marker_visible_in_observed_region,
    select_registration_bbox,
    should_collect_markers,
    should_collect_registration_inputs,
)
from src.tracking.target_state import MarkerDetection, PersonDetection


class VerifyDemoSmokeTests(unittest.TestCase):
    def test_loads_nested_yaml_config_without_pyyaml_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "\n".join(
                    [
                        "source: rtsp://192.168.11.88:554/stream2",
                        "region_mode: identity",
                        "detector:",
                        "  backend: ncnn",
                        "  ncnn_input_size: 640",
                        "camera:",
                        "  configure_supervisor_actuator: true",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(str(path))

        self.assertEqual(config_value(config, "source"), "rtsp://192.168.11.88:554/stream2")
        self.assertEqual(config_value(config, "detector.backend"), "ncnn")
        self.assertEqual(config_value(config, "detector.ncnn_input_size"), 640)
        self.assertIs(config_value(config, "camera.configure_supervisor_actuator"), True)

    def test_marker_registration_defaults_to_observed_reference(self):
        args = SimpleNamespace(
            registration_mode="marker",
            registration_reference="observed",
            target_marker_id=7,
        )
        observed = (40, 10, 120, 110)
        people = [PersonDetection((10, 10, 40, 80))]
        markers = [MarkerDetection(7, (30, 40), ((25, 35), (35, 35), (35, 45), (25, 45)))]

        bbox = select_registration_bbox(args, None, observed, people, markers, None, None)

        self.assertEqual(bbox, observed)

    def test_marker_registration_can_use_selected_person_reference(self):
        args = SimpleNamespace(
            registration_mode="marker",
            registration_reference="selected",
            target_marker_id=7,
        )
        observed = (40, 10, 120, 110)
        people = [PersonDetection((10, 10, 40, 80))]
        markers = [MarkerDetection(7, (30, 40), ((25, 35), (35, 35), (35, 45), (25, 45)))]

        bbox = select_registration_bbox(args, None, observed, people, markers, None, None)

        self.assertEqual(bbox, (10, 10, 50, 90))

    def test_center_region_verification_writes_verified_log_without_yolo(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            video_path = directory_path / "input.mp4"
            log_path = directory_path / "verify.csv"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
            )
            for _ in range(3):
                frame = np.zeros((120, 160, 3), dtype=np.uint8)
                frame[10:110, 45:115] = (0, 255, 0)
                writer.write(frame)
            writer.release()

            argv = [
                "main.py",
                "--source",
                str(video_path),
                "--register-on-start",
                "--verify-every-frames",
                "1",
                "--no-window",
                "--log-csv",
                str(log_path),
            ]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                result = main()

            with log_path.open(newline="", encoding="utf-8") as log_file:
                rows = list(csv.DictReader(log_file))

        self.assertEqual(result, 0)
        self.assertEqual(rows[0]["event"], "PRESENTER_REGISTERED")
        self.assertEqual(rows[1]["state"], "VERIFIED")
        self.assertEqual(rows[1]["region_source"], "center_crop")

    def test_marker_positive_requires_marker_inside_observed_region(self):
        marker = MarkerDetection(7, (50, 40), ((45, 35), (55, 35), (55, 45), (45, 45)))

        self.assertTrue(marker_visible_in_observed_region([marker], (20, 20, 80, 80), 7))
        self.assertFalse(marker_visible_in_observed_region([marker], (90, 20, 140, 80), 7))
        self.assertFalse(marker_visible_in_observed_region([marker], (20, 20, 80, 80), 8))

    def test_supervisor_disables_tracking_after_confirmed_mismatch(self):
        class FakeControl:
            def __init__(self):
                self.calls = []

            def enable_auto_tracking(self):
                self.calls.append("tracking")

            def set_presenter_mode(self):
                self.calls.append("presenter")

            def enable_zone_tracking(self):
                self.calls.append("zone")

            def stop(self):
                self.calls.append("stop")

        control = FakeControl()
        supervisor = CameraTrackingSupervisor(control, "stop")

        supervisor.enable_tracking("registered")
        supervisor.confirm_mismatch()
        supervisor.confirm_mismatch()

        self.assertEqual(control.calls, ["presenter", "tracking", "zone", "stop"])
        self.assertEqual(supervisor.last_action, "tracking_off:mismatch:stop")

    def test_supervisor_actuator_mode_keeps_camera_tracking_off_after_registration(self):
        class FakeControl:
            def __init__(self):
                self.calls = []

            def set_supervisor_actuator_mode(self):
                self.calls.append("actuator")

            def enable_auto_tracking(self):
                self.calls.append("tracking")

            def set_presenter_mode(self):
                self.calls.append("presenter")

        control = FakeControl()
        supervisor = CameraTrackingSupervisor(control, "stop")

        supervisor.configure_actuator()
        supervisor.enable_tracking("registered")

        self.assertEqual(control.calls, ["actuator"])
        self.assertEqual(supervisor.last_action, "tracking_left_off:registered")

    def test_ptz_follow_target_accepts_supervisor_actuator_camera_control(self):
        args = SimpleNamespace(
            ptz_follow_target=True,
            ptz_dead_zone=0.12,
            ptz_min_speed=2,
            ptz_max_speed=10,
        )

        controller = build_ptz_controller(args, object())

        self.assertIsNotNone(controller)

    def test_registration_detector_stops_after_headless_registration(self):
        args = SimpleNamespace(register_on_start=True, no_window=True, marker_positive=False)

        self.assertTrue(should_collect_registration_inputs(args, verifier_registered=False))
        self.assertFalse(should_collect_registration_inputs(args, verifier_registered=True))
        self.assertFalse(should_collect_markers(args, verifier_registered=True, should_verify=True))

    def test_marker_positive_collects_markers_only_on_verify_frames(self):
        args = SimpleNamespace(register_on_start=True, no_window=True, marker_positive=True)

        self.assertFalse(should_collect_markers(args, verifier_registered=True, should_verify=False))
        self.assertTrue(should_collect_markers(args, verifier_registered=True, should_verify=True))


if __name__ == "__main__":
    unittest.main()
