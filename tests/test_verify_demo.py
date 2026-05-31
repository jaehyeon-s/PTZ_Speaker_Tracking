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

from src.app.verify_demo import main, select_registration_bbox
from src.tracking.target_state import MarkerDetection, PersonDetection


class VerifyDemoSmokeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
