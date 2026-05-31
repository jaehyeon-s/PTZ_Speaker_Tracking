import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.app.verify_demo import main


class VerifyDemoSmokeTests(unittest.TestCase):
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
