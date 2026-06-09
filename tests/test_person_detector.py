import unittest

import numpy as np

from src.tracking.person_detector import _parse_yolo_ncnn_output
from src.tracking.target_state import PersonDetection


class YoloNCNNOutputParserTests(unittest.TestCase):
    def test_parses_channel_first_yolo_output(self):
        output = np.array(
            [
                [320.0, 100.0, 120.0, 140.0, 160.0, 180.0],
                [320.0, 100.0, 120.0, 140.0, 160.0, 180.0],
                [160.0, 40.0, 40.0, 40.0, 40.0, 40.0],
                [200.0, 80.0, 80.0, 80.0, 80.0, 80.0],
                [0.90, 0.10, 0.10, 0.10, 0.10, 0.10],
            ],
            dtype=np.float32,
        )

        detections = _parse_yolo_ncnn_output(output, (480, 640, 3), 640, 0.35)

        self.assertEqual(detections, [PersonDetection((240, 165, 160, 150), 0.8999999761581421)])

    def test_parses_anchor_first_yolo_output_and_filters_low_confidence(self):
        output = np.array(
            [
                [320.0, 320.0, 160.0, 200.0, 0.90],
                [100.0, 100.0, 40.0, 80.0, 0.10],
            ],
            dtype=np.float32,
        )

        detections = _parse_yolo_ncnn_output(output, (480, 640, 3), 640, 0.35)

        self.assertEqual(detections, [PersonDetection((240, 165, 160, 150), 0.8999999761581421)])


if __name__ == "__main__":
    unittest.main()
