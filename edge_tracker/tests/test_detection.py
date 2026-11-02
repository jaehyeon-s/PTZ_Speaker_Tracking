from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from marker_tracker.core import BoundingBox, Detection
from marker_tracker.detection import (
    DecoupledPersonSource,
    OnnxAppearanceEncoder,
    TrackerStage,
    hailo_nms_person_boxes,
    input_boxes_to_detections,
    letterbox_rgb,
    reid_crop_rgb,
    yolo26_raw_person_boxes,
)


class FakeDetector:
    def __init__(self, frames):
        self._frames = iter(frames)
        self.confidences = []

    def detect(self, frame, confidence):
        self.confidences.append(confidence)
        return next(self._frames)


def person(x, confidence=0.9):
    return Detection("person", confidence, BoundingBox(x, 100.0, x + 60.0, 260.0))


def args(**overrides):
    values = {"confidence": None, "person_confidence": 0.35, "imgsz": None, "inference_device": "cpu"}
    values.update(overrides)
    return Namespace(**values)


class DecoupledPersonSourceTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((360, 640, 3), dtype=np.uint8)

    def test_external_detections_get_stable_track_ids(self):
        detector = FakeDetector([[person(100.0), person(400.0)], [person(104.0), person(396.0)],
                                 [person(108.0), person(392.0)]])
        source = DecoupledPersonSource(args(), detector=detector)
        ids = []
        for _ in range(3):
            result, detections = source.update(self.frame, None, 0.1, "bytetrack-pi.yaml")
            self.assertIsNone(result)
            ids.append({d.track_id: round(d.box.x1) for d in detections})
        self.assertEqual(len(ids[0]), 2)
        self.assertNotIn(None, ids[0])
        self.assertEqual(set(ids[0]), set(ids[2]))
        left = min(ids[0], key=ids[0].get)
        self.assertLess(ids[2][left], 320)
        self.assertEqual(detector.confidences, [0.1, 0.1, 0.1])
        self.assertEqual(set(source.last_timings), {"person_detect", "person_track"})

    def test_low_confidence_tracks_are_not_reported_as_people(self):
        detector = FakeDetector([[person(100.0), person(400.0, confidence=0.3)]])
        source = DecoupledPersonSource(args(), detector=detector)
        _, detections = source.update(self.frame, None, 0.1, "bytetrack-pi.yaml")
        self.assertEqual(len(detections), 1)
        self.assertLess(detections[0].box.x1, 320)

    def test_weak_box_of_the_target_is_kept_only_with_target_low_confidence(self):
        frames = [[person(100.0), person(400.0, confidence=0.3)]]
        hint = (None, BoundingBox(400.0, 100.0, 460.0, 260.0))
        default = DecoupledPersonSource(args(), detector=FakeDetector(frames))
        default.target_hint = hint
        enabled = DecoupledPersonSource(
            args(target_low_confidence=0.25), detector=FakeDetector(frames)
        )
        enabled.target_hint = hint
        far = DecoupledPersonSource(args(target_low_confidence=0.25), detector=FakeDetector(frames))
        far.target_hint = (None, BoundingBox(100.0, 100.0, 160.0, 260.0))

        self.assertEqual(len(default.update(self.frame, None, 0.1, "bytetrack-pi.yaml")[1]), 1)
        kept = enabled.update(self.frame, None, 0.1, "bytetrack-pi.yaml")[1]
        self.assertEqual(sorted(round(d.box.x1) for d in kept), [100, 400])
        self.assertEqual(len(far.update(self.frame, None, 0.1, "bytetrack-pi.yaml")[1]), 1)

    def test_tracked_boxes_are_clipped_to_the_frame(self):
        edge = Detection("person", 0.9, BoundingBox(600.0, 100.0, 640.0, 260.0))
        moved = Detection("person", 0.9, BoundingBox(615.0, 100.0, 655.0, 260.0))
        source = DecoupledPersonSource(args(), detector=FakeDetector([[edge], [moved]]))
        source.update(self.frame, None, 0.1, "bytetrack-pi.yaml")
        _, detections = source.update(self.frame, None, 0.1, "bytetrack-pi.yaml")
        self.assertEqual(len(detections), 1)
        self.assertLessEqual(detections[0].box.x2, 640.0)

    def test_empty_frame_returns_no_detections(self):
        source = DecoupledPersonSource(args(), detector=FakeDetector([[]]))
        self.assertEqual(source.update(self.frame, None, 0.1, "botsort-pi.yaml")[1], [])

    def test_tracker_level_reid_is_rejected(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "with-reid.yaml"
            config.write_text(Path("botsort-pi.yaml").read_text().replace("with_reid: False", "with_reid: True"))
            with self.assertRaises(ValueError):
                TrackerStage(str(config))


class HailoDetectionMappingTest(unittest.TestCase):
    def test_letterbox_pads_and_converts_to_rgb(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # blue in BGR
        image, scale, pad_x, pad_y = letterbox_rgb(frame, 640, 640)
        self.assertEqual(image.shape, (640, 640, 3))
        self.assertEqual((scale, pad_x, pad_y), (1.0, 0, 140))
        self.assertEqual(image[0, 0].tolist(), [114, 114, 114])
        self.assertEqual(image[320, 320].tolist(), [0, 0, 255])

    def test_nms_rows_map_back_to_frame_pixels(self):
        # 1280x720 frame in a 640x640 input: scale 0.5, 140 px padding top and bottom.
        rows = np.array(
            [[0.5, 0.25, 0.75, 0.5, 0.9], [0.5, 0.25, 0.75, 0.5, 0.05]], dtype=np.float32
        )
        boxes = hailo_nms_person_boxes(rows, (640, 640))
        detections = input_boxes_to_detections(boxes, 0.1, (1280, 720), 0.5, (0, 140))
        self.assertEqual(len(detections), 1)
        box = detections[0].box
        self.assertEqual((box.x1, box.y1, box.x2, box.y2), (320.0, 360.0, 640.0, 680.0))
        self.assertIsNone(detections[0].track_id)
        self.assertAlmostEqual(detections[0].confidence, 0.9, places=5)
        for value in (box.x1, box.y1, box.x2, box.y2, detections[0].confidence):
            self.assertIs(type(value), float)

    def test_empty_nms_output_and_boxes_in_padding_give_no_detections(self):
        empty = hailo_nms_person_boxes(np.zeros((0, 5)), (640, 640))
        self.assertEqual(input_boxes_to_detections(empty, 0.1, (1280, 720), 0.5, (0, 140)), [])
        in_padding = hailo_nms_person_boxes(
            np.array([[0.0, 0.1, 0.1, 0.3, 0.9]], dtype=np.float32), (640, 640)
        )
        self.assertEqual(
            input_boxes_to_detections(in_padding, 0.1, (1280, 720), 0.5, (0, 140)), []
        )

    def test_yolo26_raw_outputs_decode_person_cells_only(self):
        outputs = []
        for grid in (80, 40, 20):
            outputs.append(np.zeros((grid, grid, 4), dtype=np.float32))
            outputs.append(np.full((grid, grid, 80), -10.0, dtype=np.float32))
        # Stride 16 cell (x=10, y=20): person logit 2.0, another class higher elsewhere.
        outputs[3][20, 10, 0] = 2.0
        outputs[3][5, 5, 1] = 9.0
        outputs[2][20, 10] = [1.5, 2.5, 0.5, 3.5]
        boxes = yolo26_raw_person_boxes(outputs[::-1], 0.35, (640, 640))
        self.assertEqual(boxes.shape, (1, 5))
        np.testing.assert_allclose(boxes[0, :4], [144.0, 288.0, 176.0, 384.0])
        self.assertAlmostEqual(float(boxes[0, 4]), 1.0 / (1.0 + np.exp(-2.0)), places=5)

    def test_yolo26_raw_outputs_without_people_give_no_boxes(self):
        outputs = [np.zeros((20, 20, 4), np.float32), np.full((20, 20, 80), -10.0, np.float32)]
        self.assertEqual(yolo26_raw_person_boxes(outputs, 0.1, (640, 640)).shape, (0, 5))


class ReidCropTest(unittest.TestCase):
    def test_crop_is_resized_to_the_model_input_in_rgb(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[100:300, 200:280] = (255, 0, 0)  # blue person in BGR
        image = reid_crop_rgb(frame, BoundingBox(200.0, 100.0, 280.0, 300.0), 128, 256)
        self.assertEqual(image.shape, (256, 128, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image[128, 64].tolist(), [0, 0, 255])

    def test_box_outside_the_frame_is_clamped(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        image = reid_crop_rgb(frame, BoundingBox(600.0, 300.0, 700.0, 420.0), 128, 256)
        self.assertEqual(image.shape, (256, 128, 3))
        image = reid_crop_rgb(frame, BoundingBox(650.0, 370.0, 700.0, 420.0), 128, 256)
        self.assertEqual(image.shape, (256, 128, 3))


if __name__ == "__main__":
    unittest.main()


REPVGG_ONNX = Path(__file__).resolve().parent.parent / "repvgg_a0_person_reid_512.onnx"


@unittest.skipUnless(REPVGG_ONNX.exists(), "repvgg_a0_person_reid_512.onnx not present")
class OnnxAppearanceEncoderTest(unittest.TestCase):
    def test_same_person_is_closer_than_a_different_one(self):
        encoder = OnnxAppearanceEncoder(str(REPVGG_ONNX))
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[100:300, 100:160] = (40, 40, 200)
        frame[100:300, 400:460] = (200, 160, 40)
        frame[100:300, 250:310] = (40, 40, 200)
        red_a, blue, red_b = encoder.encode(frame, [person(100.0), person(400.0), person(250.0)])
        unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
        self.assertEqual(red_a.shape, (512,))
        self.assertGreater(float(unit(red_a) @ unit(red_b)), float(unit(red_a) @ unit(blue)))

    def test_gpu_copy_gives_the_same_features(self):
        try:
            import onnx2torch  # noqa: F401
            import torch
        except ImportError:
            self.skipTest("onnx2torch not installed")
        if not torch.cuda.is_available():
            self.skipTest("no CUDA device")
        frame = np.random.default_rng(1).integers(0, 255, (360, 640, 3), dtype=np.uint8)
        people = [person(100.0), person(300.0)]
        cpu = OnnxAppearanceEncoder(str(REPVGG_ONNX)).encode(frame, people)
        gpu = OnnxAppearanceEncoder(str(REPVGG_ONNX), device="cuda").encode(frame, people)
        for a, b in zip(cpu, gpu):
            self.assertGreater(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), 0.9999)
