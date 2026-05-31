import json
import tempfile
import unittest
from pathlib import Path

from src.camera.region_provider import CenterCropRegionProvider, JsonlRegionProvider


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


if __name__ == "__main__":
    unittest.main()
