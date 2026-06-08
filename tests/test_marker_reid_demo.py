import unittest

import numpy as np

from src.reid.appearance import AppearanceReIdentifier
from src.tracking.marker_reid_demo import _reacquire_by_reid
from src.tracking.target_state import PersonDetection


class MarkerReidDemoTests(unittest.TestCase):
    def test_reid_reacquires_matching_person_candidate(self):
        frame = np.zeros((120, 180, 3), dtype=np.uint8)
        frame[10:90, 10:50] = (255, 0, 0)
        frame[10:90, 90:130] = (0, 255, 0)

        reid = AppearanceReIdentifier(threshold=0.68)
        self.assertTrue(reid.register(frame, (10, 10, 50, 90)))

        people = [
            PersonDetection((90, 10, 40, 80), 0.9),
            PersonDetection((10, 10, 40, 80), 0.9),
        ]
        person, score = _reacquire_by_reid(reid, frame, people)

        self.assertEqual(person, people[1])
        self.assertGreaterEqual(score, 0.68)


if __name__ == "__main__":
    unittest.main()
