import unittest

import numpy as np

from src.reid.appearance import AppearanceReIdentifier
from src.vision.models import Candidate


class AppearanceReIdentifierTests(unittest.TestCase):
    def test_selects_candidate_matching_registered_appearance(self):
        frame = np.zeros((120, 300, 3), dtype=np.uint8)
        frame[10:110, 10:70] = (0, 255, 0)
        frame[10:110, 100:160] = (255, 0, 0)
        frame[10:110, 200:260] = (0, 255, 0)
        reid = AppearanceReIdentifier(threshold=0.68)
        reid.register(frame, (10, 10, 70, 110))

        match, score = reid.best_match(
            frame,
            [
                Candidate((100, 10, 160, 110), candidate_id=2),
                Candidate((200, 10, 260, 110), candidate_id=3),
            ],
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.candidate_id, 3)
        self.assertGreaterEqual(score, 0.68)


if __name__ == "__main__":
    unittest.main()
