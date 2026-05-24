"""OpenCV ArUco marker detector."""

from __future__ import annotations

from typing import List

import numpy as np

from target_state import MarkerDetection


class ArucoMarkerDetector:
    def __init__(self, dictionary_name: str = "DICT_4X4_50") -> None:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise RuntimeError("OpenCV is required: pip install opencv-contrib-python") from exc

        self.cv2 = cv2
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV ArUco support is required: pip install opencv-contrib-python")

        dictionary_id = getattr(cv2.aruco, dictionary_name)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        if hasattr(cv2.aruco, "DetectorParameters"):
            params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)
        else:
            self.detector = None
            self.params = cv2.aruco.DetectorParameters_create()

    def detect(self, frame: np.ndarray) -> List[MarkerDetection]:
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(frame)
        else:
            corners, ids, _ = self.cv2.aruco.detectMarkers(frame, self.dictionary, parameters=self.params)

        detections: List[MarkerDetection] = []
        if ids is None:
            return detections

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            pts = marker_corners.reshape(4, 2)
            center = tuple(pts.mean(axis=0).tolist())
            detections.append(
                MarkerDetection(
                    marker_id=int(marker_id),
                    center=(float(center[0]), float(center[1])),
                    corners=tuple((float(x), float(y)) for x, y in pts),
                )
            )
        return detections
