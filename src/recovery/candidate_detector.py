"""YOLO candidate detection invoked only while diagnosing a mismatch."""

from __future__ import annotations

from src.vision.models import Candidate


class YoloRecoveryDetector:
    def __init__(self, model_path: str, image_size: int = 416, confidence: float = 0.4, device: str | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path, task="detect")
        self.image_size = image_size
        self.confidence = confidence
        self.device = device

    def detect(self, frame) -> list[Candidate]:
        kwargs = {"imgsz": self.image_size, "conf": self.confidence, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        result = self.model.predict(frame, **kwargs)[0]
        candidates = []
        if result.boxes is None:
            return candidates
        for index, box in enumerate(result.boxes):
            if int(box.cls.item()) != 0:
                continue
            bbox = tuple(int(round(value)) for value in box.xyxy[0].tolist())
            candidates.append(Candidate(bbox, float(box.conf.item()), index))
        return candidates
