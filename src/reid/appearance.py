"""Appearance features for presenter verification and recovery."""

from __future__ import annotations

import numpy as np

from src.vision.models import BBox, Candidate


class AppearanceReIdentifier:
    def __init__(self, threshold: float = 0.68) -> None:
        self.threshold = threshold
        self.reference: dict[str, np.ndarray | float] | None = None

    def register(self, frame, bbox: BBox) -> bool:
        self.reference = self._features(frame, bbox)
        return self.reference is not None

    def score(self, frame, bbox: BBox) -> float:
        if self.reference is None:
            return 0.0
        return self._compare(self.reference, self._features(frame, bbox))

    def best_match(self, frame, candidates: list[Candidate]) -> tuple[Candidate | None, float]:
        if self.reference is None:
            return None, 0.0
        best_candidate = None
        best_score = 0.0
        for candidate in candidates:
            score = self.score(frame, candidate.bbox)
            if score > best_score:
                best_candidate, best_score = candidate, score
        if best_score < self.threshold:
            return None, best_score
        return best_candidate, best_score

    def reset(self) -> None:
        self.reference = None

    @staticmethod
    def _features(frame, bbox: BBox):
        import cv2

        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        upper = crop[: max(1, crop.shape[0] // 2)]
        hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
        hue = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
        saturation = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        hue /= hue.sum() + 1e-6
        saturation /= saturation.sum() + 1e-6
        return {
            "hue": hue,
            "saturation": saturation,
            "aspect": float((x2 - x1) / max(y2 - y1, 1)),
        }

    @staticmethod
    def _compare(reference, candidate) -> float:
        import cv2

        if candidate is None:
            return 0.0
        hue = cv2.compareHist(reference["hue"], candidate["hue"], cv2.HISTCMP_INTERSECT)
        saturation = cv2.compareHist(
            reference["saturation"], candidate["saturation"], cv2.HISTCMP_INTERSECT
        )
        aspect = max(0.0, 1.0 - abs(reference["aspect"] - candidate["aspect"]) * 2.0)
        return float(np.clip(hue * 0.5 + saturation * 0.3 + aspect * 0.2, 0.0, 1.0))


class OnnxEmbeddingReIdentifier:
    """Re-ID matcher for OSNet-style ONNX embedding models.

    The model is expected to accept an NCHW RGB tensor and return one embedding
    vector per crop. This keeps the runtime optional while allowing the same
    verifier state machine to use an OSNet export when available.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.72,
        input_size: tuple[int, int] = (128, 256),
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self.threshold = threshold
        self.input_size = input_size
        self.session = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.batch_size = model_input.shape[0] if isinstance(model_input.shape[0], int) else 1
        self.reference: np.ndarray | None = None

    def register(self, frame, bbox: BBox) -> bool:
        self.reference = self._embedding(frame, bbox)
        return self.reference is not None

    def score(self, frame, bbox: BBox) -> float:
        if self.reference is None:
            return 0.0
        embedding = self._embedding(frame, bbox)
        if embedding is None:
            return 0.0
        return float(np.clip(np.dot(self.reference, embedding), -1.0, 1.0))

    def best_match(self, frame, candidates: list[Candidate]) -> tuple[Candidate | None, float]:
        if self.reference is None:
            return None, 0.0
        best_candidate = None
        best_score = -1.0
        for candidate in candidates:
            score = self.score(frame, candidate.bbox)
            if score > best_score:
                best_candidate, best_score = candidate, score
        if best_candidate is None or best_score < self.threshold:
            return None, max(0.0, best_score)
        return best_candidate, best_score

    def reset(self) -> None:
        self.reference = None

    def _embedding(self, frame, bbox: BBox) -> np.ndarray | None:
        import cv2

        crop = self._crop(frame, bbox)
        if crop is None:
            return None
        width, height = self.input_size
        resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")
        tensor = ((rgb - mean) / std).transpose(2, 0, 1)[None, ...]
        if self.batch_size > 1:
            tensor = np.repeat(tensor, self.batch_size, axis=0)
        output = self.session.run(None, {self.input_name: tensor})[0]
        embedding = np.asarray(output)[0].reshape(-1).astype("float32")
        norm = np.linalg.norm(embedding)
        if norm <= 1e-6:
            return None
        return embedding / norm

    @staticmethod
    def _crop(frame, bbox: BBox):
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(x2)), min(height, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]


def build_re_identifier(backend: str, threshold: float, model_path: str | None = None):
    if backend == "hsv":
        return AppearanceReIdentifier(threshold)
    if backend == "onnx":
        if not model_path:
            raise ValueError("--reid-backend onnx requires --reid-model")
        return OnnxEmbeddingReIdentifier(model_path, threshold)
    raise ValueError("--reid-backend must be one of hsv, onnx")
