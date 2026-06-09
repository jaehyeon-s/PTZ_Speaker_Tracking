"""Person detector adapters for the demo entry point."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from src.tracking.target_state import PersonDetection


class PersonDetector:
    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        raise NotImplementedError


class HOGPersonDetector(PersonDetector):
    """OpenCV HOG fallback when no project detector is configured."""

    def __init__(self) -> None:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise RuntimeError("OpenCV is required: pip install opencv-contrib-python") from exc

        self.cv2 = cv2
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        boxes, weights = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        return [
            PersonDetection(bbox=(int(x), int(y), int(w), int(h)), confidence=float(weight))
            for (x, y, w, h), weight in zip(boxes, weights)
        ]


class YoloOpenCVPersonDetector(PersonDetector):
    """Temporary OpenCV DNN YOLO adapter for local ONNX models.

    Warning: YOLO ONNX output layouts vary by exporter/model version. Prefer
    ExistingProjectPersonDetectorAdapter or NCNNPersonDetector for the project
    demo path, and adjust _parse_yolo_like_output only for a known model.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.35) -> None:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise RuntimeError("OpenCV is required: pip install opencv-contrib-python") from exc

        self.cv2 = cv2
        self.confidence_threshold = confidence_threshold
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)
        self.net = cv2.dnn.readNetFromONNX(str(path))

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        # TODO: Adjust parsing if the selected YOLO export uses a different output layout.
        blob = self.cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        return _parse_yolo_like_output(outputs, frame.shape, self.confidence_threshold)


class NCNNPersonDetector(PersonDetector):
    """NCNN YOLO detector adapter for Raspberry Pi 5 demos.

    This class owns only the NCNN runtime boundary and returns PersonDetection
    objects. If the project already has an NCNN YOLO parser, pass it as
    output_parser so that model-specific decode/NMS logic is reused instead of
    duplicated here.
    """

    def __init__(
        self,
        param_path: str,
        bin_path: str,
        input_size: int = 640,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        output_parser: Optional[Callable[[Any, tuple[int, int, int], float, float], List[PersonDetection]]] = None,
        input_name: str = "in0",
        output_names: tuple[str, ...] = ("out0", "output", "output0"),
        debug_detector: bool = False,
        debug_frames: int = 5,
    ) -> None:
        try:
            import ncnn
        except ModuleNotFoundError as exc:
            raise RuntimeError("NCNN Python bindings are required for --detector ncnn") from exc

        self.ncnn = ncnn
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.output_parser = output_parser
        self.input_name = input_name
        self.output_names = output_names
        self.debug_detector = debug_detector
        self.debug_frames = debug_frames
        self._debug_frame_count = 0
        self._debug_warnings: set[str] = set()
        self._last_output_shapes: dict[str, tuple[int, ...]] = {}
        self.net = ncnn.Net()

        param = Path(param_path)
        model_bin = Path(bin_path)
        if not param.exists():
            raise FileNotFoundError(param)
        if not model_bin.exists():
            raise FileNotFoundError(model_bin)
        self.net.load_param(str(param))
        self.net.load_model(str(model_bin))

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        raw_output = self._forward(frame)
        if self.output_parser is not None:
            detections = self.output_parser(
                raw_output,
                frame.shape,
                self.confidence_threshold,
                self.nms_threshold,
            )
        else:
            detections = _parse_yolo_ncnn_output(
                raw_output,
                frame.shape,
                self.input_size,
                self.confidence_threshold,
            )
            if not detections:
                detections = detections_to_person_detections(
                    raw_output,
                    bbox_format="xyxy",
                    confidence_threshold=self.confidence_threshold,
                )
            detections = nms_person_detections(detections, self.nms_threshold)

        if not detections:
            self._debug_warn("NCNN output may not match YOLO parser")
        detections = self._clamp_detections(detections, frame.shape)
        self._debug_frame(raw_output, detections)
        return detections

    def _forward(self, frame: np.ndarray) -> Any:
        frame_h, frame_w = frame.shape[:2]
        mat = self.ncnn.Mat.from_pixels_resize(
            frame,
            self.ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            frame_w,
            frame_h,
            self.input_size,
            self.input_size,
        )
        mat.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])

        extractor = self.net.create_extractor()
        extractor.input(self.input_name, mat)
        self._last_output_shapes = {}
        first_output = None
        for output_name in self.output_names:
            ret, output = extractor.extract(output_name)
            if ret == 0:
                array = np.asarray(output)
                self._last_output_shapes[output_name] = tuple(array.shape)
                if first_output is None:
                    first_output = array
        if first_output is not None:
            return first_output
        raise RuntimeError(
            "Could not extract NCNN output. Set the model's input/output blob names "
            "in NCNNPersonDetector or reuse the existing project NCNN YOLO adapter."
        )

    def _clamp_detections(
        self,
        detections: List[PersonDetection],
        frame_shape: tuple[int, int, int],
    ) -> List[PersonDetection]:
        frame_h, frame_w = frame_shape[:2]
        clamped: List[PersonDetection] = []
        for detection in detections:
            x, y, w, h = detection.bbox
            x1 = max(0, min(frame_w - 1, x))
            y1 = max(0, min(frame_h - 1, y))
            x2 = max(0, min(frame_w, x + w))
            y2 = max(0, min(frame_h, y + h))
            new_bbox = (x1, y1, max(0, x2 - x1), max(0, y2 - y1))
            if new_bbox != detection.bbox:
                self._debug_warn(f"clamped bbox from {detection.bbox} to {new_bbox}")
            clamped.append(
                PersonDetection(
                    bbox=new_bbox,
                    confidence=detection.confidence,
                    class_name=detection.class_name,
                )
            )
        return clamped

    def _debug_frame(self, raw_output: Any, detections: List[PersonDetection]) -> None:
        if not self.debug_detector or self._debug_frame_count >= self.debug_frames:
            return
        self._debug_frame_count += 1
        print(f"[NCNN DEBUG] frame={self._debug_frame_count}")
        print(f"[NCNN DEBUG] input_blob={self.input_name}")
        print(f"[NCNN DEBUG] output_blobs={','.join(self.output_names)}")
        if self._last_output_shapes:
            for name, shape in self._last_output_shapes.items():
                print(f"[NCNN DEBUG] output_shape {name}={shape}")
        else:
            print("[NCNN DEBUG] output_shape none")
        print(f"[NCNN DEBUG] raw_output_shape={tuple(np.asarray(raw_output).shape)}")
        print(f"[NCNN DEBUG] detections_after_parser={len(detections)}")
        for index, detection in enumerate(detections[:5]):
            print(
                "[NCNN DEBUG] detection "
                f"{index}: bbox={detection.bbox} confidence={detection.confidence:.4f} class_id=0"
            )

    def _debug_warn(self, message: str) -> None:
        if self.debug_detector and message not in self._debug_warnings:
            self._debug_warnings.add(message)
            print(f"[NCNN WARNING] {message}")


class ManualPersonDetector(PersonDetector):
    """Deterministic detector for smoke tests and controlled demo videos."""

    def __init__(self, bbox: tuple[int, int, int, int]) -> None:
        self.bbox = bbox

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        del frame
        return [PersonDetection(bbox=self.bbox, confidence=1.0)]


class DetectionResultAdapter:
    """Converts existing detector/tracker outputs into PersonDetection."""

    def __init__(
        self,
        bbox_format: str = "auto",
        person_class_ids: tuple[int, ...] = (0,),
        confidence_threshold: float = 0.0,
    ) -> None:
        self.bbox_format = bbox_format
        self.person_class_ids = person_class_ids
        self.confidence_threshold = confidence_threshold

    def convert(self, raw_result: Any) -> List[PersonDetection]:
        return detections_to_person_detections(
            raw_result,
            bbox_format=self.bbox_format,
            person_class_ids=self.person_class_ids,
            confidence_threshold=self.confidence_threshold,
        )


class ExistingProjectPersonDetectorAdapter(PersonDetector):
    """Adapter for existing YOLO/ByteTrack project detectors.

    The wrapped detector is expected to keep its own implementation unchanged.
    This adapter only converts its per-frame result into the MVP's
    PersonDetection(bbox=(x, y, w, h), confidence=...) shape.

    Supported item shapes:
    - dict: {"bbox"/"tlwh"/"xywh"/"tlbr"/"xyxy", "confidence"/"score", "class_id"}
    - object: attributes with the same names, including ByteTrack-style tlwh
    - ndarray/list: [x1, y1, x2, y2, score, class_id] by default
    """

    def __init__(
        self,
        detector: Any,
        result_getter: Optional[Callable[[Any, np.ndarray], Any]] = None,
        bbox_format: str = "auto",
        person_class_ids: tuple[int, ...] = (0,),
        confidence_threshold: float = 0.0,
    ) -> None:
        self.detector = detector
        self.result_getter = result_getter
        self.result_adapter = DetectionResultAdapter(
            bbox_format=bbox_format,
            person_class_ids=person_class_ids,
            confidence_threshold=confidence_threshold,
        )

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        raw_result = self._run_detector(frame)
        return self.result_adapter.convert(raw_result)

    def _run_detector(self, frame: np.ndarray) -> Any:
        if self.result_getter is not None:
            return self.result_getter(self.detector, frame)
        if hasattr(self.detector, "detect"):
            return self.detector.detect(frame)
        if hasattr(self.detector, "track"):
            return self.detector.track(frame)
        if callable(self.detector):
            return self.detector(frame)
        raise TypeError("detector must be callable or expose detect(frame)/track(frame)")


def build_person_detector(
    detector_mode: str = "hog",
    yolo_model: Optional[str] = None,
    manual_bbox: Optional[tuple[int, int, int, int]] = None,
    ncnn_param: Optional[str] = None,
    ncnn_bin: Optional[str] = None,
    ncnn_input_size: int = 640,
    ncnn_input_name: str = "in0",
    ncnn_output_names: tuple[str, ...] = ("out0", "output", "output0"),
    conf_threshold: float = 0.35,
    nms_threshold: float = 0.45,
    debug_detector: bool = False,
    existing_detector: Any = None,
    result_getter: Optional[Callable[[Any, np.ndarray], Any]] = None,
) -> PersonDetector:
    if existing_detector is not None:
        return ExistingProjectPersonDetectorAdapter(
            existing_detector,
            result_getter=result_getter,
            confidence_threshold=conf_threshold,
        )
    if detector_mode == "manual":
        if manual_bbox is None:
            raise ValueError("--detector manual requires --person-box x,y,w,h")
        return ManualPersonDetector(manual_bbox)
    if detector_mode == "opencv-yolo":
        if yolo_model is None:
            raise ValueError("--detector opencv-yolo requires --yolo-model")
        return YoloOpenCVPersonDetector(yolo_model, confidence_threshold=conf_threshold)
    if detector_mode == "ncnn":
        if ncnn_param is None or ncnn_bin is None:
            raise ValueError("--detector ncnn requires --ncnn-param and --ncnn-bin")
        return NCNNPersonDetector(
            ncnn_param,
            ncnn_bin,
            input_size=ncnn_input_size,
            input_name=ncnn_input_name,
            output_names=ncnn_output_names,
            confidence_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            debug_detector=debug_detector,
        )
    if detector_mode == "hog":
        return HOGPersonDetector()
    raise ValueError(f"Unsupported detector mode: {detector_mode}")


def detections_to_person_detections(
    raw_result: Any,
    bbox_format: str = "auto",
    person_class_ids: tuple[int, ...] = (0,),
    confidence_threshold: float = 0.0,
) -> List[PersonDetection]:
    """Convert YOLO/ByteTrack-like outputs to PersonDetection objects."""
    items = _iter_detection_items(raw_result)
    detections: List[PersonDetection] = []
    for item in items:
        class_id = _read_class_id(item)
        if class_id is not None and person_class_ids and class_id not in person_class_ids:
            continue

        confidence = _read_confidence(item)
        if confidence < confidence_threshold:
            continue

        bbox = _read_bbox(item, bbox_format)
        if bbox is None:
            continue
        detections.append(PersonDetection(bbox=bbox, confidence=confidence))
    return detections


def _parse_yolo_ncnn_output(
    raw_output: Any,
    frame_shape: tuple[int, int, int],
    input_size: int,
    confidence_threshold: float,
) -> List[PersonDetection]:
    """Parse Ultralytics YOLO NCNN detect tensors.

    Common exports return a tensor shaped like (classes + 4, anchors) or
    (anchors, classes + 4), where the first four values are xywh in resized
    square input coordinates and class 0 is the person score for COCO models.
    """
    output = np.asarray(raw_output, dtype=np.float32).squeeze()
    if output.ndim != 2:
        return []
    if output.shape[0] >= 5 and output.shape[0] < output.shape[1]:
        rows = output.T
    elif output.shape[1] >= 5:
        rows = output
    else:
        return []

    frame_h, frame_w = frame_shape[:2]
    scale_x = frame_w / max(float(input_size), 1.0)
    scale_y = frame_h / max(float(input_size), 1.0)
    detections: List[PersonDetection] = []
    for row in rows:
        if row.shape[0] < 5:
            continue
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        if class_id != 0:
            continue
        confidence = float(class_scores[class_id])
        if confidence < confidence_threshold:
            continue

        cx, cy, width, height = (float(value) for value in row[:4])
        x1 = int(round((cx - width / 2.0) * scale_x))
        y1 = int(round((cy - height / 2.0) * scale_y))
        box_w = int(round(width * scale_x))
        box_h = int(round(height * scale_y))
        if box_w <= 0 or box_h <= 0:
            continue
        detections.append(PersonDetection((x1, y1, box_w, box_h), confidence))
    return detections


def nms_person_detections(
    detections: List[PersonDetection],
    nms_threshold: float,
) -> List[PersonDetection]:
    if not detections:
        return []

    remaining = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept: List[PersonDetection] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [
            candidate
            for candidate in remaining
            if _bbox_iou(current.bbox, candidate.bbox) <= nms_threshold
        ]
    return kept


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _iter_detection_items(raw_result: Any) -> Iterable[Any]:
    if raw_result is None:
        return []
    if hasattr(raw_result, "detections"):
        return raw_result.detections
    if hasattr(raw_result, "tracks"):
        return raw_result.tracks
    if hasattr(raw_result, "boxes"):
        boxes = raw_result.boxes
        if hasattr(boxes, "data"):
            return np.asarray(boxes.data)
        return boxes
    if isinstance(raw_result, np.ndarray):
        if raw_result.ndim == 1:
            return [raw_result]
        return raw_result
    return raw_result


def _read_bbox(item: Any, bbox_format: str) -> Optional[tuple[int, int, int, int]]:
    value = _first_present(item, ("tlwh", "xywh", "bbox", "tlbr", "xyxy"))
    if value is None and _is_sequence(item):
        value = item
    if value is None:
        return None

    values = [float(v) for v in np.asarray(value).flatten()[:4]]
    if len(values) < 4:
        return None

    fmt = bbox_format
    if fmt == "auto":
        fmt = _infer_bbox_format(item)

    if fmt in ("tlwh", "xywh", "bbox"):
        x, y, w, h = values
    elif fmt in ("tlbr", "xyxy"):
        x1, y1, x2, y2 = values
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
    else:
        raise ValueError(f"Unsupported bbox_format: {bbox_format}")

    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def _infer_bbox_format(item: Any) -> str:
    for name in ("tlwh", "xywh", "bbox"):
        if _has_value(item, name):
            return name
    for name in ("tlbr", "xyxy"):
        if _has_value(item, name):
            return name
    return "xyxy"


def _read_confidence(item: Any) -> float:
    value = _first_present(item, ("confidence", "conf", "score", "det_score"))
    if value is None and _is_sequence(item) and len(item) >= 5:
        value = item[4]
    return float(np.asarray(value).flatten()[0]) if value is not None else 1.0


def _read_class_id(item: Any) -> Optional[int]:
    value = _first_present(item, ("class_id", "cls", "label", "category_id"))
    if value is None and _is_sequence(item) and len(item) >= 6:
        value = item[5]
    return int(np.asarray(value).flatten()[0]) if value is not None else None


def _first_present(item: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if _has_value(item, name):
            return _get_value(item, name)
    return None


def _has_value(item: Any, name: str) -> bool:
    value = _get_value(item, name)
    return value is not None


def _get_value(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _is_sequence(item: Any) -> bool:
    if isinstance(item, np.ndarray):
        return item.ndim == 1
    return isinstance(item, (list, tuple))


def _parse_yolo_like_output(
    output: np.ndarray,
    frame_shape: tuple[int, int, int],
    threshold: float,
) -> List[PersonDetection]:
    frame_h, frame_w = frame_shape[:2]
    rows = np.squeeze(output)
    if rows.ndim != 2:
        return []

    detections: List[PersonDetection] = []
    for row in rows:
        if len(row) < 6:
            continue
        cx, cy, w, h = row[:4]
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if class_id != 0 or confidence < threshold:
            continue
        x = int((cx - w / 2) * frame_w / 640.0)
        y = int((cy - h / 2) * frame_h / 640.0)
        bw = int(w * frame_w / 640.0)
        bh = int(h * frame_h / 640.0)
        detections.append(PersonDetection(bbox=(x, y, bw, bh), confidence=confidence))
    return detections
