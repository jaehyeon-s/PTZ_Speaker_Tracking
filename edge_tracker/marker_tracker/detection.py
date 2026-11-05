"""Person detection separated from multi-object tracking.

``model.track()`` runs the detector and the tracker in one Ultralytics call, so
the detector cannot be replaced. Here the two are separate stages: any
``PersonDetector`` (CPU or a Hailo NPU) feeds the same Ultralytics
BYTETracker/BoT-SORT classes and tracker YAML that ``model.track()`` uses.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Protocol, Sequence

import numpy as np

from .core import BoundingBox, Detection

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PersonDetector(Protocol):
    def detect(self, frame, confidence: float) -> list[Detection]:
        """Return person boxes in frame pixels with score >= confidence, without track IDs."""
        ...


class UltralyticsPersonDetector:
    def __init__(self, model, imgsz: int | None = None, device: str = "auto") -> None:
        self._model = model
        self._kwargs = {"verbose": False}
        names = getattr(model, "names", None) or {}
        person_class_ids = [int(i) for i, name in dict(names).items() if name == "person"]
        if person_class_ids:
            self._kwargs["classes"] = person_class_ids
        if imgsz is not None:
            self._kwargs["imgsz"] = imgsz
        if device != "auto":
            self._kwargs["device"] = device

    def detect(self, frame, confidence: float) -> list[Detection]:
        result = self._model.predict(frame, conf=confidence, **self._kwargs)[0]
        boxes = result.boxes.cpu().numpy()
        detections = []
        for xyxy, score, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
            if result.names[int(class_id)] != "person":
                continue
            detections.append(
                Detection("person", float(score), BoundingBox(*(float(v) for v in xyxy)))
            )
        return detections


class HailoPersonDetector:
    """Person detection on a Hailo NPU through HailoRT (``hailo_platform``).

    Accepts two kinds of COCO YOLO HEF:

    - built-in NMS post-processing (e.g. ``yolov8s_h8.hef``): one output with
      one array per class, rows ``[y_min, x_min, y_max, x_max, score]``
      normalized to the network input;
    - the Hailo Model Zoo ``yolo26n.hef``: the six raw one-to-one head outputs
      (box distances and class logits per stride), decoded here without NMS.
    """

    def __init__(
        self,
        hef_path: str,
        person_class_index: int = 0,
        shared_device: bool = False,
    ) -> None:
        from contextlib import ExitStack

        from hailo_platform import (
            HEF,
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )

        if not Path(hef_path).exists():
            raise FileNotFoundError(f"Hailo HEF not found: {hef_path}")
        self._person_class_index = person_class_index
        hef = HEF(hef_path)
        self._stack = ExitStack()
        # A second network (speaker ReID) on the same chip needs the HailoRT
        # scheduler, which activates networks itself. Alone, the detector keeps
        # the exclusive device and explicit activation verified on the Pi.
        device = (
            shared_hailo_device() if shared_device else self._stack.enter_context(VDevice())
        )
        configure_params = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
        network_group = device.configure(hef, configure_params)[0]
        input_info = hef.get_input_vstream_infos()[0]
        self._input_name = input_info.name
        self._input_height, self._input_width = (int(v) for v in input_info.shape[:2])
        self._output_names = [info.name for info in hef.get_output_vstream_infos()]
        self._pipeline = self._stack.enter_context(
            InferVStreams(
                network_group,
                InputVStreamParams.make(network_group, format_type=FormatType.UINT8),
                OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32),
            )
        )
        if not shared_device:
            self._stack.enter_context(network_group.activate(network_group.create_params()))

    def detect(self, frame, confidence: float) -> list[Detection]:
        image, scale, pad_x, pad_y = letterbox_rgb(frame, self._input_width, self._input_height)
        outputs = self._pipeline.infer({self._input_name: image[np.newaxis]})
        if len(self._output_names) == 1:
            boxes = hailo_nms_person_boxes(
                outputs[self._output_names[0]][0][self._person_class_index],
                (self._input_width, self._input_height),
            )
        else:
            boxes = yolo26_raw_person_boxes(
                [outputs[name][0] for name in self._output_names],
                confidence,
                (self._input_width, self._input_height),
                self._person_class_index,
            )
        frame_height, frame_width = frame.shape[:2]
        return input_boxes_to_detections(
            boxes, confidence, (frame_width, frame_height), scale, (pad_x, pad_y)
        )


_SHARED_HAILO_DEVICE = None


def shared_hailo_device():
    """One VDevice with the round-robin scheduler, shared by every network in the process."""
    global _SHARED_HAILO_DEVICE
    if _SHARED_HAILO_DEVICE is None:
        from hailo_platform import HailoSchedulingAlgorithm, VDevice

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        _SHARED_HAILO_DEVICE = VDevice(params)
    return _SHARED_HAILO_DEVICE


class HailoAppearanceEncoder:
    """Person ReID embeddings on a Hailo NPU (same ``encode`` interface as the CPU encoder).

    Expects a fixed-size ReID HEF with normalization built in, such as the
    Hailo Model Zoo ``repvgg_a0_person_reid_512.hef`` (256x128 RGB in, 512 out).
    """

    def __init__(self, hef_path: str) -> None:
        from contextlib import ExitStack

        from hailo_platform import (
            HEF,
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
        )

        if not Path(hef_path).exists():
            raise FileNotFoundError(f"Hailo HEF not found: {hef_path}")
        hef = HEF(hef_path)
        configure_params = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
        network_group = shared_hailo_device().configure(hef, configure_params)[0]
        input_info = hef.get_input_vstream_infos()[0]
        self._input_name = input_info.name
        self._input_height, self._input_width = (int(v) for v in input_info.shape[:2])
        self._output_name = hef.get_output_vstream_infos()[0].name
        self._stack = ExitStack()
        self._pipeline = self._stack.enter_context(
            InferVStreams(
                network_group,
                InputVStreamParams.make(network_group, format_type=FormatType.UINT8),
                OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32),
            )
        )

    def encode(self, frame, people: Sequence[Detection]) -> list[np.ndarray]:
        features = []
        for person in people:
            image = reid_crop_rgb(frame, person.box, self._input_width, self._input_height)
            output = self._pipeline.infer({self._input_name: image[np.newaxis]})
            features.append(np.asarray(output[self._output_name][0], dtype=np.float32).reshape(-1))
        return features


class OnnxAppearanceEncoder:
    """The float ONNX of the Hailo ReID model on the CPU (same ``encode`` interface).

    Lets a computer without a Hailo NPU run the model the Pi runs, for example
    ``repvgg_a0_person_reid_512.onnx`` from the Hailo Model Zoo. The HEF has
    the normalisation built in; the ONNX gets it here (the Model Zoo values).
    """

    _MEAN = np.array([111.296, 104.263, 104.396], dtype=np.float32)
    _STD = np.array([64.193, 62.547, 60.324], dtype=np.float32)

    def __init__(self, onnx_path: str, device: str = "cpu") -> None:
        import onnxruntime

        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ReID ONNX not found: {onnx_path}")
        self._session = onnxruntime.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self._input_height, self._input_width = (int(v) for v in model_input.shape[2:4])
        self._torch_model = None
        if device.startswith("cuda"):
            # On a computer with a GPU: the same network converted to PyTorch
            # (onnx2torch), one crop per call (the model has a fixed batch of 1).
            import onnx
            import torch
            from onnx2torch import convert

            self._torch = torch
            self._device = torch.device(device)
            self._torch_model = convert(onnx.load(onnx_path)).eval().to(self._device)

    def encode(self, frame, people: Sequence[Detection]) -> list[np.ndarray]:
        features = []
        for person in people:
            image = reid_crop_rgb(frame, person.box, self._input_width, self._input_height)
            tensor = ((image.astype(np.float32) - self._MEAN) / self._STD).transpose(2, 0, 1)
            if self._torch_model is not None:
                with self._torch.no_grad():
                    output = self._torch_model(
                        self._torch.from_numpy(np.ascontiguousarray(tensor[np.newaxis])).to(self._device)
                    ).cpu().numpy()
            else:
                output = self._session.run(None, {self._input_name: tensor[np.newaxis]})[0]
            features.append(np.asarray(output[0], dtype=np.float32).reshape(-1))
        return features


def reid_crop_rgb(frame, box: BoundingBox, width: int, height: int) -> np.ndarray:
    """Cut a person box out of a BGR frame and resize it to the RGB ReID input."""
    import cv2

    frame_height, frame_width = frame.shape[:2]
    x1 = min(max(int(box.x1), 0), frame_width - 1)
    y1 = min(max(int(box.y1), 0), frame_height - 1)
    x2 = min(max(int(np.ceil(box.x2)), x1 + 1), frame_width)
    y2 = min(max(int(np.ceil(box.y2)), y1 + 1), frame_height)
    crop = cv2.resize(frame[y1:y2, x1:x2], (width, height))
    return np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def letterbox_rgb(frame, width: int, height: int):
    """Resize a BGR frame into a padded RGB network input; return it with scale and padding."""
    import cv2

    frame_height, frame_width = frame.shape[:2]
    scale = min(width / frame_width, height / frame_height)
    new_width, new_height = round(frame_width * scale), round(frame_height * scale)
    pad_x, pad_y = (width - new_width) // 2, (height - new_height) // 2
    image = np.full((height, width, 3), 114, dtype=np.uint8)
    image[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = cv2.cvtColor(
        cv2.resize(frame, (new_width, new_height)), cv2.COLOR_BGR2RGB
    )
    return image, scale, pad_x, pad_y


def hailo_nms_person_boxes(rows, input_size: tuple[int, int]) -> np.ndarray:
    """Hailo NMS rows for one class as ``[x1, y1, x2, y2, score]`` in input pixels."""
    input_width, input_height = input_size
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
    return np.stack(
        [
            rows[:, 1] * input_width,
            rows[:, 0] * input_height,
            rows[:, 3] * input_width,
            rows[:, 2] * input_height,
            rows[:, 4],
        ],
        axis=1,
    )


def yolo26_raw_person_boxes(
    outputs: Sequence[np.ndarray],
    confidence: float,
    input_size: tuple[int, int],
    person_class_index: int = 0,
) -> np.ndarray:
    """Decode raw YOLO26 one-to-one head outputs to ``[x1, y1, x2, y2, score]`` in input pixels.

    ``outputs`` are the per-stride ``(H, W, 4)`` box maps (left, top, right,
    bottom distances in grid cells) and ``(H, W, classes)`` class-logit maps, in
    any order. The one-to-one head needs no NMS.
    """
    input_height = input_size[1]
    box_maps = {o.shape[:2]: o for o in outputs if o.shape[-1] == 4}
    class_maps = {o.shape[:2]: o for o in outputs if o.shape[-1] != 4}
    decoded = []
    for grid, class_map in class_maps.items():
        logits = class_map[..., person_class_index].astype(np.float32)
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        ys, xs = np.nonzero(scores >= confidence)
        if len(ys) == 0:
            continue
        stride = input_height / grid[0]
        left, top, right, bottom = box_maps[grid][ys, xs].astype(np.float32).T
        decoded.append(
            np.stack(
                [
                    (xs + 0.5 - left) * stride,
                    (ys + 0.5 - top) * stride,
                    (xs + 0.5 + right) * stride,
                    (ys + 0.5 + bottom) * stride,
                    scores[ys, xs],
                ],
                axis=1,
            )
        )
    if not decoded:
        return np.zeros((0, 5), dtype=np.float32)
    return np.concatenate(decoded)


def input_boxes_to_detections(
    boxes,
    confidence: float,
    frame_size: tuple[int, int],
    scale: float,
    padding: tuple[int, int],
) -> list[Detection]:
    """Map ``[x1, y1, x2, y2, score]`` rows in letterboxed input pixels back to frame pixels."""
    frame_width, frame_height = frame_size
    pad_x, pad_y = padding
    detections = []
    for x1, y1, x2, y2, score in np.asarray(boxes, dtype=np.float32).reshape(-1, 5):
        if score < confidence:
            continue
        x1 = min(max((x1 - pad_x) / scale, 0.0), frame_width)
        y1 = min(max((y1 - pad_y) / scale, 0.0), frame_height)
        x2 = min(max((x2 - pad_x) / scale, 0.0), frame_width)
        y2 = min(max((y2 - pad_y) / scale, 0.0), frame_height)
        if x2 <= x1 or y2 <= y1:
            continue
        # [room_01 수정] (room_01 이전 미커밋 수정) numpy float32가 JSON trace 기록에서 오류를 냄.
        # Plain floats: these boxes reach the JSON identity trace when no track is active.
        detections.append(
            Detection("person", float(score), BoundingBox(float(x1), float(y1), float(x2), float(y2)))
        )
    return detections


class _TrackerInput:
    """The subset of ultralytics Boxes that BYTETracker/BOTSORT read."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = np.zeros(len(conf), dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray:
        xywh = np.empty_like(self.xyxy)
        xywh[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2
        xywh[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2
        xywh[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        xywh[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return xywh

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index) -> "_TrackerInput":
        return _TrackerInput(self.xyxy[index], self.conf[index])


class TrackerStage:
    """Ultralytics BYTETracker/BoT-SORT driven with externally supplied detections."""

    def __init__(self, tracker_config: str) -> None:
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import YAML, IterableSimpleNamespace

        path = Path(tracker_config)
        if not path.is_absolute() and not path.exists():
            path = PROJECT_ROOT / path
        if not path.exists():
            from ultralytics.utils.checks import check_yaml

            path = Path(check_yaml(tracker_config))
        cfg = IterableSimpleNamespace(**YAML.load(path))
        if cfg.tracker_type not in TRACKER_MAP:
            raise ValueError(f"Unsupported tracker_type: {cfg.tracker_type}")
        if getattr(cfg, "with_reid", False):
            raise ValueError(
                "--person-tracker decoupled needs a tracker YAML with with_reid: False "
                "(speaker ReID is handled by --speaker-identity-backend)"
            )
        self._tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)

    def update(self, detections: Sequence[Detection], frame) -> list[Detection]:
        xyxy = np.array(
            [[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in detections], dtype=np.float32
        ).reshape(-1, 4)
        conf = np.array([d.confidence for d in detections], dtype=np.float32)
        tracks = self._tracker.update(_TrackerInput(xyxy, conf), frame)
        if len(tracks) == 0:
            # Same as model.track(): with no active track the raw boxes pass through.
            return list(detections)
        # model.track() clips the Kalman boxes to the image; do the same.
        height, width = frame.shape[:2]
        boxes = tracks[:, :4].copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        return [
            Detection(
                "person",
                float(row[5]),
                BoundingBox(*(float(v) for v in box)),
                track_id=int(row[4]),
            )
            for row, box in zip(tracks, boxes)
        ]


class DecoupledPersonSource:
    """PersonSource built from an exchangeable detector plus a tracker stage."""

    def __init__(self, args, detector: PersonDetector | None = None, tracker_stage=None) -> None:
        self._person_confidence = (
            args.confidence if args.confidence is not None else args.person_confidence
        )
        self._imgsz = args.imgsz
        self._inference_device = getattr(args, "inference_device", "auto")
        if detector is None and getattr(args, "hailo_hef", None):
            detector = HailoPersonDetector(
                args.hailo_hef,
                shared_device=bool(getattr(args, "speaker_reid_hef", None)),
            )
        self._detector = detector
        self._tracker_stage = tracker_stage
        self.last_timings: dict[str, float] = {}
        # --target-low-confidence: also keep weak boxes of the current target
        # (same track, or where it was last seen). None keeps only
        # --person-confidence, as before.
        self._target_low_confidence = getattr(args, "target_low_confidence", None)
        self.target_hint: tuple[int | None, BoundingBox | None] | None = None

    def update(self, frame, model, inference_confidence: float, bytetrack_config: str):
        if self._detector is None:
            self._detector = UltralyticsPersonDetector(model, self._imgsz, self._inference_device)
        if self._tracker_stage is None:
            self._tracker_stage = TrackerStage(bytetrack_config)
        started_at = time.perf_counter()
        detections = self._detector.detect(frame, inference_confidence)
        detected_at = time.perf_counter()
        tracked = self._tracker_stage.update(detections, frame)
        self.last_timings = {
            "person_detect": detected_at - started_at,
            "person_track": time.perf_counter() - detected_at,
        }
        return None, [
            d
            for d in tracked
            if d.confidence >= self._person_confidence or self._keep_weak_target(d)
        ]

    def _keep_weak_target(self, detection: Detection) -> bool:
        if (
            self._target_low_confidence is None
            or self.target_hint is None
            or detection.confidence < self._target_low_confidence
        ):
            return False
        hint_track_id, hint_box = self.target_hint
        if detection.track_id is not None and detection.track_id == hint_track_id:
            return True
        return hint_box is not None and near_target_box(detection.box, hint_box)


def near_target_box(box: BoundingBox, target: BoundingBox) -> bool:
    """Same place as the target: centre within one box width and a similar top edge."""
    width = max(box.x2 - box.x1, target.x2 - target.x1, 1.0)
    height = max(target.y2 - target.y1, 1.0)
    return (
        abs(box.center[0] - target.center[0]) <= width
        and abs(box.y1 - target.y1) <= 0.35 * height
    )
