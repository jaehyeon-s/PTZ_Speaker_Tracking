from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from .camera import LoggingCameraAdapter, QonHttpCameraAdapter, ViscaOverIpCameraAdapter
from .core import BoundingBox, Detection, HandoffDecision, MarkerTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from . import local_settings as LOCAL_SETTINGS
except ImportError:
    LOCAL_SETTINGS = None

LOCAL_RTSP_URL = getattr(LOCAL_SETTINGS, "RTSP_URL", None)
LOCAL_QON_CAMERA_URL = getattr(LOCAL_SETTINGS, "QON_CAMERA_URL", "")


RUNTIME_PROFILES = {
    "pi-qon-http": {
        "model": "yolo26n_ncnn_model",
        "imgsz": 416,
        "bytetrack_config": "bytetrack-pi.yaml",
        "aruco_marker_id": 0,
        "aruco_detect_flip": "auto",
        "confirmation_seconds": 2.0,
        "release_grace_seconds": 3.0,
        "marker_dropout_seconds": 0.75,
        "switch_score_margin": 12.0,
        "switch_cooldown_seconds": 1.0,
        "area_growth_penalty_threshold": 1.8,
        "tracking_status_diagnostics": True,
        "framing_mode": "upper-body",
        "track_buffer_seconds": 3.75,
        "camera_control": "qon-http",
        "qon_min_speed": 2,
        "qon_max_speed": 10,
        "qon_deadzone": 0.12,
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a person using an ArUco marker from an RTSP stream."
    )
    parser.add_argument(
        "--runtime-profile",
        choices=tuple(RUNTIME_PROFILES),
        help=(
            "Apply a deployment profile. pi-qon-http configures Raspberry Pi 5, "
            "NCNN model, ArUco marker 0, and Qon HTTP PTZ defaults."
        ),
    )
    parser.add_argument(
        "--rtsp-url",
        default=os.getenv("MARKER_TRACKER_RTSP_URL", LOCAL_RTSP_URL),
        help="RTSP stream URL. Defaults to MARKER_TRACKER_RTSP_URL or local_settings.py.",
    )
    parser.add_argument(
        "--video-file",
        help="Local video file to read at its native FPS instead of an RTSP stream.",
    )
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size. Lower this, e.g. 576 or 512, to improve FPS.",
    )
    parser.add_argument(
        "--person-source",
        choices=("yolo", "fixture"),
        default="yolo",
        help="Use YOLO/ByteTrack people or fixed two-person boxes for fixture videos.",
    )
    parser.add_argument(
        "--bytetrack-config",
        default="bytetrack.yaml",
        help="ByteTrack YAML config. Defaults to the project config with track_buffer=75.",
    )
    parser.add_argument(
        "--tracker-preset",
        choices=("bytetrack", "botsort"),
        default="bytetrack",
        help="Use the project ByteTrack config or Ultralytics BoT-SORT config.",
    )
    parser.add_argument(
        "--tracker-config",
        help=(
            "Explicit Ultralytics tracker YAML. Overrides --tracker-preset and "
            "--bytetrack-config."
        ),
    )
    parser.add_argument(
        "--person-tracker",
        choices=("ultralytics", "simple-iou"),
        default="ultralytics",
        help=(
            "Assign person IDs with Ultralytics model.track() or with a simple "
            "in-process IoU tracker over model.predict() detections."
        ),
    )
    parser.add_argument(
        "--simple-iou-threshold",
        type=float,
        default=0.3,
        help="Minimum IoU for matching detections to existing simple-iou tracks.",
    )
    parser.add_argument(
        "--simple-iou-max-lost-frames",
        type=int,
        default=15,
        help="How many missing frames a simple-iou track can survive.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        help="Override the minimum confidence for person detections.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("aruco", "marker", "manual", "yolo-class"),
        default="aruco",
        help=(
            "aruco/marker: ArUco selects the target. manual: click a tracked person "
            "box. yolo-class: reserve a YOLO class as a marker fallback."
        ),
    )
    parser.add_argument("--person-confidence", type=float, default=0.35)
    parser.add_argument(
        "--marker-class",
        help="YOLO class name to treat as a marker when --target-mode yolo-class.",
    )
    parser.add_argument("--marker-confidence", type=float, default=0.35)
    parser.add_argument("--aruco-dictionary", default="DICT_4X4_50")
    parser.add_argument(
        "--aruco-marker-id",
        type=int,
        help="Track only this ArUco marker ID. Defaults to accepting any single marker.",
    )
    parser.add_argument(
        "--aruco-detect-flip",
        choices=("none", "horizontal", "vertical", "both", "auto"),
        default="none",
        help="Flip frames for ArUco detection when the camera stream is mirrored.",
    )
    parser.add_argument(
        "--aruco-diagnostics",
        action="store_true",
        help="Log ArUco ids/rejected candidates and save diagnostic frames.",
    )
    parser.add_argument(
        "--aruco-diagnostics-dir",
        default="aruco_diagnostics",
        help="Directory for frames saved by --aruco-diagnostics.",
    )
    parser.add_argument(
        "--aruco-diagnostics-interval-seconds",
        type=float,
        default=1.0,
        help="Minimum interval between ArUco diagnostic logs/saved frames.",
    )
    parser.add_argument("--confirmation-seconds", type=float, default=3.0)
    parser.add_argument("--release-grace-seconds", type=float, default=3.0)
    parser.add_argument("--marker-dropout-seconds", type=float, default=0.5)
    parser.add_argument(
        "--switch-score-margin",
        type=float,
        default=12.0,
        help="New marker holder must beat the active marker score by this margin.",
    )
    parser.add_argument(
        "--switch-cooldown-seconds",
        type=float,
        default=1.0,
        help="Minimum time after handoff before another automatic switch can start.",
    )
    parser.add_argument(
        "--area-growth-penalty-threshold",
        type=float,
        default=1.8,
        help="Penalize the active track if its bbox grows above this area ratio.",
    )
    parser.add_argument(
        "--score-diagnostics",
        action="store_true",
        help="Log marker-holder candidate scores for tuning.",
    )
    parser.add_argument(
        "--score-diagnostics-interval-seconds",
        type=float,
        default=1.0,
        help="Minimum interval between candidate score diagnostic logs.",
    )
    parser.add_argument(
        "--tracking-status-diagnostics",
        action="store_true",
        help="Log target state, handoff reason, PTZ command, and recovery status.",
    )
    parser.add_argument(
        "--tracking-status-interval-seconds",
        type=float,
        default=1.0,
        help="Minimum interval between tracking status diagnostic logs.",
    )
    parser.add_argument(
        "--framing-mode",
        choices=("center", "upper-body", "full-body", "board-left", "board-right"),
        default="center",
        help=(
            "Composition policy for PTZ centering. board-left/right leaves room "
            "toward the board side."
        ),
    )
    parser.add_argument(
        "--track-buffer-seconds",
        type=float,
        default=3.0,
        help="Expected lost-track retention target used for ByteTrack diagnostics.",
    )
    parser.add_argument(
        "--diagnostics-interval-seconds",
        type=float,
        default=5.0,
        help="How often to log FPS and ByteTrack ID stability diagnostics.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening an OpenCV display window.",
    )
    parser.add_argument(
        "--reid-backend",
        choices=("none", "osnet"),
        default="none",
        help="Optional person ReID backend. osnet requires torchreid and torch.",
    )
    parser.add_argument("--reid-model-name", default="osnet_x1_0")
    parser.add_argument(
        "--reid-model-path",
        default="",
        help="Optional OSNet checkpoint path. Empty value uses torchreid defaults.",
    )
    parser.add_argument(
        "--reid-device",
        default="auto",
        help="OSNet device: auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--reid-score-interval-seconds",
        type=float,
        default=1.0,
        help="How often to log OSNet scores for visible people.",
    )
    parser.add_argument(
        "--camera-control",
        choices=("log", "visca", "qon-http"),
        default=os.getenv("MARKER_TRACKER_CAMERA_CONTROL", "log"),
        help="Use log-only camera events, VISCA over IP, or Qon HTTP PTZ commands.",
    )
    parser.add_argument(
        "--visca-host",
        default=os.getenv("MARKER_TRACKER_VISCA_HOST", ""),
        help="VISCA over IP camera host. Required when --camera-control visca.",
    )
    parser.add_argument(
        "--visca-port",
        type=int,
        default=int(os.getenv("MARKER_TRACKER_VISCA_PORT", "52381")),
    )
    parser.add_argument(
        "--visca-packet-mode",
        choices=("sony-ip", "raw"),
        default=os.getenv("MARKER_TRACKER_VISCA_PACKET_MODE", "sony-ip"),
        help="Use Sony VISCA-over-IP framing or raw VISCA UDP payloads.",
    )
    parser.add_argument("--visca-pan-speed", type=int, default=8)
    parser.add_argument("--visca-tilt-speed", type=int, default=6)
    parser.add_argument("--visca-deadzone-x", type=float, default=0.12)
    parser.add_argument("--visca-deadzone-y", type=float, default=0.12)
    parser.add_argument("--visca-min-interval-seconds", type=float, default=0.12)
    parser.add_argument("--visca-invert-pan", action="store_true")
    parser.add_argument("--visca-invert-tilt", action="store_true")
    parser.add_argument(
        "--visca-log-commands",
        action="store_true",
        help="Log VISCA bytes before sending them.",
    )
    parser.add_argument(
        "--qon-camera-url",
        default=os.getenv("MARKER_TRACKER_QON_CAMERA_URL", LOCAL_QON_CAMERA_URL),
        help="Qon camera origin or host, for example http://CAMERA_IP. Required for --camera-control qon-http.",
    )
    parser.add_argument("--qon-username", default=os.getenv("MARKER_TRACKER_QON_USERNAME"))
    parser.add_argument("--qon-password", default=os.getenv("MARKER_TRACKER_QON_PASSWORD"))
    parser.add_argument("--qon-password-env", default=os.getenv("MARKER_TRACKER_QON_PASSWORD_ENV"))
    parser.add_argument(
        "--qon-auth-mode",
        choices=("none", "basic", "digest"),
        default=os.getenv("MARKER_TRACKER_QON_AUTH_MODE", "basic"),
    )
    parser.add_argument("--qon-min-speed", type=int, default=int(os.getenv("MARKER_TRACKER_QON_MIN_SPEED", "2")))
    parser.add_argument("--qon-max-speed", type=int, default=int(os.getenv("MARKER_TRACKER_QON_MAX_SPEED", "10")))
    parser.add_argument("--qon-deadzone", type=float, default=float(os.getenv("MARKER_TRACKER_QON_DEADZONE", "0.12")))
    parser.add_argument(
        "--qon-command-ttl-seconds",
        type=float,
        default=float(os.getenv("MARKER_TRACKER_QON_COMMAND_TTL_SECONDS", "0.35")),
    )
    parser.add_argument(
        "--qon-timeout-seconds",
        type=float,
        default=float(os.getenv("MARKER_TRACKER_QON_TIMEOUT_SECONDS", "1.0")),
    )
    parser.add_argument(
        "--no-qon-configure-supervisor-actuator",
        action="store_true",
        help="Skip Qon common.track/auto PTZ/debug initialization before sending PTZ commands.",
    )
    parser.add_argument(
        "--no-qon-async-commands",
        action="store_true",
        help="Send Qon PTZ commands on the main loop instead of using the background worker.",
    )
    parser.add_argument(
        "--qon-log-commands",
        action="store_true",
        help="Log Qon HTTP PTZ and camera-parameter requests and responses.",
    )
    parser.add_argument(
        "--qon-close-join-timeout-seconds",
        type=float,
        default=float(os.getenv("MARKER_TRACKER_QON_CLOSE_JOIN_TIMEOUT_SECONDS", "1.0")),
    )
    args = parser.parse_args(argv)
    provided_dests = _provided_option_dests(parser, sys.argv[1:] if argv is None else argv)
    return _apply_runtime_profile(args, provided_dests)


def _provided_option_dests(
    parser: argparse.ArgumentParser, argv: Sequence[str]
) -> set[str]:
    option_dests = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_dests[option] = action.dest

    provided = set()
    for token in argv:
        option = token.split("=", 1)[0]
        dest = option_dests.get(option)
        if dest is not None:
            provided.add(dest)
    return provided


def _apply_runtime_profile(
    args: argparse.Namespace, provided_dests: set[str]
) -> argparse.Namespace:
    if args.runtime_profile is None:
        return args
    for dest, value in RUNTIME_PROFILES[args.runtime_profile].items():
        if dest not in provided_dests:
            setattr(args, dest, value)
    return args


def run(args: argparse.Namespace) -> None:
    _configure_runtime_cache()
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "Install runtime dependencies first: pip install -r requirements.txt"
        ) from exc
    YOLO = None
    if args.person_source == "yolo":
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit(
                "Install runtime dependencies first: pip install -r requirements.txt"
            ) from exc

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not args.video_file and not args.rtsp_url:
        raise SystemExit(
            "Provide --video-file, provide --rtsp-url, set MARKER_TRACKER_RTSP_URL, "
            "or create marker_tracker/local_settings.py."
        )
    if args.person_source == "fixture" and args.target_mode == "yolo-class":
        raise SystemExit("--person-source fixture cannot be used with --target-mode yolo-class")
    if args.no_display and args.target_mode == "manual":
        raise SystemExit("--no-display cannot be used with --target-mode manual")

    capture = _create_frame_capture(cv2, args)
    if not capture.is_opened():
        source = args.video_file if args.video_file else _redact_url(args.rtsp_url)
        raise SystemExit(f"Could not open video source: {source}")

    model = YOLO(args.model, task="detect") if YOLO is not None else None
    _warn_if_model_imgsz_mismatch(args.model, args.imgsz)
    person_source = _create_person_source(args)
    target_mode = "aruco" if args.target_mode == "marker" else args.target_mode
    target_source = _create_target_source(cv2, args, target_mode)
    tracker = MarkerTracker(
        confirmation_seconds=args.confirmation_seconds,
        release_grace_seconds=args.release_grace_seconds,
        marker_dropout_seconds=args.marker_dropout_seconds,
        switch_score_margin=args.switch_score_margin,
        switch_cooldown_seconds=args.switch_cooldown_seconds,
        area_growth_penalty_threshold=args.area_growth_penalty_threshold,
    )
    score_diagnostics = _create_score_diagnostics(args)
    status_diagnostics = _create_status_diagnostics(args)
    framing = FramingPolicy(args.framing_mode)
    reidentifier = _create_reidentifier(args)
    diagnostics = ByteTrackDiagnostics(
        expected_buffer_seconds=args.track_buffer_seconds,
        log_interval_seconds=args.diagnostics_interval_seconds,
    )
    camera = _create_camera_adapter(args)
    previous_frame_at = None
    fps = 0.0
    frame_sequence = 0
    window_name = "ArUco marker PTZ prototype"
    display_enabled = not args.no_display
    if display_enabled:
        cv2.namedWindow(window_name)
        if target_source.needs_mouse:
            cv2.setMouseCallback(window_name, target_source.on_mouse)
    inference_confidence = (
        args.confidence
        if args.confidence is not None
        else args.person_confidence
    )

    try:
        while True:
            ok, frame, frame_sequence = capture.read_latest(frame_sequence)
            if not ok:
                logging.info("Video source ended or frame read failed")
                break

            result, detections = person_source.update(
                frame,
                model,
                inference_confidence,
                _resolve_tracker_config(args),
            )
            current_frame_at = time.monotonic()
            if previous_frame_at is not None:
                current_fps = 1.0 / max(current_frame_at - previous_frame_at, 1e-9)
                fps = current_fps if fps == 0.0 else (0.9 * fps + 0.1 * current_fps)
            previous_frame_at = current_frame_at
            source_result = target_source.update(frame, result, detections)
            detections.extend(source_result.detections)
            requested_track_id = source_result.requested_track_id
            decision = tracker.update(
                detections,
                time.monotonic(),
                requested_track_id=requested_track_id,
            )
            if decision.emit_handoff and decision.target_box is not None:
                camera.handoff(decision.target_box)
                reidentifier.remember(decision.target_box, frame)
            if decision.emit_release:
                camera.release()
            camera_target = (
                None
                if decision.emit_release or decision.state in {"confirming", "switching"}
                else framing.camera_target(decision.target_box, frame.shape)
            )
            camera.update(camera_target, frame.shape)
            reidentifier.update(detections, frame, decision.target_track_id)
            diagnostics.update(detections, fps, decision.target_track_id)
            score_diagnostics.update(decision)
            status_diagnostics.update(
                decision,
                detections,
                fps,
                camera.status(),
                args.framing_mode,
            )

            if display_enabled:
                _draw(
                    cv2,
                    frame,
                    detections,
                    decision,
                    decision.target_box,
                    camera_target,
                    fps,
                    target_mode,
                    args.framing_mode,
                    camera.status(),
                )
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        camera.close()
        capture.release()
        if display_enabled:
            cv2.destroyAllWindows()


def _to_detections(
    names: dict[int, str],
    boxes: Sequence,
    person_confidence: float = 0.35,
) -> list[Detection]:
    detections = []
    for box in boxes:
        class_id = int(box.cls[0].item())
        label = names[class_id]
        if label != "person":
            continue
        confidence = float(box.conf[0].item())
        if confidence < person_confidence:
            continue
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        track_id = _box_track_id(box)
        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                box=BoundingBox(x1, y1, x2, y2),
                track_id=track_id,
            )
        )
    return detections


def _to_marker_detections(
    names: dict[int, str],
    boxes: Sequence,
    marker_class: str,
    marker_confidence: float,
) -> list[Detection]:
    detections = []
    for box in boxes:
        class_id = int(box.cls[0].item())
        label = names[class_id]
        if label != marker_class:
            continue
        confidence = float(box.conf[0].item())
        if confidence < marker_confidence:
            continue
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        detections.append(
            Detection(
                label="marker",
                confidence=confidence,
                box=BoundingBox(x1, y1, x2, y2),
                track_id=_box_track_id(box),
            )
        )
    return detections


def _box_track_id(box) -> int | None:
    track_id = getattr(box, "id", None)
    if track_id is None:
        return None
    if hasattr(track_id, "tolist"):
        values = track_id.tolist()
        if isinstance(values, list):
            if not values:
                return None
            return int(values[0])
        return int(values)
    return int(track_id)


class PersonSource(Protocol):
    def update(
        self,
        frame,
        model,
        inference_confidence: float,
        bytetrack_config: str,
    ):
        ...


def _create_person_source(args: argparse.Namespace) -> PersonSource:
    if args.person_source == "yolo":
        if args.person_tracker == "simple-iou":
            return PredictIouPersonSource(args)
        return YoloPersonSource(args)
    if args.person_source == "fixture":
        return FixturePersonSource()
    raise SystemExit(f"Unknown person source: {args.person_source}")


def _resolve_tracker_config(args: argparse.Namespace) -> str:
    if args.tracker_config:
        return args.tracker_config
    if args.tracker_preset == "botsort":
        return "botsort.yaml"
    return args.bytetrack_config


def _warn_if_model_imgsz_mismatch(model_path: str, runtime_imgsz: int) -> None:
    exported_imgsz = _read_exported_imgsz(model_path)
    if exported_imgsz is None:
        return
    expected = tuple(int(value) for value in exported_imgsz)
    runtime = (runtime_imgsz, runtime_imgsz)
    if expected != runtime:
        logging.warning(
            "Runtime imgsz=%sx%s does not match exported model imgsz=%sx%s for %s. "
            "For NCNN models, prefer matching --imgsz to the export size or re-export "
            "the model at the desired size.",
            runtime[0],
            runtime[1],
            expected[0],
            expected[1],
            model_path,
        )


def _read_exported_imgsz(model_path: str) -> tuple[int, int] | None:
    metadata_path = Path(model_path) / "metadata.yaml"
    if not metadata_path.exists():
        return None
    try:
        import yaml
    except ImportError:
        return _read_exported_imgsz_without_yaml(metadata_path)
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = yaml.safe_load(handle) or {}
    except OSError as exc:
        logging.warning("Failed to read model metadata %s: %s", metadata_path, exc)
        return None
    imgsz = metadata.get("imgsz") if isinstance(metadata, dict) else None
    return _normalize_exported_imgsz(imgsz)


def _read_exported_imgsz_without_yaml(metadata_path: Path) -> tuple[int, int] | None:
    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logging.warning("Failed to read model metadata %s: %s", metadata_path, exc)
        return None
    for index, line in enumerate(lines):
        if line.strip() != "imgsz:":
            continue
        values = []
        for child in lines[index + 1 : index + 3]:
            stripped = child.strip()
            if not stripped.startswith("- "):
                break
            try:
                values.append(int(stripped[2:]))
            except ValueError:
                return None
        return _normalize_exported_imgsz(values)
    return None


def _normalize_exported_imgsz(imgsz) -> tuple[int, int] | None:
    if isinstance(imgsz, int):
        return (imgsz, imgsz)
    if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
        try:
            return (int(imgsz[0]), int(imgsz[1]))
        except (TypeError, ValueError):
            return None
    return None


class YoloPersonSource:
    def __init__(self, args: argparse.Namespace) -> None:
        self._person_confidence = (
            args.confidence if args.confidence is not None else args.person_confidence
        )
        self._imgsz = args.imgsz

    def update(
        self,
        frame,
        model,
        inference_confidence: float,
        bytetrack_config: str,
    ):
        result = model.track(
            frame,
            conf=inference_confidence,
            imgsz=self._imgsz,
            verbose=False,
            persist=True,
            tracker=bytetrack_config,
        )[0]
        return result, _to_detections(
            result.names,
            result.boxes,
            person_confidence=self._person_confidence,
        )


class PredictIouPersonSource:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.simple_iou_threshold < 0 or args.simple_iou_threshold > 1:
            raise ValueError("--simple-iou-threshold must be between 0 and 1")
        if args.simple_iou_max_lost_frames < 0:
            raise ValueError("--simple-iou-max-lost-frames must be non-negative")
        self._person_confidence = (
            args.confidence if args.confidence is not None else args.person_confidence
        )
        self._imgsz = args.imgsz
        self._tracker = SimpleIouTracker(
            match_threshold=args.simple_iou_threshold,
            max_lost_frames=args.simple_iou_max_lost_frames,
        )

    def update(
        self,
        frame,
        model,
        inference_confidence: float,
        bytetrack_config: str,
    ):
        del bytetrack_config
        result = model.predict(
            frame,
            conf=inference_confidence,
            imgsz=self._imgsz,
            verbose=False,
        )[0]
        detections = _to_detections(
            result.names,
            result.boxes,
            person_confidence=self._person_confidence,
        )
        return result, self._tracker.update(detections)


@dataclass
class _IouTrack:
    track_id: int
    box: BoundingBox
    lost_frames: int = 0


class SimpleIouTracker:
    def __init__(self, match_threshold: float = 0.3, max_lost_frames: int = 15) -> None:
        self._match_threshold = match_threshold
        self._max_lost_frames = max_lost_frames
        self._next_track_id = 1
        self._tracks: list[_IouTrack] = []

    def update(self, detections: Sequence[Detection]) -> list[Detection]:
        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_detections = set(range(len(detections)))
        candidates = sorted(
            (
                (_iou(track.box, detection.box), track_index, detection_index)
                for track_index, track in enumerate(self._tracks)
                for detection_index, detection in enumerate(detections)
            ),
            reverse=True,
        )
        for score, track_index, detection_index in candidates:
            if score < self._match_threshold:
                break
            if (
                track_index not in unmatched_tracks
                or detection_index not in unmatched_detections
            ):
                continue
            matches.append((track_index, detection_index))
            unmatched_tracks.remove(track_index)
            unmatched_detections.remove(detection_index)

        visible: list[Detection] = []
        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            detection = detections[detection_index]
            track.box = detection.box
            track.lost_frames = 0
            visible.append(_detection_with_track_id(detection, track.track_id))

        for detection_index in sorted(unmatched_detections):
            detection = detections[detection_index]
            track = _IouTrack(self._next_track_id, detection.box)
            self._next_track_id += 1
            self._tracks.append(track)
            visible.append(_detection_with_track_id(detection, track.track_id))

        for track_index in unmatched_tracks:
            self._tracks[track_index].lost_frames += 1
        self._tracks = [
            track for track in self._tracks if track.lost_frames <= self._max_lost_frames
        ]
        return visible


def _detection_with_track_id(detection: Detection, track_id: int) -> Detection:
    return Detection(
        detection.label,
        detection.confidence,
        detection.box,
        track_id=track_id,
    )


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first.area + second.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


class FixturePersonSource:
    """Fixed two-person detections for deterministic marker-transfer videos."""

    def update(
        self,
        frame,
        model,
        inference_confidence: float,
        bytetrack_config: str,
    ):
        del model, inference_confidence, bytetrack_config
        height, width = frame.shape[:2]
        left = BoundingBox(
            0.08 * width,
            0.18 * height,
            0.42 * width,
            0.92 * height,
        )
        right = BoundingBox(
            0.58 * width,
            0.18 * height,
            0.92 * width,
            0.92 * height,
        )
        return None, [
            Detection("person", 1.0, left, track_id=1),
            Detection("person", 1.0, right, track_id=2),
        ]


def _create_aruco_detector(cv2, dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        raise SystemExit(
            "OpenCV ArUco support is unavailable. Install opencv-contrib-python."
        )
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise SystemExit(f"Unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    parameters = cv2.aruco.DetectorParameters_create()
    return dictionary, parameters


def _detect_aruco_markers(
    cv2,
    detector,
    frame,
    marker_id: int | None,
    flip_mode: str = "none",
    diagnostics=None,
) -> list[Detection]:
    frame_height, frame_width = frame.shape[:2]
    attempts = _aruco_detection_attempts(cv2, frame, flip_mode)
    last_ids = []
    last_rejected = []
    last_flip_mode = "none"
    for attempt_frame, attempt_flip_mode in attempts:
        corners, ids, rejected = _detect_aruco_candidates(
            cv2, detector, attempt_frame
        )
        last_ids = [] if ids is None else ids.flatten().tolist()
        last_rejected = rejected
        last_flip_mode = attempt_flip_mode
        if ids is None:
            continue

        detections = []
        for marker_corners, detected_id in zip(corners, ids.flatten()):
            if marker_id is not None and int(detected_id) != marker_id:
                continue
            points = _unflip_aruco_points(
                marker_corners.reshape(-1, 2),
                frame_width,
                frame_height,
                attempt_flip_mode,
            )
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            detections.append(
                Detection(
                    label="marker",
                    confidence=1.0,
                    box=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        if detections:
            if diagnostics is not None:
                diagnostics.update(
                    cv2,
                    frame,
                    ids.flatten().tolist(),
                    rejected,
                    detections,
                    attempt_flip_mode,
                )
            return detections

    if diagnostics is not None:
        diagnostics.update(
            cv2,
            frame,
            last_ids,
            last_rejected,
            [],
            last_flip_mode,
        )
    return []


def _detect_aruco_candidates(cv2, detector, frame):
    if hasattr(cv2.aruco, "ArucoDetector"):
        return detector.detectMarkers(frame)
    dictionary, parameters = detector
    return cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)


def _aruco_detection_attempts(cv2, frame, flip_mode: str):
    if flip_mode == "none":
        return [(frame, "none")]
    if flip_mode == "horizontal":
        return [(cv2.flip(frame, 1), "horizontal")]
    if flip_mode == "vertical":
        return [(cv2.flip(frame, 0), "vertical")]
    if flip_mode == "both":
        return [(cv2.flip(frame, -1), "both")]
    if flip_mode == "auto":
        return [
            (frame, "none"),
            (cv2.flip(frame, 1), "horizontal"),
            (cv2.flip(frame, 0), "vertical"),
            (cv2.flip(frame, -1), "both"),
        ]
    raise ValueError(f"Unknown ArUco flip mode: {flip_mode}")


def _unflip_aruco_points(points, frame_width: int, frame_height: int, flip_mode: str):
    points = points.copy()
    if flip_mode in ("horizontal", "both"):
        points[:, 0] = frame_width - 1 - points[:, 0]
    if flip_mode in ("vertical", "both"):
        points[:, 1] = frame_height - 1 - points[:, 1]
    return points


@dataclass(frozen=True)
class TargetSourceResult:
    detections: list[Detection]
    requested_track_id: int | None = None


class TargetSignalSource(Protocol):
    @property
    def needs_mouse(self) -> bool: ...

    def on_mouse(self, event, x: int, y: int, flags, param) -> None: ...

    def update(self, frame, result, detections: list[Detection]) -> TargetSourceResult: ...


class ArucoTargetSource:
    def __init__(
        self,
        cv2,
        dictionary_name: str,
        marker_id: int | None,
        flip_mode: str,
        diagnostics=None,
    ) -> None:
        self.needs_mouse = False
        self._cv2 = cv2
        self._detector = _create_aruco_detector(cv2, dictionary_name)
        self._marker_id = marker_id
        self._flip_mode = flip_mode
        self._diagnostics = diagnostics

    def on_mouse(self, event, x: int, y: int, flags, param) -> None:
        del event, x, y, flags, param

    def update(self, frame, result, detections: list[Detection]) -> TargetSourceResult:
        del result, detections
        return TargetSourceResult(
            _detect_aruco_markers(
                self._cv2,
                self._detector,
                frame,
                marker_id=self._marker_id,
                flip_mode=self._flip_mode,
                diagnostics=self._diagnostics,
            )
        )


class YoloClassTargetSource:
    def __init__(self, marker_class: str, marker_confidence: float) -> None:
        if not marker_class:
            raise SystemExit("--marker-class is required with --target-mode yolo-class")
        self.needs_mouse = False
        self._marker_class = marker_class
        self._marker_confidence = marker_confidence

    def on_mouse(self, event, x: int, y: int, flags, param) -> None:
        del event, x, y, flags, param

    def update(self, frame, result, detections: list[Detection]) -> TargetSourceResult:
        del frame, detections
        return TargetSourceResult(
            _to_marker_detections(
                result.names,
                result.boxes,
                self._marker_class,
                self._marker_confidence,
            )
        )


class ManualTargetSource:
    needs_mouse = True

    def __init__(self, select_event: int) -> None:
        self._selector = ClickTargetSelector(select_event)

    def on_mouse(self, event, x: int, y: int, flags, param) -> None:
        self._selector.on_mouse(event, x, y, flags, param)

    def update(self, frame, result, detections: list[Detection]) -> TargetSourceResult:
        del frame, result
        return TargetSourceResult(
            [],
            self._selector.consume_requested_track_id(detections),
        )


def _create_target_source(cv2, args: argparse.Namespace, target_mode: str):
    if target_mode == "aruco":
        diagnostics = None
        if args.aruco_diagnostics:
            diagnostics = ArucoDiagnostics(
                Path(args.aruco_diagnostics_dir),
                args.aruco_diagnostics_interval_seconds,
            )
        return ArucoTargetSource(
            cv2,
            args.aruco_dictionary,
            args.aruco_marker_id,
            args.aruco_detect_flip,
            diagnostics,
        )
    if target_mode == "manual":
        return ManualTargetSource(cv2.EVENT_LBUTTONDOWN)
    if target_mode == "yolo-class":
        return YoloClassTargetSource(args.marker_class, args.marker_confidence)
    raise SystemExit(f"Unknown target mode: {target_mode}")


class ArucoDiagnostics:
    def __init__(
        self,
        output_dir: Path,
        interval_seconds: float,
        now_fn=time.monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._output_dir = output_dir
        self._interval_seconds = interval_seconds
        self._now_fn = now_fn
        self._last_log_at: float | None = None
        self._frame_index = 0
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        cv2,
        frame,
        ids,
        rejected,
        detections: Sequence[Detection],
        flip_mode: str,
    ) -> None:
        now = self._now_fn()
        if (
            self._last_log_at is not None
            and now - self._last_log_at < self._interval_seconds
        ):
            return
        self._last_log_at = now
        rejected_count = len(rejected) if rejected is not None else 0
        logging.info(
            "ArUco diagnostics: ids=%s accepted=%s rejected_candidates=%s "
            "flip=%s frame=%sx%s",
            ids,
            len(detections),
            rejected_count,
            flip_mode,
            frame.shape[1],
            frame.shape[0],
        )
        self._save_frame(cv2, frame, detections, ids, rejected_count, flip_mode)

    def _save_frame(
        self,
        cv2,
        frame,
        detections: Sequence[Detection],
        ids,
        rejected_count: int,
        flip_mode: str,
    ) -> None:
        annotated = frame.copy()
        for detection in detections:
            box = detection.box
            cv2.rectangle(
                annotated,
                (int(box.x1), int(box.y1)),
                (int(box.x2), int(box.y2)),
                (0, 255, 255),
                3,
            )
        cv2.putText(
            annotated,
            f"aruco ids: {ids} rejected: {rejected_count} flip: {flip_mode}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        output_path = self._output_dir / f"aruco_diag_{self._frame_index:04d}.jpg"
        self._frame_index += 1
        if not cv2.imwrite(str(output_path), annotated):
            logging.warning("Failed to save ArUco diagnostic frame: %s", output_path)


def _create_score_diagnostics(args: argparse.Namespace):
    if not args.score_diagnostics:
        return NoopScoreDiagnostics()
    return ScoreDiagnostics(args.score_diagnostics_interval_seconds)


def _create_status_diagnostics(args: argparse.Namespace):
    if not args.tracking_status_diagnostics:
        return NoopTrackingStatusDiagnostics()
    return TrackingStatusDiagnostics(args.tracking_status_interval_seconds)


def _create_camera_adapter(args: argparse.Namespace):
    if args.camera_control == "log":
        return LoggingCameraAdapter()
    if args.camera_control == "visca":
        if not args.visca_host:
            raise SystemExit("--visca-host is required when --camera-control visca")
        return ViscaOverIpCameraAdapter(
            host=args.visca_host,
            port=args.visca_port,
            packet_mode=args.visca_packet_mode,
            pan_speed=args.visca_pan_speed,
            tilt_speed=args.visca_tilt_speed,
            deadzone_x=args.visca_deadzone_x,
            deadzone_y=args.visca_deadzone_y,
            invert_pan=args.visca_invert_pan,
            invert_tilt=args.visca_invert_tilt,
            min_interval_seconds=args.visca_min_interval_seconds,
            log_commands=args.visca_log_commands,
        )
    if args.camera_control == "qon-http":
        qon_camera_url = args.qon_camera_url or _qon_camera_url_from_rtsp(args.rtsp_url)
        if not qon_camera_url:
            raise SystemExit(
                "--qon-camera-url is required when --camera-control qon-http "
                "unless the RTSP URL contains a camera host"
            )
        qon_password = os.getenv(args.qon_password_env) if args.qon_password_env else args.qon_password
        return QonHttpCameraAdapter(
            camera_url=qon_camera_url,
            username=args.qon_username,
            password=qon_password,
            auth_mode=args.qon_auth_mode,
            min_speed=args.qon_min_speed,
            max_speed=args.qon_max_speed,
            deadzone=args.qon_deadzone,
            command_ttl_seconds=args.qon_command_ttl_seconds,
            timeout_seconds=args.qon_timeout_seconds,
            configure_supervisor_actuator=not args.no_qon_configure_supervisor_actuator,
            async_commands=not args.no_qon_async_commands,
            log_commands=args.qon_log_commands,
            close_join_timeout_seconds=args.qon_close_join_timeout_seconds,
        )
    raise SystemExit(f"Unknown camera control mode: {args.camera_control}")


def _qon_camera_url_from_rtsp(rtsp_url: str | None) -> str:
    if not rtsp_url:
        return ""
    host = urlsplit(rtsp_url).hostname
    if not host:
        return ""
    return f"http://{host}"


class NoopScoreDiagnostics:
    def update(self, decision: HandoffDecision) -> None:
        del decision


class ScoreDiagnostics:
    def __init__(
        self,
        interval_seconds: float,
        now_fn=time.monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._interval_seconds = interval_seconds
        self._now_fn = now_fn
        self._last_log_at: float | None = None

    def update(self, decision: HandoffDecision) -> None:
        if not decision.candidate_scores:
            return
        now = self._now_fn()
        if (
            self._last_log_at is not None
            and now - self._last_log_at < self._interval_seconds
        ):
            return
        self._last_log_at = now
        scores = []
        for candidate in decision.candidate_scores:
            growth = (
                "none"
                if candidate.area_growth_ratio is None
                else f"{candidate.area_growth_ratio:.2f}"
            )
            scores.append(
                "track={track} score={score:.1f} active={active} "
                "area={area:.0f} growth={growth}".format(
                    track=candidate.track_id,
                    score=candidate.score,
                    active=candidate.is_active,
                    area=candidate.box.area,
                    growth=growth,
                )
            )
        logging.info(
            "Candidate scores: state=%s target_track_id=%s %s",
            decision.state,
            decision.target_track_id,
            " | ".join(scores),
        )


class NoopTrackingStatusDiagnostics:
    def update(
        self,
        decision: HandoffDecision,
        detections: Sequence[Detection],
        fps: float,
        ptz_status: str,
        framing_mode: str,
    ) -> None:
        del decision, detections, fps, ptz_status, framing_mode


class TrackingStatusDiagnostics:
    def __init__(
        self,
        interval_seconds: float,
        now_fn=time.monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._interval_seconds = interval_seconds
        self._now_fn = now_fn
        self._last_log_at: float | None = None

    def update(
        self,
        decision: HandoffDecision,
        detections: Sequence[Detection],
        fps: float,
        ptz_status: str,
        framing_mode: str,
    ) -> None:
        now = self._now_fn()
        if (
            self._last_log_at is not None
            and now - self._last_log_at < self._interval_seconds
        ):
            return
        self._last_log_at = now
        people_count = sum(1 for detection in detections if detection.label == "person")
        marker_count = sum(1 for detection in detections if detection.label == "marker")
        best_score = (
            max(score.score for score in decision.candidate_scores)
            if decision.candidate_scores
            else None
        )
        logging.info(
            "Tracking status: fps=%.1f state=%s target_id=%s reason=%s "
            "people=%s markers=%s marker_score=%s ptz=%s framing=%s recovery=%s",
            fps,
            decision.state,
            decision.target_track_id,
            decision.reason or "none",
            people_count,
            marker_count,
            "none" if best_score is None else f"{best_score:.1f}",
            ptz_status,
            framing_mode,
            _recovery_status(decision, marker_count, people_count),
        )


class FramingPolicy:
    def __init__(self, mode: str) -> None:
        if mode not in {"center", "upper-body", "full-body", "board-left", "board-right"}:
            raise ValueError(f"Unknown framing mode: {mode}")
        self.mode = mode

    def camera_target(
        self,
        target_box: BoundingBox | None,
        frame_shape: tuple[int, ...],
    ) -> BoundingBox | None:
        if target_box is None or len(frame_shape) < 2:
            return target_box
        height, width = frame_shape[:2]
        if width <= 0 or height <= 0:
            return target_box
        center_x, center_y = target_box.center
        box_width = max(1.0, target_box.x2 - target_box.x1)
        box_height = max(1.0, target_box.y2 - target_box.y1)
        if self.mode == "center":
            return target_box
        if self.mode == "upper-body":
            center_y = target_box.y1 + box_height * 0.35
        elif self.mode == "full-body":
            center_y = target_box.y1 + box_height * 0.48
        elif self.mode == "board-left":
            center_x = target_box.x1 + box_width * 0.68
        elif self.mode == "board-right":
            center_x = target_box.x1 + box_width * 0.32
        return _box_with_center(target_box, center_x, center_y)


def _box_with_center(box: BoundingBox, center_x: float, center_y: float) -> BoundingBox:
    half_width = (box.x2 - box.x1) / 2
    half_height = (box.y2 - box.y1) / 2
    return BoundingBox(
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def _recovery_status(
    decision: HandoffDecision,
    marker_count: int,
    people_count: int,
) -> str:
    if decision.emit_release:
        return "released_to_default"
    if decision.state == "grace":
        return "holding_last_target"
    if decision.state == "confirming" and marker_count == 0:
        return "marker_dropout_hold"
    if decision.state == "switching":
        return "handoff_confirmation"
    if decision.target_track_id is None and people_count > 1:
        return "waiting_for_marker"
    if marker_count > 1:
        return "multiple_markers_ignored"
    if marker_count == 0 and decision.target_track_id is None:
        return "no_marker"
    return "stable"


class ReIdentifier(Protocol):
    def remember(self, target_box: BoundingBox, frame) -> None: ...

    def update(
        self,
        detections: Sequence[Detection],
        frame,
        target_track_id: int | None,
    ) -> None: ...


def _configure_runtime_cache() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    for path in (
        cache_root / "matplotlib",
        cache_root / "ultralytics",
        cache_root / "torch",
    ):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_root / "ultralytics"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))


def _create_reidentifier(args: argparse.Namespace) -> ReIdentifier:
    if args.reid_backend == "none":
        return NoopReIdentifier()
    if args.reid_backend == "osnet":
        return OSNetReIdentifier(
            model_name=args.reid_model_name,
            model_path=args.reid_model_path,
            device=args.reid_device,
            score_interval_seconds=args.reid_score_interval_seconds,
        )
    raise SystemExit(f"Unknown ReID backend: {args.reid_backend}")


class NoopReIdentifier:
    """Disabled ReID backend."""

    def remember(self, target_box: BoundingBox, frame) -> None:
        del target_box, frame

    def update(
        self,
        detections: Sequence[Detection],
        frame,
        target_track_id: int | None,
    ) -> None:
        del detections, frame, target_track_id


class OSNetReIdentifier:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str,
        score_interval_seconds: float,
        now_fn=time.monotonic,
    ) -> None:
        if score_interval_seconds <= 0:
            raise ValueError("score_interval_seconds must be positive")
        os.environ.setdefault(
            "TORCH_HOME",
            str(Path(__file__).resolve().parents[1] / ".cache" / "torch"),
        )
        try:
            try:
                from torchreid.utils import FeatureExtractor
            except ModuleNotFoundError:
                from torchreid.reid.utils import FeatureExtractor
        except ImportError as exc:
            raise SystemExit(
                "OSNet ReID requires torchreid and torch. Install the optional "
                "ReID dependencies before running --reid-backend osnet."
            ) from exc
        resolved_device = _resolve_reid_device(device)
        self._extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=resolved_device,
        )
        self._score_interval_seconds = score_interval_seconds
        self._now_fn = now_fn
        self._reference_feature = None
        self._last_score_log_at: float | None = None
        logging.info(
            "OSNet ReID enabled: model=%s device=%s checkpoint=%s",
            model_name,
            resolved_device,
            model_path or "torchreid-default",
        )

    def remember(self, target_box: BoundingBox, frame) -> None:
        crop = _crop_frame(frame, target_box)
        if crop is None:
            logging.warning("OSNet reference crop is empty; ReID reference not updated")
            return
        self._reference_feature = self._extract(crop)
        self._last_score_log_at = None
        logging.info("OSNet reference feature updated")

    def update(
        self,
        detections: Sequence[Detection],
        frame,
        target_track_id: int | None,
    ) -> None:
        if self._reference_feature is None:
            return
        now = self._now_fn()
        if (
            self._last_score_log_at is not None
            and now - self._last_score_log_at < self._score_interval_seconds
        ):
            return
        people = [detection for detection in detections if detection.label == "person"]
        scored = []
        for person in people:
            crop = _crop_frame(frame, person.box)
            if crop is None:
                continue
            score = _cosine_similarity(self._reference_feature, self._extract(crop))
            scored.append((score, person))
        if not scored:
            self._last_score_log_at = now
            return
        best_score, best_person = max(scored, key=lambda item: item[0])
        best_other = [
            item
            for item in scored
            if item[1].track_id is not None and item[1].track_id != target_track_id
        ]
        if best_other:
            other_score, other_person = max(best_other, key=lambda item: item[0])
            logging.info(
                "OSNet scores: best_track=%s best_score=%.3f "
                "best_non_target_track=%s best_non_target_score=%.3f",
                best_person.track_id,
                best_score,
                other_person.track_id,
                other_score,
            )
        else:
            logging.info(
                "OSNet scores: best_track=%s best_score=%.3f",
                best_person.track_id,
                best_score,
            )
        self._last_score_log_at = now

    def _extract(self, crop):
        return self._extractor(_to_reid_input(crop))


def _resolve_reid_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _crop_frame(frame, box: BoundingBox):
    height, width = frame.shape[:2]
    x1 = max(0, min(width, int(round(box.x1))))
    y1 = max(0, min(height, int(round(box.y1))))
    x2 = max(0, min(width, int(round(box.x2))))
    y2 = max(0, min(height, int(round(box.y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _to_reid_input(crop):
    if len(getattr(crop, "shape", ())) == 3 and crop.shape[2] >= 3:
        return crop[:, :, :3][:, :, ::-1].copy()
    return crop


def _cosine_similarity(reference_feature, candidate_feature) -> float:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for OSNet cosine similarity") from exc
    reference = reference_feature.flatten().float()
    candidate = candidate_feature.flatten().float()
    return float(torch.nn.functional.cosine_similarity(reference, candidate, dim=0))


class ByteTrackDiagnostics:
    def __init__(
        self,
        expected_buffer_seconds: float,
        log_interval_seconds: float,
        now_fn=time.monotonic,
    ) -> None:
        if expected_buffer_seconds <= 0:
            raise ValueError("expected_buffer_seconds must be positive")
        if log_interval_seconds <= 0:
            raise ValueError("log_interval_seconds must be positive")
        self.expected_buffer_seconds = expected_buffer_seconds
        self.log_interval_seconds = log_interval_seconds
        self._now_fn = now_fn
        self._last_log_at: float | None = None
        self._known_track_ids: set[int] = set()
        self._last_target_track_id: int | None = None

    def update(
        self,
        detections: Sequence[Detection],
        fps: float,
        target_track_id: int | None,
    ) -> None:
        now = self._now_fn()
        if self._last_log_at is None:
            self._last_log_at = now
            self._remember_tracks(detections, target_track_id)
            return
        self._remember_tracks(detections, target_track_id)
        if now - self._last_log_at < self.log_interval_seconds:
            return
        track_ids = {
            detection.track_id
            for detection in detections
            if detection.label == "person" and detection.track_id is not None
        }
        recommended_buffer = (
            int(round(fps * self.expected_buffer_seconds)) if fps > 0 else 0
        )
        logging.info(
            "ByteTrack diagnostics: fps=%.1f visible_tracks=%s known_tracks=%s "
            "target_track_id=%s recommended_track_buffer=%s",
            fps,
            len(track_ids),
            len(self._known_track_ids),
            target_track_id,
            recommended_buffer or "unknown",
        )
        self._last_log_at = now

    def _remember_tracks(
        self, detections: Sequence[Detection], target_track_id: int | None
    ) -> None:
        for detection in detections:
            if detection.label == "person" and detection.track_id is not None:
                self._known_track_ids.add(detection.track_id)
        if (
            self._last_target_track_id is not None
            and target_track_id is not None
            and target_track_id != self._last_target_track_id
        ):
            logging.info(
                "Target track changed: previous=%s current=%s",
                self._last_target_track_id,
                target_track_id,
            )
        if target_track_id is not None:
            self._last_target_track_id = target_track_id


def _create_frame_capture(cv2, args: argparse.Namespace):
    if args.video_file:
        return VideoFileCapture(cv2, args.video_file)
    return LatestFrameCapture(cv2, args.rtsp_url)


class VideoFileCapture:
    """Read local video files sequentially at their encoded FPS."""

    def __init__(self, cv2, path: str) -> None:
        self._capture = cv2.VideoCapture(path)
        fps = self._capture.get(cv2.CAP_PROP_FPS) if self._capture.isOpened() else 0
        self._frame_interval = 1.0 / fps if fps and fps > 0 else 1.0 / 30.0
        self._sequence = 0
        self._next_frame_at: float | None = None

    def is_opened(self) -> bool:
        return self._capture.isOpened()

    def read_latest(self, previous_sequence: int, timeout: float = 3.0):
        del previous_sequence, timeout
        if self._next_frame_at is not None:
            sleep_for = self._next_frame_at - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
        ok, frame = self._capture.read()
        if not ok:
            return False, None, self._sequence
        self._sequence += 1
        self._next_frame_at = time.monotonic() + self._frame_interval
        return True, frame, self._sequence

    def release(self) -> None:
        self._capture.release()


class LatestFrameCapture:
    """Continuously drain RTSP frames so inference receives the freshest frame."""

    def __init__(self, cv2, url: str) -> None:
        self._capture = cv2.VideoCapture(url)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._condition = threading.Condition()
        self._frame = None
        self._sequence = 0
        self._stopped = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        if self._capture.isOpened():
            self._thread.start()

    def is_opened(self) -> bool:
        return self._capture.isOpened()

    def read_latest(self, previous_sequence: int, timeout: float = 3.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._sequence <= previous_sequence and not self._stopped:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None, previous_sequence
                self._condition.wait(remaining)
            if self._frame is None:
                return False, None, previous_sequence
            return True, self._frame, self._sequence

    def release(self) -> None:
        self._stopped = True
        self._capture.release()
        with self._condition:
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _reader(self) -> None:
        while not self._stopped:
            ok, frame = self._capture.read()
            if not ok:
                break
            with self._condition:
                self._frame = frame
                self._sequence += 1
                self._condition.notify_all()
        self._stopped = True
        with self._condition:
            self._condition.notify_all()


class ClickTargetSelector:
    def __init__(self, select_event: int) -> None:
        self._select_event = select_event
        self._pending_click: tuple[int, int] | None = None

    def on_mouse(self, event, x: int, y: int, flags, param) -> None:
        del flags, param
        if event == self._select_event:
            self._pending_click = (x, y)

    def consume_requested_track_id(self, detections) -> int | None:
        if self._pending_click is None:
            return None
        click = self._pending_click
        self._pending_click = None
        candidates = [
            detection
            for detection in detections
            if detection.label == "person"
            and detection.track_id is not None
            and detection.box.contains(click)
        ]
        if not candidates:
            logging.info("Manual target click did not hit a tracked person")
            return None
        selected = min(candidates, key=lambda detection: detection.box.area)
        logging.info("Manual target selected: track_id=%s", selected.track_id)
        return selected.track_id


def _draw(
    cv2,
    frame,
    detections,
    decision: HandoffDecision,
    target_box,
    camera_target,
    fps: float,
    target_mode: str,
    framing_mode: str,
    ptz_status: str,
) -> None:
    colors = {"person": (255, 160, 0), "marker": (0, 255, 255)}
    for detection in detections:
        box = detection.box
        cv2.rectangle(
            frame,
            (int(box.x1), int(box.y1)),
            (int(box.x2), int(box.y2)),
            colors[detection.label],
            2,
        )
        label = f"{detection.label} {detection.confidence:.2f}"
        if detection.track_id is not None:
            label += f" id:{detection.track_id}"
        cv2.putText(
            frame,
            label,
            (int(box.x1), max(20, int(box.y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colors[detection.label],
            2,
        )
    if target_box is not None:
        cv2.rectangle(
            frame,
            (int(target_box.x1), int(target_box.y1)),
            (int(target_box.x2), int(target_box.y2)),
            (0, 255, 0),
            3,
        )
    if camera_target is not None:
        center_x, center_y = camera_target.center
        cv2.drawMarker(
            frame,
            (int(center_x), int(center_y)),
            (0, 220, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
    cv2.putText(
        frame,
        f"mode: {target_mode} state: {decision.state}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    if decision.target_track_id is not None:
        cv2.putText(
            frame,
            f"target id: {decision.target_track_id}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    reason = decision.reason or "none"
    cv2.putText(
        frame,
        f"reason: {reason}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"PTZ: {ptz_status} framing: {framing_mode}",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )


def _redact_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.username is None:
        return url
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit(
        (
            parsed.scheme,
            f"{parsed.username}:***@{hostname}{port}",
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
