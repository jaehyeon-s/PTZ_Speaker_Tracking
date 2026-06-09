"""Verify Qon internal auto tracking using registered presenter appearance."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

from src.camera.qon_control import QonTrackingControl, QonVelocityPTZController
from src.camera.region_provider import CenterCropRegionProvider, IdentityMatchedRegionProvider, JsonlRegionProvider
from src.recovery.candidate_detector import YoloRecoveryDetector
from src.reid.appearance import build_re_identifier
from src.reid.verifier import PresenterVerifier, VerifierConfig
from src.tracking.gesture_detector import GestureFallbackDetector
from src.tracking.marker_detector import ArucoMarkerDetector
from src.tracking.person_detector import build_person_detector
from src.tracking.target_selector import point_in_bbox
from src.tracking.target_state import PersonDetection
from src.vision.capture import VideoSource
from src.vision.models import BBox, TrackingState, VerificationResult


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Optional MVP config file")
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config(pre_args.config) if pre_args.config else {}

    parser = argparse.ArgumentParser(description="Qon internal tracking Re-ID verifier MVP")
    parser.add_argument("--config", default=pre_args.config, help="Optional MVP config file")
    parser.add_argument(
        "--source",
        required=config_value(config, "source") is None,
        default=config_value(config, "source"),
        help="Validation video or externally supplied camera RTSP URL",
    )
    parser.add_argument(
        "--region-mode",
        choices=("center", "jsonl", "identity"),
        default=config_value(config, "region_mode", "region.mode", default="center"),
    )
    parser.add_argument(
        "--bbox-jsonl",
        default=config_value(config, "bbox_jsonl", "region.bbox_jsonl"),
        help="Captured internal bbox samples for region-mode=jsonl",
    )
    parser.add_argument(
        "--registration-mode",
        choices=("observed", "marker", "gesture", "marker-or-gesture"),
        default=config_value(config, "registration_mode", "registration.mode", default="observed"),
        help="How to register the presenter's Re-ID reference",
    )
    parser.add_argument(
        "--target-marker-id",
        type=int,
        default=config_value(config, "target_marker_id", "registration.target_marker_id"),
        help="Only accept this ArUco marker id",
    )
    parser.add_argument(
        "--detector",
        choices=("hog", "opencv-yolo", "ncnn", "manual"),
        default=config_value(config, "detector", "detector.backend", default="ncnn"),
        help="Person detector used for registration and identity observations",
    )
    parser.add_argument("--yolo-model", default=config_value(config, "yolo_model", "detector.yolo_model"), help="Optional YOLO ONNX model path")
    parser.add_argument("--ncnn-param", default=config_value(config, "ncnn_param", "detector.ncnn_param", default="models/yolo26n_ncnn_model/model.ncnn.param"), help="NCNN YOLO .param path")
    parser.add_argument("--ncnn-bin", default=config_value(config, "ncnn_bin", "detector.ncnn_bin", default="models/yolo26n_ncnn_model/model.ncnn.bin"), help="NCNN YOLO .bin path")
    parser.add_argument(
        "--ncnn-input-size",
        type=int,
        default=config_value(config, "ncnn_input_size", "detector.ncnn_input_size", default=640),
        help="Square YOLO model input size; stream2 frame size remains 640x360",
    )
    parser.add_argument("--person-box", default=config_value(config, "person_box", "detector.person_box"), help="Manual detector bbox x,y,w,h")
    parser.add_argument("--conf-threshold", type=float, default=config_value(config, "conf_threshold", "detector.conf_threshold", default=0.35))
    parser.add_argument("--nms-threshold", type=float, default=config_value(config, "nms_threshold", "detector.nms_threshold", default=0.45))
    parser.add_argument("--debug-detector", action="store_true", default=bool(config_value(config, "debug_detector", "detector.debug", default=False)))
    parser.add_argument("--center-width-ratio", type=float, default=config_value(config, "center_width_ratio", "region.center_width_ratio", default=0.42))
    parser.add_argument("--center-height-ratio", type=float, default=config_value(config, "center_height_ratio", "region.center_height_ratio", default=0.86))
    parser.add_argument(
        "--register-on-start",
        action="store_true",
        default=bool(config_value(config, "register_on_start", "registration.register_on_start", default=False)),
        help="Register the first observed region as presenter",
    )
    parser.add_argument(
        "--register-bbox",
        default=config_value(config, "register_bbox", "registration.register_bbox"),
        help="Registration bbox x1,y1,x2,y2; otherwise uses observed region",
    )
    parser.add_argument(
        "--registration-reference",
        choices=("observed", "selected"),
        default=config_value(config, "registration_reference", "registration.reference", default="observed"),
        help="Use the camera-observed region or selected marker/gesture person bbox as the Re-ID reference",
    )
    parser.add_argument(
        "--registration-timeout-frames",
        type=int,
        default=config_value(config, "registration_timeout_frames", "registration.timeout_frames", default=90),
        help="Maximum frames to wait for marker/gesture registration when --register-on-start is used",
    )
    parser.add_argument("--verify-every-frames", type=int, default=config_value(config, "verify_every_frames", "reid.verify_every_frames", default=30))
    parser.add_argument("--reid-backend", choices=("hsv", "onnx"), default=config_value(config, "reid_backend", "reid.backend", default="hsv"))
    parser.add_argument("--reid-model", default=config_value(config, "reid_model", "reid.model"), help="OSNet-style ONNX Re-ID model path")
    parser.add_argument("--reid-threshold", type=float, default=config_value(config, "reid_threshold", "reid.threshold", default=0.68))
    parser.add_argument("--mismatch-limit", type=int, default=config_value(config, "mismatch_limit", "reid.mismatch_limit", default=2))
    parser.add_argument(
        "--identity-weak-min-score",
        type=float,
        default=config_value(config, "identity_weak_min_score", "reid.identity_weak_min_score", default=0.25),
        help="Minimum weak score for trusting identity observations; failed frames hold instead of mismatching",
    )
    parser.add_argument(
        "--identity-margin",
        type=float,
        default=config_value(config, "identity_margin", "reid.identity_margin", default=0.08),
        help="Required best-vs-second Re-ID margin before trusting the observed identity bbox",
    )
    parser.add_argument(
        "--center-margin",
        type=float,
        default=config_value(config, "center_margin", "reid.center_margin", default=0.12),
        help="Required best-vs-center score margin before counting a camera mis-track frame",
    )
    parser.add_argument(
        "--identity-center-dead-zone",
        type=float,
        default=config_value(config, "identity_center_dead_zone", "reid.identity_center_dead_zone", default=0.18),
        help="Normalized center distance tolerated before target is considered away from camera center",
    )
    parser.add_argument(
        "--identity-hold-limit",
        type=int,
        default=config_value(config, "identity_hold_limit", "reid.identity_hold_limit", default=30),
        help="Consecutive untrusted identity frames before declaring LOST",
    )
    parser.add_argument(
        "--marker-positive",
        action="store_true",
        default=bool(config_value(config, "marker_positive", "registration.marker_positive", default=False)),
        help="Treat the target marker visible inside the observed region as a strong verification signal",
    )
    parser.add_argument("--camera-url", default=config_value(config, "camera_url", "camera.url"), help="Qon camera origin, for example http://192.168.11.88")
    parser.add_argument("--camera-username", default=config_value(config, "camera_username", "camera.username"))
    parser.add_argument("--camera-password-env", default=config_value(config, "camera_password_env", "camera.password_env"))
    parser.add_argument("--camera-auth-mode", choices=("none", "basic", "digest"), default=config_value(config, "camera_auth_mode", "camera.auth_mode", default="digest"))
    parser.add_argument(
        "--control-camera",
        action="store_true",
        default=bool(config_value(config, "control_camera", "camera.control_camera", default=False)),
        help="Enable Qon tracking on registration and disable it on confirmed mismatch",
    )
    parser.add_argument(
        "--enable-camera-tracking-on-start",
        action="store_true",
        default=bool(config_value(config, "enable_camera_tracking_on_start", "camera.enable_tracking_on_start", default=False)),
        help="Enable Qon internal tracking as soon as the verifier starts",
    )
    parser.add_argument(
        "--configure-supervisor-actuator",
        action="store_true",
        default=bool(config_value(config, "configure_supervisor_actuator", "camera.configure_supervisor_actuator", default=False)),
        help="Apply Qon common.track=0, auto_zoom=0, auto_tilt=0, debug_mode=3 with write+VISCA+read-back",
    )
    parser.add_argument(
        "--ptz-follow-target",
        action="store_true",
        default=bool(config_value(config, "ptz_follow_target", "ptz.follow_target", default=False)),
        help="Drive Qon PTZ directly toward the identity-selected bbox",
    )
    parser.add_argument("--ptz-dead-zone", type=float, default=config_value(config, "ptz_dead_zone", "ptz.dead_zone", default=0.12))
    parser.add_argument("--ptz-min-speed", type=int, default=config_value(config, "ptz_min_speed", "ptz.min_speed", default=2))
    parser.add_argument("--ptz-max-speed", type=int, default=config_value(config, "ptz_max_speed", "ptz.max_speed", default=10))
    parser.add_argument(
        "--recovery-action",
        choices=("none", "stop", "home", "zoomout"),
        default=config_value(config, "recovery_action", "recovery.action", default="home"),
        help="Camera action after confirmed mismatch when --control-camera is enabled",
    )
    parser.add_argument("--recovery-model", default=config_value(config, "recovery_model", "recovery.model"), help="Optional YOLO model used only after confirmed mismatch")
    parser.add_argument("--recovery-imgsz", type=int, default=config_value(config, "recovery_imgsz", "recovery.imgsz", default=416))
    parser.add_argument("--recovery-confidence", type=float, default=config_value(config, "recovery_confidence", "recovery.confidence", default=0.4))
    parser.add_argument("--device", default=config_value(config, "device"))
    parser.add_argument("--rtsp-drop-frames", type=int, default=config_value(config, "rtsp_drop_frames", "video.rtsp_drop_frames", default=0))
    parser.add_argument("--output", default=config_value(config, "output", "logging.output"), help="Write annotated MP4")
    parser.add_argument("--log-csv", default=config_value(config, "log_csv", "logging.csv"))
    parser.add_argument("--no-window", action="store_true", default=bool(config_value(config, "no_window", "display.no_window", default=False)))
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        import json

        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("--config JSON root must be an object")
        return data
    return parse_simple_yaml(text)


def parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"YAML indentation must use multiples of two spaces at line {line_number}")
        item = line.strip()
        if ":" not in item:
            raise ValueError(f"Expected key: value at line {line_number}")
        key, value = item.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_config_scalar(value)
    return root


def parse_config_scalar(value: str) -> Any:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def config_value(config: dict[str, Any], *paths: str, default=None):
    for path in paths:
        current: Any = config
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            return current
    return default


def main() -> int:
    args = parse_args()
    if args.no_window and not args.register_on_start:
        print("--no-window requires --register-on-start for unattended registration.")
        return 2
    if args.region_mode == "jsonl" and not args.bbox_jsonl:
        print("--region-mode jsonl requires --bbox-jsonl.")
        return 2
    if args.ptz_follow_target and args.region_mode != "identity":
        print("--ptz-follow-target requires --region-mode identity.")
        return 2

    import cv2

    video = VideoSource(args.source, rtsp_drop_frames=args.rtsp_drop_frames)
    if not video.is_opened():
        print("Could not open validation video source.")
        return 1

    matcher = build_re_identifier(args.reid_backend, args.reid_threshold, args.reid_model)
    provider = build_region_provider(args, matcher)
    camera_control = build_camera_control(args)
    supervisor = CameraTrackingSupervisor(camera_control, args.recovery_action)
    if args.configure_supervisor_actuator:
        supervisor.configure_actuator()
    if args.enable_camera_tracking_on_start:
        supervisor.enable_tracking("startup")
    registration_detector = build_registration_detector(args)
    marker_detector = (
        ArucoMarkerDetector()
        if args.registration_mode in ("marker", "marker-or-gesture") or args.marker_positive
        else None
    )
    gesture_detector = (
        GestureFallbackDetector()
        if args.registration_mode in ("gesture", "marker-or-gesture")
        else None
    )
    verifier = PresenterVerifier(
        matcher,
        VerifierConfig(max(1, args.mismatch_limit)),
    )
    ptz_controller = build_ptz_controller(args, camera_control)
    recovery_detector = (
        YoloRecoveryDetector(args.recovery_model, args.recovery_imgsz, args.recovery_confidence, args.device)
        if args.recovery_model
        else None
    )
    registration_bbox = parse_bbox(args.register_bbox)
    writer = None
    log_file = None
    log_writer = None
    last_result = VerificationResult(TrackingState.UNREGISTERED, None, event="REGISTER_REQUIRED")
    recovery_bbox = None
    recovery_score = 0.0

    if args.log_csv:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        log_file = Path(args.log_csv).open("w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        log_writer.writerow(
            [
                "frame",
                "state",
                "event",
                "region_source",
                "bbox",
                "score",
                "recovery_bbox",
                "recovery_score",
                "camera_action",
                "best_score",
                "center_score",
                "second_score",
                "identity_margin",
                "center_margin",
                "candidate_count",
                "hold_count",
                "mismatch_count",
                "ptz_action",
            ]
        )

    frame_index = 0
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frame_index += 1
            bbox, region_source = provider.region_for_frame(frame_index, frame)
            registered_this_frame = False
            should_verify = frame_index == 1 or frame_index % max(1, args.verify_every_frames) == 0

            people = []
            markers = []
            if registration_detector is not None and should_collect_registration_inputs(args, verifier.registered):
                people = registration_detector.detect(frame)
            if marker_detector is not None and should_collect_markers(args, verifier.registered, should_verify):
                markers = marker_detector.detect(frame)

            if args.register_on_start and not verifier.registered:
                selected_bbox = registration_bbox or select_registration_bbox(
                    args,
                    frame,
                    bbox,
                    people,
                    markers,
                    marker_detector,
                    gesture_detector,
                )
                if selected_bbox is None or not verifier.register(frame, selected_bbox):
                    if args.no_window and frame_index >= max(1, args.registration_timeout_frames):
                        print("Could not register presenter from the available frame.")
                        return 1
                else:
                    last_result = VerificationResult(
                        TrackingState.VERIFIED, selected_bbox, 1.0, "PRESENTER_REGISTERED", registration_source(args)
                    )
                    supervisor.enable_tracking("registered")
                    registered_this_frame = True

            if verifier.registered and should_verify and not registered_this_frame:
                marker_verified = (
                    args.marker_positive
                    and marker_visible_in_observed_region(markers, bbox, args.target_marker_id)
                )
                if marker_verified:
                    last_result = verifier.mark_verified(bbox, 1.0, "MARKER_VERIFIED", region_source)
                elif is_identity_provider(provider):
                    last_result = result_from_identity_observation(provider.last_observation)
                else:
                    last_result = verifier.verify(frame, bbox, region_source)
                recovery_bbox, recovery_score = None, 0.0
                if last_result.event == "PRESENTER_MISMATCH" and recovery_detector is not None:
                    candidate, recovery_score = verifier.find_presenter(frame, recovery_detector.detect(frame))
                    if candidate is not None:
                        recovery_bbox = candidate.bbox
                        last_result = VerificationResult(
                            last_result.state,
                            last_result.bbox,
                            last_result.score,
                            "RECOVERY_CANDIDATE_FOUND",
                            last_result.source,
                        )
                if last_result.event in ("PRESENTER_MISMATCH", "PRESENTER_MISTRACK_CONFIRMED"):
                    supervisor.confirm_mismatch()
                print_status(frame_index, last_result, recovery_bbox, recovery_score)

            ptz_action = ""
            if ptz_controller is not None and verifier.registered:
                observation = getattr(provider, "last_observation", None)
                if observation is not None and observation.state in (TrackingState.CAMERA_ALIGNED, TrackingState.SUSPECT):
                    ptz_action = ptz_controller.follow_bbox(observation.bbox, frame.shape)
                else:
                    ptz_controller.stop()
                    ptz_action = "ptzstop"

            if log_writer and should_verify:
                log_writer.writerow(
                    [
                        frame_index,
                        last_result.state.value,
                        last_result.event,
                        last_result.source,
                        format_bbox(last_result.bbox),
                        f"{last_result.score:.3f}",
                        format_bbox(recovery_bbox),
                        f"{recovery_score:.3f}",
                        supervisor.last_action,
                        f"{get_identity_metric(provider, 'best_score'):.3f}",
                        f"{get_identity_metric(provider, 'center_score'):.3f}",
                        f"{get_identity_metric(provider, 'second_score'):.3f}",
                        f"{get_identity_metric(provider, 'identity_margin'):.3f}",
                        f"{get_identity_metric(provider, 'center_margin'):.3f}",
                        int(get_identity_metric(provider, "candidate_count")),
                        int(get_identity_metric(provider, "hold_count")),
                        int(get_identity_metric(provider, "mismatch_count")),
                        ptz_action,
                    ]
                )
                log_file.flush()

            draw_debug(cv2, frame, last_result, recovery_bbox)
            writer = write_video(cv2, writer, args.output, frame, video.fps())
            if not args.no_window:
                cv2.imshow("Qon tracking Re-ID verifier", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r") and bbox is not None:
                    selected_bbox = registration_bbox or select_registration_bbox(
                        args,
                        frame,
                        bbox,
                        people,
                        markers,
                        marker_detector,
                        gesture_detector,
                    )
                    if selected_bbox is None:
                        continue
                    verifier.register(frame, selected_bbox)
                    last_result = VerificationResult(
                        TrackingState.VERIFIED, selected_bbox, 1.0, "PRESENTER_REGISTERED", registration_source(args)
                    )
                    supervisor.enable_tracking("manual_register")
        if args.register_on_start and args.no_window and not verifier.registered:
            print("Could not register presenter before the video source ended.")
            return 1
    finally:
        video.release()
        if writer:
            writer.release()
        if log_file:
            log_file.close()
        if ptz_controller is not None:
            ptz_controller.stop()
        if not args.no_window:
            cv2.destroyAllWindows()
    return 0


def build_region_provider(args, matcher):
    if args.region_mode == "jsonl":
        return JsonlRegionProvider(args.bbox_jsonl)
    if args.region_mode == "identity":
        detector = build_person_detector(
            detector_mode=resolve_detector_mode(args),
            yolo_model=args.yolo_model,
            manual_bbox=parse_xywh(args.person_box),
            ncnn_param=args.ncnn_param,
            ncnn_bin=args.ncnn_bin,
            ncnn_input_size=args.ncnn_input_size,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold,
            debug_detector=args.debug_detector,
        )
        return IdentityMatchedRegionProvider(
            detector,
            matcher,
            CenterCropRegionProvider(args.center_width_ratio, args.center_height_ratio),
            weak_min_score=args.identity_weak_min_score,
            identity_margin=args.identity_margin,
            center_margin=args.center_margin,
            center_dead_zone_ratio=args.identity_center_dead_zone,
            mismatch_limit=args.mismatch_limit,
            hold_limit=args.identity_hold_limit,
        )
    return CenterCropRegionProvider(args.center_width_ratio, args.center_height_ratio)


def build_camera_control(args):
    if (
        not args.control_camera
        and not args.enable_camera_tracking_on_start
        and not args.configure_supervisor_actuator
        and not args.ptz_follow_target
    ):
        return None
    if not args.camera_url:
        raise ValueError("camera control options require --camera-url")
    password = os.environ.get(args.camera_password_env) if args.camera_password_env else None
    return QonTrackingControl(
        args.camera_url,
        args.camera_username,
        password,
        auth_mode=args.camera_auth_mode,
    )


class CameraTrackingSupervisor:
    """Coordinate Qon internal tracking ownership with verifier state."""

    def __init__(self, control: QonTrackingControl | None, recovery_action: str = "stop") -> None:
        self.control = control
        self.recovery_action = recovery_action
        self.tracking_enabled = False
        self.mismatch_handled = False
        self.actuator_owned_by_supervisor = False
        self.last_action = ""

    def enable_tracking(self, reason: str) -> None:
        if self.actuator_owned_by_supervisor:
            self.last_action = f"tracking_left_off:{reason}"
            return
        if self.control is None or self.tracking_enabled:
            return
        self.control.set_presenter_mode()
        self.control.enable_auto_tracking()
        self.tracking_enabled = True
        self.mismatch_handled = False
        self.last_action = f"tracking_on:{reason}"

    def configure_actuator(self) -> None:
        if self.control is None:
            return
        self.control.set_supervisor_actuator_mode()
        self.tracking_enabled = False
        self.actuator_owned_by_supervisor = True
        self.last_action = "supervisor_actuator_configured"

    def confirm_mismatch(self) -> None:
        if self.control is None or self.mismatch_handled:
            return
        if not self.actuator_owned_by_supervisor:
            self.control.enable_zone_tracking()
        self.tracking_enabled = False
        if self.recovery_action == "stop":
            self.control.stop()
        elif self.recovery_action == "home":
            self.control.stop()
            self.control.home()
        elif self.recovery_action == "zoomout":
            self.control.stop()
            self.control.zoom_out()
            self.control.zoom_stop()
        self.mismatch_handled = True
        self.last_action = f"tracking_off:mismatch:{self.recovery_action}"


def build_ptz_controller(args, camera_control):
    if not args.ptz_follow_target:
        return None
    if camera_control is None:
        raise ValueError("--ptz-follow-target requires camera control options and --camera-url")
    return QonVelocityPTZController(
        camera_control,
        dead_zone_ratio=args.ptz_dead_zone,
        min_speed=args.ptz_min_speed,
        max_speed=args.ptz_max_speed,
    )


def should_collect_registration_inputs(args, verifier_registered: bool) -> bool:
    return (args.register_on_start and not verifier_registered) or not args.no_window


def should_collect_markers(args, verifier_registered: bool, should_verify: bool) -> bool:
    return should_collect_registration_inputs(args, verifier_registered) or (
        args.marker_positive and verifier_registered and should_verify
    )


def is_identity_provider(provider) -> bool:
    return isinstance(provider, IdentityMatchedRegionProvider)


def result_from_identity_observation(observation) -> VerificationResult:
    score = observation.best_score
    return VerificationResult(observation.state, observation.bbox, score, observation.event, observation.source)


def get_identity_metric(provider, name: str) -> float:
    observation = getattr(provider, "last_observation", None)
    if observation is None:
        return 0.0
    return float(getattr(observation, name, 0.0))


def build_registration_detector(args):
    if args.registration_mode == "observed":
        return None
    return build_person_detector(
        detector_mode=resolve_detector_mode(args),
        yolo_model=args.yolo_model,
        manual_bbox=parse_xywh(args.person_box),
        ncnn_param=args.ncnn_param,
        ncnn_bin=args.ncnn_bin,
        ncnn_input_size=args.ncnn_input_size,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        debug_detector=args.debug_detector,
    )


def resolve_detector_mode(args) -> str:
    if args.detector == "hog" and args.person_box is not None:
        return "manual"
    if args.detector == "hog" and args.yolo_model is not None:
        return "opencv-yolo"
    return args.detector


def select_registration_bbox(
    args,
    frame,
    observed_bbox: BBox | None,
    people: list[PersonDetection],
    markers,
    marker_detector,
    gesture_detector,
) -> BBox | None:
    del marker_detector
    if args.registration_mode == "observed":
        return observed_bbox
    if args.registration_mode in ("marker", "marker-or-gesture"):
        marker_bbox = select_marker_person_bbox(people, markers, args.target_marker_id)
        if marker_bbox is not None:
            return selected_reference_bbox(args, observed_bbox, marker_bbox)
    if args.registration_mode in ("gesture", "marker-or-gesture") and gesture_detector is not None:
        raised = gesture_detector.detect_raised_hand_indices(frame, people)
        person = gesture_detector.update(people, raised)
        if person is not None:
            return selected_reference_bbox(args, observed_bbox, xywh_to_xyxy(person.bbox))
    return None


def selected_reference_bbox(args, observed_bbox: BBox | None, selected_bbox: BBox) -> BBox | None:
    if args.registration_reference == "observed":
        return observed_bbox
    return selected_bbox


def select_marker_person_bbox(people: list[PersonDetection], markers, target_marker_id: int | None) -> BBox | None:
    best = None
    for marker in markers:
        if target_marker_id is not None and marker.marker_id != target_marker_id:
            continue
        for person in people:
            if point_in_bbox(marker.center, person.bbox):
                px, py = person.center
                mx, my = marker.center
                score = (px - mx) ** 2 + (py - my) ** 2
                if best is None or score < best[0]:
                    best = (score, person)
    return None if best is None else xywh_to_xyxy(best[1].bbox)


def marker_visible_in_observed_region(markers, observed_bbox: BBox | None, target_marker_id: int | None) -> bool:
    if observed_bbox is None:
        return False
    x1, y1, x2, y2 = observed_bbox
    observed_xywh = (x1, y1, x2 - x1, y2 - y1)
    for marker in markers:
        if target_marker_id is not None and marker.marker_id != target_marker_id:
            continue
        if point_in_bbox(marker.center, observed_xywh):
            return True
    return False


def xywh_to_xyxy(bbox: tuple[int, int, int, int]) -> BBox:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def registration_source(args) -> str:
    return f"registration_{args.registration_mode}"


def parse_bbox(value: str | None) -> BBox | None:
    if value is None:
        return None
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4:
        raise ValueError("--register-bbox must be x1,y1,x2,y2")
    return parts


def parse_xywh(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4:
        raise ValueError("--person-box must be x,y,w,h")
    return parts


def format_bbox(bbox: BBox | None) -> str:
    return "" if bbox is None else ",".join(str(value) for value in bbox)


def print_status(frame_index, result, recovery_bbox, recovery_score) -> None:
    message = (
        f"frame={frame_index} state={result.state.value} event={result.event} "
        f"score={result.score:.3f} source={result.source}"
    )
    if recovery_bbox is not None:
        message += f" recovery_bbox={format_bbox(recovery_bbox)} recovery_score={recovery_score:.3f}"
    print(message)


def draw_debug(cv2, frame, result, recovery_bbox) -> None:
    if result.bbox is not None:
        color = (0, 200, 0) if result.state is TrackingState.VERIFIED else (0, 0, 255)
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if recovery_bbox is not None:
        x1, y1, x2, y2 = recovery_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    text = f"{result.state.value} {result.event} score={result.score:.2f} src={result.source}"
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)


def write_video(cv2, writer, output: str | None, frame, source_fps: float):
    if not output:
        return writer
    if writer is None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        fps = source_fps if source_fps > 0 else 15.0
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output video: {output}")
    writer.write(frame)
    return writer


if __name__ == "__main__":
    raise SystemExit(main())
