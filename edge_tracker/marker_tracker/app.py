from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from .camera import LoggingCameraAdapter, QonHttpCameraAdapter, ViscaOverIpCameraAdapter
from .core import BoundingBox, Detection, HandoffDecision, MarkerTracker
from .detection import DecoupledPersonSource, HailoAppearanceEncoder, OnnxAppearanceEncoder
from .identity import NoopSpeakerIdentityManager, SpeakerIdentityManager
from .zoom import ZoomController, ZoomFovTable, ZoomState

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
        "inference_device": "cpu",
        "bytetrack_config": "bytetrack-pi.yaml",
        "aruco_marker_id": 0,
        # A mirrored marker cannot decode, so on an unmirrored stream the two
        # mirrored attempts of "auto" only cost time (room_01: 0 of 223 sightings).
        "aruco_detect_flip": "unmirrored",
        "confirmation_seconds": 2.0,
        "release_grace_seconds": 3.0,
        # [room_01 수정] marker_dropout_seconds 0.75 -> 1.5.
        # 이유: 640x360 영상에서 마커가 약 10px이라 0.5~1.2초 간격으로만 검출됨.
        # 0.75초로는 확인 타이머가 매번 끊겨 등록이 되지 않았음.
        # A small, distant marker is decoded only now and then.
        "marker_dropout_seconds": 1.5,
        "switch_score_margin": 12.0,
        "switch_cooldown_seconds": 1.0,
        "area_growth_penalty_threshold": 1.8,
        "tracker_input_confidence": 0.1,
        "recovery_confirmation_seconds": 0.12,
        "recovery_min_iou": 0.2,
        "recovery_max_center_distance": 0.75,
        "recovery_score_margin": 0.25,
        "speaker_resolution_min_frames": 8,
        "speaker_resolution_min_seconds": 0.50,
        "speaker_resolution_min_margin": 0.06,
        "speaker_resolution_winner_ratio": 0.85,
        "speaker_resolution_search_after_seconds": 1.2,
        "speaker_resolution_search_interval_frames": 2,
        "speaker_single_resolution_min_frames": 5,
        "speaker_single_resolution_min_seconds": 0.35,
        "speaker_single_resolution_min_similarity": 0.50,
        "speaker_coast_max_seconds": 1.2,
        # [room_01 수정] 신규 기본값 2개.
        # - speaker_resolution_timeout_seconds=10: 근거 없이 10초 탐색하면 대상 해제
        #   (기존에는 identity_resolving 상태가 끝없이 유지됨).
        # - provisional_speed_limit=4: 미확정 후보를 따라갈 때 속도 상한 2 -> 4
        #   (걷는 사람은 약 107px/s, 속도 2는 약 45px/s라 따라가지 못했음).
        "speaker_resolution_timeout_seconds": 10.0,
        "provisional_speed_limit": 4,
        "tracking_status_diagnostics": True,
        "track_continuity_diagnostics": True,
        "performance_diagnostics": True,
        "framing_mode": "upper-body",
        "track_buffer_seconds": 3.75,
        "camera_control": "qon-http",
        "qon_min_speed": 2,
        "qon_max_speed": 10,
        "qon_deadzone": 0.10,
        "qon_stop_deadzone": 0.07,
        "qon_stop_debounce_seconds": 0.08,
        "qon_lost_target_hold_seconds": 0.35,
        "qon_command_min_interval_seconds": 0.08,
        # [room_01 수정] PTZ 응답 조정 4개 (구현은 camera.py QonHttpCameraAdapter).
        # - full_speed_error 1.0 -> 0.5: 오차가 화면 반폭의 50%면 최대 속도.
        # - uniform_error_scale: tilt 오차도 화면 반폭 기준으로 계산(상하 떨림 원인 제거).
        # - axis_switch_ratio 1.5: pan/tilt 축이 프레임마다 번갈아 바뀌는 것 방지.
        # - response_delay_seconds 0.5: 카메라+영상 지연만큼 미리 감속.
        "qon_full_speed_error": 0.5,
        "qon_uniform_error_scale": True,
        "qon_axis_switch_ratio": 1.5,
        "qon_response_delay_seconds": 0.5,
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
    parser.add_argument(
        "--record-input-video",
        help=(
            "Save the exact pre-overlay frames processed by the pipeline. "
            "Use an .mp4 path such as test_video/raw/crossing_01.mp4."
        ),
    )
    parser.add_argument("--record-input-fps", type=float, default=15.0)
    parser.add_argument(
        "--identity-trace-file",
        help="Write frame-by-frame detections and identity decisions as JSONL.",
    )
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument(
        "--inference-device",
        default="auto",
        help=(
            "Ultralytics inference device (auto, cpu, 0, cuda:0, etc.). "
            "The pi-qon-http profile forces cpu to avoid accidental CUDA "
            "provider selection for exported detector/ReID models."
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        help=(
            "YOLO inference image size. Omit this to use the model/default "
            "Ultralytics behavior, matching the original yolo26n.pt ByteTrack path."
        ),
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
        choices=("ultralytics", "simple-iou", "decoupled"),
        default="ultralytics",
        help=(
            "Assign person IDs with Ultralytics model.track(), with a simple "
            "in-process IoU tracker over model.predict() detections, or with "
            "'decoupled': a separate detector stage feeding the same tracker "
            "YAML, so the detector can run on an NPU (see --hailo-hef)."
        ),
    )
    parser.add_argument(
        "--overlay",
        choices=("simple", "debug"),
        default="simple",
        help=(
            "What the display window shows. simple: boxes without labels, the "
            "speaker in thick green, the marker, the camera aim cross, FPS and "
            "one state word. debug: labels, IDs, reason and PTZ status as before."
        ),
    )
    parser.add_argument(
        "--serial-target-signal",
        action="store_true",
        help=(
            "Detect the marker after the person source instead of in parallel "
            "with it (the earlier behaviour; results are the same)."
        ),
    )
    parser.add_argument(
        "--hailo-hef",
        help=(
            "Hailo HEF person detector to run on the NPU instead of --model, "
            "for example yolo26n_h8.hef. Requires --person-tracker decoupled."
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
        "--tracker-input-confidence",
        type=float,
        help=(
            "Confidence floor passed into YOLO tracking. This can be lower than "
            "--person-confidence so ByteTrack can associate weak detections."
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=("aruco", "marker", "manual"),
        default="aruco",
        help=(
            "aruco/marker: ArUco selects the target. manual: click a tracked person "
            "box."
        ),
    )
    parser.add_argument("--person-confidence", type=float, default=0.35)
    parser.add_argument("--aruco-dictionary", default="DICT_4X4_50")
    parser.add_argument(
        "--aruco-marker-id",
        type=int,
        help="Track only this ArUco marker ID. Defaults to accepting any single marker.",
    )
    parser.add_argument(
        "--aruco-detect-flip",
        choices=("none", "horizontal", "vertical", "both", "auto", "unmirrored"),
        default="none",
        help=(
            "Flip frames for ArUco detection when the camera stream is mirrored. "
            "'auto' tries all four orientations on every frame without a marker; "
            "'unmirrored' tries only the frame and its 180-degree rotation, which "
            "is every orientation that can decode when the stream is not mirrored."
        ),
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
    parser.add_argument(
        "--target-low-confidence",
        type=float,
        default=None,
        help=(
            "Also keep person detections down to this confidence when they are the "
            "current target's track or lie where the target was last seen. "
            "Default: off (only --person-confidence applies)."
        ),
    )
    parser.add_argument(
        "--occlusion-head-anchor",
        action="store_true",
        help=(
            "When the target box suddenly becomes much shorter (body hidden behind "
            "furniture), keep framing from the top of the box using the height seen before."
        ),
    )
    parser.add_argument("--occlusion-height-ratio", type=float, default=0.6)
    parser.add_argument("--occlusion-hold-seconds", type=float, default=5.0)
    parser.add_argument(
        "--recovery-min-area-ratio",
        type=float,
        default=None,
        help="Smallest box area, relative to the last target box, accepted as the same person. Default 0.2.",
    )
    parser.add_argument("--marker-dropout-seconds", type=float, default=0.5)
    parser.add_argument(
        "--recovery-confirmation-seconds",
        type=float,
        default=0.12,
        help="How long a nearby replacement track ID must remain stable before recovery.",
    )
    parser.add_argument(
        "--recovery-min-iou",
        type=float,
        default=0.2,
        help="Minimum last-box IoU for target ID recovery.",
    )
    parser.add_argument(
        "--recovery-max-center-distance",
        type=float,
        default=0.75,
        help="Maximum normalized center distance for target ID recovery.",
    )
    parser.add_argument(
        "--recovery-score-margin",
        type=float,
        default=0.25,
        help="Required score lead over the second-best ID recovery candidate.",
    )
    parser.add_argument(
        "--speaker-identity-backend",
        choices=("none", "yolo-reid"),
        default="none",
        help=(
            "Maintain a stable logical speaker identity with selective appearance "
            "inference. yolo-reid is intended for botsort-pi.yaml (tracker ReID off)."
        ),
    )
    parser.add_argument("--speaker-reid-model", default="yolo26n-reid.onnx")
    parser.add_argument(
        "--speaker-reid-hef",
        help=(
            "Hailo ReID HEF to run speaker ReID on the NPU instead of "
            "--speaker-reid-model, for example repvgg_a0_person_reid_512_h8.hef. "
            "Requires --speaker-reid-similarity-scale and -offset."
        ),
    )
    parser.add_argument(
        "--speaker-reid-onnx",
        help=(
            "Float ONNX of the Hailo ReID model run on the CPU, for example "
            "repvgg_a0_person_reid_512.onnx, so a computer without the NPU uses "
            "the Pi's ReID model. Same similarity options as --speaker-reid-hef."
        ),
    )
    parser.add_argument(
        "--speaker-reid-similarity-scale",
        type=float,
        help=(
            "Multiply the encoder's cosine similarity by this before the identity "
            "thresholds, which were tuned on yolo26n-reid (default 1)."
        ),
    )
    parser.add_argument(
        "--speaker-reid-similarity-offset",
        type=float,
        help="Add this to the scaled similarity (default 0).",
    )
    parser.add_argument(
        "--speaker-reid-device",
        default="cpu",
        help="Device for selective speaker ReID. Defaults to cpu independently of detector device.",
    )
    parser.add_argument("--speaker-gallery-size", type=int, default=12)
    parser.add_argument(
        "--speaker-gallery-interval-seconds", type=float, default=0.35
    )
    parser.add_argument(
        "--speaker-gallery-novelty-similarity", type=float, default=0.98
    )
    parser.add_argument("--speaker-gallery-max-overlap", type=float, default=0.25)
    parser.add_argument("--speaker-reid-trigger-overlap", type=float, default=0.20)
    parser.add_argument("--speaker-reid-burst-frames", type=int, default=4)
    parser.add_argument("--speaker-reid-min-similarity", type=float, default=0.55)
    parser.add_argument("--speaker-reid-margin", type=float, default=0.08)
    parser.add_argument("--speaker-reid-weight", type=float, default=2.0)
    parser.add_argument("--speaker-resolution-min-frames", type=int, default=6)
    parser.add_argument("--speaker-resolution-min-seconds", type=float, default=0.50)
    parser.add_argument("--speaker-resolution-min-similarity", type=float, default=0.35)
    parser.add_argument("--speaker-resolution-min-margin", type=float, default=0.02)
    parser.add_argument("--speaker-resolution-winner-ratio", type=float, default=0.75)
    parser.add_argument("--speaker-crossing-recent-seconds", type=float, default=1.0)
    parser.add_argument("--speaker-negative-gallery-weight", type=float, default=0.5)
    parser.add_argument("--speaker-resolution-close-distance", type=float, default=1.75)
    parser.add_argument("--speaker-resolution-exit-distance", type=float, default=1.9)
    parser.add_argument("--speaker-resolution-exit-frames", type=int, default=3)
    parser.add_argument("--speaker-resolution-cooldown-seconds", type=float, default=0.75)
    parser.add_argument("--speaker-coast-max-seconds", type=float, default=0.8)
    parser.add_argument(
        "--speaker-single-resolution-min-similarity", type=float, default=0.45
    )
    parser.add_argument(
        "--speaker-single-resolution-min-frames", type=int, default=4
    )
    parser.add_argument(
        "--speaker-single-resolution-min-seconds", type=float, default=0.25
    )
    parser.add_argument(
        "--speaker-single-path-max-distance", type=float, default=1.25
    )
    parser.add_argument(
        "--speaker-single-path-max-gap-seconds", type=float, default=0.5
    )
    parser.add_argument(
        "--speaker-resolution-search-after-seconds", type=float, default=2.0
    )
    parser.add_argument(
        "--speaker-resolution-search-interval-frames", type=int, default=3
    )
    # [room_01 수정] 신규 옵션 5개 (재식별/재등록/임시 추종 속도).
    # 아래 5개는 모두 room_01 로그 분석 후 추가했으며 pi-qon-http 프로파일이 기본값을 덮어씀.
    parser.add_argument(
        "--speaker-resolution-strong-margin",
        type=float,
        default=0.15,
        help=(
            "Similarity lead over every other person that lets a candidate with "
            "a new track ID be verified below --speaker-reid-min-similarity."
        ),
    )
    parser.add_argument(
        "--speaker-resolution-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Release the speaker after searching this long without seeing the "
            "target or a credible candidate. 0 searches indefinitely."
        ),
    )
    parser.add_argument(
        "--reacquire-marker-observations",
        type=int,
        default=3,
        help="Marker sightings on one person that re-register a lost target.",
    )
    parser.add_argument(
        "--reacquire-marker-window-seconds",
        type=float,
        default=3.0,
        help="Time window in which those marker sightings must fall.",
    )
    parser.add_argument(
        "--reacquire-marker-min-seconds",
        type=float,
        default=0.5,
        help=(
            "Shortest time the marker must stay with one new holder before a "
            "lost speaker is re-registered from it."
        ),
    )
    parser.add_argument(
        "--provisional-speed-limit",
        type=int,
        default=2,
        help="PTZ speed cap while following an unverified identity candidate.",
    )
    parser.add_argument("--speaker-anchor-gallery-size", type=int, default=6)
    parser.add_argument("--speaker-gallery-freeze-seconds", type=float, default=1.0)
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
        "--track-continuity-diagnostics",
        action="store_true",
        help=(
            "Log target-ID loss, visible replacement candidates, reappearance, "
            "and release events without changing tracker decisions."
        ),
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
        "--performance-diagnostics",
        action="store_true",
        help="Log average and p95 processing time for major pipeline stages.",
    )
    parser.add_argument(
        "--performance-diagnostics-interval-seconds",
        type=float,
        default=5.0,
        help="How often to report pipeline stage timing.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening an OpenCV display window.",
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
        "--qon-stop-deadzone",
        type=float,
        default=float(os.getenv("MARKER_TRACKER_QON_STOP_DEADZONE", "0.08")),
    )
    parser.add_argument(
        "--qon-stop-debounce-seconds",
        type=float,
        default=float(
            os.getenv("MARKER_TRACKER_QON_STOP_DEBOUNCE_SECONDS", "0.12")
        ),
    )
    parser.add_argument(
        "--qon-lost-target-hold-seconds",
        type=float,
        default=float(
            os.getenv("MARKER_TRACKER_QON_LOST_TARGET_HOLD_SECONDS", "0.20")
        ),
    )
    parser.add_argument(
        "--qon-command-min-interval-seconds",
        type=float,
        default=float(
            os.getenv("MARKER_TRACKER_QON_COMMAND_MIN_INTERVAL_SECONDS", "0.12")
        ),
    )
    # [room_01 수정] 신규 옵션 6개 (Qon PTZ 응답 조정). 기본값은 기존 동작과 동일.
    parser.add_argument(
        "--qon-full-speed-error",
        type=float,
        default=1.0,
        help="Normalized target offset at which the PTZ reaches --qon-max-speed.",
    )
    parser.add_argument(
        "--qon-uniform-error-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Measure pan and tilt offsets in the same pixels (half frame width).",
    )
    parser.add_argument(
        "--qon-axis-switch-ratio",
        type=float,
        default=1.0,
        help="How much larger the other axis offset must be before the PTZ changes axis.",
    )
    parser.add_argument(
        "--qon-response-delay-seconds",
        type=float,
        default=0.0,
        help=(
            "Camera plus video delay to compensate when braking. 0 disables "
            "the compensation."
        ),
    )
    parser.add_argument(
        "--qon-pan-rate-per-speed",
        type=float,
        default=0.075,
        help="Picture travel in half frame widths per second per pan speed step.",
    )
    parser.add_argument(
        "--qon-tilt-rate-per-speed",
        type=float,
        default=0.05,
        help="Picture travel in half frame widths per second per tilt speed step.",
    )
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
        "--qon-speed-map",
        default="",
        help=(
            "Optional error->level steps, e.g. '0.10:2,0.14:3,0.22:4'. Error is the "
            "target offset as a fraction of half the frame. Replaces the linear ramp."
        ),
    )
    parser.add_argument(
        "--qon-predict-target-seconds",
        type=float,
        default=0.0,
        help="Aim at where the target will be after this delay (needs --qon-level-speeds). 0 = off.",
    )
    parser.add_argument(
        "--qon-stop-lookahead-seconds",
        type=float,
        default=0.0,
        help=(
            "Subtract camera motion already commanded within this delay but not yet "
            "visible, so the stop is issued early (needs --qon-level-speeds). 0 = off."
        ),
    )
    parser.add_argument("--qon-tilt-deadzone", type=float, default=None,
                        help="Two-axis modes: deadzone for tilt; default = --qon-deadzone.")
    parser.add_argument("--qon-tilt-stop-deadzone", type=float, default=None)
    parser.add_argument("--qon-predict-axes", choices=("both", "pan"), default="both",
                        help="Axes that use --qon-predict-target-seconds.")
    parser.add_argument("--qon-axis-reverse-hold-seconds", type=float, default=0.0,
                        help="Two-axis modes: an axis waits this long before reversing. 0 = off.")
    parser.add_argument(
        "--zoom-control",
        action="store_true",
        help="Adjust zoom to keep the upper body at --framing-size-target. Needs --zoom-fov-table.",
    )
    parser.add_argument("--zoom-fov-table", default="",
                        help="Zoom position to horizontal FOV pairs, e.g. '0:60.7,5543:20,16384:2.3'.")
    parser.add_argument("--zoom-start-position", type=float, default=None,
                        help="Zoom position at start when it cannot be read from the camera.")
    parser.add_argument("--zoom-rates", default="",
                        help="Zoom position units per second for zoom speeds, e.g. '1:640,4:1940'. Used when the position cannot be read.")
    parser.add_argument("--zoom-inquiry-port", type=int, default=0,
                        help="VISCA-over-TCP port for reading the zoom position (e.g. 5678). 0 = estimate from commands.")
    parser.add_argument("--framing-size-target", type=float, default=0.55,
                        help="Wanted upper-body height as a fraction of the frame height.")
    parser.add_argument("--framing-size-tolerance", type=float, default=0.15)
    parser.add_argument("--zoom-upper-body-fraction", type=float, default=0.5,
                        help="Share of the full standing height counted as upper body.")
    parser.add_argument("--zoom-delay-seconds", type=float, default=0.4)
    parser.add_argument("--zoom-response-seconds", type=float, default=0.6)
    parser.add_argument("--zoom-cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--zoom-lost-hfov", type=float, default=40.0)
    parser.add_argument("--zoom-ceiling-factor", type=float, default=1.5,
                        help="k in: minimum hfov = 2 * (k * target speed * (delay + response) + target half width).")
    parser.add_argument("--zoom-in-max-screen-speed", type=float, default=0.10,
                        help="Zoom in only while the target crosses less than this share of the frame width per second.")
    parser.add_argument("--zoom-lost-search", action="store_true",
                        help="After a loss: widen until a person is seen, then steer to and narrow onto that person until the marker can be confirmed.")
    parser.add_argument("--zoom-search-marker-pixels", type=float, default=14.0)
    parser.add_argument("--zoom-max-burst-seconds", type=float, default=0.0,
                        help="Longest continuous zoom before the tracker sends zoomstop by itself (e.g. 0.5). 0 = no limit.")
    parser.add_argument("--zoom-burst-rest-seconds", type=float, default=0.3,
                        help="Pause after a zoom burst was cut off, so the new zoom position is seen before zooming on.")
    parser.add_argument("--zoom-mode", choices=("continuous", "absolute"), default="continuous",
                        help="continuous: zoomin/zoomout until zoomstop. absolute: zoomto the wanted field of view (camera must support zoomto).")
    parser.add_argument("--zoom-safety-margin", type=float, default=1.3,
                        help="Edge and confidence zoom-outs may widen to this multiple of the minimum hfov from the speed limit (or the size-based hfov, or --zoom-safety-step), and never pass --zoom-lost-hfov.")
    parser.add_argument("--zoom-safety-step", type=float, default=1.5,
                        help="One edge or confidence zoom-out may also widen to this multiple of the hfov it started from.")
    parser.add_argument("--disable-bystander-rules", action="store_true",
                        help="Comparison only: turn off the pi_unity_04/05 bystander rules (quick re-registration "
                             "needs no minimum time, ID recovery may pick bystanders and non-holders).")
    parser.add_argument("--holder-geometry", action="store_true",
                        help="When the marker centre lies in several people's boxes, credit the person it sits on at chest "
                             "height and whose box height fits the marker size, instead of the smaller box; leave close "
                             "calls unattributed until the boxes separate.")
    parser.add_argument("--holder-marker-meters", type=float, default=0.093,
                        help="Side of the marker's black square in metres, for --holder-geometry.")
    parser.add_argument("--holder-person-meters", type=float, default=1.7,
                        help="Standing person height in metres, for --holder-geometry.")
    parser.add_argument("--holder-ambiguity-margin", type=float, default=20.0,
                        help="--holder-geometry: score lead below which the holder is left undecided.")
    parser.add_argument("--max-process-fps", type=float, default=0.0,
                        help="Process frames at most this often and hold each frame about that long before acting on it, "
                             "to mimic a slower device on a fast PC (Pi 5 + Hailo-8: about 21.9). 0 = no limit.")
    parser.add_argument("--switch-follow-candidate", action="store_true",
                        help="While the target is out of view and a marker holder is being confirmed as its replacement, steer to that marker holder instead of standing still.")
    parser.add_argument("--lost-search", action="store_true",
                        help="After the target is released: steer to a person in view; with nobody in view, look toward where the target was last seen. Works without zoom control.")
    parser.add_argument("--lost-search-wait-seconds", type=float, default=2.0,
                        help="Nobody in view for this long before the search moves the camera.")
    parser.add_argument("--lost-search-sweep-seconds", type=float, default=6.0,
                        help="Length of one look toward the last seen side (and of the way back).")
    parser.add_argument("--lost-search-pause-seconds", type=float, default=3.0)
    parser.add_argument("--lost-search-speed", type=int, default=4)
    parser.add_argument("--lost-search-give-up-seconds", type=float, default=60.0,
                        help="The search stops moving the camera this long after the release.")
    parser.add_argument("--tilt-head-anchor", action="store_true",
                        help="Take the vertical framing point from the smoothed top of the box plus the usual body height, instead of the box centre line.")
    parser.add_argument("--tilt-head-smoothing-seconds", type=float, default=0.5)
    parser.add_argument("--zoom-upper-body-meters", type=float, default=None,
                        help="Real length to keep at --framing-size-target (e.g. half the body height). With it, a marker seen large enough sets the zoom from its known size.")
    parser.add_argument("--zoom-marker-meters", type=float, default=0.093,
                        help="Side of the marker's black square in metres.")
    parser.add_argument("--zoom-marker-min-pixels", type=float, default=14.0)
    parser.add_argument("--marker-min-pixels", type=float, default=0.0,
                        help="Ignore markers narrower than this many pixels (640-wide frame scale not applied). 0 = accept all.")
    parser.add_argument("--grace-predict-follow", action="store_true",
                        help="While the target is undetected but predicted to be well inside the picture, extend the grace period and keep steering to the prediction.")
    parser.add_argument("--grace-predict-max-seconds", type=float, default=5.0)
    parser.add_argument("--grace-predict-edge-margin", type=float, default=0.15,
                        help="The prediction must be further than this share of the frame from every edge.")
    parser.add_argument("--zoom-in-speed", type=int, default=1)
    parser.add_argument("--zoom-out-speed", type=int, default=4)
    parser.add_argument("--qon-level-speeds-hfov", type=float, default=None,
                        help="Horizontal FOV at which --qon-level-speeds were measured; with zoom control the speeds are scaled from it.")
    parser.add_argument(
        "--qon-axis-mode",
        choices=("dominant", "diagonal", "alternate"),
        default="dominant",
        help=(
            "dominant: one axis per command (current behaviour). diagonal: pan and tilt "
            "together with leftup/rightdown... (camera must support them). "
            "alternate: both axes taking turns, for cameras without diagonal commands."
        ),
    )
    parser.add_argument("--qon-axis-alternate-seconds", type=float, default=0.35)
    parser.add_argument(
        "--qon-lost-coast-seconds",
        type=float,
        default=0.0,
        help="After the target is lost, keep panning in its last direction for this long. 0 = off.",
    )
    parser.add_argument("--qon-lost-coast-speed", type=int, default=2)
    parser.add_argument(
        "--qon-level-speeds",
        default="",
        help="Measured pan image speed per level in px/s on a 640x360 frame, e.g. '2:43,3:64,4:119'.",
    )
    parser.add_argument(
        "--qon-tilt-level-speeds",
        default="",
        help="Same for tilt; defaults to --qon-level-speeds.",
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
        if dest == "imgsz" and "model" in provided_dests and "imgsz" not in provided_dests:
            continue
        if dest not in provided_dests:
            setattr(args, dest, value)
    return args


def run(args: argparse.Namespace) -> None:
    _configure_runtime_cache()
    if args.inference_device == "cpu":
        # Must be set before importing Ultralytics/torch. The tracker creates
        # its ReID encoder internally, so passing device only to model.track()
        # is not sufficient to constrain that nested model.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.inference_device != "auto":
        # A GPU person detector next to the CPU speaker ReID: Ultralytics
        # hides every GPU from the process when the ReID model is set up with
        # device="cpu", which only works on torch's CUDA state if that has
        # not been initialised yet. Initialise it first.
        try:
            import torch

            torch.cuda.is_available()
        except ImportError:
            pass
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
            _install_ultralytics_reid_feature_compatibility()
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
    if args.hailo_hef and args.person_tracker != "decoupled":
        raise SystemExit("--hailo-hef requires --person-tracker decoupled")
    if args.no_display and args.target_mode == "manual":
        raise SystemExit("--no-display cannot be used with --target-mode manual")

    capture = _create_frame_capture(cv2, args)
    if not capture.is_opened():
        source = args.video_file if args.video_file else _redact_url(args.rtsp_url)
        raise SystemExit(f"Could not open video source: {source}")

    model = YOLO(args.model) if YOLO is not None else None
    if args.imgsz is not None:
        _warn_if_model_imgsz_mismatch(args.model, args.imgsz)
    person_source = _create_person_source(args)
    resolved_tracker_config = _resolve_tracker_config(args)
    if (
        args.speaker_identity_backend != "none"
        and "with_reid=true" in _tracker_config_summary(resolved_tracker_config).lower()
    ):
        logging.warning(
            "Tracker-level ReID and selective speaker ReID are both enabled; "
            "use botsort-pi.yaml to avoid duplicate appearance inference"
        )
    identity_manager = _create_speaker_identity_manager(args)
    target_mode = "aruco" if args.target_mode == "marker" else args.target_mode
    target_source = _create_target_source(cv2, args, target_mode)
    overlay_state_word = OverlayStateWord()
    signal_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="target-signal")
        if getattr(target_source, "frame_only", False)
        and not getattr(args, "serial_target_signal", False)
        else None
    )
    tracker = MarkerTracker(
        confirmation_seconds=args.confirmation_seconds,
        release_grace_seconds=args.release_grace_seconds,
        marker_dropout_seconds=args.marker_dropout_seconds,
        switch_score_margin=args.switch_score_margin,
        switch_cooldown_seconds=args.switch_cooldown_seconds,
        area_growth_penalty_threshold=args.area_growth_penalty_threshold,
        recovery_confirmation_seconds=args.recovery_confirmation_seconds,
        recovery_min_area_ratio=getattr(args, "recovery_min_area_ratio", None),
        bystander_rules=not getattr(args, "disable_bystander_rules", False),
        holder_geometry=getattr(args, "holder_geometry", False),
        holder_marker_meters=getattr(args, "holder_marker_meters", 0.093),
        holder_person_meters=getattr(args, "holder_person_meters", 1.7),
        holder_ambiguity_margin=getattr(args, "holder_ambiguity_margin", 20.0),
        recovery_min_iou=args.recovery_min_iou,
        recovery_max_center_distance=args.recovery_max_center_distance,
        recovery_score_margin=args.recovery_score_margin,
        recovery_appearance_weight=args.speaker_reid_weight,
        recovery_min_appearance=args.speaker_reid_min_similarity,
        recovery_appearance_margin=args.speaker_reid_margin,
        # [room_01 수정] 놓친 대상의 마커 재등록 조건을 MarkerTracker로 전달.
        reacquire_marker_observations=args.reacquire_marker_observations,
        reacquire_marker_window_seconds=args.reacquire_marker_window_seconds,
        reacquire_marker_min_seconds=(
            0.0
            if getattr(args, "disable_bystander_rules", False)
            else args.reacquire_marker_min_seconds
        ),
    )
    score_diagnostics = _create_score_diagnostics(args)
    status_diagnostics = _create_status_diagnostics(args)
    continuity_diagnostics = _create_continuity_diagnostics(args)
    framing = FramingPolicy(args.framing_mode)
    occlusion_anchor = (
        OcclusionAnchor(args.occlusion_height_ratio, args.occlusion_hold_seconds)
        if getattr(args, "occlusion_head_anchor", False)
        else None
    )
    zoom = _create_zoom_control(args)
    lost_search = (
        LostSearch(
            sweep=getattr(args, "lost_search", False),
            wait_seconds=getattr(args, "lost_search_wait_seconds", 2.0),
            sweep_seconds=getattr(args, "lost_search_sweep_seconds", 6.0),
            pause_seconds=getattr(args, "lost_search_pause_seconds", 3.0),
            speed=getattr(args, "lost_search_speed", 4),
            give_up_seconds=getattr(args, "lost_search_give_up_seconds", 60.0),
        )
        if getattr(args, "lost_search", False) or getattr(args, "zoom_lost_search", False)
        else None
    )
    grace_follow = GraceFollowLog()
    head_anchor = (
        HeadAnchor(args.tilt_head_smoothing_seconds)
        if getattr(args, "tilt_head_anchor", False)
        else None
    )
    diagnostics = ByteTrackDiagnostics(
        expected_buffer_seconds=args.track_buffer_seconds,
        log_interval_seconds=args.diagnostics_interval_seconds,
    )
    performance_diagnostics = _create_performance_diagnostics(args)
    camera = _create_camera_adapter(args)
    input_recorder = (
        InputVideoRecorder(cv2, args.record_input_video, args.record_input_fps)
        if args.record_input_video
        else None
    )
    identity_trace = (
        IdentityTraceRecorder(Path(args.identity_trace_file))
        if args.identity_trace_file
        else None
    )
    previous_frame_at = None
    fps = 0.0
    frame_sequence = 0
    window_name = "ArUco marker PTZ prototype"
    display_enabled = not args.no_display
    if display_enabled:
        cv2.namedWindow(window_name)
        if target_source.needs_mouse:
            cv2.setMouseCallback(window_name, target_source.on_mouse)
    inference_confidence = _resolve_inference_confidence(args)
    _log_person_tracking_configuration(
        args,
        resolved_tracker_config,
        inference_confidence,
    )

    max_process_fps = getattr(args, "max_process_fps", 0.0) or 0.0
    pace_interval = 1.0 / max_process_fps if max_process_fps > 0 else 0.0
    pace_work_started_at = None
    pace_work_seconds = None
    try:
        while True:
            if pace_work_started_at is not None:
                work = time.perf_counter() - pace_work_started_at
                pace_work_seconds = (
                    work if pace_work_seconds is None else 0.8 * pace_work_seconds + 0.2 * work
                )
            ok, frame, frame_sequence = capture.read_latest(frame_sequence)
            if not ok:
                logging.info("Video source ended or frame read failed")
                break
            if pace_interval > 0:
                # Like a slower device, let the frame age by the rest of the
                # interval before working on it: both the frame rate and the
                # frame-to-command delay then match the interval.
                wait = pace_interval - (pace_work_seconds or 0.0)
                if wait > 0:
                    time.sleep(wait)
                pace_work_started_at = time.perf_counter()
            if input_recorder is not None:
                input_recorder.write(frame)

            pipeline_started_at = time.perf_counter()
            # Marker detection is CPU work that does not depend on the person
            # detections, so it runs while the detector (NPU) and tracker run.
            signal_future = (
                signal_executor.submit(_timed_target_signal, target_source, frame)
                if signal_executor is not None
                else None
            )
            person_source_started_at = time.perf_counter()
            result, detections = person_source.update(
                frame,
                model,
                inference_confidence,
                resolved_tracker_config,
            )
            person_source_seconds = time.perf_counter() - person_source_started_at
            current_frame_at = time.monotonic()
            if previous_frame_at is not None:
                current_fps = 1.0 / max(current_frame_at - previous_frame_at, 1e-9)
                fps = current_fps if fps == 0.0 else (0.9 * fps + 0.1 * current_fps)
            previous_frame_at = current_frame_at
            if signal_future is not None:
                source_result, target_signal_seconds = signal_future.result()
            else:
                target_signal_started_at = time.perf_counter()
                source_result = target_source.update(frame, result, detections)
                target_signal_seconds = time.perf_counter() - target_signal_started_at
            marker_min_pixels = getattr(args, "marker_min_pixels", 0.0)
            detections.extend(
                detection
                for detection in source_result.detections
                if detection.label != "marker"
                or detection.box.x2 - detection.box.x1 >= marker_min_pixels
            )
            requested_track_id = source_result.requested_track_id
            decision_at = time.monotonic()
            identity_started_at = time.perf_counter()
            detections = identity_manager.score_detections(
                frame,
                detections,
                decision_at,
            )
            identity_reid_seconds = time.perf_counter() - identity_started_at
            target_logic_started_at = time.perf_counter()
            predicted_inside = False
            if getattr(args, "grace_predict_follow", False) and hasattr(camera, "predicted_error"):
                predicted = camera.predicted_error()
                limit = 1.0 - 2.0 * args.grace_predict_edge_margin
                predicted_inside = (
                    predicted is not None
                    and abs(predicted[0]) <= limit
                    and abs(predicted[1]) <= limit
                )
            decision = tracker.update(
                detections,
                decision_at,
                requested_track_id=requested_track_id,
                identity_resolution=identity_manager.resolution_context(),
                view_scale=zoom.view_scale() if zoom is not None else None,
                grace_extension=(
                    max(0.0, args.grace_predict_max_seconds - args.release_grace_seconds)
                    if predicted_inside
                    else 0.0
                ),
            )
            target_logic_seconds = time.perf_counter() - target_logic_started_at
            identity_started_at = time.perf_counter()
            decision = identity_manager.observe(
                decision,
                detections,
                frame,
                decision_at,
            )
            identity_reid_seconds += time.perf_counter() - identity_started_at
            if decision.emit_handoff and decision.target_box is not None:
                camera.handoff(decision.target_box)
            if decision.emit_release:
                camera.release()
            search_plan = (
                lost_search.update(
                    decision,
                    detections,
                    decision_at,
                    camera,
                    scale=zoom.view_scale() if zoom is not None else 1.0,
                    zoomable=zoom is not None,
                )
                if lost_search is not None
                else None
            )
            search_box = search_plan.candidate if search_plan is not None else None
            if hasattr(person_source, "target_hint"):
                # Lets the person source keep weak detections of the current target.
                person_source.target_hint = (
                    (decision.target_track_id, decision.target_box)
                    if decision.state in {"tracking", "grace", "recovering", "switching"}
                    else None
                )
            camera_target = _camera_target_for_decision(
                decision,
                framing,
                frame.shape,
                occlusion_anchor,
                detections,
                decision_at,
                head_anchor,
            )
            if (
                camera_target is None
                and getattr(args, "switch_follow_candidate", False)
                and decision.candidate_box is not None
            ):
                camera_target = framing.camera_target(decision.candidate_box, frame.shape)
            if camera_target is None and search_box is not None:
                # Lost-target search: bring the candidate to the centre so the marker can be checked.
                camera_target = framing.camera_target(search_box, frame.shape)
            following_prediction = (
                predicted_inside
                and decision.state == "grace"
                and camera_target is None
                and camera.follow_prediction()
            )
            grace_follow.update(decision, following_prediction, decision_at)
            if following_prediction:
                pass
            elif search_plan is not None and search_plan.active and hasattr(camera, "search_pan"):
                camera.search_pan(search_plan.pan, lost_search.speed)
            else:
                camera.update(
                    camera_target,
                    frame.shape,
                    # [room_01 수정] 하드코딩 2 대신 --provisional-speed-limit 사용.
                    speed_limit=_camera_speed_limit_for_decision(decision, args),
                    immediate_stop=decision.state != "grace" and search_box is None,
                    allow_coast=decision.state in {"grace", "default"} and search_box is None,
                )
            if zoom is not None:
                zoom.step(camera, decision, detections, frame.shape, decision_at, occlusion_anchor, search_plan)
            diagnostics.update(detections, fps, decision.target_track_id)
            score_diagnostics.update(decision)
            status_diagnostics.update(
                decision,
                detections,
                fps,
                camera.status(),
                args.framing_mode,
            )
            continuity_diagnostics.update(decision, detections, decision_at)
            performance_diagnostics.update(
                person_source_seconds=person_source_seconds,
                target_signal_seconds=target_signal_seconds,
                target_logic_seconds=target_logic_seconds,
                identity_reid_seconds=identity_reid_seconds,
                pipeline_seconds=time.perf_counter() - pipeline_started_at,
                extra_seconds=getattr(person_source, "last_timings", None),
            )
            if identity_trace is not None:
                identity_trace.write(frame_sequence, decision_at, detections, decision)

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
                    overlay=getattr(args, "overlay", "simple"),
                    state_word=overlay_state_word,
                )
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        if identity_trace is not None:
            identity_trace.close()
        if input_recorder is not None:
            input_recorder.close()
        if signal_executor is not None:
            signal_executor.shutdown(wait=False)
        if zoom is not None:
            if hasattr(camera, "zoom"):
                camera.zoom("stop", 0)
            zoom.close()
        camera.close()
        capture.release()
        if display_enabled:
            cv2.destroyAllWindows()


def _near_target_box(box: BoundingBox, target: BoundingBox) -> bool:
    """Same place as the target: centre within one box width and a similar top edge."""
    width = max(box.x2 - box.x1, target.x2 - target.x1, 1.0)
    height = max(target.y2 - target.y1, 1.0)
    return (
        abs(box.center[0] - target.center[0]) <= width
        and abs(box.y1 - target.y1) <= 0.35 * height
    )


def _to_detections(
    names: dict[int, str],
    boxes: Sequence,
    person_confidence: float = 0.35,
    target_low_confidence: float | None = None,
    target_hint: tuple[int | None, BoundingBox | None] | None = None,
) -> list[Detection]:
    detections = []
    for box in boxes:
        class_id = int(box.cls[0].item())
        label = names[class_id]
        if label != "person":
            continue
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        track_id = _box_track_id(box)
        if confidence < person_confidence:
            if (
                target_low_confidence is None
                or target_hint is None
                or confidence < target_low_confidence
            ):
                continue
            hint_track_id, hint_box = target_hint
            same_track = track_id is not None and track_id == hint_track_id
            if not same_track and (
                hint_box is None
                or not _near_target_box(BoundingBox(x1, y1, x2, y2), hint_box)
            ):
                continue
        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                box=BoundingBox(x1, y1, x2, y2),
                track_id=track_id,
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
        if args.person_tracker == "decoupled":
            return DecoupledPersonSource(args)
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


def _log_person_tracking_configuration(
    args: argparse.Namespace,
    tracker_config: str,
    inference_confidence: float,
) -> None:
    person_confidence = (
        args.confidence if args.confidence is not None else args.person_confidence
    )
    if args.person_source == "fixture":
        backend = "fixture"
        details = "fixed_test_detections=True"
        tracker_config_value = "ignored"
    elif args.person_tracker == "simple-iou":
        backend = args.person_tracker
        details = (
            f"iou_threshold={args.simple_iou_threshold} "
            f"max_lost_frames={args.simple_iou_max_lost_frames}"
        )
        tracker_config_value = "ignored"
    else:
        backend = args.person_tracker
        details = _tracker_config_summary(tracker_config)
        tracker_config_value = tracker_config
    logging.info(
        "Person tracking configuration: person_source=%s backend=%s model=%s imgsz=%s "
        "device=%s inference_confidence=%.3f person_confidence=%.3f "
        "tracker_config=%s %s",
        args.person_source,
        backend,
        args.model,
        args.imgsz if args.imgsz is not None else "default",
        getattr(args, "inference_device", "auto"),
        inference_confidence,
        person_confidence,
        tracker_config_value,
        details,
    )


def _resolve_inference_confidence(args: argparse.Namespace) -> float:
    confidence = (
        args.tracker_input_confidence
        if args.tracker_input_confidence is not None
        else args.confidence
        if args.confidence is not None
        else args.person_confidence
    )
    if not 0 <= confidence <= 1:
        raise SystemExit("--tracker-input-confidence must be between 0 and 1")
    return confidence


def _tracker_config_summary(tracker_config: str) -> str:
    path = Path(tracker_config)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "settings=unavailable"
    wanted = {
        "tracker_type",
        "track_high_thresh",
        "track_low_thresh",
        "new_track_thresh",
        "track_buffer",
        "match_thresh",
        "fuse_score",
        "gmc_method",
        "proximity_thresh",
        "appearance_thresh",
        "with_reid",
        "model",
    }
    values = []
    for raw_line in lines:
        content = raw_line.split("#", 1)[0].strip()
        key, separator, value = content.partition(":")
        if separator and key.strip() in wanted:
            values.append(f"{key.strip()}={value.strip()}")
    return "settings=" + (",".join(values) if values else "unavailable")


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


def _normalize_reid_feature_vectors(features):
    """Return one flat embedding per crop for tracker association.

    Ultralytics 8.4.60 leaves batched ONNX outputs shaped ``(1, D)`` while its
    BoT-SORT matching code expects ``(D,)``. Keeping this normalization in the
    project makes the exported ReID path reproducible without editing the
    virtualenv package in place.
    """
    import numpy as np

    return [np.asarray(feature).reshape(-1) for feature in features]


def _install_ultralytics_reid_feature_compatibility() -> None:
    try:
        from ultralytics.trackers.bot_sort import ReID
    except (ImportError, AttributeError):
        return
    original_call = ReID.__call__
    if getattr(original_call, "_marker_tracker_flattens_features", False):
        return

    def normalized_call(self, img, dets):
        return _normalize_reid_feature_vectors(original_call(self, img, dets))

    normalized_call._marker_tracker_flattens_features = True
    ReID.__call__ = normalized_call


class UltralyticsAppearanceEncoder:
    """Small adapter around Ultralytics' exported-model ReID runner."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit(
                "Selective speaker ReID requires Ultralytics and ONNX Runtime."
            ) from exc
        self._model = YOLO(model_path)
        embed_layer = len(self._model.model.model) - 2 if ".pt" in model_path else -1
        # Ultralytics' tracker ReID wrapper initializes without an explicit
        # device. On CUDA-capable hosts that can auto-install onnxruntime-gpu
        # even when CUDA runtime libraries are absent. Initialize the predictor
        # ourselves so selective ReID remains on the requested provider.
        self._model(
            embed=[embed_layer],
            verbose=False,
            save=False,
            device=device,
        )
        logging.info(
            "Selective speaker ReID enabled: model=%s device=%s",
            model_path,
            device,
        )

    def encode(self, frame, people: Sequence[Detection]):
        import numpy as np
        import torch
        from ultralytics.utils.ops import xywh2xyxy
        from ultralytics.utils.plotting import save_one_box

        boxes = np.asarray(
            [
                [
                    person.box.center[0],
                    person.box.center[1],
                    person.box.x2 - person.box.x1,
                    person.box.y2 - person.box.y1,
                ]
                for person in people
            ],
            dtype=np.float32,
        )
        crops = [
            save_one_box(box, frame, save=False)
            for box in xywh2xyxy(torch.from_numpy(boxes[:, :4]))
        ]
        # [room_01 수정] 기존: predictor(crops) 한 번에 호출 -> 변경: crop 한 장씩 호출.
        # 이유: 크기가 다른 crop을 한 배치로 넣으면 모두 448x448로 padding 되어
        # 3명 기준 약 350ms 동안 메인 루프가 멈췄고, 그 사이 추적기가 대상을 놓쳤음.
        # One crop per call. A batch of differently shaped crops is padded to
        # the full square export size (448x448 each, ~130 ms per crop on a
        # Pi 5), while a single crop keeps its minimal rectangle (~40 ms).
        # Per-crop calls also give gallery and query crops the same
        # preprocessing regardless of how many people are in the frame.
        vectors = []
        for crop in crops:
            feature = self._model.predictor([crop])[0]
            vectors.append(
                feature.detach().cpu().numpy()
                if hasattr(feature, "detach")
                else np.asarray(feature)
            )
        return _normalize_reid_feature_vectors(vectors)


def _create_speaker_identity_manager(args: argparse.Namespace):
    if args.speaker_identity_backend == "none":
        return NoopSpeakerIdentityManager()
    if args.speaker_identity_backend == "yolo-reid":
        scale = getattr(args, "speaker_reid_similarity_scale", None)
        offset = getattr(args, "speaker_reid_similarity_offset", None)
        hef = getattr(args, "speaker_reid_hef", None)
        onnx_model = getattr(args, "speaker_reid_onnx", None)
        if hef and onnx_model:
            raise SystemExit("Use either --speaker-reid-hef or --speaker-reid-onnx")
        if hef or onnx_model:
            # Another ReID model has another similarity scale; running it on
            # thresholds tuned for yolo26n-reid would accept bystanders.
            option = "--speaker-reid-hef" if hef else "--speaker-reid-onnx"
            if scale is None or offset is None:
                raise SystemExit(
                    f"{option} requires --speaker-reid-similarity-scale "
                    "and --speaker-reid-similarity-offset"
                )
            encoder = (
                HailoAppearanceEncoder(hef)
                if hef
                else OnnxAppearanceEncoder(onnx_model, device=args.speaker_reid_device)
            )
            logging.info(
                "Selective speaker ReID enabled: %s=%s similarity_scale=%s "
                "similarity_offset=%s",
                "hef" if hef else "onnx",
                hef or onnx_model,
                scale,
                offset,
            )
        else:
            encoder = UltralyticsAppearanceEncoder(
                args.speaker_reid_model,
                device=args.speaker_reid_device,
            )
        return SpeakerIdentityManager(
            encoder,
            similarity_scale=1.0 if scale is None else scale,
            similarity_offset=0.0 if offset is None else offset,
            gallery_size=args.speaker_gallery_size,
            gallery_interval_seconds=args.speaker_gallery_interval_seconds,
            gallery_novelty_similarity=args.speaker_gallery_novelty_similarity,
            gallery_max_overlap=args.speaker_gallery_max_overlap,
            trigger_overlap=args.speaker_reid_trigger_overlap,
            score_burst_frames=args.speaker_reid_burst_frames,
            min_similarity=args.speaker_reid_min_similarity,
            resolution_min_frames=args.speaker_resolution_min_frames,
            resolution_min_seconds=args.speaker_resolution_min_seconds,
            resolution_min_similarity=args.speaker_resolution_min_similarity,
            resolution_min_margin=args.speaker_resolution_min_margin,
            resolution_winner_ratio=args.speaker_resolution_winner_ratio,
            crossing_recent_seconds=args.speaker_crossing_recent_seconds,
            negative_gallery_weight=args.speaker_negative_gallery_weight,
            resolution_close_distance=args.speaker_resolution_close_distance,
            resolution_exit_distance=args.speaker_resolution_exit_distance,
            resolution_exit_frames=args.speaker_resolution_exit_frames,
            resolution_cooldown_seconds=args.speaker_resolution_cooldown_seconds,
            coast_max_seconds=args.speaker_coast_max_seconds,
            single_resolution_min_similarity=args.speaker_single_resolution_min_similarity,
            single_resolution_min_frames=args.speaker_single_resolution_min_frames,
            single_resolution_min_seconds=args.speaker_single_resolution_min_seconds,
            single_path_max_distance=args.speaker_single_path_max_distance,
            single_path_max_gap_seconds=args.speaker_single_path_max_gap_seconds,
            resolution_search_after_seconds=args.speaker_resolution_search_after_seconds,
            resolution_search_interval_frames=args.speaker_resolution_search_interval_frames,
            anchor_gallery_size=args.speaker_anchor_gallery_size,
            gallery_freeze_seconds=args.speaker_gallery_freeze_seconds,
            # [room_01 수정] 신규 인자 2개를 SpeakerIdentityManager로 전달.
            resolution_strong_margin=args.speaker_resolution_strong_margin,
            resolution_timeout_seconds=args.speaker_resolution_timeout_seconds,
        )
    raise SystemExit(
        f"Unknown speaker identity backend: {args.speaker_identity_backend}"
    )


class YoloPersonSource:
    def __init__(self, args: argparse.Namespace) -> None:
        self._person_confidence = (
            args.confidence if args.confidence is not None else args.person_confidence
        )
        self._imgsz = args.imgsz
        self._inference_device = getattr(args, "inference_device", "auto")
        self._target_low_confidence = getattr(args, "target_low_confidence", None)
        self.target_hint: tuple[int | None, BoundingBox | None] | None = None

    def update(
        self,
        frame,
        model,
        inference_confidence: float,
        bytetrack_config: str,
    ):
        track_kwargs = {
            "conf": inference_confidence,
            "verbose": False,
            "persist": True,
            "tracker": bytetrack_config,
        }
        person_class_ids = _class_ids_for_label(getattr(model, "names", None), "person")
        if person_class_ids:
            track_kwargs["classes"] = person_class_ids
        if self._imgsz is not None:
            track_kwargs["imgsz"] = self._imgsz
        if self._inference_device != "auto":
            track_kwargs["device"] = self._inference_device
        result = model.track(frame, **track_kwargs)[0]
        return result, _to_detections(
            result.names,
            result.boxes,
            person_confidence=self._person_confidence,
            target_low_confidence=self._target_low_confidence,
            target_hint=self.target_hint,
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
        self._inference_device = getattr(args, "inference_device", "auto")
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
        predict_kwargs = {
            "conf": inference_confidence,
            "verbose": False,
        }
        person_class_ids = _class_ids_for_label(getattr(model, "names", None), "person")
        if person_class_ids:
            predict_kwargs["classes"] = person_class_ids
        if self._imgsz is not None:
            predict_kwargs["imgsz"] = self._imgsz
        if self._inference_device != "auto":
            predict_kwargs["device"] = self._inference_device
        result = model.predict(frame, **predict_kwargs)[0]
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


def _class_ids_for_label(names, label: str) -> list[int]:
    if isinstance(names, dict):
        return [int(class_id) for class_id, name in names.items() if name == label]
    if isinstance(names, (list, tuple)):
        return [class_id for class_id, name in enumerate(names) if name == label]
    return []


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
    if flip_mode == "unmirrored":
        return [(frame, "none"), (cv2.flip(frame, -1), "both")]
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


class ArucoTargetSource:
    # Reads only the frame, so it can run while the person source is busy.
    frame_only = True

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


def _timed_target_signal(target_source, frame) -> tuple["TargetSourceResult", float]:
    """Run a frame-only target source and return its result with its own duration."""
    started_at = time.perf_counter()
    result = target_source.update(frame, None, [])
    return result, time.perf_counter() - started_at


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


def _create_continuity_diagnostics(args: argparse.Namespace):
    if not args.track_continuity_diagnostics:
        return NoopTrackContinuityDiagnostics()
    return TrackContinuityDiagnostics()


def _create_performance_diagnostics(args: argparse.Namespace):
    if not args.performance_diagnostics:
        return NoopPerformanceDiagnostics()
    return PerformanceDiagnostics(args.performance_diagnostics_interval_seconds)


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
            stop_deadzone=getattr(args, "qon_stop_deadzone", 0.08),
            stop_debounce_seconds=getattr(args, "qon_stop_debounce_seconds", 0.12),
            lost_target_hold_seconds=getattr(args, "qon_lost_target_hold_seconds", 0.20),
            command_min_interval_seconds=getattr(
                args, "qon_command_min_interval_seconds", 0.12
            ),
            command_ttl_seconds=args.qon_command_ttl_seconds,
            timeout_seconds=args.qon_timeout_seconds,
            configure_supervisor_actuator=not args.no_qon_configure_supervisor_actuator,
            async_commands=not args.no_qon_async_commands,
            log_commands=args.qon_log_commands,
            close_join_timeout_seconds=args.qon_close_join_timeout_seconds,
            # [room_01 수정] 신규 PTZ 응답 인자 6개 전달. getattr 기본값은 기존 동작과 동일.
            full_speed_error=getattr(args, "qon_full_speed_error", 1.0),
            uniform_error_scale=getattr(args, "qon_uniform_error_scale", False),
            axis_switch_ratio=getattr(args, "qon_axis_switch_ratio", 1.0),
            response_delay_seconds=getattr(args, "qon_response_delay_seconds", 0.0),
            pan_rate_per_speed=getattr(args, "qon_pan_rate_per_speed", 0.075),
            tilt_rate_per_speed=getattr(args, "qon_tilt_rate_per_speed", 0.05),
            speed_map=[
                (float(error), int(level))
                for error, level in _parse_pairs(getattr(args, "qon_speed_map", ""))
            ],
            predict_target_seconds=getattr(args, "qon_predict_target_seconds", 0.0),
            stop_lookahead_seconds=getattr(args, "qon_stop_lookahead_seconds", 0.0),
            level_speeds={
                int(level): float(speed)
                for level, speed in _parse_pairs(getattr(args, "qon_level_speeds", ""))
            },
            tilt_level_speeds={
                int(level): float(speed)
                for level, speed in _parse_pairs(getattr(args, "qon_tilt_level_speeds", ""))
            },
            axis_mode=getattr(args, "qon_axis_mode", "dominant"),
            axis_alternate_seconds=getattr(args, "qon_axis_alternate_seconds", 0.35),
            lost_coast_seconds=getattr(args, "qon_lost_coast_seconds", 0.0),
            lost_coast_speed=getattr(args, "qon_lost_coast_speed", 2),
            zoom_max_burst_seconds=getattr(args, "zoom_max_burst_seconds", 0.0),
            zoom_burst_rest_seconds=getattr(args, "zoom_burst_rest_seconds", 0.3),
            tilt_deadzone=getattr(args, "qon_tilt_deadzone", None),
            tilt_stop_deadzone=getattr(args, "qon_tilt_stop_deadzone", None),
            predict_axes=getattr(args, "qon_predict_axes", "both"),
            axis_reverse_hold_seconds=getattr(args, "qon_axis_reverse_hold_seconds", 0.0),
        )
    raise SystemExit(f"Unknown camera control mode: {args.camera_control}")


def _parse_pairs(text: str) -> list[tuple[str, str]]:
    """Parse 'a:b,c:d' option values."""
    pairs = []
    for item in (text or "").split(","):
        if item.strip():
            key, _, value = item.partition(":")
            if not value:
                raise SystemExit(f"Expected key:value pairs, got: {item!r}")
            pairs.append((key.strip(), value.strip()))
    return pairs


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
            "Tracking status: fps=%.1f state=%s speaker_id=%s target_id=%s reason=%s "
            "people=%s tracks=%s markers=%s marker_score=%s ptz=%s framing=%s recovery=%s",
            fps,
            decision.state,
            decision.speaker_id,
            decision.target_track_id,
            decision.reason or "none",
            people_count,
            _person_track_summary(detections),
            marker_count,
            "none" if best_score is None else f"{best_score:.1f}",
            ptz_status,
            framing_mode,
            _recovery_status(decision, marker_count, people_count),
        )


class NoopTrackContinuityDiagnostics:
    def update(
        self,
        decision: HandoffDecision,
        detections: Sequence[Detection],
        now: float,
    ) -> None:
        del decision, detections, now


class TrackContinuityDiagnostics:
    """Report target-ID continuity events without influencing tracking decisions."""

    def __init__(self) -> None:
        self._target_track_id: int | None = None
        self._target_box: BoundingBox | None = None
        self._missing_since: float | None = None
        self._candidate_signature: tuple[int | None, ...] = ()

    def update(
        self,
        decision: HandoffDecision,
        detections: Sequence[Detection],
        now: float,
    ) -> None:
        people = [detection for detection in detections if detection.label == "person"]
        visible_ids = {
            person.track_id for person in people if person.track_id is not None
        }
        reference_box = self._target_box or decision.target_box

        if decision.emit_release:
            missing_seconds = self._missing_duration(now)
            logging.info(
                "Track continuity: event=target_released target_id=%s "
                "missing_seconds=%.3f visible=%s candidates=%s",
                decision.target_track_id,
                missing_seconds,
                _visible_track_ids(people),
                _continuity_candidates(reference_box, people),
            )
            self._reset()
            return

        target_track_id = decision.target_track_id
        if target_track_id is None or decision.state not in {
            "tracking",
            "grace",
            "recovering",
            "switching",
        }:
            return

        if target_track_id != self._target_track_id:
            logging.info(
                "Track continuity: event=target_selected previous_target_id=%s "
                "target_id=%s reason=%s visible=%s",
                self._target_track_id,
                target_track_id,
                decision.reason or "none",
                _visible_track_ids(people),
            )
            self._target_track_id = target_track_id
            self._target_box = decision.target_box
            reference_box = self._target_box
            self._missing_since = None
            self._candidate_signature = ()

        if target_track_id in visible_ids:
            visible_target = next(
                person for person in people if person.track_id == target_track_id
            )
            self._target_box = visible_target.box
            if self._missing_since is not None:
                logging.info(
                    "Track continuity: event=target_reappeared target_id=%s "
                    "missing_seconds=%.3f visible=%s",
                    target_track_id,
                    self._missing_duration(now),
                    _visible_track_ids(people),
                )
            self._missing_since = None
            self._candidate_signature = ()
            return

        signature = tuple(
            sorted(
                (person.track_id for person in people),
                key=lambda track_id: -1 if track_id is None else track_id,
            )
        )
        if self._missing_since is None:
            self._missing_since = now
            event = "target_missing"
        elif signature != self._candidate_signature:
            event = "missing_candidates_changed"
        else:
            return
        self._candidate_signature = signature
        logging.info(
            "Track continuity: event=%s target_id=%s missing_seconds=%.3f "
            "visible=%s candidates=%s",
            event,
            target_track_id,
            self._missing_duration(now),
            _visible_track_ids(people),
            _continuity_candidates(reference_box, people),
        )
        if decision.provisional_target and decision.target_box is not None:
            self._target_box = decision.target_box

    def _missing_duration(self, now: float) -> float:
        if self._missing_since is None:
            return 0.0
        return max(0.0, now - self._missing_since)

    def _reset(self) -> None:
        self._target_track_id = None
        self._target_box = None
        self._missing_since = None
        self._candidate_signature = ()


def _person_track_summary(detections: Sequence[Detection]) -> str:
    people = [detection for detection in detections if detection.label == "person"]
    if not people:
        return "none"
    summaries = []
    for person in people:
        appearance = (
            ""
            if person.appearance_similarity is None
            else f"/reid={person.appearance_similarity:.3f}"
        )
        summaries.append(
            f"{person.track_id if person.track_id is not None else 'none'}:"
            f"{person.confidence:.2f}{appearance}"
            f"@({person.box.x1:.1f},{person.box.y1:.1f},"
            f"{person.box.x2:.1f},{person.box.y2:.1f})"
        )
    return ",".join(summaries)


def _visible_track_ids(people: Sequence[Detection]) -> str:
    ids = sorted(
        person.track_id for person in people if person.track_id is not None
    )
    return "none" if not ids else ",".join(str(track_id) for track_id in ids)


def _continuity_candidates(
    reference_box: BoundingBox | None,
    people: Sequence[Detection],
) -> str:
    if not people:
        return "none"
    return "|".join(
        _continuity_candidate(reference_box, person) for person in people
    )


def _continuity_candidate(
    reference_box: BoundingBox | None,
    person: Detection,
) -> str:
    box = person.box
    if reference_box is None:
        iou = 0.0
        center_distance = float("inf")
        area_ratio = 0.0
        width_ratio = 0.0
        top_distance = float("inf")
        horizontal_distance = float("inf")
    else:
        iou = _iou(reference_box, box)
        reference_x, reference_y = reference_box.center
        person_x, person_y = box.center
        reference_width = max(1.0, reference_box.x2 - reference_box.x1)
        reference_height = max(1.0, reference_box.y2 - reference_box.y1)
        center_distance = math.hypot(
            (person_x - reference_x) / reference_width,
            (person_y - reference_y) / reference_height,
        )
        area_ratio = box.area / max(reference_box.area, 1.0)
        person_width = max(1.0, box.x2 - box.x1)
        person_height = max(1.0, box.y2 - box.y1)
        width_ratio = person_width / reference_width
        top_distance = abs(box.y1 - reference_box.y1) / max(
            reference_height, person_height
        )
        horizontal_distance = abs(person_x - reference_x) / max(
            reference_width, person_width
        )
    bbox = f"({box.x1:.1f},{box.y1:.1f},{box.x2:.1f},{box.y2:.1f})"
    return (
        f"track={person.track_id} conf={person.confidence:.3f} iou={iou:.3f} "
        f"appearance={'none' if person.appearance_similarity is None else f'{person.appearance_similarity:.3f}'} "
        f"center_distance={center_distance:.3f} area_ratio={area_ratio:.3f} "
        f"width_ratio={width_ratio:.3f} top_distance={top_distance:.3f} "
        f"horizontal_distance={horizontal_distance:.3f} bbox={bbox}"
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


class OcclusionAnchor:
    """Keeps the framing point steady when the target box suddenly shrinks.

    A person stepping behind a desk is often detected as head and shoulders
    only. The framing point is a fraction of the box height, so it would jump
    up to the head. While the box is much shorter than it was a moment ago,
    this returns a box of the remembered height hanging from the same top edge.
    """

    def __init__(self, height_ratio: float = 0.6, hold_seconds: float = 5.0) -> None:
        if not 0 < height_ratio < 1 or hold_seconds < 0:
            raise ValueError("Occlusion height ratio must be in (0, 1) and hold time non-negative")
        self._height_ratio = height_ratio
        self._hold_seconds = hold_seconds
        self._reference_height: float | None = None
        self._reference_confidence: float | None = None
        self._occluded_since: float | None = None

    @property
    def occluded(self) -> bool:
        return self._occluded_since is not None

    def rescale(self, factor: float) -> None:
        """The picture was magnified by `factor` (zoom): the remembered height follows."""
        if self._reference_height is not None:
            self._reference_height *= factor

    def reset(self) -> None:
        self._reference_height = None
        self._reference_confidence = None
        self._occluded_since = None

    def adjust(self, box: BoundingBox, confidence: float | None, now: float) -> BoundingBox:
        height = box.y2 - box.y1
        if self._reference_height is not None:
            weaker = (
                confidence is None
                or self._reference_confidence is None
                or confidence < self._reference_confidence - 0.1
            )
            if height < self._height_ratio * self._reference_height and weaker:
                if self._occluded_since is None:
                    self._occluded_since = now
                if now - self._occluded_since <= self._hold_seconds:
                    return BoundingBox(box.x1, box.y1, box.x2, box.y1 + self._reference_height)
                # Hidden for too long: accept the short box as the new normal.
                self._reference_height = None
        self._occluded_since = None
        if self._reference_height is None:
            self._reference_height = height
            self._reference_confidence = confidence
        else:
            self._reference_height += 0.2 * (height - self._reference_height)
            if confidence is not None and self._reference_confidence is not None:
                self._reference_confidence += 0.2 * (confidence - self._reference_confidence)
        return box


class HeadAnchor:
    """Vertical framing from the head instead of the whole box.

    The bottom of a walking person's box moves with every step and with every
    desk in front of the legs, and the usual framing point (a fraction of the
    box height) moves with it. Here the point hangs from the smoothed top of
    the box, at a fixed share of the height the person normally has.
    """

    def __init__(self, smoothing_seconds: float = 0.5) -> None:
        if smoothing_seconds < 0:
            raise ValueError("Head smoothing time must not be negative")
        self._smoothing_seconds = smoothing_seconds
        self._top: float | None = None
        self._height: float | None = None
        self._at: float | None = None

    def reset(self) -> None:
        self._top = self._height = self._at = None

    def adjust(self, box: BoundingBox, frame_height: int, now: float) -> BoundingBox:
        height = box.y2 - box.y1
        if self._top is None or self._at is None or now - self._at > 1.0:
            self._top, self._height = box.y1, height
        else:
            elapsed = now - self._at
            blend = 1.0 if self._smoothing_seconds == 0 else min(1.0, elapsed / self._smoothing_seconds)
            self._top += blend * (box.y1 - self._top)
            # The usual height is learned slowly, and not from a box cut off by the frame.
            if box.y2 < frame_height - 3 and box.y1 > 2:
                self._height += min(1.0, elapsed / 2.0) * (height - self._height)
        self._at = now
        return BoundingBox(box.x1, self._top, box.x2, self._top + self._height)


@dataclass(frozen=True)
class SearchPlan:
    candidate: BoundingBox | None = None  # person to steer to
    active: bool = False  # the search drives the pan itself
    pan: str | None = None  # "left", "right" or None (hold still)
    narrow: bool = False  # with zoom: narrow onto where the target was last seen


class LostSearch:
    """What to do after the target was released.

    A person in view is steered to, so that the marker can be confirmed. With
    nobody in view for a while the camera looks toward where the target was
    last seen: without zoom it pans that way and back; with zoom it points
    there and narrows, then widens again.
    """

    def __init__(
        self,
        sweep: bool = True,
        wait_seconds: float = 2.0,
        sweep_seconds: float = 6.0,
        pause_seconds: float = 3.0,
        speed: int = 4,
        give_up_seconds: float = 60.0,
    ) -> None:
        self._sweep = sweep
        self._wait_seconds = wait_seconds
        self._sweep_seconds = sweep_seconds
        self._pause_seconds = pause_seconds
        self.speed = speed
        self._give_up_seconds = give_up_seconds
        self._had_target = False
        self._last_target_x: float | None = None
        self._tracking_scale = 1.0
        self._lost_since: float | None = None
        self._empty_since: float | None = None
        self._side: str | None = None
        self._phase = "idle"

    def _log(self, phase: str, now: float) -> None:
        if phase != self._phase:
            self._phase = phase
            since = 0.0 if self._lost_since is None else now - self._lost_since
            logging.info("Lost search: %s (%.1fs after release, side=%s)", phase, since, self._side)

    def update(self, decision, detections, now, camera=None, scale: float = 1.0, zoomable: bool = False) -> SearchPlan:
        if decision.state == "tracking" and decision.target_box is not None:
            self._had_target = True
            self._last_target_x = decision.target_box.center[0]
            self._tracking_scale = scale
        if decision.state != "default" or not self._had_target:
            if self._lost_since is not None:
                self._log("ended", now)
            self._lost_since = self._empty_since = None
            self._phase = "idle"
            return SearchPlan()
        if self._lost_since is None:
            self._lost_since = now
            predicted = camera.predicted_error(now) if hasattr(camera, "predicted_error") else None
            self._side = None
            if predicted is not None and abs(predicted[0]) > 0.1:
                self._side = "right" if predicted[0] > 0 else "left"
        people = [d for d in detections if d.label == "person"]
        if people:
            self._empty_since = None
            if self._last_target_x is None:
                box = max(people, key=lambda d: d.confidence).box
            else:
                box = min(people, key=lambda d: abs(d.box.center[0] - self._last_target_x)).box
            self._last_target_x = box.center[0]
            self._log("steering to a person in view", now)
            return SearchPlan(candidate=box)
        if self._empty_since is None:
            self._empty_since = now
        elapsed = now - self._empty_since - self._wait_seconds
        if not self._sweep or elapsed < 0:
            self._log("nobody in view: waiting", now)
            return SearchPlan()
        if now - self._lost_since > self._give_up_seconds:
            self._log("gave up", now)
            return SearchPlan()
        if zoomable:
            # Point at where the target was last seen and narrow, then widen again.
            if elapsed % (self._sweep_seconds + self._pause_seconds) >= self._sweep_seconds:
                self._log("widening again", now)
                return SearchPlan()
            last = camera.last_seen_error(now) if hasattr(camera, "last_seen_error") else None
            offset = 0.0 if last is None else last[0] * scale / self._tracking_scale
            pan = None if abs(offset) <= 0.15 else "right" if offset > 0 else "left"
            self._log("narrowing to where it was last seen", now)
            return SearchPlan(active=True, pan=pan, narrow=True)
        if self._side is None:
            self._log("nobody in view: last seen side unknown", now)
            return SearchPlan()
        phase = elapsed % (2 * (self._sweep_seconds + self._pause_seconds))
        if phase < self._sweep_seconds:
            self._log("looking toward the last seen side", now)
            return SearchPlan(active=True, pan=self._side)
        if phase < self._sweep_seconds + self._pause_seconds:
            self._log("holding at the last seen side", now)
            return SearchPlan(active=True)
        if phase < 2 * self._sweep_seconds + self._pause_seconds:
            self._log("coming back", now)
            return SearchPlan(active=True, pan="left" if self._side == "right" else "right")
        self._log("holding at the start", now)
        return SearchPlan(active=True)


class GraceFollowLog:
    """Logs each stretch of following the prediction through a dropout, with its outcome."""

    def __init__(self) -> None:
        self._since: float | None = None
        self._last: float = 0.0

    def update(self, decision, following: bool, now: float) -> None:
        if following:
            if self._since is None:
                self._since = now
                logging.info("Grace follow: started")
            self._last = now
            return
        if self._since is None:
            return
        if decision.state == "tracking":
            outcome = "reacquired"
        elif decision.state == "default" or decision.emit_release:
            outcome = "released"
        else:
            return  # still undecided (grace, recovering, switching)
        logging.info("Grace follow: ended after %.1fs: %s", self._last - self._since, outcome)
        self._since = None


class ZoomControl:
    """Ties the zoom controller to the camera adapter and the zoom position."""

    def __init__(
        self,
        controller: ZoomController,
        state: ZoomState,
        reference_hfov: float | None,
        search: bool = False,
        widen_during_grace: bool = True,
        table: ZoomFovTable | None = None,
    ) -> None:
        self._widen_during_grace = widen_during_grace
        # With a table the zoom is commanded as absolute positions (zoomto).
        self._table = table
        self._sent_to: tuple[float, float] | None = None
        self._tracking_hfov: float | None = None
        self._controller = controller
        self._state = state
        self._reference_hfov = reference_hfov
        self._search = search
        self._last_target_x: float | None = None
        self._had_target = False
        self._grace_since: float | None = None
        self._previous_scale: float | None = None
        self._logged_at = 0.0

    def view_scale(self) -> float:
        return 1.0 / math.tan(math.radians(self._state.hfov() / 2))

    def search_candidate(self, decision, detections) -> BoundingBox | None:
        """After a loss: the person to steer to and narrow onto, if any is in view."""
        if not self._search or not self._had_target or decision.state != "default":
            return None
        people = [d for d in detections if d.label == "person"]
        if not people:
            return None
        if self._last_target_x is None:
            return max(people, key=lambda d: d.confidence).box
        return min(people, key=lambda d: abs(d.box.center[0] - self._last_target_x)).box

    def step(self, camera, decision, detections, frame_shape, now, occlusion_anchor=None, plan=None) -> None:
        hfov = self._state.hfov()
        scale = self.view_scale()
        if occlusion_anchor is not None and self._previous_scale:
            occlusion_anchor.rescale(scale / self._previous_scale)
        self._previous_scale = scale
        if self._reference_hfov and hasattr(camera, "set_view_gain"):
            camera.set_view_gain(scale * math.tan(math.radians(self._reference_hfov / 2)))
        tracking = decision.state == "tracking" and decision.target_box is not None
        if tracking:
            self._had_target = True
            self._last_target_x = decision.target_box.center[0]
            self._tracking_hfov = hfov
        candidate = plan.candidate if plan is not None else self.search_candidate(decision, detections)
        narrow_to = None
        if plan is not None and plan.narrow:
            # Back to about the field of view the target was last tracked at.
            narrow_to = min(self._tracking_hfov or hfov, 0.6 * self._controller.lost_hfov)
        if candidate is not None:
            self._last_target_x = candidate.center[0]
        self._grace_since = (self._grace_since or now) if decision.state == "grace" else None
        # A short dropout is not a loss yet; widen once it has lasted a second or the target is released.
        lost = self._had_target and (
            decision.state == "default"
            or (
                self._widen_during_grace
                and self._grace_since is not None
                and now - self._grace_since > 1.0
            )
        )
        confidence = next(
            (
                d.confidence
                for d in detections
                if d.label == "person" and d.track_id is not None and d.track_id == decision.target_track_id
            ),
            None,
        )
        marker = next((d.box for d in detections if d.label == "marker"), None)
        action, speed = self._controller.update(
            now,
            hfov,
            decision.target_box if tracking else None,
            frame_shape,
            confidence=confidence,
            occluded=bool(occlusion_anchor is not None and occlusion_anchor.occluded),
            target_velocity=camera.target_velocity() if hasattr(camera, "target_velocity") else 0.0,
            marker_box=marker,
            lost=lost,
            search_box=candidate,
            narrow_to=narrow_to,
        )
        target_hfov = self._controller.last_target_hfov
        if self._table is not None and action != "stop" and target_hfov and hasattr(camera, "zoom_to"):
            position = round(self._table.position(target_hfov))
            span = self._table.max_position - self._table.min_position
            if (
                self._sent_to is None
                or abs(position - self._sent_to[0]) > 0.02 * span
                or now - self._sent_to[1] >= 1.0
            ):
                camera.zoom_to(position, speed)
                self._state.commanded_to(position, speed)
                self._sent_to = (position, now)
        else:
            self._sent_to = None
            effective = camera.zoom(action, speed) if hasattr(camera, "zoom") else action
            self._state.commanded(effective or action, speed)
        if now - self._logged_at >= 1.0:
            self._logged_at = now
            ratio = self._controller.last_ratio
            logging.info(
                "Zoom status: hfov=%.1f position=%.0f source=%s action=%s reason=%s ratio=%s min_hfov=%.1f size_from=%s safety_hfov=%.1f",
                hfov,
                self._state.position(),
                self._state.source,
                action,
                self._controller.last_reason,
                "none" if ratio is None else f"{ratio:.2f}",
                self._controller.last_min_hfov,
                self._controller.last_size_source,
                self._controller.last_safety_hfov,
            )

    def close(self) -> None:
        self._state.close()


def _create_zoom_control(args: argparse.Namespace) -> "ZoomControl | None":
    if not getattr(args, "zoom_control", False):
        return None
    pairs = [(float(position), float(hfov)) for position, hfov in _parse_pairs(args.zoom_fov_table)]
    if len(pairs) < 2:
        raise SystemExit("--zoom-control needs --zoom-fov-table with at least two position:hfov pairs")
    table = ZoomFovTable(pairs)
    camera_url = args.qon_camera_url or _qon_camera_url_from_rtsp(args.rtsp_url)
    host = urlsplit(camera_url if "://" in camera_url else f"http://{camera_url}").hostname if camera_url else None
    start = args.zoom_start_position
    if args.zoom_inquiry_port and host:
        read = ZoomState.inquire(host, args.zoom_inquiry_port, timeout=1.0)
        if read is not None:
            start = read
        else:
            logging.warning("Zoom position inquiry failed; estimating zoom from commands")
    if start is None:
        raise SystemExit("--zoom-control needs --zoom-start-position when the zoom position cannot be read")
    state = ZoomState(
        table,
        start,
        {int(speed): float(rate) for speed, rate in _parse_pairs(args.zoom_rates)},
        inquiry_host=host if args.zoom_inquiry_port else None,
        inquiry_port=args.zoom_inquiry_port,
    )
    controller = ZoomController(
        table,
        target_ratio=args.framing_size_target,
        tolerance=args.framing_size_tolerance,
        upper_body_fraction=args.zoom_upper_body_fraction,
        delay_seconds=args.zoom_delay_seconds,
        response_seconds=args.zoom_response_seconds,
        cooldown_seconds=args.zoom_cooldown_seconds,
        lost_hfov=args.zoom_lost_hfov,
        in_speed=args.zoom_in_speed,
        out_speed=args.zoom_out_speed,
        ceiling_factor=args.zoom_ceiling_factor,
        in_max_screen_speed=args.zoom_in_max_screen_speed,
        search_marker_pixels=args.zoom_search_marker_pixels,
        upper_body_meters=args.zoom_upper_body_meters,
        marker_meters=args.zoom_marker_meters,
        marker_min_pixels=args.zoom_marker_min_pixels,
        safety_margin=getattr(args, "zoom_safety_margin", 1.3),
        safety_step=getattr(args, "zoom_safety_step", 1.5),
    )
    logging.info(
        "Zoom control enabled: start_position=%.0f hfov=%.1f target_ratio=%.2f tolerance=%.2f inquiry_port=%s",
        start,
        table.hfov(start),
        args.framing_size_target,
        args.framing_size_tolerance,
        args.zoom_inquiry_port or "none",
    )
    return ZoomControl(
        controller,
        state,
        getattr(args, "qon_level_speeds_hfov", None),
        search=getattr(args, "zoom_lost_search", False),
        widen_during_grace=not getattr(args, "grace_predict_follow", False),
        table=table if getattr(args, "zoom_mode", "continuous") == "absolute" else None,
    )


def _camera_target_for_decision(
    decision: HandoffDecision,
    framing: FramingPolicy,
    frame_shape: tuple[int, ...],
    occlusion_anchor: "OcclusionAnchor | None" = None,
    detections: Sequence[Detection] = (),
    now: float = 0.0,
    head_anchor: "HeadAnchor | None" = None,
) -> BoundingBox | None:
    if decision.emit_release or decision.state in {
        "confirming",
        "grace",
        "switching",
    }:
        if occlusion_anchor is not None and decision.emit_release:
            occlusion_anchor.reset()
        if head_anchor is not None and decision.emit_release:
            head_anchor.reset()
        return None
    if decision.state == "recovering" and not decision.provisional_target:
        return None
    target_box = decision.target_box
    if occlusion_anchor is not None and target_box is not None:
        confidence = next(
            (
                detection.confidence
                for detection in detections
                if detection.label == "person"
                and detection.track_id is not None
                and detection.track_id == decision.target_track_id
            ),
            None,
        )
        target_box = occlusion_anchor.adjust(target_box, confidence, now)
    if head_anchor is not None and target_box is not None:
        target_box = head_anchor.adjust(target_box, frame_shape[0], now)
    return framing.camera_target(target_box, frame_shape)


# [room_01 수정] 신규 함수. run()에 박혀 있던 "identity_resolving이면 속도 2" 규칙을
# 옵션으로 분리함(프로파일 기본 4).
def _camera_speed_limit_for_decision(
    decision: HandoffDecision,
    args: argparse.Namespace,
) -> int | None:
    if decision.state == "identity_resolving" and decision.provisional_target:
        return getattr(args, "provisional_speed_limit", 2)
    return None


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
    if decision.state == "recovering":
        return "id_recovery_confirmation"
    if decision.state == "identity_resolving":
        return "identity_evidence_collecting"
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


class NoopPerformanceDiagnostics:
    def update(
        self,
        person_source_seconds: float,
        target_signal_seconds: float,
        target_logic_seconds: float,
        pipeline_seconds: float,
        identity_reid_seconds: float = 0.0,
        extra_seconds: dict[str, float] | None = None,
    ) -> None:
        del (
            person_source_seconds,
            target_signal_seconds,
            target_logic_seconds,
            pipeline_seconds,
            identity_reid_seconds,
            extra_seconds,
        )


class PerformanceDiagnostics:
    """Measure comparable stage timings without changing pipeline behavior."""

    _METRIC_NAMES = (
        "person_source",
        "target_signal",
        "target_logic",
        "identity_reid",
        "pipeline",
    )

    def __init__(
        self,
        interval_seconds: float,
        max_samples: int = 600,
        now_fn=time.monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self._interval_seconds = interval_seconds
        self._max_samples = max_samples
        self._now_fn = now_fn
        self._last_log_at: float | None = None
        self._samples = {
            name: deque(maxlen=max_samples) for name in self._METRIC_NAMES
        }

    def update(
        self,
        person_source_seconds: float,
        target_signal_seconds: float,
        target_logic_seconds: float,
        pipeline_seconds: float,
        identity_reid_seconds: float = 0.0,
        extra_seconds: dict[str, float] | None = None,
    ) -> None:
        values = {
            "person_source": person_source_seconds,
            "target_signal": target_signal_seconds,
            "target_logic": target_logic_seconds,
            "identity_reid": identity_reid_seconds,
            "pipeline": pipeline_seconds,
        }
        # Optional sub-stage timings (e.g. detector vs tracker) from the source.
        values.update(extra_seconds or {})
        for name, seconds in values.items():
            if name not in self._samples:
                self._samples[name] = deque(maxlen=self._max_samples)
            self._samples[name].append(max(0.0, seconds) * 1000.0)

        now = self._now_fn()
        if self._last_log_at is None:
            self._last_log_at = now
            return
        if now - self._last_log_at < self._interval_seconds:
            return

        sample_count = len(self._samples["pipeline"])
        summary = " ".join(
            f"{name}_avg_ms={self._average(self._samples[name]):.2f} "
            f"{name}_p95_ms={self._percentile_95(self._samples[name]):.2f}"
            for name in self._samples
        )
        logging.info("Performance diagnostics: samples=%s %s", sample_count, summary)
        for samples in self._samples.values():
            samples.clear()
        self._last_log_at = now

    @staticmethod
    def _average(samples) -> float:
        return sum(samples) / len(samples) if samples else 0.0

    @staticmethod
    def _percentile_95(samples) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        index = max(0, math.ceil(len(ordered) * 0.95) - 1)
        return ordered[index]


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
            "Tracker diagnostics: fps=%.1f visible_tracks=%s known_tracks=%s "
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


class InputVideoRecorder:
    """Record the unannotated frames actually consumed by inference."""

    def __init__(self, cv2, path: str, fps: float) -> None:
        if fps <= 0:
            raise ValueError("--record-input-fps must be positive")
        self._cv2 = cv2
        self._path = Path(path)
        self._fps = fps
        self._writer = None

    def write(self, frame) -> None:
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = self._cv2.VideoWriter(
                str(self._path),
                fourcc,
                self._fps,
                (width, height),
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open input video recorder: {self._path}")
            logging.info(
                "Raw input recording enabled: path=%s fps=%.2f size=%sx%s",
                self._path,
                self._fps,
                width,
                height,
            )
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class IdentityTraceRecorder:
    """Persist deterministic identity inputs and decisions for replay/debugging."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("w", encoding="utf-8")
        logging.info("Identity trace enabled: path=%s", path)

    def write(
        self,
        frame_sequence: int,
        timestamp: float,
        detections: Sequence[Detection],
        decision: HandoffDecision,
    ) -> None:
        payload = {
            "frame": frame_sequence,
            "timestamp": timestamp,
            "detections": [
                {
                    "label": item.label,
                    "confidence": item.confidence,
                    "track_id": item.track_id,
                    "appearance_similarity": item.appearance_similarity,
                    "box": [item.box.x1, item.box.y1, item.box.x2, item.box.y2],
                }
                for item in detections
            ],
            "decision": {
                "state": decision.state,
                "speaker_id": decision.speaker_id,
                "target_track_id": decision.target_track_id,
                "reason": decision.reason,
                "emit_handoff": decision.emit_handoff,
                "emit_release": decision.emit_release,
                "provisional_target": decision.provisional_target,
            },
        }
        self._stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


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
        with self._condition:
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        # Most live streams yield another frame quickly, allowing the reader
        # to observe _stopped and exit before VideoCapture is released. This
        # avoids calling OpenCV release() concurrently with read(), which can
        # segfault during exception/timeout shutdown.
        self._capture.release()
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
    overlay: str = "debug",
    state_word=None,
) -> None:
    if overlay == "simple":
        _draw_simple(
            cv2, frame, detections, decision, target_box, camera_target, fps, state_word
        )
        return
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
    if decision.speaker_id is not None or decision.target_track_id is not None:
        cv2.putText(
            frame,
            f"speaker id: {decision.speaker_id} target id: {decision.target_track_id}",
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


# State word and BGR colour shown by the simple overlay.
_OVERLAY_STATES = {
    "default": ("WAITING", (180, 180, 180)),
    "confirming": ("REGISTERING", (0, 255, 255)),
    "switching": ("REGISTERING", (0, 255, 255)),
    "tracking": ("TRACKING", (0, 255, 0)),
}
_OVERLAY_SEARCHING = ("SEARCHING", (0, 165, 255))


class OverlayStateWord:
    """State word for the simple overlay, without flicker on one-frame dropouts.

    The detector loses the speaker for single frames now and then; the state
    then leaves "tracking" and returns on the next frame. TRACKING is kept
    until another state has lasted for hold_seconds.
    """

    def __init__(self, hold_seconds: float = 0.4, now_fn=time.monotonic) -> None:
        self._hold_seconds = hold_seconds
        self._now_fn = now_fn
        self._left_tracking_at: float | None = None
        self._was_tracking = False

    def word(self, state: str) -> tuple[str, tuple[int, int, int]]:
        shown = _OVERLAY_STATES.get(state, _OVERLAY_SEARCHING)
        if state == "tracking":
            self._left_tracking_at = None
            self._was_tracking = True
            return shown
        if self._was_tracking:
            now = self._now_fn()
            if self._left_tracking_at is None:
                self._left_tracking_at = now
            if now - self._left_tracking_at < self._hold_seconds:
                return _OVERLAY_STATES["tracking"]
            self._was_tracking = False
        return shown


def _draw_simple(
    cv2, frame, detections, decision, target_box, camera_target, fps, state_word=None
) -> None:
    """Boxes without labels, the speaker by colour and thickness, FPS and one state word."""
    colors = {"person": (255, 160, 0), "marker": (0, 255, 255)}
    for detection in detections:
        box = detection.box
        cv2.rectangle(
            frame,
            (int(box.x1), int(box.y1)),
            (int(box.x2), int(box.y2)),
            colors[detection.label],
            2 if detection.label == "marker" else 1,
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
    word, color = (
        state_word.word(decision.state)
        if state_word is not None
        else _OVERLAY_STATES.get(decision.state, _OVERLAY_SEARCHING)
    )
    for text, origin, text_color in (
        (f"FPS {fps:.0f}", (10, 28), (255, 255, 255)),
        (word, (10, 58), color),
    ):
        # Dark outline first so the text stays readable on a bright background.
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)


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
