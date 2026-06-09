"""Runnable marker-based PTZ auto-tracking MVP demo."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional

from src.reid.appearance import AppearanceReIdentifier
from src.tracking.gesture_detector import GestureFallbackDetector
from src.tracking.marker_detector import ArucoMarkerDetector
from src.tracking.person_detector import build_person_detector
from src.tracking.ptz_controller import PTZCommand, PTZSimulator
from src.tracking.target_selector import TargetSelector, point_in_bbox
from src.tracking.target_state import ActiveTarget, DemoConfig, PersonDetection, TrackingState
from src.vision.models import Candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PTZ marker tracking MVP demo")
    parser.add_argument("--source", default="0", help="RTSP URL, video file, or webcam index")
    parser.add_argument(
        "--detector",
        choices=("hog", "opencv-yolo", "ncnn", "manual"),
        default="ncnn",
        help="Person detector backend",
    )
    parser.add_argument("--yolo-model", default=None, help="Optional YOLO ONNX model path")
    parser.add_argument("--ncnn-param", default="models/yolo26n_ncnn_model/model.ncnn.param", help="NCNN YOLO .param path")
    parser.add_argument("--ncnn-bin", default="models/yolo26n_ncnn_model/model.ncnn.bin", help="NCNN YOLO .bin path")
    parser.add_argument(
        "--ncnn-input-size",
        type=int,
        default=640,
        help="Square YOLO model input size; stream2 frame size remains 640x360",
    )
    parser.add_argument("--ncnn-input-name", default="in0", help="NCNN input blob name")
    parser.add_argument("--ncnn-output-names", default="out0", help="Comma-separated NCNN output blob names to try")
    parser.add_argument("--conf-threshold", type=float, default=0.35, help="Person confidence threshold")
    parser.add_argument("--nms-threshold", type=float, default=0.45, help="NMS threshold for detector adapters")
    parser.add_argument("--debug-detector", action="store_true", help="Print NCNN detector debug output")
    parser.add_argument("--save-debug-video", default=None, help="Save debug overlay video to this MP4 path")
    parser.add_argument("--log-csv", default=None, help="Save per-frame tracking/PTZ log to this CSV path")
    parser.add_argument(
        "--person-box",
        default=None,
        help="Manual bbox x,y,w,h for deterministic smoke tests or controlled demos",
    )
    parser.add_argument("--no-window", action="store_true", help="Run without debug display")
    parser.add_argument("--gesture", action="store_true", help="Enable gesture fallback at startup")
    parser.add_argument("--marker-only", action="store_true", help="Require marker reacquisition")
    parser.add_argument("--target-marker-id", type=int, default=None, help="Only acquire this ArUco marker id")
    parser.add_argument("--max-suspended-frames", type=int, default=45)
    parser.add_argument("--disable-reid", action="store_true", help="Disable Re-ID fallback after marker lock")
    parser.add_argument("--reid-threshold", type=float, default=0.68, help="Minimum appearance score for Re-ID fallback")
    return parser.parse_args()


def open_capture(cv2, source: str):
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def parse_bbox(value: Optional[str]) -> Optional[tuple[int, int, int, int]]:
    if value is None:
        return None
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--person-box must be formatted as x,y,w,h")
    return tuple(parts)


def resolve_detector_mode(args: argparse.Namespace) -> str:
    if args.detector == "hog" and args.person_box is not None:
        return "manual"
    if args.detector == "hog" and args.yolo_model is not None:
        return "opencv-yolo"
    return args.detector


def main() -> int:
    args = parse_args()
    try:
        import cv2
    except ModuleNotFoundError:
        print("OpenCV is required. Install with: pip install opencv-contrib-python")
        return 2

    config = DemoConfig(
        max_suspended_frames=args.max_suspended_frames,
        marker_only=args.marker_only,
        gesture_enabled=args.gesture,
        target_marker_id=args.target_marker_id,
    )
    marker_detector = ArucoMarkerDetector()
    person_detector = build_person_detector(
        detector_mode=resolve_detector_mode(args),
        yolo_model=args.yolo_model,
        manual_bbox=parse_bbox(args.person_box),
        ncnn_param=args.ncnn_param,
        ncnn_bin=args.ncnn_bin,
        ncnn_input_size=args.ncnn_input_size,
        ncnn_input_name=args.ncnn_input_name,
        ncnn_output_names=tuple(part.strip() for part in args.ncnn_output_names.split(",") if part.strip()),
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        debug_detector=args.debug_detector,
    )
    selector = TargetSelector(config)
    gesture_detector = GestureFallbackDetector(config.gesture_hold_seconds)
    reid = None if args.disable_reid else AppearanceReIdentifier(args.reid_threshold)
    ptz = PTZSimulator(dead_zone_ratio=config.dead_zone_ratio)

    capture = open_capture(cv2, args.source)
    if not capture.isOpened():
        print(f"Could not open video source: {args.source}")
        return 1

    debug_video = None
    csv_file = None
    csv_writer = None
    state = TrackingState.IDLE
    target: Optional[ActiveTarget] = None
    marker_only = config.marker_only
    gesture_enabled = config.gesture_enabled
    last_command = PTZCommand()
    last_reid_score = 0.0

    while state is not TrackingState.ENDED:
        ok, frame = capture.read()
        if not ok:
            state = TrackingState.ENDED
            break

        people = person_detector.detect(frame)
        markers = marker_detector.detect(frame)

        if target is None:
            target, state = update_active_target_for_frame(
                selector,
                config,
                target,
                state,
                people,
                markers,
                frame.shape,
                marker_only,
            )
            if target is None and gesture_enabled and not marker_only:
                raised = gesture_detector.detect_raised_hand_indices(frame, people)
                gesture_person = gesture_detector.update(people, raised)
                if gesture_person is not None:
                    target = selector.acquire_from_gesture(gesture_person)
                    state = TrackingState.ACTIVE
                else:
                    state = TrackingState.IDLE
        else:
            target, state = update_active_target_for_frame(
                selector,
                config,
                target,
                state,
                people,
                markers,
                frame.shape,
                marker_only,
            )

        if target is not None and reid is not None and _target_marker_visible(selector, target, people, markers):
            reid.register(frame, _xywh_to_xyxy(target.bbox))
            last_reid_score = 1.0

        if target is not None and state is TrackingState.SUSPENDED and reid is not None and not marker_only:
            reid_person, last_reid_score = _reacquire_by_reid(reid, frame, people)
            if reid_person is not None:
                target.bbox = reid_person.bbox
                target.source = "reid"
                target.missed_frames = 0
                state = TrackingState.REID_TRACKING

        if target is not None and state in (TrackingState.ACTIVE, TrackingState.REID_TRACKING):
            last_command = ptz.command_for_bbox(target.bbox, frame.shape)
            ptz.send(last_command)
        elif state in (TrackingState.IDLE, TrackingState.SUSPENDED):
            last_command = PTZCommand()
            ptz.send(last_command)

        if args.save_debug_video is not None or not args.no_window:
            draw_debug(
                cv2,
                frame,
                people,
                markers,
                target,
                state,
                last_command,
                marker_only,
                gesture_enabled,
                last_reid_score,
                show_window=not args.no_window,
            )

        if args.save_debug_video is not None:
            if debug_video is None:
                debug_video = open_debug_video_writer(cv2, args.save_debug_video, frame, capture)
            debug_video.write(frame)

        if args.log_csv is not None:
            if csv_writer is None:
                csv_file, csv_writer = open_csv_logger(args.log_csv)
            write_csv_row(csv_writer, state, target, last_command, last_reid_score)

        if not args.no_window:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                state = TrackingState.ENDED
            elif key == ord("r"):
                target = None
                selector.reset_ids()
                state = TrackingState.IDLE
            elif key == ord("s"):
                target = None
                state = TrackingState.IDLE
                ptz.send(PTZCommand())
            elif key == ord("m"):
                marker_only = not marker_only
            elif key == ord("g"):
                gesture_enabled = not gesture_enabled

    capture.release()
    if debug_video is not None:
        debug_video.release()
    if csv_file is not None:
        csv_file.close()
    if not args.no_window:
        cv2.destroyAllWindows()
    return 0


def open_debug_video_writer(cv2, output_path: str, frame, capture):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_h, frame_w = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open debug video writer: {output}")
    return writer


def open_csv_logger(output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_file = output.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        ["timestamp", "state", "target_id", "marker_id", "bbox", "ptz_command", "error_x", "error_y", "reid_score"]
    )
    return csv_file, writer


def write_csv_row(
    writer,
    state: TrackingState,
    target: Optional[ActiveTarget],
    command: PTZCommand,
    reid_score: float = 0.0,
) -> None:
    bbox = ""
    target_id = ""
    marker_id = ""
    if target is not None:
        bbox = ",".join(str(value) for value in target.bbox)
        target_id = target.target_id
        marker_id = "" if target.marker_id is None else target.marker_id
    writer.writerow(
        [
            f"{time.time():.3f}",
            state.name,
            target_id,
            marker_id,
            bbox,
            command.as_text(),
            f"{command.error_x:.4f}",
            f"{command.error_y:.4f}",
            f"{reid_score:.4f}",
        ]
    )


def _target_from_same_marker(selector: TargetSelector, people, markers, marker_id: Optional[int]):
    if marker_id is None:
        return None
    if not selector._marker_allowed(marker_id):
        return None
    same_markers = [marker for marker in markers if marker.marker_id == marker_id]
    for marker in same_markers:
        for person in people:
            if point_in_bbox(marker.center, person.bbox):
                return person
    return None


def _target_marker_visible(selector: TargetSelector, target: ActiveTarget, people, markers) -> bool:
    marker_person = _target_from_same_marker(selector, people, markers, target.marker_id)
    return marker_person is not None and marker_person.bbox == target.bbox


def _xywh_to_xyxy(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def _reacquire_by_reid(
    reid: AppearanceReIdentifier,
    frame,
    people: list[PersonDetection],
) -> tuple[Optional[PersonDetection], float]:
    candidates = [
        Candidate(bbox=_xywh_to_xyxy(person.bbox), confidence=person.confidence, candidate_id=index)
        for index, person in enumerate(people)
    ]
    candidate, score = reid.best_match(frame, candidates)
    if candidate is None or candidate.candidate_id is None:
        return None, score
    return people[candidate.candidate_id], score


def update_active_target_for_frame(
    selector: TargetSelector,
    config: DemoConfig,
    target: Optional[ActiveTarget],
    state: TrackingState,
    people,
    markers,
    frame_shape,
    marker_only: bool,
) -> tuple[Optional[ActiveTarget], TrackingState]:
    """Pure marker/tracker state transition used by the demo and tests."""
    if target is None:
        target = selector.acquire_from_marker(people, markers)
        return (target, TrackingState.ACTIVE) if target is not None else (None, TrackingState.IDLE)

    marker_refresh = _target_from_same_marker(selector, people, markers, target.marker_id)
    if marker_refresh is not None:
        target.bbox = marker_refresh.bbox
        target.source = "marker"
        target.missed_frames = 0
        state = TrackingState.ACTIVE
    elif marker_only:
        target.missed_frames += 1
        state = TrackingState.SUSPENDED
    else:
        updated = selector.update_existing(target, people, frame_shape)
        if updated is not None:
            target = updated
            state = TrackingState.ACTIVE
        else:
            target.missed_frames += 1
            state = TrackingState.SUSPENDED

    if target is not None and target.missed_frames > config.max_suspended_frames:
        target = None
        state = TrackingState.IDLE
    return target, state


def draw_debug(
    cv2,
    frame,
    people,
    markers,
    target,
    state,
    command,
    marker_only,
    gesture_enabled,
    reid_score,
    show_window: bool = True,
) -> None:
    for person in people:
        x, y, w, h = person.bbox
        color = (90, 180, 255)
        thickness = 2
        if target is not None and person.bbox == target.bbox:
            color = (40, 255, 40)
            thickness = 3
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(
            frame,
            f"person {person.confidence:.2f}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    for marker in markers:
        pts = [(int(x), int(y)) for x, y in marker.corners]
        for start, end in zip(pts, pts[1:] + pts[:1]):
            cv2.line(frame, start, end, (255, 80, 80), 2)
        cx, cy = marker.center
        cv2.circle(frame, (int(cx), int(cy)), 4, (255, 80, 80), -1)
        cv2.putText(
            frame,
            f"aruco {marker.marker_id}",
            (int(cx) + 6, int(cy) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 80, 80),
            1,
            cv2.LINE_AA,
        )

    frame_h, frame_w = frame.shape[:2]
    dz_w = int(frame_w * 0.12)
    dz_h = int(frame_h * 0.12)
    cv2.rectangle(
        frame,
        (frame_w // 2 - dz_w, frame_h // 2 - dz_h),
        (frame_w // 2 + dz_w, frame_h // 2 + dz_h),
        (180, 180, 180),
        1,
    )

    target_text = "none"
    if target is not None:
        target_text = f"{target.target_id} src={target.source} marker={target.marker_id}"
    overlay = (
        f"state={state.name} target={target_text} ptz={command.as_text()} "
        f"marker_only={marker_only} gesture={gesture_enabled} reid={reid_score:.2f}"
    )
    cv2.putText(frame, overlay, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(frame, overlay, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    if show_window:
        cv2.imshow("PTZ marker tracking demo", frame)


if __name__ == "__main__":
    raise SystemExit(main())
