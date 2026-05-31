"""Verify Qon internal auto tracking using registered presenter appearance."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.camera.region_provider import CenterCropRegionProvider, JsonlRegionProvider
from src.recovery.candidate_detector import YoloRecoveryDetector
from src.reid.appearance import AppearanceReIdentifier
from src.reid.verifier import PresenterVerifier, VerifierConfig
from src.tracking.gesture_detector import GestureFallbackDetector
from src.tracking.marker_detector import ArucoMarkerDetector
from src.tracking.person_detector import build_person_detector
from src.tracking.target_selector import point_in_bbox
from src.tracking.target_state import PersonDetection
from src.vision.capture import VideoSource
from src.vision.models import BBox, TrackingState, VerificationResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qon internal tracking Re-ID verifier MVP")
    parser.add_argument("--source", required=True, help="Validation video or externally supplied camera RTSP URL")
    parser.add_argument("--region-mode", choices=("center", "jsonl"), default="center")
    parser.add_argument("--bbox-jsonl", default=None, help="Captured internal bbox samples for region-mode=jsonl")
    parser.add_argument(
        "--registration-mode",
        choices=("observed", "marker", "gesture", "marker-or-gesture"),
        default="observed",
        help="How to register the presenter's Re-ID reference",
    )
    parser.add_argument("--target-marker-id", type=int, default=None, help="Only accept this ArUco marker id")
    parser.add_argument(
        "--detector",
        choices=("hog", "opencv-yolo", "ncnn", "manual"),
        default="hog",
        help="Person detector used for marker/gesture registration",
    )
    parser.add_argument("--yolo-model", default=None, help="Optional YOLO ONNX model path")
    parser.add_argument("--ncnn-param", default=None, help="NCNN .param path")
    parser.add_argument("--ncnn-bin", default=None, help="NCNN .bin path")
    parser.add_argument("--ncnn-input-size", type=int, default=640)
    parser.add_argument("--person-box", default=None, help="Manual detector bbox x,y,w,h")
    parser.add_argument("--conf-threshold", type=float, default=0.35)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--debug-detector", action="store_true")
    parser.add_argument("--center-width-ratio", type=float, default=0.42)
    parser.add_argument("--center-height-ratio", type=float, default=0.86)
    parser.add_argument("--register-on-start", action="store_true", help="Register the first observed region as presenter")
    parser.add_argument("--register-bbox", default=None, help="Registration bbox x1,y1,x2,y2; otherwise uses observed region")
    parser.add_argument("--verify-every-frames", type=int, default=30)
    parser.add_argument("--reid-threshold", type=float, default=0.68)
    parser.add_argument("--mismatch-limit", type=int, default=2)
    parser.add_argument("--recovery-model", default=None, help="Optional YOLO model used only after confirmed mismatch")
    parser.add_argument("--recovery-imgsz", type=int, default=416)
    parser.add_argument("--recovery-confidence", type=float, default=0.4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rtsp-drop-frames", type=int, default=0)
    parser.add_argument("--output", default=None, help="Write annotated MP4")
    parser.add_argument("--log-csv", default=None)
    parser.add_argument("--no-window", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.no_window and not args.register_on_start:
        print("--no-window requires --register-on-start for unattended registration.")
        return 2
    if args.region_mode == "jsonl" and not args.bbox_jsonl:
        print("--region-mode jsonl requires --bbox-jsonl.")
        return 2

    import cv2

    video = VideoSource(args.source, rtsp_drop_frames=args.rtsp_drop_frames)
    if not video.is_opened():
        print("Could not open validation video source.")
        return 1

    provider = build_region_provider(args)
    registration_detector = build_registration_detector(args)
    marker_detector = ArucoMarkerDetector() if args.registration_mode in ("marker", "marker-or-gesture") else None
    gesture_detector = (
        GestureFallbackDetector()
        if args.registration_mode in ("gesture", "marker-or-gesture")
        else None
    )
    verifier = PresenterVerifier(
        AppearanceReIdentifier(args.reid_threshold),
        VerifierConfig(max(1, args.mismatch_limit)),
    )
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
            ["frame", "state", "event", "region_source", "bbox", "score", "recovery_bbox", "recovery_score"]
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

            people = []
            markers = []
            if registration_detector is not None and (args.register_on_start or not args.no_window):
                people = registration_detector.detect(frame)
            if marker_detector is not None and (args.register_on_start or not args.no_window):
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
                    if args.no_window:
                        print("Could not register presenter from the available frame.")
                        return 1
                else:
                    last_result = VerificationResult(
                        TrackingState.VERIFIED, selected_bbox, 1.0, "PRESENTER_REGISTERED", registration_source(args)
                    )
                    registered_this_frame = True

            should_verify = frame_index == 1 or frame_index % max(1, args.verify_every_frames) == 0
            if verifier.registered and should_verify and not registered_this_frame:
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
                print_status(frame_index, last_result, recovery_bbox, recovery_score)

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
    finally:
        video.release()
        if writer:
            writer.release()
        if log_file:
            log_file.close()
        if not args.no_window:
            cv2.destroyAllWindows()
    return 0


def build_region_provider(args):
    if args.region_mode == "jsonl":
        return JsonlRegionProvider(args.bbox_jsonl)
    return CenterCropRegionProvider(args.center_width_ratio, args.center_height_ratio)


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
            return marker_bbox
    if args.registration_mode in ("gesture", "marker-or-gesture") and gesture_detector is not None:
        raised = gesture_detector.detect_raised_hand_indices(frame, people)
        person = gesture_detector.update(people, raised)
        if person is not None:
            return xywh_to_xyxy(person.bbox)
    return None


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
