"""Verify Qon internal auto tracking using registered presenter appearance."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.camera.region_provider import CenterCropRegionProvider, JsonlRegionProvider
from src.recovery.candidate_detector import YoloRecoveryDetector
from src.reid.appearance import AppearanceReIdentifier
from src.reid.verifier import PresenterVerifier, VerifierConfig
from src.vision.capture import VideoSource
from src.vision.models import BBox, TrackingState, VerificationResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qon internal tracking Re-ID verifier MVP")
    parser.add_argument("--source", required=True, help="Validation video or externally supplied camera RTSP URL")
    parser.add_argument("--region-mode", choices=("center", "jsonl"), default="center")
    parser.add_argument("--bbox-jsonl", default=None, help="Captured internal bbox samples for region-mode=jsonl")
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

            if frame_index == 1 and args.register_on_start:
                selected_bbox = registration_bbox or bbox
                if selected_bbox is None or not verifier.register(frame, selected_bbox):
                    print("Could not register presenter from the initial frame.")
                    return 1
                last_result = VerificationResult(
                    TrackingState.VERIFIED, selected_bbox, 1.0, "PRESENTER_REGISTERED", region_source
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
                    verifier.register(frame, registration_bbox or bbox)
                    last_result = VerificationResult(
                        TrackingState.VERIFIED, registration_bbox or bbox, 1.0, "PRESENTER_REGISTERED", region_source
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


def parse_bbox(value: str | None) -> BBox | None:
    if value is None:
        return None
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4:
        raise ValueError("--register-bbox must be x1,y1,x2,y2")
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
