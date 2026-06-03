from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from collections.abc import Sequence
from urllib.parse import urlsplit, urlunsplit

from .camera import LoggingCameraAdapter
from .core import BoundingBox, Detection, MarkerTracker

try:
    from .local_settings import RTSP_URL as LOCAL_RTSP_URL
except ImportError:
    LOCAL_RTSP_URL = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a person using a phone marker from an RTSP stream."
    )
    parser.add_argument(
        "--rtsp-url",
        default=os.getenv("MARKER_TRACKER_RTSP_URL", LOCAL_RTSP_URL),
        help="RTSP stream URL. Defaults to MARKER_TRACKER_RTSP_URL or local_settings.py.",
    )
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument(
        "--confidence",
        type=float,
        help="Override the minimum confidence for all detected objects.",
    )
    parser.add_argument("--person-confidence", type=float, default=0.35)
    parser.add_argument("--phone-confidence", type=float, default=0.15)
    parser.add_argument("--confirmation-seconds", type=float, default=3.0)
    parser.add_argument("--release-grace-seconds", type=float, default=3.0)
    parser.add_argument("--marker-dropout-seconds", type=float, default=0.5)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Install runtime dependencies first: pip install -r requirements.txt"
        ) from exc

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not args.rtsp_url:
        raise SystemExit(
            "Provide --rtsp-url, set MARKER_TRACKER_RTSP_URL, "
            "or create marker_tracker/local_settings.py."
        )

    capture = LatestFrameCapture(cv2, args.rtsp_url)
    if not capture.is_opened():
        raise SystemExit(f"Could not open RTSP stream: {_redact_url(args.rtsp_url)}")

    model = YOLO(args.model)
    tracker = MarkerTracker(
        confirmation_seconds=args.confirmation_seconds,
        release_grace_seconds=args.release_grace_seconds,
        marker_dropout_seconds=args.marker_dropout_seconds,
    )
    camera = LoggingCameraAdapter()
    previous_frame_at = None
    fps = 0.0
    frame_sequence = 0
    inference_confidence = (
        args.confidence
        if args.confidence is not None
        else min(args.person_confidence, args.phone_confidence)
    )

    try:
        while True:
            ok, frame, frame_sequence = capture.read_latest(frame_sequence)
            if not ok:
                logging.warning("RTSP frame read failed")
                break

            result = model.predict(frame, conf=inference_confidence, verbose=False)[0]
            current_frame_at = time.monotonic()
            if previous_frame_at is not None:
                current_fps = 1.0 / max(current_frame_at - previous_frame_at, 1e-9)
                fps = current_fps if fps == 0.0 else (0.9 * fps + 0.1 * current_fps)
            previous_frame_at = current_frame_at
            detections = _to_detections(
                result.names,
                result.boxes,
                person_confidence=(
                    args.confidence
                    if args.confidence is not None
                    else args.person_confidence
                ),
                phone_confidence=(
                    args.confidence
                    if args.confidence is not None
                    else args.phone_confidence
                ),
            )
            decision = tracker.update(detections, time.monotonic())
            if decision.emit_handoff and decision.target_box is not None:
                camera.handoff(decision.target_box)
            if decision.emit_release:
                camera.release()

            _draw(cv2, frame, detections, decision.state, decision.target_box, fps)
            cv2.imshow("phone marker PTZ prototype", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def _to_detections(
    names: dict[int, str],
    boxes: Sequence,
    person_confidence: float = 0.35,
    phone_confidence: float = 0.15,
) -> list[Detection]:
    detections = []
    for box in boxes:
        class_id = int(box.cls[0].item())
        label = names[class_id]
        if label not in {"person", "cell phone"}:
            continue
        confidence = float(box.conf[0].item())
        threshold = phone_confidence if label == "cell phone" else person_confidence
        if confidence < threshold:
            continue
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                box=BoundingBox(x1, y1, x2, y2),
            )
        )
    return detections


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


def _draw(cv2, frame, detections, state: str, target_box, fps: float) -> None:
    colors = {"person": (255, 160, 0), "cell phone": (0, 255, 255)}
    for detection in detections:
        box = detection.box
        cv2.rectangle(
            frame,
            (int(box.x1), int(box.y1)),
            (int(box.x2), int(box.y2)),
            colors[detection.label],
            2,
        )
        cv2.putText(
            frame,
            f"{detection.label} {detection.confidence:.2f}",
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
    cv2.putText(
        frame,
        f"state: {state}",
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
