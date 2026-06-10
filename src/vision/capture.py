"""Video input handling without storing camera credentials in source code."""

from __future__ import annotations

import os
import threading
import time


class VideoSource:
    def __init__(self, source: str, rtsp_drop_frames: int = 0) -> None:
        import cv2

        self.cv2 = cv2
        self.source = source
        self.rtsp_drop_frames = max(0, rtsp_drop_frames)
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_ok = False
        self._running = False
        self._thread: threading.Thread | None = None
        parsed_source = int(source) if source.isdigit() else source
        if source.lower().startswith("rtsp://"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|max_delay;500000")
            self.capture = cv2.VideoCapture(parsed_source, cv2.CAP_FFMPEG)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._running = True
            self._thread = threading.Thread(target=self._capture_latest, daemon=True)
            self._thread.start()
        else:
            self.capture = cv2.VideoCapture(parsed_source)

    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def read(self):
        if self.source.lower().startswith("rtsp://"):
            deadline = time.monotonic() + 2.0
            while True:
                with self._lock:
                    if self._latest_frame is not None:
                        return self._latest_ok, self._latest_frame.copy()
                if time.monotonic() >= deadline:
                    return False, None
                time.sleep(0.01)
        return self.capture.read()

    def fps(self) -> float:
        return float(self.capture.get(self.cv2.CAP_PROP_FPS))

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.capture.release()

    def _capture_latest(self) -> None:
        while self._running:
            ok, frame = self.capture.read()
            with self._lock:
                if ok:
                    self._latest_ok = True
                    self._latest_frame = frame
                elif self._latest_frame is None:
                    self._latest_ok = False
