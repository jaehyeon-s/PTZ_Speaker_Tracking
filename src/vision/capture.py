"""Video input handling without storing camera credentials in source code."""

from __future__ import annotations

import os
import threading
import time


class VideoSource:
    def __init__(
        self,
        source: str,
        rtsp_drop_frames: int = 0,
        read_timeout_seconds: float = 2.0,
        first_read_timeout_seconds: float = 10.0,
    ) -> None:
        import cv2

        self.cv2 = cv2
        self.source = source
        self.rtsp_drop_frames = max(0, rtsp_drop_frames)
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_ok = False
        self._latest_seq = 0
        self._consumed_seq = 0
        self._consecutive_failures = 0
        self._max_consecutive_failures = 60
        self._read_timeout_seconds = read_timeout_seconds
        self._first_read_timeout_seconds = first_read_timeout_seconds
        self._running = False
        self._thread: threading.Thread | None = None
        # Reconnect backoff after the stream stays down for _max_consecutive_failures
        # reads in a row (a camera reboot or Wi-Fi drop), so a transient outage ends
        # the RTSP capture object instead of the whole tracking session.
        self._reconnect_backoff_seconds = 1.0
        self._max_reconnect_backoff_seconds = 10.0
        self._reopen_source: int | str | None = None
        parsed_source = int(source) if source.isdigit() else source
        if source.lower().startswith("rtsp://"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|max_delay;500000")
            self._reopen_source = parsed_source
            self.capture = cv2.VideoCapture(parsed_source, cv2.CAP_FFMPEG)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._running = True
            self._thread = threading.Thread(target=self._capture_latest, daemon=True)
            self._thread.start()
        else:
            self.capture = cv2.VideoCapture(parsed_source)

    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def is_live(self) -> bool:
        """True for a live RTSP stream, where a read timeout means "still reconnecting",
        not "end of source" (which is only meaningful for finite files/webcams)."""
        return self.source.lower().startswith("rtsp://")

    def read(self):
        if self.source.lower().startswith("rtsp://"):
            timeout = self._first_read_timeout_seconds if self._consumed_seq == 0 else self._read_timeout_seconds
            deadline = time.monotonic() + timeout
            while True:
                with self._lock:
                    if self._latest_frame is not None and not self._latest_ok:
                        return False, None
                    if self._latest_seq > self._consumed_seq and self._latest_frame is not None:
                        self._consumed_seq = self._latest_seq
                        return self._latest_ok, self._latest_frame.copy()
                if time.monotonic() >= deadline:
                    return False, None
                time.sleep(0.005)
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
            self._capture_once()

    def _capture_once(self) -> None:
        """Run a single read (+ reconnect-if-needed) cycle; split out from
        _capture_latest's infinite loop so tests can drive it deterministically."""
        for _ in range(self.rtsp_drop_frames):
            self.capture.grab()
        ok, frame = self.capture.read()
        reconnect_needed = False
        with self._lock:
            if ok:
                self._latest_ok = True
                self._latest_frame = frame
                self._latest_seq += 1
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
            if self._latest_frame is None or self._consecutive_failures >= self._max_consecutive_failures:
                self._latest_ok = False
            if self._consecutive_failures >= self._max_consecutive_failures:
                reconnect_needed = True
        if reconnect_needed:
            self._reconnect()

    def _reconnect(self) -> None:
        """Reopen the RTSP capture after repeated read failures.

        A camera reboot or a Wi-Fi drop should end this capture object, not the
        whole tracking session: keep retrying with a capped backoff until the
        stream comes back, instead of leaving the source permanently failed.
        """
        backoff = self._reconnect_backoff_seconds
        while self._running:
            try:
                self.capture.release()
            except Exception:
                pass
            time.sleep(backoff)
            if not self._running:
                return
            self.capture = self.cv2.VideoCapture(self._reopen_source, self.cv2.CAP_FFMPEG)
            self.capture.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.capture.isOpened():
                with self._lock:
                    self._consecutive_failures = 0
                return
            backoff = min(backoff * 2, self._max_reconnect_backoff_seconds)
