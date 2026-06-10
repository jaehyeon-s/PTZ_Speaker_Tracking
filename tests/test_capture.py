import threading
import time
import unittest

import numpy as np

from src.vision.capture import VideoSource


def make_rtsp_source(timeout: float = 0.001) -> VideoSource:
    source = VideoSource.__new__(VideoSource)
    source.source = "rtsp://camera/stream"
    source._lock = threading.Lock()
    source._latest_frame = None
    source._latest_ok = False
    source._latest_seq = 0
    source._consumed_seq = 0
    source._read_timeout_seconds = timeout
    source._first_read_timeout_seconds = timeout
    return source


class VideoSourceTests(unittest.TestCase):
    def test_rtsp_read_consumes_each_latest_frame_once(self):
        source = make_rtsp_source()
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        with source._lock:
            source._latest_ok = True
            source._latest_frame = frame
            source._latest_seq = 1

        ok, first = source.read()
        ok_again, second = source.read()

        self.assertTrue(ok)
        self.assertIsNot(first, frame)
        self.assertFalse(ok_again)
        self.assertIsNone(second)

    def test_rtsp_read_stops_after_capture_thread_marks_stream_failed(self):
        source = make_rtsp_source()
        with source._lock:
            source._latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)
            source._latest_ok = False
            source._latest_seq = 1

        ok, frame = source.read()

        self.assertFalse(ok)
        self.assertIsNone(frame)

    def test_rtsp_uses_first_read_timeout_only_before_first_consumed_frame(self):
        source = make_rtsp_source(timeout=0.1)
        source._read_timeout_seconds = 0.001
        source._first_read_timeout_seconds = 0.02

        start = time.monotonic()
        ok, frame = source.read()
        first_elapsed = time.monotonic() - start

        with source._lock:
            source._latest_ok = True
            source._latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)
            source._latest_seq = 1
        ok_frame, frame = source.read()

        start = time.monotonic()
        ok_again, frame_again = source.read()
        next_elapsed = time.monotonic() - start

        self.assertFalse(ok)
        self.assertGreaterEqual(first_elapsed, 0.015)
        self.assertTrue(ok_frame)
        self.assertIsNotNone(frame)
        self.assertFalse(ok_again)
        self.assertIsNone(frame_again)
        self.assertLess(next_elapsed, first_elapsed)


if __name__ == "__main__":
    unittest.main()
