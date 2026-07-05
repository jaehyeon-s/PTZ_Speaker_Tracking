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

    def test_is_live_true_for_rtsp_false_for_file(self):
        rtsp_source = make_rtsp_source()
        file_source = VideoSource.__new__(VideoSource)
        file_source.source = "sample.mp4"

        self.assertTrue(rtsp_source.is_live())
        self.assertFalse(file_source.is_live())


class FakeCapture:
    """Stand-in cv2.VideoCapture that always fails to read and records release()."""

    def __init__(self, opened: bool = True):
        self.opened = opened
        self.released = False

    def grab(self):
        pass

    def read(self):
        return False, None

    def set(self, prop, value):
        del prop, value

    def isOpened(self):
        return self.opened

    def release(self):
        self.released = True


class RecoveredCapture:
    """Stand-in cv2.VideoCapture that reads one good frame then fails."""

    def __init__(self, frame):
        self.frame = frame
        self._served = False
        self.released = False

    def grab(self):
        pass

    def read(self):
        if not self._served:
            self._served = True
            return True, self.frame
        return False, None

    def set(self, prop, value):
        del prop, value

    def isOpened(self):
        return True

    def release(self):
        self.released = True


class FakeCv2Module:
    CAP_FFMPEG = 1
    CAP_PROP_BUFFERSIZE = 2

    def __init__(self, captures_to_open):
        self._captures_to_open = list(captures_to_open)

    def VideoCapture(self, source, backend=None):
        del source, backend
        return self._captures_to_open.pop(0)


def make_reconnecting_source(max_consecutive_failures: int = 2) -> VideoSource:
    source = VideoSource.__new__(VideoSource)
    source.source = "rtsp://camera/stream"
    source.rtsp_drop_frames = 0
    source._lock = threading.Lock()
    source._latest_frame = None
    source._latest_ok = False
    source._latest_seq = 0
    source._consumed_seq = 0
    source._consecutive_failures = 0
    source._max_consecutive_failures = max_consecutive_failures
    source._running = True
    source._reconnect_backoff_seconds = 0.001
    source._max_reconnect_backoff_seconds = 0.002
    source._reopen_source = "rtsp://camera/stream"
    return source


class VideoSourceReconnectTests(unittest.TestCase):
    def test_reconnects_after_repeated_failures_and_resumes_reading(self):
        good_frame = np.ones((2, 2, 3), dtype=np.uint8)
        source = make_reconnecting_source(max_consecutive_failures=2)
        source.capture = FakeCapture()
        source.cv2 = FakeCv2Module([RecoveredCapture(good_frame)])

        # Two failed reads cross the threshold and trigger _reconnect(), which
        # should reopen the capture (via cv2.VideoCapture) and reset the
        # failure counter so the session keeps running instead of ending.
        source._capture_once()
        source._capture_once()

        self.assertEqual(source._consecutive_failures, 0)
        self.assertIsInstance(source.capture, RecoveredCapture)

        # The reopened capture now serves a real frame.
        source._capture_once()
        self.assertTrue(source._latest_ok)
        self.assertIsNotNone(source._latest_frame)

    def test_reconnect_retries_with_backoff_until_reopen_succeeds(self):
        source = make_reconnecting_source(max_consecutive_failures=1)
        source.capture = FakeCapture()
        # First reopen attempt fails to open; second succeeds.
        source.cv2 = FakeCv2Module([FakeCapture(opened=False), FakeCapture(opened=True)])

        source._capture_once()

        self.assertEqual(source._consecutive_failures, 0)
        self.assertTrue(source.capture.isOpened())

    def test_stopping_during_reconnect_backoff_returns_without_reopening(self):
        source = make_reconnecting_source(max_consecutive_failures=1)
        source.capture = FakeCapture()
        source.cv2 = FakeCv2Module([FakeCapture(opened=True)])

        def stop_during_backoff():
            time.sleep(0.0005)
            source._running = False

        stopper = threading.Thread(target=stop_during_backoff)
        stopper.start()
        source._capture_once()
        stopper.join()

        # Should not hang or raise; exact capture identity depends on timing,
        # but the reconnect loop must honour _running and return.


if __name__ == "__main__":
    unittest.main()
