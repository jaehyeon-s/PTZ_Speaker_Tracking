"""Video input handling without storing camera credentials in source code."""

from __future__ import annotations


class VideoSource:
    def __init__(self, source: str, rtsp_drop_frames: int = 0) -> None:
        import cv2

        self.cv2 = cv2
        self.source = source
        self.rtsp_drop_frames = max(0, rtsp_drop_frames)
        parsed_source = int(source) if source.isdigit() else source
        if source.lower().startswith("rtsp://"):
            self.capture = cv2.VideoCapture(parsed_source, cv2.CAP_FFMPEG)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.capture = cv2.VideoCapture(parsed_source)

    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def read(self):
        if self.source.lower().startswith("rtsp://") and self.rtsp_drop_frames:
            for _ in range(self.rtsp_drop_frames):
                self.capture.grab()
            return self.capture.retrieve()
        return self.capture.read()

    def fps(self) -> float:
        return float(self.capture.get(self.cv2.CAP_PROP_FPS))

    def release(self) -> None:
        self.capture.release()
