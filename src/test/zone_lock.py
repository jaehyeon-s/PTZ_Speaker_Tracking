import cv2
import json
import numpy as np

class ZoneLock:
    def __init__(self, config_path="zone_config.json"):
        self.zone = None
        self.config_path = config_path
        self._load()

    def _load(self):
        try:
            with open(self.config_path) as f:
                self.zone = json.load(f)
            print(f"Zone 로드 완료: {self.zone}")
        except:
            print("zone_config.json 없음 → 캘리브레이션 필요")

    def calibrate(self, frame):
        """마우스로 강의 구역 지정"""
        points = []
        clone = frame.copy()

        def click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(clone, (x,y), 5, (0,255,0), -1)
                cv2.imshow("Calibration", clone)

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", click)
        cv2.putText(clone, "좌상단 클릭 -> 우하단 클릭 (2번)",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,255), 2)
        cv2.imshow("Calibration", clone)

        while len(points) < 2:
            cv2.waitKey(1)

        cv2.destroyWindow("Calibration")

        h, w = frame.shape[:2]
        x1, y1 = points[0]
        x2, y2 = points[1]

        self.zone = {
            "x_min": min(x1,x2) / w,
            "x_max": max(x1,x2) / w,
            "y_min": min(y1,y2) / h,
            "y_max": max(y1,y2) / h
        }

        with open(self.config_path, "w") as f:
            json.dump(self.zone, f)

        print(f"Zone 저장 완료: {self.zone}")
        return self.zone

    def is_in_zone(self, xyxy, frame_w, frame_h):
        """bbox 중심이 강의 구역 안인지 확인"""
        if self.zone is None:
            return True  # Zone 미설정 시 모두 허용

        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2 / frame_w
        cy = (y1 + y2) / 2 / frame_h

        return (self.zone["x_min"] < cx < self.zone["x_max"] and
                self.zone["y_min"] < cy < self.zone["y_max"])

    def draw(self, frame):
        """Zone 시각화"""
        if self.zone is None:
            return frame

        h, w = frame.shape[:2]
        x1 = int(self.zone["x_min"] * w)
        y1 = int(self.zone["y_min"] * h)
        x2 = int(self.zone["x_max"] * w)
        y2 = int(self.zone["y_max"] * h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), -1)
        frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, "Lecture Zone", (x1+5, y1+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return frame
