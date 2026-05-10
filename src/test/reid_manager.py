# reid_manager.py
import cv2
import numpy as np
import time


class ReIDManager:
    def __init__(self, similarity_threshold=0.6):
        self.lecturer_features = None   # 등록된 교수님 특징
        self.session_state = "IDLE"     # IDLE/ACTIVE/SUSPENDED/ENDED
        self.last_seen_time = None      # 마지막 감지 시각
        self.suspend_timeout = 15 * 60  # 15분 (SUSPENDED → ENDED)
        self.threshold = similarity_threshold

    # 특징 추출
    def extract_features(self, frame, xyxy):
        """bbox에서 색상 히스토그램 + 체형 비율 추출"""
        x1, y1, x2, y2 = map(int, xyxy)
        h, w = frame.shape[:2]

        # 경계 클리핑
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # 상체만 사용 (하체는 가려지는 경우 많음)
        upper_h = max(1, (y2 - y1) // 2)
        upper_crop = crop[:upper_h, :]

        # HSV 색상 히스토그램
        hsv = cv2.cvtColor(upper_crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [36], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])

        # 정규화
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()

        # 체형 비율
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        aspect_ratio = bbox_w / (bbox_h + 1e-5)

        return {
            "hist_h": hist_h.tolist(),
            "hist_s": hist_s.tolist(),
            "aspect_ratio": float(aspect_ratio)
        }

    def compare_features(self, feat1, feat2):
        """두 특징 간 유사도 계산 (0~1, 높을수록 유사)"""
        if feat1 is None or feat2 is None:
            return 0.0

        hist_h1 = np.array(feat1["hist_h"]).astype(np.float32)
        hist_h2 = np.array(feat2["hist_h"]).astype(np.float32)
        hist_s1 = np.array(feat1["hist_s"]).astype(np.float32)
        hist_s2 = np.array(feat2["hist_s"]).astype(np.float32)

        # 히스토그램 교차 유사도
        sim_h = cv2.compareHist(hist_h1, hist_h2, cv2.HISTCMP_INTERSECT)
        sim_s = cv2.compareHist(hist_s1, hist_s2, cv2.HISTCMP_INTERSECT)

        # 체형 비율 유사도
        ar_diff = abs(feat1["aspect_ratio"] - feat2["aspect_ratio"])
        sim_ar = max(0.0, 1.0 - ar_diff * 2)

        # 가중 합산
        total = sim_h * 0.5 + sim_s * 0.3 + sim_ar * 0.2
        return float(np.clip(total, 0.0, 1.0))

    # 세션 관리
    def start_session(self, frame, xyxy):
        """교수님 특징 저장 + ACTIVE 전환"""
        features = self.extract_features(frame, xyxy)
        if features is None:
            print("[Re-ID] 특징 추출 실패")
            return False

        self.lecturer_features = features
        self.session_state = "ACTIVE"
        self.last_seen_time = time.time()
        print("[Re-ID] 세션 시작 → ACTIVE")
        return True

    def end_session(self):
        """세션 완전 초기화 → IDLE"""
        self.lecturer_features = None
        self.session_state = "IDLE"
        self.last_seen_time = None
        print("[Re-ID] 세션 종료 → IDLE, Re-ID 초기화 완료")

    def update(self, detected_persons, frame):
        """
        매 프레임 호출
        detected_persons: [(track_id, xyxy), ...]
        반환: 교수님으로 판단된 track_id (없으면 None)
        """
        # IDLE이면 추적 안 함
        if self.session_state == "IDLE":
            return None

        # SUSPENDED 타임아웃 체크
        if self.session_state == "SUSPENDED":
            if time.time() - self.last_seen_time > self.suspend_timeout:
                self.end_session()
                print("[Re-ID] 15분 초과 → ENDED → IDLE")
                return None

        # 감지된 인물 없음
        if not detected_persons:
            if self.session_state == "ACTIVE":
                self.session_state = "SUSPENDED"
                self.last_seen_time = time.time()
                print("[Re-ID] 교수님 미감지 → SUSPENDED")
            return None

        # Re-ID 매칭
        best_id = None
        best_score = 0.0

        for track_id, xyxy in detected_persons:
            feat = self.extract_features(frame, xyxy)
            score = self.compare_features(self.lecturer_features, feat)

            print(f"  ID:{track_id} 유사도: {score:.3f}", end="")
            if score > best_score:
                best_score = score
                best_id = track_id
            print()

        if best_score >= self.threshold:
            # 매칭 성공
            if self.session_state == "SUSPENDED":
                print(f"[Re-ID] 재식별 성공 (유사도:{best_score:.2f}) → ACTIVE")
            self.session_state = "ACTIVE"
            self.last_seen_time = time.time()
            return best_id
        else:
            # 매칭 실패
            if self.session_state == "ACTIVE":
                self.session_state = "SUSPENDED"
                self.last_seen_time = time.time()
                print(f"[Re-ID] 매칭 실패 (최고:{best_score:.2f}) → SUSPENDED")
            return None

    def get_state_info(self):
        return {
            "state": self.session_state,
            "has_lecturer": self.lecturer_features is not None,
            "last_seen": self.last_seen_time
        }
