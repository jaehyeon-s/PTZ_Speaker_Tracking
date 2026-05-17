# reid_manager.py
import cv2
import numpy as np
import time


class ReIDManager:
    def __init__(self, similarity_threshold=0.6, low_score_frames=10, reacquire_threshold=0.65):
        self.lecturer_features = None   # 등록된 교수님 특징
        self.session_state = "IDLE"     # IDLE/ACTIVE/SUSPENDED/ENDED
        
        self.last_seen_time = None      # 마지막 감지 시각
        self.suspend_timeout = 15 * 60  # 15분 (SUSPENDED → ENDED)
        
        self.threshold = similarity_threshold
        self.reacquire_threshold = reacquire_threshold
        self.low_score_count = 0 
        self.low_score_frames = low_score_frames
        
        self.target_id = None           # 처음 등록했을 때의 ID
        self.current_id = None          # 현재 따라가야 하는 ID
        self.last_scores = {}

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
        hist_h = hist_h.flatten()
        hist_h = hist_h / (hist_h.sum() + 1e-6)

        hist_s = hist_s.flatten()
        hist_s = hist_s / (hist_s.sum() + 1e-6)

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
    def start_session(self, frame, xyxy, track_id=None):
        """교수님 특징 저장 + ACTIVE 전환"""
        features = self.extract_features(frame, xyxy)
        if features is None:
            print("[Re-ID] 특징 추출 실패")
            return False
            
        self.lecturer_features = features
        self.session_state = "ACTIVE"
        self.last_seen_time = time.time()
        self.low_score_count = 0
        
        self.target_id = track_id   # 처음 등록했을 때의 ID
        self.current_id = track_id  # 현재 따라가야 하는 ID
        self.last_scores = {}
        
        print("[Re-ID] 세션 시작 → ACTIVE")
        return True

    def end_session(self):
        """세션 완전 초기화 → IDLE"""
        self.lecturer_features = None
        self.session_state = "IDLE"
        self.last_seen_time = None
        
        self.low_score_count = 0
        self.target_id = None
        self.current_id = None
        self.last_scores = {}
        
        print("[Re-ID] 세션 종료 → IDLE, Re-ID 초기화 완료")
            
    def update(self, detected_persons, frame):
        """
        반환: (lecturer_id, best_id, best_score)
        """
        # IDLE이면 추적 안 함
        if self.session_state == "IDLE":
            return None, None, 0.0

        # SUSPENDED 타임아웃 체크
        if (
            self.session_state == "SUSPENDED"
            and self.last_seen_time is not None
            and time.time() - self.last_seen_time > self.suspend_timeout
        ):
            self.end_session()
            return None, None, 0.0

        # 감지된 인물 없음
        if not detected_persons:
            self.low_score_count += 1

            if (
                self.session_state == "ACTIVE"
                and self.low_score_count >= self.low_score_frames
            ):
                self.session_state = "SUSPENDED"
                self.last_seen_time = time.time()

            self.current_id = None
            self.last_scores = {}
            return None, None, 0.0

        # 매칭
        best_id = None
        best_score = 0.0
        self.last_scores = {}

        for track_id, xyxy in detected_persons:
            feat = self.extract_features(frame, xyxy)
            score = self.compare_features(self.lecturer_features, feat)

            self.last_scores[track_id] = score

            if score > best_score:
                best_score = score
                best_id = track_id

        required_score = (
            self.reacquire_threshold
            if self.session_state == "SUSPENDED"
            else self.threshold
        )

        # 매칭 성공
        if best_score >= required_score:
            self.low_score_count = 0
            self.session_state = "ACTIVE"
            self.last_seen_time = time.time()
            self.current_id = best_id
            return best_id, best_id, best_score

        # 매칭 실패
        self.low_score_count += 1

        if (
            self.session_state == "ACTIVE"
            and self.low_score_count >= self.low_score_frames
        ):
            self.session_state = "SUSPENDED"
            self.last_seen_time = time.time()
            self.current_id = None

        return None, best_id, best_score
				
    def get_state_info(self):
        return {
            "state": self.session_state,
            "has_lecturer": self.lecturer_features is not None,
            "last_seen": self.last_seen_time
        }
