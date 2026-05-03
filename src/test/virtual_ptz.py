# virtual_ptz.py

class VirtualPTZ:
    def __init__(self, frame_w, frame_h,
                 view_w=640, view_h=360):

	# 전체 입력 프레임 크기
        self.frame_w = frame_w  
        self.frame_h = frame_h

	# 출력 뷰포트 크기
        self.view_w = view_w
        self.view_h = view_h

        # 뷰포트 중심 (처음엔 프레임 중앙)
        self.cx = frame_w / 2
        self.cy = frame_h / 2

        # 줌 레벨 (1.0 = 광각, 3.0 = 망원)
        self.zoom = 1.0
        self.zoom_min = 1.0
        self.zoom_max = 4.0

        # 이동 속도 제한
        self.max_speed = 20       # 픽셀/프레임
        self.pan_speed = 0.0
        self.tilt_speed = 0.0

        # 저역통과 필터 (급격한 이동 방지)
        self.smooth_factor = 0.15

    # 현재 Pan/Tilt/Zoom 기준으로 뷰포트 반환
    def get_view(self, frame):
        import cv2

        # 줌에 따른 실제 뷰포트 크기 계산
        actual_w = int(self.view_w / self.zoom)
        actual_h = int(self.view_h / self.zoom)

        # 뷰포트 좌상단 좌표 계산
        x1 = int(self.cx - actual_w / 2)
        y1 = int(self.cy - actual_h / 2)
        x2 = x1 + actual_w
        y2 = y1 + actual_h

        # 경계 클리핑
        x1 = max(0, min(x1, self.frame_w - actual_w))
        y1 = max(0, min(y1, self.frame_h - actual_h))
        x2 = x1 + actual_w
        y2 = y1 + actual_h
        self.cx = (x1 + x2) / 2
        self.cy = (y1 + y2) / 2

        # 크롭 후 출력 크기로 리사이즈
        crop = frame[y1:y2, x1:x2]
        view = cv2.resize(crop, (self.view_w, self.view_h))
        return view

    def move(self, pan_speed, tilt_speed):
        self.pan_speed = pan_speed * self.max_speed  # -1.0(왼쪽) ~ +1.0(오른쪽)
        self.tilt_speed = tilt_speed * self.max_speed  # -1.0(위)   ~ +1.0(아래)

    def stop(self):
        self.pan_speed = 0.0
        self.tilt_speed = 0.0

    def zoom_in(self, speed=0.05):
        self.zoom = min(self.zoom_max, self.zoom + speed)

    def zoom_out(self, speed=0.05):
        self.zoom = max(self.zoom_min, self.zoom - speed)

    # Re-ID로 찾은 교수님 bbox 중심으로 이동
    def move_to_target(self, target_cx, target_cy,
                       frame_w, frame_h):
        # 현재 뷰포트 중심과 전체 프레임 기준 좌표 간 오차
        dx = target_cx - self.cx
        dy = target_cy - self.cy

        # 오차가 지정해 둔 값보다 작으면 이동 안 함
        deadband = 30  # 픽셀
        if abs(dx) < deadband and abs(dy) < deadband:
            self.stop()
            return

        # 저역통과 필터로 부드럽게 이동
        self.cx += dx * self.smooth_factor
        self.cy += dy * self.smooth_factor

    def update(self):
        # 매 프레임 위치 업데이트 (수동 이동 시)
        self.cx += self.pan_speed
        self.cy += self.tilt_speed

        # 경계 처리
        margin_w = (self.view_w / self.zoom) / 2
        margin_h = (self.view_h / self.zoom) / 2
        self.cx = max(margin_w,
                      min(self.frame_w - margin_w, self.cx))
        self.cy = max(margin_h,
                      min(self.frame_h - margin_h, self.cy))

    def get_state(self):
        return {
            "cx": int(self.cx),
            "cy": int(self.cy),
            "zoom": round(self.zoom, 2)
        }
