from virtual_ptz import VirtualPTZ
import cv2

cap = cv2.VideoCapture("result3.mp4")
ret, first = cap.read()
if not ret:
    print("영상 열기 실패")
    exit()

orig_h, orig_w = first.shape[:2]
print(f"원본 영상 크기: {orig_w}×{orig_h}")

# 원본 영상을 2배로 업스케일해서 여백 확보
SCALE = 2
big_w = orig_w * SCALE
big_h = orig_h * SCALE
print(f"업스케일 크기: {big_w}×{big_h}")
print(f"뷰포트 크기:   640×360")
print(f"이동 가능 여백: {big_w-640}×{big_h-360} 픽셀\n")

ptz = VirtualPTZ(frame_w=big_w, frame_h=big_h,
                 view_w=640, view_h=360)

# VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('ptz_result.mp4', fourcc, 15.0, (1280, 360))

# 시나리오 (프레임 기준)
scenarios = [
    (0,   30,  0.0,  0.0,  0.0),    # 0~2초: 정지
    (30,  75,  0.8,  0.0,  0.0),    # 2~5초: 오른쪽 이동
    (75,  105, 0.0,  0.0,  0.0),    # 5~7초: 정지
    (105, 150, -0.8, 0.0,  0.0),    # 7~10초: 왼쪽 이동
    (150, 195, 0.0,  0.0,  0.0),    # 10~13초: 정지
    (195, 240, 0.0,  0.8,  0.0),    # 13~16초: 아래 이동
    (240, 285, 0.0,  -0.8, 0.0),    # 16~19초: 위 이동
    (285, 330, 0.0,  0.0,  0.08),   # 19~22초: 줌 인
    (330, 375, 0.0,  0.0, -0.08),   # 22~25초: 줌 아웃
    (375, 450, 0.0,  0.0,  0.0),    # 25~30초: 정지
]

MAX_FRAMES = 450  # 30초 (15 FPS × 30)
frame_count = 0

print("녹화 시작...")

while frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            break

    frame_count += 1

    # 업스케일
    big_frame = cv2.resize(frame, (big_w, big_h),
                           interpolation=cv2.INTER_LINEAR)

    # 시나리오 적용
    for start, end, pan, tilt, zoom_delta in scenarios:
        if start <= frame_count < end:
            if zoom_delta > 0:
                ptz.zoom_in(zoom_delta)
                ptz.stop()
            elif zoom_delta < 0:
                ptz.zoom_out(abs(zoom_delta))
                ptz.stop()
            else:
                ptz.move(pan, tilt)
            break

    ptz.update()

    # Overview (원본 크기로 표시)
    overview = cv2.resize(big_frame, (640, 360))

    # PTZ 뷰박스 (업스케일 좌표 → 640×360 표시 좌표)
    state = ptz.get_state()
    vw = int(640 / ptz.zoom)
    vh = int(360 / ptz.zoom)
    vx1 = max(0, int(state["cx"] - vw/2))
    vy1 = max(0, int(state["cy"] - vh/2))
    vx2 = min(big_w, vx1 + vw)
    vy2 = min(big_h, vy1 + vh)

    # 업스케일 좌표 → 640×360 표시 좌표로 변환
    sx = 640 / big_w
    sy = 360 / big_h
    disp_x1 = int(vx1 * sx)
    disp_y1 = int(vy1 * sy)
    disp_x2 = int(vx2 * sx)
    disp_y2 = int(vy2 * sy)

    cv2.rectangle(overview,
                  (disp_x1, disp_y1),
                  (disp_x2, disp_y2),
                  (0, 255, 255), 2)

    # 정보 표시
    sec = frame_count // 15
    cv2.putText(overview,
                f"Time: {sec}s",
                (470, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0), 1)
    cv2.putText(overview,
                f"Zoom: {state['zoom']}x",
                (470, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0), 1)
    cv2.putText(overview,
                f"PTZ cx={state['cx']} cy={state['cy']}",
                (470, 71),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 255, 0), 1)
    cv2.putText(overview, "OVERVIEW",
                (530, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # PTZ 뷰포트
    ptz_view = ptz.get_view(big_frame)
    cv2.putText(ptz_view, "PTZ VIEW",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # 저장
    combined = cv2.hconcat([overview, ptz_view])
    out.write(combined)

    if frame_count % 15 == 0:
        print(f"  {frame_count//15}초 / 30초 처리 중...")

cap.release()
out.release()
print(f"\n완료: {frame_count}프레임 처리")
print("ptz_result.mp4 저장 완료")
