# reid_bytetrack_demo.py
from ultralytics import YOLO
from zone_lock import ZoneLock
from reid_manager import ReIDManager
import cv2
import csv
import time
import torch
import os

# 설정
SOURCE = "one-by-one-person-detection.mp4"  # 영상 파일 또는 RTSP URL
OUTPUT = "demo_result.mp4"
LOG_FILE = "event_log.csv"
AUTO_REGISTER = True
AUTO_REGISTER_FRAME = 30

# 환경 자동 감지
if torch.cuda.is_available():
    MODEL = "yolo26n.pt"
    DEVICE = "cuda"
    print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
else:
    MODEL = "yolo26n_ncnn_model"
    DEVICE = "cpu"
    print("CPU 사용 (NCNN)")

# Re-ID 파라미터
SIMILARITY_THRESHOLD = 0.62   # 유사도 임계값
LOW_SCORE_FRAMES = 10         # 연속 N프레임 낮으면 SUSPENDED
REACQUIRE_SCORE = 0.75        # 재식별 임계값 (등록보다 약간 높게)

# 초기화
model = YOLO(MODEL, task="detect")
zone = ZoneLock()
reid = ReIDManager(
    similarity_threshold=SIMILARITY_THRESHOLD,
    low_score_frames=LOW_SCORE_FRAMES,
    reacquire_threshold=REACQUIRE_SCORE
)

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print(f"영상을 열 수 없습니다: {SOURCE}")
    exit()

ret, first = cap.read()
if not ret:
    print("첫 프레임 읽기 실패")
    exit()

h, w = first.shape[:2]
print(f"영상 크기: {w}x{h}")

# Zone 설정
if zone.zone is None:
    print("zone_config.json 없음 → 전체 화면을 Zone으로 설정")
    zone.zone = {"x_min": 0.05, "x_max": 0.95,
                 "y_min": 0.0,  "y_max": 1.0}

# VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT, fourcc, 15.0, (w, h))

# CSV 로그
log_f = open(LOG_FILE, "w", newline="")
log_writer = csv.writer(log_f)
log_writer.writerow([
    "time", "frame", "state", "event",
    "target_id", "best_id", "best_score"
])

# 상태 변수
frame_count = 0
fps_list = []
results = None
zone_persons = []
lecturer_id = None
last_event = ""
last_best_score = 0.0
last_best_id = None
is_rtsp = isinstance(SOURCE, str) and SOURCE.startswith("rtsp")
start_time = time.time()

print("\n조작법:")
print("  [s] 교수님 등록")
print("  [e] 세션 종료")
print("  [q] 종료")
print("\n처리 시작...\n")

MAX_FRAMES = 9999  # SSH 환경 자동 종료 (영상 끝까지)

def log_event(frame_n, state, event, target_id,
              best_id, best_score):
    elapsed = time.time() - start_time
    log_writer.writerow([
        f"{elapsed:.1f}", frame_n, state, event,
        target_id, best_id, f"{best_score:.3f}"
    ])
    log_f.flush()
    print(f"  [{frame_n:4d}f | {elapsed:5.1f}s] "
          f"{state:10s} | {event:30s} | "
          f"target={target_id} best={best_id} "
          f"score={best_score:.3f}")

# 메인 루프
while frame_count < MAX_FRAMES:
    if is_rtsp:
        cap.grab(); cap.grab(); cap.grab()
        ret, frame = cap.retrieve()
    else:
        ret, frame = cap.read()

    if not ret:
        print("영상 종료")
        break

    frame_count += 1

    # YOLO 추론 (5프레임에 1번)
    if frame_count % 5 == 0 or results is None:
        inf_start = time.time()
        results = model.track(
            frame, imgsz=416, conf=0.4,
            tracker="bytetrack.yaml",
            persist=True, verbose=False,
            device=DEVICE
        )
        inf_fps = 1 / (time.time() - inf_start)
        fps_list.append(inf_fps)

        # Zone 내 인물 수집
        zone_persons = []
        if results[0].boxes.id is not None:
            for box in results[0].boxes:
                if box.id is None or int(box.cls) != 0:
                    continue
                tid = int(box.id)
                xyxy = box.xyxy[0].tolist()
                if zone.is_in_zone(xyxy, w, h):
                    zone_persons.append((tid, xyxy))

        if (
            AUTO_REGISTER
            and reid.session_state == "IDLE"
            and frame_count >= AUTO_REGISTER_FRAME
            and zone_persons
        ):
            best = max(
                zone_persons,
                key=lambda p: (
                    (p[1][2] - p[1][0]) *
                    (p[1][3] - p[1][1])
                )
            )

            ok = reid.start_session(frame, best[1], track_id=best[0])
            if ok:
                lecturer_id = best[0]
                last_event = "AUTO_REGISTER"
                log_event(frame_count, "ACTIVE",
                        "AUTO_REGISTER",
                        best[0], best[0], 1.0)
        
        # Re-ID 업데이트
        prev_state = reid.session_state
        lecturer_id, last_best_id, last_best_score = \
            reid.update(zone_persons, frame)
        curr_state = reid.session_state

        # 이벤트 감지 및 로깅
        event = ""
        if prev_state != curr_state:
            if curr_state == "ACTIVE" and prev_state == "IDLE":
                event = "SESSION_STARTED"
            elif curr_state == "SUSPENDED" and prev_state == "ACTIVE":
                event = "WRONG_TARGET_SUSPECTED"
            elif curr_state == "ACTIVE" and prev_state == "SUSPENDED":
                event = "TARGET_REACQUIRED"
            elif curr_state == "IDLE":
                event = "SESSION_ENDED"

        elif curr_state == "ACTIVE" and lecturer_id is not None:
            event = "TARGET_CONFIRMED"

        if event:
            last_event = event
            log_event(frame_count, curr_state, event,
                      reid.target_id, last_best_id,
                      last_best_score)

    # 시각화
    if results is not None and results[0].boxes.id is not None:
        for box in results[0].boxes:
            if box.id is None or int(box.cls) != 0:
                continue

            tid = int(box.id)
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            in_zone = zone.is_in_zone(xyxy, w, h)
            is_lec = (tid == lecturer_id)

            if is_lec:
                # 교수님: 파란 박스
                color = (255, 100, 0)
                thickness = 3
                label = f"ID:{tid} LECTURER"
            elif in_zone:
                # Zone 내 일반 인물: 초록 박스
                color = (0, 255, 0)
                thickness = 2
                label = f"ID:{tid} IN"
            else:
                # Zone 외: 빨간 박스
                color = (0, 0, 255)
                thickness = 1
                label = f"ID:{tid} OUT"

            cv2.rectangle(frame, (x1,y1), (x2,y2),
                          color, thickness)
            cv2.putText(frame, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

            # 유사도 점수 표시 (Zone 내 인물만)
            if in_zone and reid.session_state != "IDLE":
                score = reid.last_scores.get(tid, 0.0)
                score_color = (0, 255, 0) if score >= SIMILARITY_THRESHOLD \
                    else (0, 165, 255) if score >= 0.45 \
                    else (0, 0, 255)
                cv2.putText(frame,
                            f"score:{score:.2f}",
                            (x1, y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, score_color, 1)

    # Zone 시각화
    frame = zone.draw(frame)

    # 상태 오버레이
    state_color = {
        "IDLE":      (128, 128, 128),
        "ACTIVE":    (0, 255, 0),
        "SUSPENDED": (0, 100, 255),
        "ENDED":     (0, 0, 255)
    }.get(reid.session_state, (255, 255, 255))

    # 반투명 배경
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 130),
                  (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:]) \
        if fps_list else 0

    cv2.putText(frame,
                f"State: {reid.session_state}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, state_color, 2)
    cv2.putText(frame,
                f"Lecturer: {lecturer_id}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 100, 0), 2)
    cv2.putText(frame,
                f"BestScore: {last_best_score:.2f} "
                f"(ID:{last_best_id})",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)
    cv2.putText(frame,
                f"Event: {last_event}",
                (10, 98), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1)
    cv2.putText(frame,
                f"FPS: {avg_fps:.1f} | "
                f"Frame: {frame_count}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (180, 180, 180), 1)

    out.write(frame)

    # 키 입력 (SSH에서는 동작 안 함, 로컬에서만)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if zone_persons:
            best = max(zone_persons,
                key=lambda p: (
                    (p[1][2]-p[1][0]) *
                    (p[1][3]-p[1][1])
                ))
            ok = reid.start_session(frame, best[1], track_id=best[0])
            if ok:
                lecturer_id = best[0]
                log_event(frame_count, "ACTIVE",
                        "MANUAL_REGISTER",
                        best[0], best[0], 1.0)
    elif key == ord('e'):
        reid.end_session()
        log_event(frame_count, "IDLE",
                  "MANUAL_END", None, None, 0.0)

# 종료
cap.release()
out.release()
log_f.close()
cv2.destroyAllWindows()

print(f"\n=== 완료 ===")
print(f"총 프레임: {frame_count}")
print(f"평균 FPS:  {sum(fps_list)/len(fps_list):.1f}" if fps_list else "")
print(f"결과 영상: {OUTPUT}")
print(f"이벤트 로그: {LOG_FILE}")
