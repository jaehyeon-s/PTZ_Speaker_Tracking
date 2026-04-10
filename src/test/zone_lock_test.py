from ultralytics import YOLO
from zone_lock import ZoneLock
import cv2
import time

RTSP = "rtsp://admin:비밀번호@ip주소/profile4/media.smp"
model = YOLO("yolo26n_ncnn_model")
zone = ZoneLock()

cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 캘리브레이션 (zone_config.json 없을 때만)
if zone.zone is None:
    ret, frame = cap.read()
    zone.calibrate(frame)

fps_list = []
track_history = {}

while True:
    cap.grab(); cap.grab(); cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    h, w = frame.shape[:2]

    start = time.time()
    results = model.track(
        frame, imgsz=416, conf=0.4,
        tracker="bytetrack.yaml",
        persist=True, verbose=False
    )
    fps = 1 / (time.time() - start)
    fps_list.append(fps)

    candidates = []  # Zone 내 후보만

    if results[0].boxes.id is not None:
        for box in results[0].boxes:
            if box.id is None or int(box.cls) != 0:
                continue

            tid = int(box.id)
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = (x1+x2)//2, (y1+y2)//2

            in_zone = zone.is_in_zone(xyxy, w, h)

            if in_zone:
                candidates.append(tid)
                # 초록: 강의 구역 내 (타겟 후보)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"ID:{tid} IN",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

                # 이동 경로
                if tid not in track_history:
                    track_history[tid] = []
                track_history[tid].append((cx,cy))
                if len(track_history[tid]) > 30:
                    track_history[tid].pop(0)
                for i in range(1, len(track_history[tid])):
                    cv2.line(frame, track_history[tid][i-1],
                             track_history[tid][i], (255,255,0), 1)
            else:
                # 빨강: 강의 구역 외 (제외됨)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 1)
                cv2.putText(frame, f"ID:{tid} OUT",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,0,255), 1)

    frame = zone.draw(frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, f"Candidates: {candidates}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("ByteTrack + Zone Lock", frame)

    if len(fps_list) % 30 == 0:
        print(f"FPS: {sum(fps_list[-30:])/30:.1f} | "
              f"Zone 내 후보: {candidates}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Zone 재설정: 'c' 누르면 캘리브레이션 재시작
    if cv2.waitKey(1) & 0xFF == ord('c'):
        ret, frame = cap.read()
        zone.calibrate(frame)

cap.release()
cv2.destroyAllWindows()
print(f"전체 평균 FPS: {sum(fps_list)/len(fps_list):.1f}")
