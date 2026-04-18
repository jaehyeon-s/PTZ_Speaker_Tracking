from ultralytics import YOLO
from zone_lock import ZoneLock
import cv2
import time

model = YOLO("yolo26n_ncnn_model")
zone = ZoneLock()
cap = cv2.VideoCapture("test4.mp4")

# 캘리브레이션 (zone_config.json 없을 때만)
if zone.zone is None:
    ret, frame = cap.read()
    zone.calibrate(frame)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps_list = []
track_history = {}
frame_count = 0
results = None
out_result = cv2.VideoWriter('result3.mp4', fourcc, 15.0, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 5프레임에 1번만 YOLO 추론
    if frame_count % 5 == 0 or results is None:
        start = time.time()
        results = model.track(
            frame, imgsz=(480, 640), conf=0.4,
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

    out_result.write(frame)

    cv2.imshow("ByteTrack + Zone Lock", frame)

    if len(fps_list) % 30 == 0:
        print(f"FPS: {sum(fps_list[-30:])/30:.1f} | "
              f"Zone my candidate: {candidates}")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        ret, frame = cap.read()
        zone.calibrate(frame)

cap.release()
out_result.release()
cv2.destroyAllWindows()

if fps_list:
    print(f"All AVG FPS: {sum(fps_list)/len(fps_list):.1f}")
