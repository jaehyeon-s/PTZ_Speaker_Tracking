# bytetrack_test.py
from ultralytics import YOLO
import cv2
import time

RTSP = "rtsp://admin:비밀번호@ip주소/profile5/media.smp"
model = YOLO("yolo26n_ncnn_model")

cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

fps_list = []
track_history = {}  # ID별 이동 경로 저장

while True:
    cap.grab()
    cap.grab()
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    start = time.time()

    # ByteTrack 적용 (YOLO + 추적)
    results = model.track(
        frame,
        imgsz=416,
        conf=0.4,
        tracker="bytetrack.yaml",
        persist=True,      # 프레임 간 ID 유지 핵심
        verbose=False
    )

    fps = 1 / (time.time() - start)
    fps_list.append(fps)

    # 결과 처리
    if results[0].boxes.id is not None:
        for box in results[0].boxes:
            if box.id is None:
                continue

            track_id = int(box.id)
            cls = int(box.cls)

            if cls != 0:  # person 클래스만
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # ID별 이동 경로 저장 (최근 30프레임)
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((cx, cy))
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

            # 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 이동 경로 그리기
            pts = track_history[track_id]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (255, 255, 0), 1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Tracks: {len(track_history)}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("ByteTrack Test", frame)

    if len(fps_list) % 30 == 0:
        print(f"평균 FPS: {sum(fps_list[-30:])/30:.1f} | "
              f"추적 중 ID 수: {len(track_history)}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"전체 평균 FPS: {sum(fps_list)/len(fps_list):.1f}")
