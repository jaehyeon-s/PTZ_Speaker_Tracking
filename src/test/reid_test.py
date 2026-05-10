from ultralytics import YOLO
from zone_lock import ZoneLock
from reid_manager import ReIDManager
import cv2

SOURCE = "one-by-one-person-detection.mp4"
model = YOLO("yolo26n.pt", task="detect")
zone = ZoneLock()
reid = ReIDManager(similarity_threshold=0.6)

cap = cv2.VideoCapture(SOURCE)
ret, first = cap.read()
if not ret:
    print("영상 열기 실패")
    exit()

h, w = first.shape[:2]
print(f"영상 크기: {w}x{h}")

# VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('reid_result.mp4', fourcc, 15.0, (w, h))

frame_count = 0
zone_persons = []
lecturer_id = None
results = None

# 자동 시나리오
AUTO_REGISTER_FRAME = 100   # 2초 후 자동 등록
AUTO_END_FRAME = 60       # 약 13초 후 세션 종료

print("처리 시작...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 자동 시나리오
    if frame_count == AUTO_REGISTER_FRAME:
        if zone_persons:
            best = max(zone_persons,
                key=lambda p: (p[1][2]-p[1][0]) * (p[1][3]-p[1][1]))
            success = reid.start_session(frame, best[1])
            if success:
                print(f"[자동 등록] 프레임 {frame_count}: ID {best[0]} 교수님 등록")

    if frame_count == AUTO_END_FRAME:
        reid.end_session()
        print(f"[자동 종료] 프레임 {frame_count}: 세션 종료")

    # YOLO 추론 (5프레임에 1번)
    if frame_count % 5 == 0 or results is None:
        results = model.track(
            frame, imgsz=416, conf=0.4,
            tracker="bytetrack.yaml",
            persist=True, verbose=False,
            device="cuda"
        )

        zone_persons = []
        if results[0].boxes.id is not None:
            for box in results[0].boxes:
                if box.id is None or int(box.cls) != 0:
                    continue
                tid = int(box.id)
                xyxy = box.xyxy[0].tolist()
                if zone.is_in_zone(xyxy, w, h):
                    zone_persons.append((tid, xyxy))

        lecturer_id = reid.update(zone_persons, frame)

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

            color = (255, 100, 0) if is_lec \
                else (0, 255, 0) if in_zone \
                else (0, 0, 255)
            thickness = 3 if is_lec else 2 if in_zone else 1
            label = "LECTURER" if is_lec \
                else "IN" if in_zone else "OUT"

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
            cv2.putText(frame, f"ID:{tid} {label}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    frame = zone.draw(frame)

    info = reid.get_state_info()
    state_color = {
        "IDLE":      (128,128,128),
        "ACTIVE":    (0,255,0),
        "SUSPENDED": (0,165,255),
        "ENDED":     (0,0,255)
    }.get(info["state"], (255,255,255))

    cv2.putText(frame, f"State: {info['state']}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, state_color, 2)
    cv2.putText(frame, f"Lecturer: {lecturer_id}",
                (10,55), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,100,0), 2)
    cv2.putText(frame, f"Candidates: {[p[0] for p in zone_persons]}",
                (10,80), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200,200,200), 1)

    out.write(frame)

    if frame_count % 30 == 0:
        print(f"  [{frame_count}프레임] "
              f"State={info['state']} | "
              f"Lecturer={lecturer_id} | "
              f"Zone내={[p[0] for p in zone_persons]}")

cap.release()
out.release()
print(f"\n완료: {frame_count}프레임 처리")
print("reid_result.mp4 저장 완료")
