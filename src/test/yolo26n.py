from ultralytics import YOLO
import cv2
import time

RTSP = "rtsp://admin:비밀번호@ip주소/profile5/media.smp"
model = YOLO("yolo26n.pt")  # CPU로 자동 동작

cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

fps_list = []

while True:
    cap.grab()
    cap.grab()
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    start = time.time()

    # imgsz 두 가지 비교 테스트
    results = model(frame, imgsz=416, conf=0.4, verbose=False)

    fps = 1 / (time.time() - start)
    fps_list.append(fps)

    annotated = results[0].plot()
    cv2.imshow("PIt Test", annotated)

    if len(fps_list) % 30 == 0:
        print(f"평균 FPS (imgsz=416): {sum(fps_list[-30:])/30:.1f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("전체 평균 FPS: {sum(fps_list)/len(fps_list):.1f}")
