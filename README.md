# PTZ Marker Tracking MVP

이 프로젝트는 마커(Marker)를 통해 추적 대상을 명확히 지정하는 PTZ 자동 추적 시스템의 졸업 작품용 MVP입니다.

웹캠, RTSP 스트림, 또는 동영상 파일에서 영상 프레임을 읽어와 사람과 ArUco 마커를 동시에 감지합니다. 감지된 마커의 중심점이 특정 사람의 바운딩 박스(bbox) 안에 들어가면 그 사람을 '최종 타겟'으로 고정(Lock-on)합니다. 이후 마커 중심이 아닌 선택된 사람의 bbox 중심을 기준으로 PTZ 시뮬레이터 조작 명령을 내보냅니다.

## Run

```bash
pip install -r requirements.txt
python3 main_marker_tracking_demo.py --source 0
python3 main_marker_tracking_demo.py --source path/to/video.mp4
python3 main_marker_tracking_demo.py --source rtsp://user:pass@camera/stream
```

주요 실행 옵션(플래그)
`--no-window`: 화면(GUI 창)을 띄우지 않고, 터미널에 시뮬레이터 로그만 출력하는 헤드리스(Headless) 모드입니다.
`--gesture`: 마커가 없을 때 작동하는 '제스처 보조 타겟 지정' 기능을 활성화합니다.
`--target-marker-id 7`: 특정 ID(여기서는 7번)를 가진 마커만 타겟으로 인식하도록 제한합니다. 다른 ID의 마커도 화면에 그려지긴 하지만, 주인공 등록은 무시됩니다.
`--detector manual --person-box x,y,w,h`: 고정된 사람 바운딩 박스를 수동으로 입력하는 모드입니다. 알고리즘 검증이나 데모용 고정 시나리오를 테스트할 때 유용합니다.

## ArUco Marker

Generate a printable marker:

```bash
python3 tools/generate_aruco_marker.py --id 7 --size 800 --output marker_7.png
```

출력 및 부착 팁
생성기는 기본적으로 DICT_4X4_50 사전(Dictionary)을 사용합니다. 
마커를 인쇄할 때는 사각형 주변의 흰색 테두리(Quiet Zone)가 잘리지 않도록 주의하고, 
구겨지지 않게 판판한 상태로 추적 대상 인원의 가슴이나 몸에 부착해야 카메라가 잘 인식합니다.

안정적인 졸업 작품 시연을 위해 아래처럼 특정 마커 ID를 지정하여 실행하는 것을 권장합니다.

```bash
python3 main_marker_tracking_demo.py --source 0 --target-marker-id 7
```

## Detector Backends

사용 환경에 맞춰 `--detector {hog,opencv-yolo,ncnn,manual}` 옵션으로 탐지기를 선택할 수 있습니다. 
hog 방식은 프로젝트 내에 다른 탐지기를 쓸 수 없을 때 사용하는 최소한의 비상용 백엔드입니다.

OpenCV DNN / ONNX 사용 시 주의사항
내장된 `YoloOpenCVPersonDetector`는 임시 데모용 어댑터입니다. 
YOLO 모델은 내보내기(Export) 방식이나 버전에 따라 ONNX 출력 레이아웃(구조)이 완전히 달라지므로, 
실제 모델을 교체할 때는 모델의 출력 포맷과 파서(Parser) 코드가 일치하는지 반드시 확인해야 합니다.

Raspberry Pi 5 NCNN example:

```bash
python3 main_marker_tracking_demo.py \
  --source 0 \
  --detector ncnn \
  --ncnn-param models/yolo.param \
  --ncnn-bin models/yolo.bin \
  --ncnn-input-size 640 \
  --conf-threshold 0.35 \
  --nms-threshold 0.45 \
  --target-marker-id 7
```

기존 프로젝트에 구현해 둔 NCNN용 YOLO 디코딩 및 NMS(비최대 억제) 로직이 있다면, 
해당 파서를 `NCNNPersonDetector(output_parser=...)`의 인자로 넘겨주어 그대로 재사용할 수 있습니다. 
기본 내장된 범용 NCNN 파서는 `[x1, y1, x2, y2, score, class_id]` 형태의 행(Row) 데이터를 기반으로 보수적으로 동작하므로, 
특수한 모델 헤드를 쓸 경우 프로젝트 전용 어댑터를 거쳐야 합니다.

## Existing Detector Adapter

기존 연구실이나 팀에서 쓰던 YOLO 또는 ByteTrack 기반 탐지기 코드를 수정하지 않고 그대로 감싸서(Wrap) 이 MVP 시스템에 연동할 수 있습니다.

```python
from person_detector import DetectionResultAdapter, ExistingProjectPersonDetectorAdapter

person_detector = ExistingProjectPersonDetectorAdapter(existing_detector, bbox_format="xyxy")
people = person_detector.detect(frame)
```

이 어댑터는 ByteTrack 스타일의 `tlwh`(top-left, width, height), 
YOLO 스타일의 `xyxy`, 딕셔너리/객체 형태 및 일반 2차원 배열(`[x1, y1, x2, y2, score, class_id]`) 출력을 모두 지원합니다. 
변환 결과는 이 MVP의 표준 데이터 구조인 `PersonDetection(bbox=(x, y, w, h), confidence=score)` 형태로 깔끔하게 정제됩니다.

## Keyboard Controls

- `q`: 프로그램 종료 (Quit)
- `r`: 현재 고정된 타겟 리셋 (Reset)
- `s`: 카메라 추적 일시 정지/재개 (Stop/Start tracking)
- `m`: 마커만 단독으로 추적하는 모드 토글 (Marker-only mode)
- `g`: 제스처 보조 기능 활성화/비활성화 (Gesture fallback)

## MVP Behavior

- 타겟 고정 알고리즘: ArUco 마커의 중심점이 사람의 바운딩 박스 내부에 들어오는 순간에만 타겟(주인공)으로 최초 등록됩니다.
- PTZ 제어 기준점: 카메라 제어 명령(팬/틸트)은 마커 위치가 아니라, 마커를 품고 있는 사람 바운딩 박스의 중심점을 기준으로 계산합니다.
- 데드존(Dead Zone): 화면 정중앙을 기준으로 일정 영역의 '안전지대(데드존)'를 둡니다.
타겟이 이 영역 안에서 미세하게 움직일 때는 카메라 조작 명령을 내리지 않아 화면이 덜덜 떨리는 지터(Jitter) 현상을 방지합니다.
- 마커 소실 예외 처리: 옷이나 장애물에 마커가 잠시 가려지더라도 시스템이 타겟을 즉시 놓치지 않습니다. 
내장된 보조 박스 추적기(Fallback bbox tracker)가 이전 프레임 위치를 기반으로 근처의 동일 인물을 계속 쫓아갑니다. 
만약 주변에 매칭되는 사람이 완전히 사라지면 시스템은 `SUSPENDED(보류)` 상태로 진입하며, 
지정한 프레임 수(`--max-suspended-frames`) 동안 대기한 후 타겟을 초기화합니다.
- 제스처 보조(Gesture Fallback): 마커 기반 타겟이 지정되지 않은 클린 상태에서만 동작합니다. 
MediaPipe가 설치되어 있고, 화면 내에 딱 한 사람만 1초 이상 손목을 어께 위로 올리는 행동(Raised Hand)을 유지하면,
그 사람을 임시 타겟으로 등록합니다. 등록이 완료된 이후에는 제스처 판단을 멈추고 바운딩 박스 추적기로 전환됩니다.

하드웨어 연동 시 참고
실제 카메라 장비와의 연동 코드는 `ptz_controller.py` 파일 내부의 `HardwarePTZController` 클래스에 작성하면 됩니다. 
현재 코드는 실제 카메라가 없어도 원활하게 시연 및 디버깅을 할 수 있도록 
터미널에 상하좌우(LEFT/RIGHT/UP/DOWN/STOP) 제어 명령을 출력해 주는 시뮬레이터 형태로 안전하게 구현되어 있습니다.
