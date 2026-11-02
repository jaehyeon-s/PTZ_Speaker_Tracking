# ArUco Marker PTZ Prototype

ArUco 마커를 사용해 PTZ 카메라의 내장 추적 대상으로 넘길 사람을 선택하는
초기 프로토타입입니다. 기본값은 로그 전용이지만, Qon 카메라는 HTTP
`ptzctrl.cgi` 제어 경로로 pan/tilt 명령을 보낼 수 있습니다.

현재 구현 범위, ID 의미, 발화자 변경 절차와 남은 실제 환경 검증 항목은
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)에 정리되어 있습니다.

## 현재 동작

- RTSP 영상에 `yolo26n.pt` 모델을 적용해 `person` 검출
- Ultralytics ByteTrack으로 `person` 추적 ID 부여
- OpenCV ArUco로 마커 검출
- ArUco 마커 중심점이 `person` 바운딩 박스 내부인지 판정해 대상 ID 선택
- 조건이 3초 동안 연속 유지되면 해당 사람의 바운딩 박스로 handoff 요청
- 마커가 사라져도 같은 ByteTrack ID가 보이면 계속 추적
- tracker ID가 바뀌거나 잠시 없어져도 bbox의 시간적 이동, 상단 위치, 너비와
  수평 위치가 연속적인 명확한 단일 후보면 같은 발화자로 재결합
- 추적 중 마커가 다른 ByteTrack ID 안에 안정적으로 들어오면 확인 시간 이후
  새 대상 ID로 handoff 요청
- 마커를 포함하는 사람 후보가 여러 명이면 detection confidence, 현재 추적
  ID 유지 보너스, bbox 면적/중심 거리 패널티를 합산해 가장 높은 후보 선택
- 새 후보 점수가 충분히 높고 전환 cooldown을 지난 경우에만 대상 전환 시작
- 현재 추적 bbox가 갑자기 커지면 큰 bbox 후보에 패널티를 부여
- 선택된 ID가 사라지면 검증된 provisional 후보는 저속 PTZ 추종 대상으로
  사용하고, 후보가 없거나 모호하면 PTZ를 정지한 채 grace 이후 release 요청
- 초기 테스트 범위를 명확히 하기 위해 마커가 정확히 하나일 때만 판정
- 겹치는 사람 박스가 있으면 마커를 포함하는 가장 작은 박스 선택
- 마커 없이 사용할 경우 화면에서 사람 박스를 클릭해 수동으로 대상 ID 선택
- 5초마다 FPS, 현재 추적 ID 수, 권장 ByteTrack `track_buffer` 값을 로그로 출력
- ArUco가 불안정할 경우 YOLO class를 `marker` 신호로 사용하는 fallback 모드 제공
- handoff/release/switching 이유, marker score, PTZ 명령, lost/recovery 상태를
  로그와 화면 overlay로 확인 가능
- `center`, `upper-body`, `full-body`, `board-left`, `board-right` framing 정책으로
  발표자 또는 보드 여백 중심의 PTZ 구도 제어 가능

## 설치 및 실행

Python 3.10 이상 환경에서 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# ONNX 파일로 CPU ReID를 쓸 때만 추가 설치 (--speaker-reid-hef로 NPU를 쓰면 필요 없음)
pip install -r requirements-onnx-reid.txt
python3 -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
cp marker_tracker/local_settings.example.py marker_tracker/local_settings.py
# local_settings.py의 RTSP_URL을 실제 주소로 수정
python3 -m marker_tracker.app
```

라즈베리 파이 5에서 Qon HTTP PTZ 제어로 실행할 때는 현장 기본값을 묶은
프로파일을 사용할 수 있습니다. ArUco 마커 ID 0, NCNN 모델, Pi용 ByteTrack
설정, 2초 handoff 확인 시간, Qon HTTP 제어가 자동 적용됩니다.

```bash
python3 -m marker_tracker.app \
  --runtime-profile pi-qon-http \
  --rtsp-url "rtsp://USER:PASSWORD@CAMERA_IP:554/STREAM"
```

`--qon-camera-url`을 생략하면 RTSP URL의 host를 사용해 `http://CAMERA_IP`를
자동으로 만듭니다. 프로파일 값은 일반 옵션으로 덮어쓸 수 있습니다. 예를 들어
전환 확인 시간을 1.5초로 낮추려면 `--confirmation-seconds 1.5`를 추가합니다.
창 없이 먼저 안정성만 확인하려면 `--no-display`를 추가하고, PTZ HTTP
요청/응답을 확인해야 할 때만 `--qon-log-commands`를 추가합니다.

`model.track()` 경로에서 native crash가 반복되면 Ultralytics tracker를
우회해 `model.predict()`와 앱 내부 IoU tracker를 사용하는 안정성 확인 경로로
실행합니다.

```bash
python3 -m marker_tracker.app \
  --runtime-profile pi-qon-http \
  --rtsp-url "rtsp://USER:PASSWORD@CAMERA_IP:554/STREAM" \
  --person-tracker simple-iou \
  --no-display
```

화면에서 `q` 또는 `Esc`를 누르면 종료합니다. 모델 파일이 로컬에 없다면
Ultralytics가 최초 실행 시 내려받으므로 네트워크 연결이 필요합니다.
RTSP URL 형식과 스트림 경로는 카메라 설정에 맞게 입력해야 합니다.
로컬 카메라 주소와 인증정보가 들어가는 `local_settings.py`는 소스 관리에서
제외됩니다. 필요하면 `--rtsp-url` 옵션이나 `MARKER_TRACKER_RTSP_URL` 환경
변수로 로컬 설정을 덮어쓸 수 있습니다.

실행 화면에는 사람과 ArUco 마커 바운딩 박스, 선택된 추적 대상, 현재 상태,
상태 전환 이유, PTZ 명령, framing 모드, 실시간 FPS가 표시됩니다.

RTSP 입력은 별도 스레드에서 계속 읽으며, 추론은 가장 최신 프레임을
사용합니다. 특정 마커 ID만 추적하거나 짧은 인식 누락 허용 시간을 조절하려면
다음과 같이 실행합니다.

```bash
python3 -m marker_tracker.app \
  --aruco-marker-id 0 \
  --marker-dropout-seconds 0.7
```

PyTorch `.pt` 모델로 실행할 때 FPS가 낮으면 `--imgsz`를 명시해 YOLO 입력
크기를 낮춰 실행합니다. `--imgsz`를 생략하면 이전 안정 경로처럼 Ultralytics
기본 동작을 사용합니다.

```bash
python3 -m marker_tracker.app \
  --aruco-marker-id 0 \
  --aruco-detect-flip auto \
  --imgsz 576
```

NCNN export 모델을 사용할 때는 런타임 `--imgsz`를 export 크기와 맞춰야
합니다. 현재 포함된 `yolo26n_ncnn_model`은 416으로 export되어 있으므로
프로파일 기본값도 416입니다. 576이나 512로 바꾸려면 같은 크기로 NCNN
모델을 다시 export한 뒤 해당 모델과 같은 `--imgsz`를 사용합니다.

마커를 다른 사람에게 넘겼을 때 전환이 느리면 `--confirmation-seconds`를 낮춥니다.
초기 선택과 대상 전환 확인 시간에 함께 적용됩니다.

```bash
python3 -m marker_tracker.app \
  --aruco-marker-id 0 \
  --confirmation-seconds 1.0
```

전환이 너무 민감하거나 bbox가 합쳐지는 상황을 튜닝하려면 다음 옵션을
조절합니다.

```bash
python3 -m marker_tracker.app \
  --aruco-marker-id 0 \
  --confirmation-seconds 1.0 \
  --switch-score-margin 12.0 \
  --switch-cooldown-seconds 1.0 \
  --area-growth-penalty-threshold 1.8 \
  --score-diagnostics
```

- `--switch-score-margin`: 새 후보가 현재 후보보다 이 점수 이상 높아야 전환
- `--switch-cooldown-seconds`: handoff 직후 재전환을 막는 최소 시간
- `--area-growth-penalty-threshold`: 현재 bbox 면적이 이전보다 이 배율 이상
  커지면 큰 bbox 후보 감점
- `--score-diagnostics`: 후보별 score, bbox 면적, active 여부를 로그로 출력

데모나 현장 튜닝 중에는 추적 품질 진단을 켜면 현재 판단 근거를 더 명확히 볼
수 있습니다.

```bash
python3 -m marker_tracker.app \
  --aruco-marker-id 0 \
  --tracking-status-diagnostics \
  --score-diagnostics
```

`Tracking status` 로그에는 FPS, 현재 target id, handoff 이유, marker score,
PTZ 명령, lost/recovery 상태가 함께 출력됩니다. 예를 들어 마커를 넘기는 중이면
`recovery=handoff_confirmation`, 대상이 잠시 사라졌지만 유지 중이면
`recovery=holding_last_target`처럼 표시됩니다.

`pi-qon-http` 프로파일은 ID 연속성 진단도 기본으로 켭니다. 시작 로그의
`Person tracking configuration`에서 실제 `simple-iou`/`ultralytics` backend와
tracker 설정을 확인할 수 있습니다. 활성 ID가 사라지면 `Track continuity` 로그에
현재 보이는 ID와 각 후보의 confidence, IoU, 정규화 중심 거리, 면적/너비 비율,
상단 및 수평 거리, bbox가 기록됩니다. 이 진단은 대상 선택이나 PTZ 제어에는
영향을 주지 않습니다. 다른
프로파일에서 같은 진단을 켜려면 `--track-continuity-diagnostics`를 사용합니다.

`pi-qon-http`는 ByteTrack 내부 연계를 위해 YOLO 입력 confidence를 `0.1`로
낮추되, 앱이 발화자 후보로 사용하는 `--person-confidence`는 `0.35`로
유지합니다. 두 값을 별도로 튜닝하려면 `--tracker-input-confidence`와
`--person-confidence`를 사용합니다. ID 재결합은 기본적으로 0.12초 동안 후보를
확인하며 `--recovery-confirmation-seconds`, `--recovery-min-iou`,
`--recovery-max-center-distance`, `--recovery-score-margin`으로 조정할 수 있습니다.
부분 가림으로 상체/전신 bbox가 전환될 때는 면적 변화만으로 동일성을 판단하지
않고, bbox 상단 정렬과 수평 이동 및 너비 변화가 함께 연속적인 경우에만 복구
경로를 유지합니다. 영상 색상이나 배경 형태는 복구 점수에 사용하지 않습니다.

발표 상황에 맞춰 구도를 바꾸려면 `--framing-mode`를 지정합니다.

```bash
python3 -m marker_tracker.app \
  --aruco-marker-id 0 \
  --framing-mode upper-body
```

사용 가능한 값은 다음과 같습니다.

- `center`: 선택된 사람 bbox 중심을 화면 중앙에 둠
- `upper-body`: 상반신 중심으로 tilt
- `full-body`: 전신 구도를 우선
- `board-left`: 왼쪽 보드 쪽 여백을 남기도록 발표자를 오른쪽에 배치
- `board-right`: 오른쪽 보드 쪽 여백을 남기도록 발표자를 왼쪽에 배치

혼자서 마커 이동 전환을 반복 테스트하려면 fixture 영상을 생성한 뒤 고정된
두 사람 bbox를 주입해 실행합니다. 이 모드는 YOLO/ByteTrack 대신 테스트용
`track_id` 1, 2를 사용하지만, ArUco 검출과 대상 전환 상태 머신은 실제 앱
경로를 그대로 사용합니다.

```bash
python3 tools/generate_marker_transfer_video.py
python3 -m marker_tracker.app \
  --video-file test_assets/marker_transfer_fixture.mp4 \
  --person-source fixture \
  --aruco-dictionary DICT_4X4_50 \
  --aruco-marker-id 0 \
  --confirmation-seconds 1.0
```

창 없이 로그만 확인하려면 `--no-display`를 추가합니다. 성공하면 로그에 첫
handoff 이후 `Target track changed: previous=1 current=2`가 출력됩니다.

기본 target mode는 `aruco`입니다. 이전 명령과의 호환을 위해 `marker`도
`aruco`와 동일하게 처리합니다. 기본 ArUco dictionary는 `DICT_4X4_50`입니다. 다른 dictionary로 출력한
마커를 사용할 경우 `--aruco-dictionary DICT_5X5_100`처럼 맞춰 지정합니다.

마커가 화면에서도 전혀 잡히지 않으면 먼저 ArUco 진단 모드로 실제 RTSP
프레임에서 OpenCV가 후보를 보고 있는지 확인합니다.

```bash
python3 -m marker_tracker.app \
  --aruco-dictionary DICT_4X4_50 \
  --aruco-marker-id 0 \
  --confirmation-seconds 0 \
  --aruco-diagnostics
```

실행 중 로그의 `ArUco diagnostics`에서 `ids=[]`이고
`rejected_candidates=0`이면 프레임 안의 마커 경계가 OpenCV 후보로도 보이지
않는 상태입니다. `aruco_diagnostics/`에 저장된 이미지를 열어 마커가 실제
RTSP 프레임에 충분히 크게, 선명하게, 잘리지 않고 들어오는지 확인합니다.
`rejected_candidates`가 계속 증가하는데 `ids=[]`이면 마커 사전 불일치,
검은 테두리/흰 여백 손상, 과도한 원근 왜곡 가능성이 큽니다.
카메라 스트림이나 표시 경로가 좌우/상하 반전되어 있으면 원본 마커가 다른
ID처럼 보일 수 있습니다. 이때는 다음처럼 ArUco 검출에만 반전 보정을
적용합니다.

```bash
python3 -m marker_tracker.app \
  --aruco-dictionary DICT_4X4_50 \
  --aruco-marker-id 0 \
  --confirmation-seconds 0 \
  --aruco-detect-flip auto \
  --aruco-diagnostics
```

마커 없이 ByteTrack ID를 직접 선택하려면 수동 모드로 실행한 뒤 화면의 사람
바운딩 박스를 클릭합니다.

```bash
python3 -m marker_tracker.app --target-mode manual
```

ByteTrack 설정 파일을 바꾸려면 다음처럼 지정합니다.

```bash
python3 -m marker_tracker.app --bytetrack-config bytetrack.yaml
```

BoT-SORT와 비교하려면 Ultralytics 기본 `botsort.yaml`을 사용합니다.

```bash
python3 -m marker_tracker.app --tracker-preset botsort
```

특정 tracker YAML을 직접 지정하면 preset보다 우선합니다.

```bash
python3 -m marker_tracker.app --tracker-config custom_tracker.yaml
```

PTZ 카메라 모션 보정 효과를 ByteTrack과 공정하게 비교하려면 공통 threshold와
`track_buffer`를 사용하는 `bytetrack-pi.yaml`, `botsort-pi.yaml`을 번갈아
지정합니다. `botsort-pi.yaml`은 `sparseOptFlow` GMC를 켜고 Re-ID는 끈 상태라
이 단계에서는 카메라 모션 보정 비용과 ID 안정성만 비교합니다.

```bash
# ByteTrack 기준
python3 -m marker_tracker.app \
  --runtime-profile pi-qon-http \
  --tracker-config bytetrack-pi.yaml

# BoT-SORT + GMC, Re-ID off
python3 -m marker_tracker.app \
  --runtime-profile pi-qon-http \
  --tracker-config botsort-pi.yaml
```

`pi-qon-http` 프로파일은 성능 진단을 기본 활성화합니다. 5초마다 출력되는
`Performance diagnostics`에서 `person_source`는 YOLO 추론과 tracker를 합친 시간,
`target_signal`은 ArUco 처리 시간, `target_logic`은 발화자 상태 머신 시간,
`pipeline`은 화면 표시를 제외한 전체 처리 시간입니다. 각 구간의 평균과 p95를
비교해 BoT-SORT GMC의 실제 비용을 판단합니다.

프로젝트 기본 `bytetrack.yaml`은 `track_buffer: 75`로 설정되어 있습니다.
컴퓨터 테스트 기준 23-25 FPS에서 약 3초 동안 lost track ID를 유지하기 위한
값입니다. 라즈베리 파이 5에서 실제 FPS가 달라지면 다음 계산으로 조정합니다.

```text
track_buffer = 목표 유지 시간(초) * 실제 추적 FPS
```

실행 중 로그에는 `recommended_track_buffer`가 주기적으로 출력됩니다. 실제
장비에서 이 값을 확인한 뒤 `bytetrack.yaml`의 `track_buffer`를 맞추면 됩니다.
`pi-qon-http`가 사용하는 `bytetrack-pi.yaml`은 현재 15 FPS 시뮬레이터 측정값에
맞춰 `track_buffer: 60`을 사용합니다.
목표 유지 시간을 바꿔서 계산하려면 다음처럼 실행합니다.

```bash
python3 -m marker_tracker.app --track-buffer-seconds 4.0
```

### 권장: 논리적 speaker ID + 선택적 ReID

원시 BoT-SORT `track_id`는 가림, 부분/전신 bbox 전환, 교차 상황에서 바뀔 수
있습니다. 이 숫자를 고정하는 대신 앱은 하나의 논리적 `speaker_id=1`을 유지하고,
현재 발화자에 해당하는 원시 `track_id`를 다시 연결합니다.

ArUco가 확실히 보이고 다른 사람과 겹치지 않는 crop만 멀티샷 외형 갤러리에
추가합니다. 평상시에는 ReID를 실행하지 않습니다. 활성 ID가 사라졌을 때 또는
두 사람이 겹쳤다가 분리된 직후에만 후보 crop을 비교합니다. 겹친 순간에는
혼합된 crop으로 잘못 확정하지 않도록 전환을 보류합니다.
사라졌던 원시 ID가 다시 나타나더라도 같은 사람이라고 즉시 신뢰하지 않고,
재등장 직후 여러 프레임 동안 모든 후보의 외형을 다시 검증합니다. 외형 기준에
미달한 후보는 ArUco를 포함해도 즉시 재지정하거나 갤러리에 추가하지 않습니다.
짧은 가림에서 bbox의 위치·IoU 경로가 강하게 이어지는 경우에는 낮아진 ReID
점수를 하드 거부 조건으로 사용하지 않고 결합 점수의 보조 증거로만 사용합니다.
초기 ArUco 확인 시간도 raw tracker ID가 아니라 연속된 물리적 bbox 경로를 따라
누적하므로 동일 인물의 ID가 잠깐 바뀌어도 등록 타이머가 초기화되지 않습니다.

```bash
python3 -m marker_tracker.app \
  --runtime-profile pi-qon-http \
  --person-tracker ultralytics \
  --tracker-config botsort-pi.yaml \
  --speaker-identity-backend yolo-reid \
  --speaker-reid-model yolo26n-reid.onnx \
  --speaker-reid-device cpu \
  --rtsp-url "http://CAMERA_IP:8081/video" \
  --qon-camera-url "http://CAMERA_IP:8080" \
  --qon-auth-mode none \
  --qon-log-commands
```

이 모드에서는 tracker-level ReID가 꺼진
`botsort-pi.yaml`을 사용합니다. tracker-level ReID를 함께 켜면 같은 crop을 중복 추론해 FPS만
낮아질 수 있습니다. 로그의 `target_id`는 바뀔 수 있지만 `speaker_id`는 같은
발화자를 의미하며 유지되어야 합니다. `Speaker ReID candidates`는 선택적 추론이
실제로 발생한 프레임의 후보 유사도를, `identity_reid_avg_ms/p95_ms`는 그 비용을
보여줍니다.

기본 안전 게이트는 절대 유사도 `0.55`, 1·2위 후보 차이 `0.08`, 재연결 확인
`0.12초`입니다. 실제 2인 교차 영상에서 오연결과 미연결을 함께 측정한 뒤
`--speaker-reid-min-similarity`, `--speaker-reid-margin`,
`--recovery-confirmation-seconds`를 조정합니다.

### ArUco 전달을 통한 발화자 변경

추적 중 마커가 다른 사람의 bbox 안에서 확인 시간 동안 안정적으로 유지되면
`confirmed_marker_transfer`로 발화자 변경을 확정합니다. 이 이벤트는 같은 사람의
raw ID 복구와 다르게 처리됩니다.
기존 대상과 크게 겹치는 중복/부분 bbox로의 이동은
`confirmed_marker_reassociation`으로 분류되어 `speaker_id`를 유지합니다.

```text
speaker_id: 1 -> 2
이전 발화자 ReID gallery 삭제
새 발화자 crop으로 gallery 재생성
새 발화자의 raw target_id를 추적 대상으로 설정
```

마커를 전달하는 도중에는 기존 발화자를 유지하고, 새 사람이 마커를 착용한 뒤
확인 시간이 완료되어야 변경합니다. 변경 후 교차 테스트에서는 `target_id`가
바뀌어도 `speaker_id=2`와 실제 새 발화자가 유지되는지를 확인합니다.

### 교차 후 신원 재검증

사람들이 겹치거나 여러 사람이 보이던 직후 활성 raw ID가 사라지면 일반적인
geometry 복구를 잠시 중단하고 `identity_resolving` 상태로 전환합니다. 이 상태는
BoT-SORT의 숫자 ID를 실제 신원으로 신뢰하지 않으며 다음 규칙을 사용합니다.

- 겹친 crop은 외형 판단에서 제외
- 같은 사람의 전신/상단 bbox가 동시에 검출되면 하나의 물리 인물로 병합
- track ID가 없는 교차 중 불완전 crop은 identity evidence에서 제외
- 후보가 잠시 한 명이 되거나 trackless 구간이 나와도 기존 물리 경로는 보존
- bbox 중심 속도로 다음 위치를 예측하여 raw ID가 교환돼도 물리 경로를 연결
- 두 사람이 접근 중일 때는 확정하지 않고, 근접 후 다시 분리된 프레임만 승인 증거로 사용
- 분리 후 bbox 물리 경로별로 최소 6개의 clean frame 수집
- ReID 유사도의 중앙값, 1·2위 margin, 프레임별 승률을 함께 확인
- 중앙값뿐 아니라 최종 프레임도 동일 후보가 margin을 만족해야 승인
- 마커 등록 중 비발화자 exclusion gallery도 만들어 대조 점수에 반영
- 최초 마커 확인 샘플은 anchor gallery로 고정하고 이후 샘플은 adaptive gallery로 분리
- 증거가 모호하면 마지막으로 갱신되는 물리 경로를 PTZ 속도 2로 soft-coast
- 유효한 다중 경로 관측이 끊긴 뒤에는 최대 0.8초까지만 예측을 연장
- 예측 시간이 만료되거나 안전한 경로가 없을 때만 PTZ 정지
- 단일 후보는 raw track ID가 바뀌어도 bbox가 같은 물리 경로를 이어가면 증거를 유지
- 가까운 연속 경로는 4프레임/0.25초와 중앙값 0.45로 빠르게 복구하고,
  멀리 나타난 후보는 기존의 높은 외형 기준을 통과해야 복구
- 2초 이상 확정되지 않으면 PTZ coast를 끝내고 `identity_searching`으로 전환하여
  ReID를 기본 3프레임마다 실행
- 교차 종료 후 0.75초 cooldown 동안 즉시 재진입과 갤러리 오염 방지

기본 identity resolution 값은 6프레임, 0.50초, 중앙값 0.35, margin 0.02,
승률 0.75, 근접 거리 1.75 bbox 폭, 종료 거리 1.9 bbox 폭, 종료 확인 3프레임입니다.
일반적인 단일 사람 ID churn은 기존 geometry 경로로 빠르게 복구하고, 최근
다중 사람/교차 문맥에서만 교차 에피소드 정책을 적용합니다.

### 원본 입력 영상과 identity trace 저장

화면 녹화 대신 모델이 실제로 처리한 overlay 이전 프레임을 저장할 수 있습니다.

```bash
mkdir -p test_video/raw test_video/traces
python3 -m marker_tracker.app \
  --runtime-profile pi-qon-http \
  --tracker-config botsort-pi.yaml \
  --speaker-identity-backend yolo-reid \
  --speaker-reid-model yolo26n-reid.onnx \
  --speaker-reid-device cpu \
  --record-input-video test_video/raw/crossing_01.mp4 \
  --record-input-fps 15 \
  --identity-trace-file test_video/traces/crossing_01.jsonl \
  --rtsp-url "http://CAMERA_IP:8081/video" \
  --qon-camera-url "http://CAMERA_IP:8080" \
  --qon-auth-mode none
```

MP4에는 추론에 들어간 원본 프레임, JSONL에는 각 프레임의 bbox, raw ID,
confidence, appearance similarity, identity 상태와 PTZ handoff 판단이 저장됩니다.
두 경로는 Git에서 제외됩니다.

## 단위 테스트

핵심 상태 머신 테스트에는 외부 패키지가 필요하지 않습니다.

```bash
python3 -m unittest discover -s tests -v
```

## Qon HTTP PTZ 제어

기본값은 로그 전용입니다. Qon 카메라는 MVPv3에서 확인된 HTTP
`ptzctrl.cgi` 명령 방식으로 움직일 수 있습니다. `qon-http` 모드는 시작할 때
MVPv3와 같은 초기 설정을 적용합니다.

```text
common.track=0
tracking.auto_zoom=0
tracking.auto_tilt=0
common.debug_mode=3
post_visca refresh
get_path read-back
```

```bash
python3 -m marker_tracker.app \
  --aruco-dictionary DICT_4X4_50 \
  --aruco-marker-id 0 \
  --aruco-detect-flip auto \
  --confirmation-seconds 1.0 \
  --camera-control qon-http \
  --qon-camera-url http://CAMERA_IP \
  --qon-log-commands
```

Qon HTTP 방식은 대상 bbox가 화면 중앙에서 벗어난 정도를 보고
`left/right/up/down` 중 가장 큰 축 하나와 속도를 보냅니다. 이동 시작/정지
deadzone을 분리하고 정지 debounce를 적용하여 경계에서 move/stop이 반복되는 것을
막습니다. `grace`의 짧은 detector dropout은 기본 0.2초 동안 현재 이동을 유지하지만,
identity ambiguity와 명시적 release는 즉시 안전 정지합니다. PTZ 명령은 기본적으로 백그라운드
worker에서 비동기로 전송하고, 종료 시 동기 `ptzstop`을 한 번 더 보냅니다.
필요하면 다음 옵션으로 조정합니다.

```bash
--qon-min-speed 2
--qon-max-speed 10
--qon-deadzone 0.12
--qon-stop-deadzone 0.08
--qon-stop-debounce-seconds 0.12
--qon-lost-target-hold-seconds 0.20
--qon-command-min-interval-seconds 0.12
--qon-command-ttl-seconds 0.35
--qon-timeout-seconds 1.0
--qon-close-join-timeout-seconds 1.0
--qon-log-commands
```

카메라 인증이 필요하면 다음 옵션을 사용합니다. 비밀번호는 직접 넘기거나
환경변수 이름으로 넘길 수 있습니다.

```bash
--qon-username USER
--qon-password-env QON_PASSWORD
--qon-auth-mode digest
```

카메라 설정을 이미 수동으로 끝냈고 시작 시 초기화를 건너뛰려면 다음 옵션을
추가합니다.

```bash
--no-qon-configure-supervisor-actuator
```

PTZ 명령을 백그라운드 worker 없이 메인 루프에서 직접 보내려면 다음 옵션을
추가합니다.

```bash
--no-qon-async-commands
```

VISCA over IP 구현은 이전 실험과 비교하기 위한 fallback/reference 경로로
남아 있습니다. 현재 Qon 현장 테스트는 위의 `qon-http` 실행 방식을 우선
사용합니다. Sony VISCA over IP로 pan/tilt를 움직이려면 카메라 IP를
지정합니다.

```bash
python3 -m marker_tracker.app \
  --aruco-dictionary DICT_4X4_50 \
  --aruco-marker-id 0 \
  --aruco-detect-flip auto \
  --confirmation-seconds 1.0 \
  --camera-control visca \
  --visca-host CAMERA_IP \
  --visca-port 52381 \
  --visca-log-commands
```

기본 패킷 모드는 Sony VISCA-over-IP UDP framing입니다. 카메라가 raw VISCA
UDP payload를 받는 모델이면 다음 옵션을 추가합니다.

```bash
--visca-packet-mode raw
```

움직임 방향이 반대면 다음 옵션으로 축을 뒤집습니다.

```bash
--visca-invert-pan
--visca-invert-tilt
```

추적이 흔들리면 `--visca-deadzone-x`, `--visca-deadzone-y`를 키우고, 반응이
느리면 `--visca-pan-speed`, `--visca-tilt-speed`를 올립니다. 선택 대상이
없거나 release가 발생하면 stop 명령을 보냅니다.

## Raspberry Pi 5 / NCNN

라즈베리파이 5에서는 기본 PyTorch 모델 대신 export된 NCNN 모델을 사용합니다.
현재 repo에는 다음 산출물이 준비되어 있습니다.

```text
yolo26n_ncnn_model/
```

Pi에서는 `requirements-pi.txt`를 설치한 뒤 `--model yolo26n_ncnn_model`로
실행합니다. 자세한 절차는 `RASPBERRY_PI.md`를 참고합니다.
