# PTZ Speaker Tracking MVP

6/22 발표 목표는 마커를 가진 발표자를 우선 선택하고, 마커가 사라지는
구간에서는 Re-ID로 같은 사람을 유지하거나 재획득하는 PTZ 추적 MVP입니다.
MediaPipe는 손들기 같은 보조 등록/복구 경로로 사용합니다.

## MVP Scope

핵심 흐름:

```text
RTSP / camera / video
  -> person detection
  -> ArUco marker detection
  -> marker inside person bbox selects target
  -> register target appearance for Re-ID
  -> PTZ command from target bbox center
  -> if marker disappears:
       bbox tracker first
       Re-ID fallback second
  -> if marker returns:
       marker lock refreshes the target and Re-ID reference
  -> optional MediaPipe raised-hand fallback
```

역할 구분:

| Feature | Role |
|---|---|
| ArUco marker | 신뢰 가능한 초기 target selection |
| Re-ID | 마커가 안 보이는 구간의 identity 유지/복구 |
| MediaPipe | 마커 테스트 후 추가할 gesture fallback |
| PTZ control | target bbox 중심 기준 pan/tilt 명령 생성 |
| Qon CGI control | optional tracking/zone mode switch 실험용 |

## Current Status

현재 기본 실행 경로는 marker-first MVP입니다.

```bash
python3 main.py --source 0
python3 main.py --source path/to/video.mp4
python3 main.py --source "$RTSP_URL"
```

마커 ID를 하나로 제한:

```bash
python3 main.py --source "$RTSP_URL" --target-marker-id 7
```

NCNN detector 사용 예:

```bash
python3 main.py \
  --source "$RTSP_URL" \
  --detector ncnn \
  --ncnn-param models/yolo.param \
  --ncnn-bin models/yolo.bin \
  --ncnn-input-size 416 \
  --target-marker-id 7
```

MediaPipe gesture fallback 활성화:

```bash
python3 main.py --source "$RTSP_URL" --gesture
```

Headless 로그/영상 저장:

```bash
python3 main.py \
  --source "$RTSP_URL" \
  --target-marker-id 7 \
  --no-window \
  --save-debug-video logs/marker_reid.mp4 \
  --log-csv logs/marker_reid.csv
```

## Runtime Behavior

상태는 다음 의미로 사용합니다.

| State | Meaning |
|---|---|
| `IDLE` | 아직 대상 없음 |
| `ACTIVE` | 마커 또는 bbox tracker로 대상 추적 중 |
| `REID_TRACKING` | 마커가 사라진 뒤 Re-ID로 재획득/유지 중 |
| `SUSPENDED` | 대상 후보를 찾지 못해 일시 정지 |
| `ENDED` | 종료 |

기본적으로 Re-ID fallback은 켜져 있습니다. 끄려면:

```bash
python3 main.py --source "$RTSP_URL" --disable-reid
```

현재 Re-ID 구현은 발표 전 빠른 검증을 위한 HSV appearance matcher입니다.
마커로 선택된 사람의 상반신 색상 히스토그램과 bbox 비율을 저장하고, 마커가
없을 때 사람 후보 중 가장 비슷한 bbox를 고릅니다. 최종 정확도를 위해서는
이 부분을 OSNet/FastReID 등 embedding 기반 모델로 교체하는 것이 다음 단계입니다.

## Keyboard Controls

```text
q  quit
r  reset target and wait for marker reacquisition
s  stop tracking
m  toggle marker-only mode
g  toggle MediaPipe gesture fallback
```

## Project Structure

```text
PTZ_Speaker_Tracking/
├── main.py                         # marker-first MVP entry point
├── src/
│   ├── tracking/
│   │   ├── marker_reid_demo.py     # marker + Re-ID + MediaPipe loop
│   │   ├── marker_detector.py      # ArUco detector
│   │   ├── person_detector.py      # HOG / OpenCV YOLO / NCNN adapters
│   │   ├── target_selector.py      # marker and bbox target selection
│   │   ├── target_state.py         # tracking state/data models
│   │   ├── gesture_detector.py     # MediaPipe raised-hand fallback
│   │   └── ptz_controller.py       # PTZ simulator / hardware placeholder
│   ├── reid/
│   │   ├── appearance.py           # current lightweight Re-ID matcher
│   │   └── verifier.py             # Qon verifier support
│   ├── camera/
│   │   ├── qon_control.py          # optional tracking/zone mode CGI control
│   │   └── region_provider.py      # Qon bbox/center crop adapter
│   └── app/
│       └── verify_demo.py          # optional Qon internal tracking verifier
└── tests/
```

## Optional Qon Mode Control

카메라 웹 UI Network 캡처로 확인된 mode switch는 보조 기능으로 남겨둡니다.
주소와 인증 정보는 저장소에 넣지 않고 실행 시 제공합니다.

```bash
export QON_CAMERA_URL='http://camera-address'

python3 -m src.camera.qon_control --camera-url "$QON_CAMERA_URL" status
python3 -m src.camera.qon_control --camera-url "$QON_CAMERA_URL" tracking
python3 -m src.camera.qon_control --camera-url "$QON_CAMERA_URL" zone
```

확인된 CGI 요청:

```text
POST /cgi-bin/param.cgi?get_path   path=/data/track.conf
POST /cgi-bin/param.cgi?write_path path=/data/track.conf&common.track=1  (tracking)
POST /cgi-bin/param.cgi?write_path path=/data/track.conf&common.track=0  (zone)
```

함께 캡처된 `post_visca` 요청은 tracking/zone 양쪽 전환에 동일하게 발생했기
때문에, 목적이 확인되기 전까지 자동 전송하지 않습니다.

## 6/22 Implementation Plan

1. Marker-first loop stability
   - camera/RTSP 입력 안정화
   - ArUco marker selection 검증
   - PTZ simulator 로그와 debug video 확보

2. Re-ID fallback
   - marker lock 시 reference 등록
   - marker missing 시 detector 후보 중 Re-ID best match 선택
   - 오추적/재획득 케이스 CSV 기록
   - lightweight matcher를 embedding 모델로 교체 검토

3. MediaPipe fallback
   - raised-hand gesture로 임시 target 등록
   - marker/Re-ID 실패 시 보조 복구 경로로 제한

4. Hardware integration
   - PTZ simulator command를 실제 VISCA/ONVIF/vendor command로 교체
   - bbox와 command 로그를 발표 테스트 자료로 정리

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Optional packages:

```bash
python3 -m pip install mediapipe
python3 -m pip install "ultralytics>=8.3.0"
```

## Test

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q main.py src tests
python3 main.py --help
```
