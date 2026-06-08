# PTZ Speaker Tracking MVP

이 저장소의 메인 목표는 **카메라 내장 Auto Tracking 위에 마커와 Re-ID
기반 정체성 검증 레이어를 얹는 PTZ 추적 MVP**입니다. 카메라가 PTZ 추적을
수행하고, Raspberry Pi 5는 마커로 등록된 발표자를 OSNet/appearance Re-ID로
검증해 잘못된 사람을 따라가는 상황을 감지하고 차단합니다.

현재 권장 실행 경로는 `python3 -m src.app.verify_demo`입니다. `main.py`가
호출하는 `src/tracking/marker_reid_demo.py`는 카메라 내장 추적을 쓰지 않는
직접 추적 fallback입니다.

```text
Qon PTZ internal tracking
  -> Pi receives RTSP/video frames
  -> marker registers trusted presenter identity
  -> Presenter mode + tracking are enabled after registration
  -> Re-ID verifies the current camera-followed region
  -> marker visible acts as a strong positive signal
  -> sustained mismatch disables camera tracking and triggers recovery action
  -> marker-based registration can resume tracking
```

Qon 카메라의 내장 Auto Tracking 검증 경로(`src/app/verify_demo.py`)와
tracking/zone/PTZ CGI 제어(`src/camera/qon_control.py`)가 현재 권장 경로입니다.
`marker_reid_demo.py`는 카메라 내장 추적을 끄고 외부에서 직접 PTZ 명령을
생성해야 할 때 쓰는 fallback/비교 경로로 남겨둡니다.

## 현재 구현 상태

| 항목 | 상태 |
|---|---|
| ArUco marker 기반 대상 선택 | 구현됨 |
| 사람 detector | HOG / OpenCV YOLO / NCNN / manual 지원 |
| bbox 기반 target 유지 | 구현됨 |
| Re-ID verifier | HSV baseline / OSNet-style ONNX backend 구현 |
| MediaPipe 손들기 fallback | 구현됨, 선택 기능 |
| PTZ 명령 생성 | simulator 구현됨 |
| 실제 PTZ 하드웨어 제어 | Qon HTTP CGI tracking/stop/home/zoom 기본 구현 |
| Qon tracking/zone CGI 제어 | 구현됨 |
| Qon live tracking bbox | 미확보 |

가장 큰 남은 리스크는 두 가지입니다.

1. **Re-ID 품질**
   - OSNet-style ONNX backend는 연결되어 있지만 모델 파일과 threshold는
     실제 카메라 영상으로 캘리브레이션해야 합니다.

2. **실제 카메라 bbox**
   - Qon live tracking bbox endpoint는 아직 미확보입니다.
   - 현재 verifier는 center crop 또는 jsonl metadata adapter를 사용합니다.

## 직접 추적 fallback 실행 방법

기본 실행:

```bash
python3 main.py --source "$RTSP_URL"
```

특정 ArUco marker id만 대상으로 허용:

```bash
python3 main.py \
  --source "$RTSP_URL" \
  --target-marker-id 7
```

마커가 사라져도 Re-ID fallback을 사용:

```bash
python3 main.py \
  --source "$RTSP_URL" \
  --target-marker-id 7 \
  --reid-threshold 0.68
```

MediaPipe 손들기 fallback 활성화:

```bash
python3 main.py \
  --source "$RTSP_URL" \
  --target-marker-id 7 \
  --gesture
```

NCNN detector 사용:

```bash
python3 main.py \
  --source "$RTSP_URL" \
  --detector ncnn \
  --ncnn-param models/yolo.param \
  --ncnn-bin models/yolo.bin \
  --ncnn-input-size 416 \
  --target-marker-id 7
```

로그와 디버그 영상 저장:

```bash
python3 main.py \
  --source "$RTSP_URL" \
  --target-marker-id 7 \
  --save-debug-video logs/marker_reid.mp4 \
  --log-csv logs/marker_reid.csv
```

창 없이 실행:

```bash
python3 main.py \
  --source "$RTSP_URL" \
  --target-marker-id 7 \
  --no-window \
  --log-csv logs/marker_reid.csv
```

## 주요 옵션

| 옵션 | 설명 |
|---|---|
| `--source` | RTSP URL, 비디오 파일, 웹캠 index |
| `--detector` | `hog`, `opencv-yolo`, `ncnn`, `manual` 중 선택 |
| `--target-marker-id` | 특정 ArUco marker id만 허용 |
| `--marker-only` | 마커 재획득 없이는 bbox/Re-ID fallback을 쓰지 않음 |
| `--gesture` | MediaPipe 손들기 fallback 활성화 |
| `--disable-reid` | 마커가 사라졌을 때 Re-ID fallback 비활성화 |
| `--reid-threshold` | Re-ID fallback 허용 threshold |
| `--max-suspended-frames` | target을 잃은 뒤 유지할 최대 프레임 수 |
| `--save-debug-video` | bbox/marker/PTZ overlay 영상 저장 |
| `--log-csv` | frame별 상태와 PTZ 명령 로그 저장 |

## 키보드 조작

```text
q  종료
r  target 초기화 후 마커 재획득 대기
s  추적 중지
m  marker-only 모드 토글
g  MediaPipe gesture fallback 토글
```

## 상태 의미

| 상태 | 의미 |
|---|---|
| `IDLE` | 아직 target 없음 |
| `ACTIVE` | 마커 또는 bbox tracker로 target 추적 중 |
| `REID_TRACKING` | 마커가 사라진 뒤 Re-ID로 target 유지/재획득 중 |
| `SUSPENDED` | target 후보를 찾지 못해 일시 정지 |
| `ENDED` | 종료 |

## 프로젝트 구조

```text
PTZ_Speaker_Tracking/
├── main.py                         # marker-first MVP 메인 진입점
├── src/
│   ├── tracking/
│   │   ├── marker_reid_demo.py     # marker + Re-ID + MediaPipe 추적 루프
│   │   ├── marker_detector.py      # ArUco marker detector
│   │   ├── person_detector.py      # HOG / OpenCV YOLO / NCNN detector
│   │   ├── target_selector.py      # marker/bbox 기반 target 선택
│   │   ├── target_state.py         # tracking 상태와 데이터 모델
│   │   ├── gesture_detector.py     # MediaPipe 손들기 fallback
│   │   └── ptz_controller.py       # PTZ simulator / hardware placeholder
│   ├── reid/
│   │   ├── appearance.py           # 현재 HSV baseline Re-ID
│   │   └── verifier.py             # Qon verifier용 검증 상태머신
│   ├── app/
│   │   └── verify_demo.py          # Qon 내장 추적 검증 실험 경로
│   ├── camera/
│   │   ├── qon_control.py          # Qon tracking/zone CGI 제어
│   │   └── region_provider.py      # Qon verifier용 center/jsonl bbox provider
│   ├── recovery/
│   │   └── candidate_detector.py   # Qon verifier용 YOLO 후보 탐색
│   └── vision/
│       ├── capture.py
│       └── models.py
└── tests/
```

## 권장 실행: Qon Auto Tracking + Pi Verifier

카메라 내장 tracking을 사용하고, Pi는 현재 중앙 영역 또는 metadata bbox가
등록된 발표자인지 검증합니다. RTSP 주소가 확인되면 `--source`에 넣습니다.

```bash
export QON_PASSWORD='camera-password'

python3 -m src.app.verify_demo \
  --source "$RTSP_URL" \
  --registration-mode marker \
  --target-marker-id 7 \
  --register-on-start \
  --marker-positive \
  --reid-backend onnx \
  --reid-model models/osnet_x0_25.onnx \
  --reid-threshold 0.72 \
  --verify-every-frames 6 \
  --mismatch-limit 8 \
  --control-camera \
  --camera-url 'http://192.168.11.88' \
  --camera-username admin \
  --camera-password-env QON_PASSWORD \
  --camera-auth-mode digest \
  --recovery-action home \
  --log-csv logs/qon_verifier.csv
```

동작 정책:

```text
IDLE/UNREGISTERED: marker가 보일 때까지 등록하지 않음
VERIFIED: marker visible 또는 Re-ID score 통과
SUSPECT: Re-ID mismatch가 누적 중인 상태
MISMATCH: mismatch-limit 초과, 카메라 tracking off + stop/복구 액션
RECOVERY/LOST: marker 기반 재등록 또는 별도 recovery detector로 후보 탐색
```

중요한 안전 정책:

```text
- marker 없는 최초 등록 금지
- marker가 보이면 강한 positive signal로 사용
- OSNet은 `--reid-backend onnx --reid-model ...`로 연결, 모델이 없을 때만 HSV baseline 사용
- Re-ID mismatch는 단발 프레임이 아니라 누적해서 확정
- mismatch 확정 시 Pi가 카메라와 싸우지 않도록 Zone 모드로 전환한 뒤 stop/home/zoomout 수행
- marker 재등록이 성공하면 Presenter 모드(`common.track_mode=tracking`)와 Track(`common.track=1`)을 다시 활성화
- MediaPipe는 identity 확정 수단이 아니라 marker가 없을 때의 보조 후보 제안 수단
```

## Qon 제어

카메라 웹 UI Network 캡처로 확인된 CGI 요청을 이용해 tracking/zone 모드를
조회하거나 바꿀 수 있습니다.

```bash
export QON_CAMERA_URL='http://camera-address'

python3 -m src.camera.qon_control --camera-url "$QON_CAMERA_URL" status
python3 -m src.camera.qon_control --camera-url "$QON_CAMERA_URL" --username admin --password-env QON_PASSWORD --auth-mode digest tracking
python3 -m src.camera.qon_control --camera-url "$QON_CAMERA_URL" --username admin --password-env QON_PASSWORD --auth-mode digest zone
python3 -m src.camera.qon_control --camera-url "$QON_CAMERA_URL" --username admin --password-env QON_PASSWORD --auth-mode digest stop
python3 -m src.camera.qon_control --camera-url "$QON_CAMERA_URL" --username admin --password-env QON_PASSWORD --auth-mode digest debug-bbox
```

확인된 요청:

```text
POST /cgi-bin/param.cgi?get_path   path=/data/track.conf
POST /cgi-bin/param.cgi?write_path path=/data/track.conf&common.track=1  (tracking)
POST /cgi-bin/param.cgi?write_path path=/data/track.conf&common.track=0  (zone)
POST /cgi-bin/param.cgi?write_path path=/data/track.conf&common.debug_mode=2  (bbox debug)
POST /cgi-bin/param.cgi?write_path path=/data/track.conf&common.osd_mode=1  (tracking hint)
GET  /cgi-bin/ptzctrl.cgi?ptzcmd&ptzstop&10&10
GET  /cgi-bin/ptzctrl.cgi?ptzcmd&home&10&10
GET  /cgi-bin/ptzctrl.cgi?ptzcmd&zoomout&5
```

모드 변경 시 함께 캡처된 `post_visca` 요청은 tracking/zone 양쪽 전환에서
동일하게 발생했습니다. 아직 목적이 확인되지 않았기 때문에 코드에서 자동으로
전송하지 않습니다.

## 6/22까지 우선순위

1. **marker-first 추적 루프 안정화**
   - RTSP 입력 안정화
   - ArUco marker selection 검증
   - bbox tracker와 Re-ID fallback 전환 로그 확보

2. **Re-ID embedding 모델 적용**
   - 현재 HSV baseline과 OSNet/FastReID embedding 비교
   - Pi 5에서는 후보 crop 수를 제한해 실시간성 확보

3. **MediaPipe fallback 테스트**
   - 마커가 없을 때 손든 사람을 target으로 등록
   - 단일 pose 기반 한계를 발표 자료에 명시

4. **실제 PTZ 제어 연결**
   - `PTZSimulator` 출력 명령을 VISCA/ONVIF/vendor command로 교체
   - bbox 중심과 PTZ command 로그를 발표 자료로 정리

5. **Qon 내장 추적 metadata 조사**
   - ONVIF Analytics metadata
   - Qon CGI/RTSP 부가 metadata
   - debug overlay가 픽셀 burn-in인지 좌표 metadata인지 확인

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

선택 설치:

```bash
python3 -m pip install mediapipe
python3 -m pip install "ultralytics>=8.3.0"
```

## 테스트

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q main.py src tests
python3 main.py --help
```
