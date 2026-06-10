# PTZ Speaker Tracking MVP v3

Qon PTZ 카메라를 단순 PTZ actuator처럼 운용하고, Raspberry Pi 쪽에서
사람 detector와 Re-ID로 발표자 identity를 추적하는 MVP입니다.

현장 검증 결과 Qon 카메라의 내부 Auto Tracking bbox는 외부 API로 받을 수
없었습니다. 따라서 v3의 기준 구조는 다음입니다.

```text
Qon RTSP stream2
  -> Pi NCNN YOLO person detector
  -> marker/gesture 기반 발표자 등록
  -> identity-first Re-ID로 target bbox 선택
  -> Pi가 Qon HTTP PTZ 명령으로 직접 pan/tilt 제어
```

## 현재 결론

카메라 내부 bbox 경로는 사용하지 않습니다.

```text
/cgi-bin/param.cgi?get_target_status -> 404
get_draw_box_flag -> issupport="0"
RTSP stream1/stream2 -> video/audio only, metadata track 없음
ONVIF Events -> bbox/coordinate 없음
```

또한 Qon의 `write_path`는 모르는 key에도 `Success`를 반환할 수 있으므로,
설정 변경은 항상 다음 순서로 검증합니다.

```text
write_path -> VISCA refresh -> get_path read-back
```

## 핵심 설계

### 1. 카메라 PTZ 소유권

v3에서는 Pi supervisor가 PTZ 제어권을 갖습니다.

`--configure-supervisor-actuator`는 다음 값을 적용합니다.

```text
common.track=0
tracking.auto_zoom=0
tracking.auto_tilt=0
common.debug_mode=3
```

의미:

```text
Qon 내부 tracking off
Qon 자동 zoom/tilt off
debug bbox burn-in off
Pi가 ptzctrl.cgi로 직접 PTZ 제어
```

Qon PTZ 명령은 위치 명령이 아니라 방향/속도 명령입니다. `ptzstop`을 보내지
않으면 계속 움직이므로 코드에서 stop을 보장합니다.

### 2. Identity-first bbox 선택

center crop이나 center-nearest person을 주 신호로 쓰지 않습니다. 카메라가
틀린 사람을 중앙에 둘 때 center 기반 방식도 같이 틀리기 때문입니다.

`IdentityMatchedRegionProvider`는 프레임 전체 사람 후보를 모두 보고, 등록된
발표자 Re-ID feature와 가장 가까운 후보를 observed bbox로 선택합니다. 기본
사람 detector는 NCNN YOLO입니다.

판정은 두 단계입니다.

```text
관측 신뢰:
best_score >= weak_min_score
AND best_score - second_score >= identity_margin

mis-track:
best_score - center_person_score >= center_margin
AND best bbox가 center dead zone 밖
AND N frames 지속
```

관측 신뢰가 낮으면 바로 mismatch나 LOST로 가지 않고 `HOLD` 상태로 둡니다.
`HOLD`가 여러 프레임 지속될 때만 `LOST`로 전환합니다.

### 3. MediaPipe gesture fallback

MediaPipe는 반영되어 있습니다. 다만 필수 의존성이 아니라 선택 기능입니다.

- 코드 위치: `src/tracking/gesture_detector.py`
- 사용 경로:
  - `src/tracking/marker_reid_demo.py --gesture`
  - `src/app/verify_demo.py --registration-mode gesture`
  - `src/app/verify_demo.py --registration-mode marker-or-gesture`
- `mediapipe`가 설치되어 있지 않으면 gesture fallback은 빈 결과를 반환하고
  메인 추적은 계속 동작합니다.

설치:

```bash
python3 -m pip install mediapipe
```

`requirements.txt`에는 선택 의존성으로 주석 처리되어 있습니다.

## 실행

### 권장 실행

```bash
cd PTZ_Speaker_Tracking
export QON_PASS='camera-password'

python3 -m src.app.verify_demo --config configs/qon_mvpv3.yaml
```

이 명령이 기본 데모 실행입니다. 세부 값은 `configs/qon_mvpv3.yaml`에서 관리합니다.

현장에서 일부 값만 바꿀 때는 필요한 옵션만 CLI로 override합니다.

```bash
python3 -m src.app.verify_demo \
  --config configs/qon_mvpv3.yaml \
  --identity-margin 0.12 \
  --ptz-max-speed 8
```

### NCNN 입력 크기

`stream2`의 영상 해상도는 640x360입니다. 하지만 `ncnn_input_size: 416`은
RTSP 프레임 해상도가 아니라 YOLO 모델의 square 입력 크기입니다.

```text
RTSP frame: 640x360
YOLO/NCNN model input: 416x416
```

따라서 `(640, 360)`으로 설정하는 값이 아닙니다. 모델 export 크기와
`ncnn_input_size`는 반드시 맞춥니다. 현재 기본 config는 416 export 모델 기준입니다.

### fallback 실행

기본 데모는 `verify_demo.py --config`입니다. `main.py`는 marker-first 직접 추적
비교용으로만 남겨둡니다.

```bash
python3 main.py --source "rtsp://192.168.11.88:554/stream2" --target-marker-id 0
```

## 주요 옵션

| 옵션 | 설명 |
|---|---|
| `--config configs/qon_mvpv3.yaml` | 권장 실행 설정 파일 |
| `--detector ncnn` | 기본 detector. NCNN YOLO 모델 사용 |
| `--ncnn-param models/yolo26n_ncnn_model/model.ncnn.param` | NCNN YOLO26n param 파일 |
| `--ncnn-bin models/yolo26n_ncnn_model/model.ncnn.bin` | NCNN YOLO26n bin 파일 |
| `--region-mode identity` | detector + Re-ID로 observed bbox 선택 |
| `--identity-weak-min-score` | 관측 신뢰 최소 score. 실패 시 HOLD |
| `--identity-margin` | best 후보와 second 후보의 최소 score 차이 |
| `--center-margin` | best 후보와 center 후보의 최소 score 차이 |
| `--identity-hold-limit` | HOLD가 LOST로 바뀌기 전 프레임 수 |
| `--configure-supervisor-actuator` | Qon 내부 tracking/auto PTZ/debug burn-in off |
| `--ptz-follow-target` | identity-selected bbox를 향해 Qon PTZ 직접 제어 |
| `--rtsp-drop-frames` | non-threaded RTSP 디버깅용 frame drop 수. 기본 실행은 최신 frame capture thread 사용 |
| `--reid-backend hsv` | HSV baseline Re-ID |
| `--reid-backend onnx --reid-model ...` | OSNet-style ONNX Re-ID |
| `--registration-mode marker` | ArUco marker로 발표자 등록 |
| `--registration-mode gesture` | MediaPipe 손들기 후보로 발표자 등록 |
| `--registration-mode marker-or-gesture` | marker 우선, 없으면 gesture |

기본 실행은 `--detector ncnn`으로 동작합니다. 따라서 실험 전 YOLO26n을
Ultralytics NCNN export로 변환해 다음 파일을 준비해야 합니다.

```text
models/yolo26n_ncnn_model/model.ncnn.param
models/yolo26n_ncnn_model/model.ncnn.bin
```

Ultralytics NCNN export의 기본 blob 이름은 보통 `in0`/`out0`입니다. 다른 이름으로
export된 모델이면 `configs/qon_mvpv3.yaml`의 `ncnn_input_name`,
`ncnn_output_names`를 맞춥니다.

모델이 없을 때만 임시 smoke test 용도로 `--detector hog`를 명시해 사용할 수
있습니다.

실행 로그와 CSV에는 처리 FPS, `read_ms`, `detect_ms`가 함께 기록됩니다.
`read_ms`가 크면 RTSP/디코드 지연, `detect_ms`가 크면 NCNN 추론/파서 병목입니다.

## 프로젝트 구조

```text
PTZ_Speaker_Tracking/
├── main.py
├── requirements.txt
├── configs/
│   └── qon_mvpv3.yaml
├── logs/
├── models/
├── src/
│   ├── app/
│   │   └── verify_demo.py
│   ├── camera/
│   │   ├── qon_control.py
│   │   └── region_provider.py
│   ├── recovery/
│   │   └── candidate_detector.py
│   ├── reid/
│   │   ├── appearance.py
│   │   └── verifier.py
│   ├── tracking/
│   │   ├── gesture_detector.py
│   │   ├── marker_detector.py
│   │   ├── marker_reid_demo.py
│   │   ├── person_detector.py
│   │   ├── ptz_controller.py
│   │   ├── target_selector.py
│   │   └── target_state.py
│   └── vision/
│       ├── capture.py
│       └── models.py
└── tests/
```

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
python3 -m unittest discover -s tests
python3 -m src.app.verify_demo --help
```

현재 확인:

```text
Ran 24 tests OK
```
