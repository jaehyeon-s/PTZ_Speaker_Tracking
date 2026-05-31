# PTZ Speaker Tracking MVP

이 저장소의 메인 목표는 **Qon PTZ 카메라의 내장 Auto Tracking을 기본
추적기로 사용하고, Raspberry Pi 5에서 Re-ID로 현재 추적 대상이 맞는지
검증·교정하는 시스템**입니다.

대상 지정은 다음 순서로 처리합니다.

```text
1. ArUco 마커를 가진 사람을 발표자로 등록
2. 마커가 없으면 MediaPipe 손들기 gesture로 발표자 등록
3. 카메라 내장 트래커가 따라가는 사람이 등록 발표자인지 Re-ID로 검증
4. mismatch가 확정되면 전체 프레임에서 사람 후보를 찾고 Re-ID로 복구 후보 확인
```

소프트웨어가 직접 사람을 추적하고 PTZ 명령을 만드는 `src/tracking` 경로는
**카메라 내장 추적 또는 bbox metadata를 못 쓰는 상황을 위한 백업 데모**입니다.
본체는 `main.py`가 실행하는 `src/app/verify_demo.py`입니다.

## 전체 구조

```text
Qon PTZ Camera
  - 내장 AI Auto Tracking
  - 실제 PTZ motor / zoom / smoothing 담당
  - RTSP 검증 영상 제공
        |
        v
Raspberry Pi 5
  - marker 우선 발표자 등록
  - marker가 없으면 MediaPipe gesture 등록
  - 현재 카메라가 따라가는 영역을 가져옴
      현재: center crop fallback 또는 jsonl bbox
      목표: Qon/ONVIF/RTSP metadata live bbox
  - Re-ID로 등록 발표자와 현재 추적 대상 비교
  - mismatch 시 YOLO/person detector로 복구 후보 탐색
        |
        v
Optional Camera Control
  - tracking / zone mode CGI 전환
  - 추후 target reassignment 명령 발견 시 교정 제어
```

## 현재 구현 상태

| 항목 | 상태 |
|---|---|
| Qon tracking/zone mode 전환 | 구현됨 |
| RTSP 기반 검증 앱 | 구현됨 |
| marker 기반 등록 | 구현됨 |
| MediaPipe 손들기 등록 | 구현됨 |
| Re-ID 검증 | 구현됨, 현재 HSV baseline |
| mismatch 후 복구 후보 탐색 | optional YOLO로 구현됨 |
| 카메라 live tracking bbox | 미확보, center crop/jsonl fallback |
| 카메라 target 재지정 명령 | 미확보 |

가장 큰 기술 리스크는 두 가지입니다.

1. **카메라가 실제로 따라가는 bbox를 실시간으로 가져오는 경로**
   - 현재 `center` 모드는 “카메라가 대상을 중앙에 둔다”는 가정입니다.
   - 발표용 MVP에서는 동작 확인이 가능하지만, 완성형은 live bbox metadata가 필요합니다.

2. **Re-ID 품질**
   - 현재는 빠른 MVP 검증을 위한 HSV histogram 기반 matcher입니다.
   - 발표 전 가능하면 OSNet/FastReID 계열 embedding 모델로 교체하는 것이 목표입니다.

## 실행 방법

기본 실행은 Qon 내장 추적을 검증하는 경로입니다.

```bash
python3 main.py --source "$RTSP_URL" --region-mode center
```

첫 프레임에서 카메라가 따라가는 중앙 영역을 바로 등록:

```bash
python3 main.py \
  --source "$RTSP_URL" \
  --region-mode center \
  --register-on-start
```

마커를 가진 사람을 발표자로 등록:

```bash
python3 main.py \
  --source "$RTSP_URL" \
  --region-mode center \
  --registration-mode marker \
  --target-marker-id 7 \
  --register-on-start
```

마커가 있으면 마커를 우선 사용하고, 없으면 MediaPipe 손들기로 등록:

```bash
python3 main.py \
  --source "$RTSP_URL" \
  --region-mode center \
  --registration-mode marker-or-gesture \
  --target-marker-id 7 \
  --register-on-start
```

수동 bbox로 등록:

```bash
python3 main.py \
  --source "$RTSP_URL" \
  --region-mode center \
  --register-on-start \
  --register-bbox 420,120,780,700
```

검증 로그와 결과 영상 저장:

```bash
python3 main.py \
  --source "$RTSP_URL" \
  --region-mode center \
  --registration-mode marker-or-gesture \
  --target-marker-id 7 \
  --register-on-start \
  --no-window \
  --output logs/qon_verify.mp4 \
  --log-csv logs/qon_verify.csv
```

## 주요 옵션

| 옵션 | 설명 |
|---|---|
| `--source` | RTSP URL, 비디오 파일, 카메라 index |
| `--region-mode center` | 카메라 추적 대상이 중앙에 있다고 보고 중앙 crop 검증 |
| `--region-mode jsonl` | 캡처한 bbox jsonl로 검증 |
| `--registration-mode observed` | 현재 관측 영역을 등록 |
| `--registration-mode marker` | 마커가 들어있는 사람 bbox를 등록 |
| `--registration-mode gesture` | MediaPipe 손들기 대상 등록 |
| `--registration-mode marker-or-gesture` | 마커 우선, 없으면 gesture |
| `--target-marker-id` | 특정 ArUco marker id만 허용 |
| `--verify-every-frames` | 몇 프레임마다 Re-ID 검증할지 |
| `--reid-threshold` | Re-ID 검증 threshold |
| `--mismatch-limit` | 연속 실패 몇 회부터 mismatch로 볼지 |
| `--recovery-model` | mismatch 후 전체 프레임 후보 탐색에 사용할 YOLO 모델 |

## Qon Mode Control

카메라 웹 UI Network 캡처로 확인된 CGI 요청을 이용해 tracking/zone 모드를
조회하거나 바꿀 수 있습니다.

```bash
export QON_CAMERA_URL='http://camera-address'

python3 -m src.camera.qon_control --camera-url "$QON_CAMERA_URL" status
python3 -m src.camera.qon_control --camera-url "$QON_CAMERA_URL" tracking
python3 -m src.camera.qon_control --camera-url "$QON_CAMERA_URL" zone
```

확인된 요청:

```text
POST /cgi-bin/param.cgi?get_path   path=/data/track.conf
POST /cgi-bin/param.cgi?write_path path=/data/track.conf&common.track=1  (tracking)
POST /cgi-bin/param.cgi?write_path path=/data/track.conf&common.track=0  (zone)
```

모드 변경 시 함께 캡처된 `post_visca` 요청은 tracking/zone 양쪽 전환에서
동일하게 발생했습니다. 아직 목적이 확인되지 않았기 때문에 코드에서 자동으로
전송하지 않습니다.

## 백업 실행 경로

`src/tracking/marker_reid_demo.py`는 카메라 내장 추적을 쓰지 않고, Pi에서
사람 검출·마커 선택·Re-ID fallback·PTZ 명령 생성을 모두 수행하는 백업
데모입니다.

```bash
python3 -m src.tracking.marker_reid_demo \
  --source "$RTSP_URL" \
  --target-marker-id 7
```

이 경로는 Qon metadata나 재지정 API를 확보하지 못했을 때 발표 데모를
유지하기 위한 fallback입니다. 최종 설계의 본체는 아닙니다.

## 프로젝트 구조

```text
PTZ_Speaker_Tracking/
├── main.py                         # Qon 내장 추적 검증 메인 진입점
├── src/
│   ├── app/
│   │   └── verify_demo.py          # 메인 Re-ID 검증/복구 앱
│   ├── camera/
│   │   ├── qon_control.py          # Qon tracking/zone CGI 제어
│   │   └── region_provider.py      # center/jsonl bbox provider
│   ├── reid/
│   │   ├── appearance.py           # 현재 HSV baseline Re-ID
│   │   └── verifier.py             # Re-ID 검증 상태머신
│   ├── recovery/
│   │   └── candidate_detector.py   # mismatch 후 YOLO 후보 탐색
│   ├── tracking/
│   │   ├── marker_reid_demo.py     # 직접 추적 fallback 데모
│   │   ├── marker_detector.py      # ArUco detector
│   │   ├── gesture_detector.py     # MediaPipe 손들기 detector
│   │   ├── person_detector.py      # HOG / OpenCV YOLO / NCNN detector
│   │   └── ptz_controller.py       # PTZ simulator / hardware placeholder
│   └── vision/
│       ├── capture.py
│       └── models.py
└── tests/
```

## 6/22까지 우선순위

1. **B 구조를 본체로 고정**
   - README, `main.py`, 발표 설명 모두 Qon 내장 추적 + Re-ID 검증 기준으로 통일

2. **카메라 추적 bbox 확보**
   - ONVIF Analytics metadata
   - Qon CGI/RTSP 부가 metadata
   - debug overlay가 픽셀에 burn-in인지, 좌표 metadata인지 확인

3. **Re-ID embedding 모델 적용**
   - 현재 HSV baseline과 OSNet/FastReID embedding 비교
   - Pi 5에서는 후보 crop 수를 제한해 실시간성 확보

4. **MediaPipe 등록 경로 안정화**
   - 마커 없을 때 손든 사람을 등록
   - 단일 person pose 가정과 한계를 발표 자료에 명시

5. **교정 제어**
   - target reassignment 명령을 찾으면 직접 교정
   - 못 찾으면 zone/tracking mode 전환 또는 전체 프레임 복구 후보 로그로 제한

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
