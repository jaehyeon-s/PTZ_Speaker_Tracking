# Qon4K6012XN Re-ID Verification MVP

이 프로젝트는 `Qon4K6012XN NDI` 카메라의 내부 AI Auto Tracking을 기본
추적기로 사용하고, Raspberry Pi 5에서 Re-ID로 현재 추적 대상이 등록된
강사인지 검증하는 MVP입니다.

카메라가 PTZ 이동과 상시 추적을 담당합니다. Pi 5는 RTSP 검증 스트림을
저주기로 확인하며, 오추적이 의심될 때만 선택적으로 YOLO 후보 탐색을
실행할 수 있습니다. RTSP 주소와 계정 정보는 코드에 저장하지 않습니다.

## Confirmed Camera Capabilities

`Qon4K6012XN NDI` 판매사 브로셔에서 확인되는 기능:

```text
4K60P / 12x optical zoom / PTZ
AI auto tracking (lecturer and zone tracking)
First Stream / Second Stream
RTSP / ONVIF / VISCA over IP / NDI HX2
```

웹 UI Network 캡처로 확인된 제어:

```text
POST /cgi-bin/param.cgi?get_path   path=/data/track.conf
POST /cgi-bin/param.cgi?write_path path=/data/track.conf&common.track=1  (tracking)
POST /cgi-bin/param.cgi?write_path path=/data/track.conf&common.track=0  (zone)
```

아직 확인되지 않은 기능:

```text
내부 tracking bbox 조회 API 또는 metadata stream
외부 프로그램의 추적 대상 재지정 명령
```

모드 변경 시 함께 캡처된 두 `post_visca` 요청은 tracking/zone 양쪽 전환에
동일하게 발생하므로, 목적을 확인하기 전에는 코드에서 자동 전송하지 않습니다.

## Architecture

```text
Qon4K6012XN
  internal AI tracking + physical PTZ movement
  RTSP verification stream
        |
        v
Pi 5 Re-ID Verifier
  register presenter appearance
  obtain observed target region
    - center crop fallback now
    - camera bbox metadata adapter later
  verify at a low frame interval
  log VERIFIED / MISMATCH events
        |
        v (optional, only after confirmed mismatch)
YOLO Recovery Detector
  find whether the registered presenter is still visible
  report a recovery candidate; automatic camera reassignment is pending API discovery
```

The `center` region mode assumes the camera keeps its active tracking subject
near the center of the frame. It is suitable for initial verification testing,
but camera-provided bbox metadata is preferred once its interface is identified.

## Structure

```text
PTZ_Speaker_Tracking/
├── main.py
├── requirements.txt
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
│   └── vision/
│       ├── capture.py
│       └── models.py
└── tests/
```

## Setup

```bash
cd /home/gpu_team/ptz/PTZ_Speaker_Tracking
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Only when using optional YOLO recovery detection:

```bash
python3 -m pip install "ultralytics>=8.3.0"
```

## Confirmed Tracking Mode Control

캡처한 CGI 요청을 이용해 현재 모드를 조회하거나 전환할 수 있습니다. 주소와
인증 정보는 저장소에 넣지 않고 실행 시 제공합니다.

```bash
export QON_CAMERA_URL='http://camera-address'

python3 -m src.camera.qon_control --camera-url "$QON_CAMERA_URL" status
python3 -m src.camera.qon_control --camera-url "$QON_CAMERA_URL" tracking
python3 -m src.camera.qon_control --camera-url "$QON_CAMERA_URL" zone
```

카메라가 HTTP Basic 인증을 요구하는 경우 비밀번호는 환경 변수로 전달합니다.

```bash
export QON_PASSWORD='camera password'

python3 -m src.camera.qon_control \
  --camera-url "$QON_CAMERA_URL" \
  --username admin \
  --password-env QON_PASSWORD \
  status
```

이 제어 모듈은 mode switch만 담당합니다. Re-ID 검증 앱이 오추적을 발견했을
때 자동으로 mode를 바꾸지는 않습니다. 자동 복구 정책은 bbox 또는 대상
재선택 제어면이 더 확인된 후 추가합니다.

## First Test Without Bbox API

1. In the camera web UI, enable Auto Tracking while the correct lecturer is
   centered and being followed.
2. Run the verifier using a low-resolution RTSP stream if available.
3. `--register-on-start` records the first centered subject as the lecturer.
4. The verifier checks the center region every N frames.

```bash
export RTSP_URL='rtsp URL supplied outside the repository'

python3 main.py \
  --source "$RTSP_URL" \
  --region-mode center \
  --register-on-start \
  --verify-every-frames 30 \
  --rtsp-drop-frames 1 \
  --no-window \
  --output logs/qon_verify.mp4 \
  --log-csv logs/qon_verify.csv
```

For interactive testing, omit `--no-window` and press `r` after the camera is
tracking the correct lecturer:

```bash
python3 main.py --source "$RTSP_URL" --region-mode center
```

Keyboard controls:

```text
r  register the currently observed center region as the lecturer
q  quit
```

Key arguments:

```text
--center-width-ratio N     observed center crop width ratio, default 0.42
--center-height-ratio N    observed center crop height ratio, default 0.86
--verify-every-frames N    verification interval, default 30
--reid-threshold N         minimum verified appearance score, default 0.68
--mismatch-limit N         consecutive low scores before confirmed mismatch
```

## Optional Recovery Diagnosis

Until an AI target reassignment command is discovered, recovery mode only
reports whether the registered lecturer remains visible after mismatch.
YOLO is not run in the normal verification path.

```bash
python3 main.py \
  --source "$RTSP_URL" \
  --region-mode center \
  --register-on-start \
  --recovery-model models/yolo26n_ncnn_model \
  --no-window \
  --log-csv logs/qon_recovery_check.csv
```

If a candidate is found after confirmed mismatch, the log event becomes
`RECOVERY_CANDIDATE_FOUND`. The code does not yet attempt to move the camera or
change its internal tracking target.

## Connecting Camera Bbox Metadata Later

If the camera debug bbox can be extracted, the verifier no longer needs to
assume the target is centered. A JSON Lines development adapter is already
provided. Each line contains the internal target bbox for a recorded frame:

```json
{"frame": 1, "bbox": [450, 120, 820, 700]}
{"frame": 31, "bbox": [470, 120, 840, 700]}
```

Run against a captured test video:

```bash
python3 main.py \
  --source test_video.mp4 \
  --region-mode jsonl \
  --bbox-jsonl captured_bbox.jsonl \
  --register-on-start \
  --no-window \
  --log-csv logs/metadata_verify.csv
```

When the live Qon metadata interface is identified, implement another provider
with the same `region_for_frame(frame_index, frame)` method in
`src/camera/region_provider.py`.

## How To Find The Debug Bbox Interface

The camera visibly drawing a debug bbox does not prove it is externally
available. Check in this order:

1. Open the camera web UI in Chrome or Edge and press `F12`.
2. In `Network`, enable `Preserve log`, then check `Fetch/XHR`, `WS`, and
   `Media` while enabling debug overlay and Auto Tracking.
3. Look for requests containing terms such as `tracking`, `ai`, `debug`,
   `metadata`, `osd`, `bbox`, `human`, or `face`.
4. If a WebSocket is present, inspect received frames while the bbox moves.
5. `get_path` 요청의 response body에서 `track.conf`에 debug 또는 metadata
   관련 설정값이 함께 내려오는지 저장합니다.
6. Debug 표시 버튼을 켜고 끌 때 새 `write_path` 또는 `post_visca` 요청이
   생기는지 캡처합니다.
7. Compare RTSP output with debug overlay enabled and disabled. If the
   rectangle is burned into the video pixels, it is display overlay rather
   than usable bbox coordinates.
8. Inspect the manual/package download material for CGI, HTTP API, VISCA
   extension, or metadata documentation.
9. Ask the distributor for the `Qon4K6012XN network control protocol`,
   `AI tracking command`, and `tracking metadata/bounding-box API`.

When sharing captured requests, remove IP addresses, credentials,
`Authorization` headers, cookies, and session tokens.

## Log Interpretation

CSV output columns:

| Field | Meaning |
|---|---|
| `state` | `UNREGISTERED`, `VERIFIED`, `MISMATCH`, or `NO_REGION` |
| `event` | registration, verification, mismatch, or recovery candidate event |
| `region_source` | `center_crop` or `metadata_jsonl` |
| `bbox` | region compared against the registered lecturer |
| `score` | appearance similarity score |
| `recovery_bbox` | candidate found only during optional recovery diagnosis |

## Test

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q main.py src tests
python3 main.py --help
```

Sources:

- [Digital Hongil Qon4K6012XN product page](https://www.redsun.co.kr/goods/goods_view.php?goodsNo=3614)
- [Qon4K6012XN / Qon4K6020XN brochure](https://data.redsun.co.kr/brochure/Qon4K60_brouchure.pdf)
