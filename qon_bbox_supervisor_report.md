# Qon PTZ BBox 검증 및 Supervisor 전환 보고서

## 목표

Qon PTZ 카메라의 내장 Auto Tracking이 현재 따라가는 사람의 bbox를 외부 프로그램에서 받을 수 있는지 확인했다. 원래 목표는 카메라 내부 tracking bbox를 받아 Pi 측 Re-ID verifier에 넣고, 카메라가 등록 발표자를 잘못 따라가는 경우를 감지하는 것이었다.

최종 결론은 다음과 같다.

```text
카메라 내부 tracking bbox API는 현재 장비/펌웨어에서 미지원으로 판단한다.
카메라 bbox 의존 경로는 폐기한다.
stream2 기반 Pi 측 detector + identity-first Re-ID + Pi 직접 PTZ 제어 구조로 전환한다.
```

## 장비 및 스트림 정보

현장 확인값:

```text
Camera IP: 192.168.11.88
Gateway: 192.168.11.1
HTTP Port: 80
RTSP Port: 554
TCP Port: 5678
UDP Port: 1259
Sony VISCA: 52381
Main stream 후보: rtsp://192.168.11.88:554/stream1
사용 예정 sub stream: rtsp://192.168.11.88:554/stream2
```

`stream2`는 640x360, 30 fps로 확인되었고, Pi 측 detector 입력으로 사용하기에 main stream보다 부담이 낮다.

## 웹 UI JS 분석

카메라 웹 UI에서 추출한 `2.js` 안에 bbox처럼 보이는 코드가 있었다.

```js
query_target_status: {
  method: "get",
  url: "/cgi-bin/param.cgi?get_target_status",
  dataType: "json",
  callback: function(t) { return t }
}
```

UI drawing loop는 `objs[]`의 `X`, `Y`, `Width`, `Height`를 canvas 좌표로 변환한다.

```js
startX: 720 * n[o].X / 640
startY: 720 * n[o].Y / 640
width: 720 * n[o].Width / 640
height: 720 * n[o].Height / 640
```

따라서 처음에는 다음 형태의 응답을 기대했다.

```json
{
  "shoulddraw": "1",
  "target": 0,
  "objs": [
    {"X": 120, "Y": 80, "Width": 160, "Height": 240}
  ]
}
```

하지만 실측 결과는 달랐다. OEM 공통 JS 번들에는 현재 펌웨어에 없는 기능 코드도 포함되어 있었다. 코드 존재는 기능 존재를 보장하지 않는다.

## HTTP BBox API 실측

시도한 요청:

```bash
curl -i -u "ID:PW" "http://192.168.11.88/cgi-bin/param.cgi?get_target_status"
```

결과:

```html
<h1>404 Not Found</h1><hr>
```

즉 JS에는 endpoint 정의가 있으나, 현재 펌웨어에서는 실제 endpoint가 존재하지 않거나 비활성화되어 있다.

추가로 `get_draw_box_flag`를 확인했다.

```bash
curl -i -u "ID:PW" "http://192.168.11.88/cgi-bin/param.cgi?get_draw_box_flag"
```

핵심 응답:

```text
istrackingbox="0"
issupport="0"
```

`issupport="0"`은 tracking box/status 기능이 현재 장비/펌웨어에서 미지원임을 강하게 시사한다. 이 값이 bbox API 폐기 판단의 핵심 근거다.

## write_path Success의 함정

`tracking.target_en=1`도 시도했다.

```bash
curl -i -u "ID:PW" -X POST \
  -d "path=/data/track.conf&tracking.target_en=1" \
  "http://192.168.11.88/cgi-bin/param.cgi?write_path"
```

응답은 성공이었다.

```json
{"Response":{"Result":"Success"}}
```

하지만 `get_path`로 read-back하면 `tracking.target_en`은 저장되지 않았다. 결론:

```text
이 카메라의 write_path Success는 키 적용을 보장하지 않는다.
모든 설정 변경은 write -> VISCA refresh -> read-back으로 검증해야 한다.
```

이 원칙은 이후 `track_mode`, `auto_zoom`, `auto_tilt`, `debug_mode`, `common.track` 검증에 그대로 적용했다.

## track.conf 확인

Auto Tracking, debug, OSD 관련 주요 값:

```text
common.track="1"
common.track_mode="tracking"
common.debug_mode="2"
common.osd_mode="1"
tracking.auto_zoom="1"
tracking.auto_tilt="1"
```

초기에는 `common.debug_mode="2"`라 debug bbox overlay가 영상에 burn-in될 수 있었다. 이는 Pi detector를 오염시킬 수 있으므로 실제 파이프라인에서는 꺼야 한다.

## ONVIF / RTSP 확인

ONVIF를 켠 뒤 Events를 확인했지만, 사람이 움직여도 bbox나 coordinate 수준의 자세한 metadata는 확인되지 않았다.

RTSP는 `ffprobe`로 확인했다.

Main 후보:

```bash
ffprobe -hide_banner -rtsp_transport tcp "rtsp://192.168.11.88:554/stream1"
```

결과:

```text
Stream #0:0: Video: h264, 1920x1080, 30 fps
Stream #0:1: Audio: aac, 48000 Hz, stereo
```

Sub stream:

```bash
ffprobe -hide_banner -rtsp_transport tcp "rtsp://192.168.11.88:554/stream2"
```

결과:

```text
Stream #0:0: Video: h264, 640x360, 30 fps
Stream #0:1: Audio: aac, 48000 Hz, stereo
```

둘 다 video/audio만 있고 다음 항목은 없었다.

```text
Data
metadata
vnd.onvif.metadata
application
```

정확히는 "현재 설정과 현재 URL 기준으로 RTSP metadata track은 확인되지 않았다"가 맞다. 다만 `issupport=0`이라는 상위 증거 때문에 ONVIF/RTSP metadata를 더 파는 기대값은 낮다고 판단했다.

## RTMP / FreeD 판단

웹 UI RTMP에는 다음 MRL이 있었다.

```text
rtmp://192.168.100.138/live/stream0
rtmp://192.168.100.138/live/stream1
```

이는 카메라가 외부 RTMP 서버로 push하는 주소로 보이며, RTSP/ONVIF metadata나 bbox 수신과는 직접 관련이 낮다.

FreeD는 PTZ pose 데이터 계열로 bbox가 아니므로 본 작업에서는 후순위로 분류했다.

## VISCA Refresh 메커니즘

웹 UI는 `track_mode` 등 설정을 바꾼 뒤 다음 VISCA payload를 보낸다.

```text
0x81,0x0a,0x01,0x04,0x1d,0x17,0xff
```

실측 결과, `post_visca` 응답의 `Result`는 보낸 바이트 echo일 뿐 적용 성공 신호가 아니다. 적용 여부는 read-back으로만 확인한다.

검증된 적용 순서:

```text
1. write_path
2. post_visca refresh
3. get_path read-back
```

`common.track_mode=region`은 이 순서로 실제 read-back에서 변경되었다.

## Supervisor 액추에이터 모드 검증

최종적으로 카메라 자체 PTZ 구동을 끄고 Pi가 PTZ 전권을 갖는 방향으로 정리했다.

적용한 설정:

```text
common.track="0"
tracking.auto_zoom="0"
tracking.auto_tilt="0"
common.debug_mode="3"
```

의미:

```text
common.track=0          카메라 내부 tracking off
tracking.auto_zoom=0    카메라 자동 zoom off
tracking.auto_tilt=0    카메라 자동 tilt off
common.debug_mode=3     debug bbox burn-in off
```

`auto_zoom=0`, `auto_tilt=0`은 read-back으로 적용 확인했고, 이후 `common.track=0`을 통해 카메라 내부 tracking 자체도 끄는 운영으로 전환한다.

외부 PTZ 제어는 `ptzctrl.cgi`로 실제 동작을 확인했다.

```bash
curl -i -u "ID:PW" \
  "http://192.168.11.88/cgi-bin/ptzctrl.cgi?ptzcmd&left&10&10"

curl -i -u "ID:PW" \
  "http://192.168.11.88/cgi-bin/ptzctrl.cgi?ptzcmd&ptzstop&10&10"
```

중요한 관찰:

```text
ptzstop을 보내지 않으면 계속 움직인다.
```

따라서 Qon PTZ는 위치 명령이 아니라 방향/속도 기반 연속 모션 모델이다. 모든 move에는 stop 보장이 필요하다.

## 설계 Trade-off

### 폐기한 설계: 카메라 내부 bbox 사용

장점:

```text
카메라가 실제 따라가는 bbox를 직접 받아 Re-ID에 넣을 수 있음
```

폐기 이유:

```text
get_target_status 404
issupport=0
tracking.target_en 저장 안 됨
RTSP metadata track 없음
ONVIF Events에서 bbox 없음
```

### 폐기한 fallback: center crop / center proximity 중심

center crop 또는 center-nearest person은 카메라가 정상 추적할 때는 그럴듯하다. 하지만 시스템의 목표는 카메라가 엉뚱한 사람을 따라갈 때 이를 감지하는 것이다.

문제:

```text
카메라가 틀린 사람을 중앙에 두면 center 기반 provider도 같은 틀린 사람을 고른다.
실패 감지기가 실패 원인과 같은 가정을 공유한다.
```

따라서 center는 등록 bootstrap, tie-break, PTZ error 계산의 보조 신호로만 둔다.

### 채택한 설계: identity-first supervisor

운영 구조:

```text
stream2 frame
-> Pi 측 person detector로 전체 사람 bbox 후보 생성
-> 각 후보를 Re-ID matcher로 등록 발표자 gallery와 비교
-> best identity 후보를 observed bbox로 선택
-> center 후보와 score margin 비교
-> target identity와 camera center의 불일치가 지속되면 mis-track
-> Pi가 ptzctrl.cgi로 closed-loop PTZ 제어
```

핵심 조건:

```text
관측 신뢰:
best_score >= weak_min_score
AND best_score - second_score >= identity_margin

mis-track 본체:
best_score - center_person_score >= center_margin
AND best bbox가 center dead zone 밖
AND N frames 지속
```

관측 신뢰 조건을 통과하지 못하면 바로 LOST나 mismatch로 가지 않고 HOLD 상태로 둔다. HOLD가 여러 프레임 지속될 때만 LOST로 전이한다. 이렇게 해야 조명, 거리, 포즈 변화로 Re-ID score가 순간적으로 흔들릴 때 오발을 줄일 수 있다.

## 사용 기술

확인 및 제어:

```text
curl
ffprobe
ONVIF Device Manager
Qon HTTP CGI: param.cgi, ptzctrl.cgi
VISCA refresh over param.cgi?post_visca
```

영상 및 비전:

```text
RTSP stream2, 640x360, 30 fps
OpenCV VideoCapture
HOG / OpenCV YOLO / NCNN detector 옵션
HSV baseline Re-ID
OSNet-style ONNX Re-ID backend 준비
ArUco marker 기반 등록
MediaPipe gesture fallback 옵션
```

제어:

```text
ptzctrl.cgi direction/speed command
ptzstop required
P-style closed-loop control
dead zone
deadman/stop 보장
```

## 코드 반영 내용

이번 코드 변경의 핵심:

```text
IdentityMatchedRegionProvider 추가
2단 quality gate 추가
identity margin / center margin / hold limit 옵션 추가
Qon write_path + VISCA refresh + read-back 유틸 추가
supervisor actuator mode 추가(common.track=0, auto_zoom=0, auto_tilt=0, debug_mode=3)
QonVelocityPTZController 추가
ptz-follow-target 옵션 추가
CSV 로그에 identity score/margin/counter/PTZ action 추가
```

권장 실행 형태:

```bash
python3 -m src.app.verify_demo \
  --source "rtsp://192.168.11.88:554/stream2" \
  --region-mode identity \
  --register-on-start \
  --registration-mode marker \
  --registration-reference selected \
  --target-marker-id 7 \
  --detector hog \
  --camera-url "http://192.168.11.88" \
  --camera-username "ID" \
  --camera-password-env QON_PASS \
  --camera-auth-mode digest \
  --configure-supervisor-actuator \
  --ptz-follow-target \
  --log-csv logs/identity_supervisor.csv
```

강의실 실험에서는 detector와 Re-ID threshold/margin 값을 로그로 캘리브레이션해야 한다. 현재 기본값은 초기값이며 확정값이 아니다.

## 보고서용 핵심 문장

```text
카메라 내부 tracking bbox endpoint는 JS 번들에 존재하나 실제 펌웨어에서 404이며, get_draw_box_flag의 issupport=0으로 미지원이 확인되었다. 또한 RTSP stream1/stream2는 현재 설정 기준 video/audio만 포함하고 metadata track은 확인되지 않았다. write_path가 미지원 키에도 Success를 반환함을 확인했으므로 이후 모든 설정 변경은 write -> VISCA refresh -> read-back으로 검증한다. 카메라 내부 bbox 의존 경로를 폐기하고, stream2 기반 Pi 측 detector + identity-first Re-ID + Pi 직접 PTZ 제어 파이프라인으로 전환한다.
```
