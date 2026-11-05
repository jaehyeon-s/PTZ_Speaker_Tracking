# 구현 현황

최종 갱신: 2026-09-04

이 프로젝트의 목표는 실제 강의실에서 PTZ 카메라가 현재 발화자를 안정적으로
추적하는 것입니다. Unity 장면은 가림, 교차, detector bbox 변화와 ID churn을
반복 재현하기 위한 테스트 수단이며 Unity 장면 자체에 최적화하지 않습니다.

## 현재 권장 파이프라인

```text
NCNN YOLO person detection
  -> BoT-SORT + sparseOptFlow GMC (tracker-level ReID off)
  -> ArUco 등록/발화자 변경 신호
  -> 논리적 speaker identity manager
  -> 선택적 512차원 person ReID
  -> bbox 경로 + appearance 결합 복구
  -> Qon HTTP PTZ
```

권장 실행 설정은 `botsort-pi.yaml`과 앱 수준 선택적 ReID 조합입니다.

## ID 의미

- `target_id`: BoT-SORT가 현재 관측에 붙인 임시 ID. 가림이나 bbox 변화로 자주
  바뀔 수 있습니다.
- `speaker_id`: 등록된 실제 발화자의 논리적 세대 ID. 같은 사람의 `target_id`가
  바뀌어도 유지됩니다.
- ArUco가 다른 사람에게 전달되어 변경이 확정되면 `speaker_id`가 1 증가합니다.

예시:

```text
첫 발화자: speaker_id=1, target_id=2 -> 4 -> 8
마커 전달: speaker_id=2, target_id=17 -> 21 -> 30
```

## 구현 완료

### 최초 등록

- ArUco 중심이 들어 있는 person bbox를 후보로 선택합니다.
- 기본 2초 동안 동일한 물리적 bbox 경로가 유지되어야 등록됩니다.
- raw tracker ID가 바뀌어도 IoU/중심 이동이 연속적이면 등록 타이머를 유지합니다.

### 동일 발화자 추적

- 평상시에는 BoT-SORT 위치와 ID를 사용합니다.
- 활성 ID 소실, 사람 겹침/분리, ID 재등장 시에만 ReID를 실행합니다.
- 겹친 crop은 `reid=-1.000`으로 표시하고 신원 판단과 갤러리 갱신에서 제외합니다.
- 짧은 거리에서는 bbox 경로가 주 증거이고 ReID는 보조 점수입니다.
- 먼 후보는 ReID 절대 기준을 통과해야 같은 사람으로 복구됩니다.
- 복구 후보는 기본 0.12초 동안 물리 경로가 이어진 뒤 현재 target으로 확정됩니다.

### 교차 ambiguity 해소

- 최근 여러 사람이 보인 상태에서 활성 raw ID가 사라지거나 bbox가 겹치면
  `identity_resolving` 상태로 진입합니다.
- 이 상태에서는 generic geometry handoff를 차단하고 PTZ를 정지합니다.
- 교차 전 raw ID와 같은 번호가 재등장해도 동일 신원으로 자동 승인하지 않습니다.
- 분리 후 후보를 raw ID가 아닌 bbox 물리 경로 가설로 연결합니다.
- top-aligned containment가 높은 전신/상단 bbox 쌍은 동일 인물의 중복 검출로
  병합하여 교차 이벤트에서 제외합니다.
- trackless/단일 후보 구간은 다중 후보 경로 evidence를 초기화하여 교차 중
  혼합 crop이 분리 후 다른 사람에게 상속되지 않게 합니다.
- 기본 6 clean frame과 0.50초 이상 동안 ReID 중앙값, 마지막 프레임의
  1·2위 margin 및
  프레임별 winner 비율을 만족해야 새 raw ID를 logical speaker에 연결합니다.
- 후보가 모호하면 오답을 선택하지 않고 보류합니다.
- 검증된 raw ID가 기존 ID와 같아 handoff가 발생하지 않아도 resolution 상태는
  즉시 종료합니다.
- 한 후보만 남은 경우에는 기존 절대 ReID 기준을 여러 프레임 통과해야 복구합니다.

### ReID 갤러리

- ArUco로 확인되고 다른 사람과 많이 겹치지 않는 crop만 저장합니다.
- 같은 시점의 비발화자 crop은 exclusion gallery로 별도 저장하며, 후보 점수는
  발화자와의 유사도와 비발화자와의 비유사도를 함께 반영합니다.
- feature cache는 ndarray 객체를 직접 보유하여 다음 프레임의 객체 ID 재사용으로
  이전 임베딩이 섞이는 문제를 방지합니다.
- 최대 12개 샘플, 기본 0.35초 간격, cosine similarity를 사용합니다.
- 낮은 유사도의 재획득 후보와 겹친 crop은 갤러리를 오염시키지 못합니다.

### 발화자 변경

- 추적 중 ArUco가 다른 person bbox에 기본 확인 시간 동안 유지되면
  `confirmed_marker_transfer` 이벤트를 생성합니다.
- 변경 확정 시 `speaker_id`를 증가시킵니다.
- 이전 발화자의 ReID 갤러리, crop cache, ambiguity/recovery 상태를 원자적으로
  초기화합니다.
- 같은 프레임의 새 발화자 crop으로 새 갤러리를 시작합니다.
- 이후 교차 복구는 새 발화자 갤러리만 기준으로 수행합니다.
- 기존 발화자의 단순 raw ID 변경인 `target_recovered_new_track_id`는
  `speaker_id`를 증가시키거나 갤러리를 초기화하지 않습니다.
- 마커를 포함한 중복/부분 bbox가 기존 대상과 크게 겹치는 경우에는
  `confirmed_marker_reassociation`으로 처리하고 같은 `speaker_id`를 유지합니다.

정상 변경 로그:

```text
reason=confirmed_marker_transfer
Speaker ReID gallery reset: speaker_id=2 previous_samples=...
Speaker changed: previous_speaker_id=1 current_speaker_id=2 ...
Speaker ReID gallery updated: speaker_id=2 ...
```

## 권장 실행 명령

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
  --qon-log-commands \
  2>&1 | tee selective_reid_crossing.log
```

재현 가능한 테스트에서는 다음 옵션도 추가합니다.

```bash
  --record-input-video test_video/raw/crossing_01.mp4 \
  --record-input-fps 15 \
  --identity-trace-file test_video/traces/crossing_01.jsonl
```

영상은 화면 overlay 이전의 실제 추론 입력이고, trace는 프레임별 detection,
appearance similarity와 identity 결정을 기록합니다.

## 발화자 변경 테스트 순서

1. A가 마커를 착용한 상태로 `speaker_id=1` 등록을 완료합니다.
2. A와 B가 분리되어 보이는 상태에서 마커를 B에게 전달합니다.
3. `confirmed_marker_transfer`와 `speaker_id=2`를 확인합니다.
4. 새 갤러리 샘플이 B의 bbox에서 생성되는 것을 확인합니다.
5. 마커가 보이지 않는 상태로 A와 B를 교차시킵니다.
6. raw `target_id`가 바뀌더라도 카메라가 B를 계속 따라가는지 확인합니다.

마커를 손에서 전달하는 중이거나 두 bbox가 겹친 상태는 변경 확정에 적합하지
않습니다. 새 착용자의 bbox 안에서 마커가 확인 시간 동안 안정적으로 보여야 합니다.

## 현재 검증 상태

- 단위 테스트 178개 통과: 최초 등록, ID churn, 가림 복구, 겹침 보류, raw ID 재등장 검증,
  마커 전환, speaker ID 증가, 갤러리 교체, 실제 실패 로그 기반 `2 -> 1 ->
  swapped 2 / correct 13` 교차 회귀와 접근 중 조기 확정/raw ID 교환 회귀를
  포함합니다.
- `crossing_02.mp4` 전체를 detector/tracker/ReID까지 오프라인 재생했습니다.
  접근 중 조기 확정이 제거됐고, 마지막 모호한 교차에서는 잘못된 handoff 없이
  resolution을 유지했습니다.
- 교차 중에는 갱신되는 물리 경로를 속도 2로 soft-coast하고, 유효한 다중 경로
  관측이 끊긴 뒤 0.8초가 지나 예측이 만료되면 PTZ를 정지합니다.
- Qon PTZ는 시작/정지 deadzone hysteresis, 정지 debounce, 짧은 grace dropout
  hold를 적용하며 ambiguity/release 안전 정지는 즉시 수행합니다.
- `crossing_04.mp4` 오프라인 재생에서 `identity_resolving`이 16.93초에서
  9.80초로 감소했고 마지막 상태가 정상 `tracking`으로 종료됐습니다.
- 최초 마커 샘플 6개는 고정 anchor gallery로 유지하며, 이후 샘플은 별도의
  adaptive gallery에서만 교체됩니다.
- `--person-tracker decoupled`(검출·추적 분리): `crossing_04.mp4` 414프레임을
  `botsort-pi.yaml`로 재생했을 때 기존 `ultralytics` 경로와 track ID, 상자 좌표,
  판단 결과가 모든 프레임에서 같았습니다(ReID 미사용 조건).
- Unity: 외형이 다른 2인이 교차할 때 기존 발화자 유지가 확인되었습니다.
- 실제 강의실: 아직 최종 검증이 필요합니다.

## 남은 작업

- 실제 강의실의 역광, 모션 블러, 비슷한 복장, 강연대 가림 데이터로 임계값 검증
- MediaPipe Pose 등록 경로를 동일 identity manager의 등록 이벤트로 연결
- 실제 원본 영상/identity trace를 사용한 반복 replay 평가

## 2026-10-07 `room_01` 실기 로그 기반 수정

`room_01` (Pi 5 + Hailo, 실제 강의실, 3~4명)에서 교차 직후 `identity_resolving`에
약 3분간 머물고 카메라가 빈 벽을 향해 계속 회전한 문제를 수정했습니다.

- 대상이 없고 coast 추정도 없으면 카메라 목표를 주지 않습니다. 이전에는 마지막
  확정 bbox(낡은 화면 좌표)를 향해 계속 회전했습니다.
- 다른 사람들과 점수 차가 뚜렷하면(`--speaker-resolution-strong-margin`, 기본 0.15)
  절대 임계값 0.55 미만이어도 새 raw ID를 검증합니다. 겹친 사람이 화면에 있으면
  이 규칙은 적용하지 않습니다.
- 발화자가 관여하지 않은 겹침은 resolution을 시작하지 않으며, 겹친 사람만 점수에서
  제외합니다.
- 대상이 사라진 상태에서는 마커가 같은 사람에게서 3초 안에 3번 보이면 재등록합니다
  (`--reacquire-marker-observations`, `--reacquire-marker-window-seconds`).
- 근거 없이 10초가 지나면 release 합니다(`--speaker-resolution-timeout-seconds`).
- ReID는 crop 한 장씩 추론합니다. 크기가 다른 crop을 한 번에 넣으면 448x448로
  padding 되어 3명 기준 약 350ms가 걸렸습니다(현재 약 240ms).
- Qon PTZ: tilt 오차를 pan과 같은 픽셀 기준으로 계산(`--qon-uniform-error-scale`),
  응답 지연 보상(`--qon-response-delay-seconds`), 축 전환 hysteresis
  (`--qon-axis-switch-ratio`), 오차 0.5에서 최대 속도(`--qon-full-speed-error`).

`room_01`의 프레임·검출·타임스탬프를 수정된 판단 로직에 다시 넣은 open-loop 재생에서
교차 후 약 8초 만에 발화자를 다시 잡았고, 이후 handoff는 모두 실제 발화자였습니다.

수정 후 같은 이름으로 다시 실행한 실기 영상(11:18~11:24, 395초, 현재의 `room_01.*`)에서는
추적 310초, `identity_resolving` 60초(대상 없이 최장 5.2초), 평균 17.1fps였고 1.5초 내
PTZ 방향 반전은 4회였습니다. 대상 전환 8회는 전후 프레임에서 같은 발화자임을 확인했습니다.
처음 분석한 실패 영상은 이 실행으로 덮어써졌습니다.

## 2026-10-12: 놓친 발화자의 마커 재등록 조건 강화 (pi_unity_04)

Unity 30fps 시험(`pi_unity_04`)에서 발표자가 앉아 있는 사람 옆을 지나가는 동안
발표자의 사람 상자가 약 0.6초 검출되지 않았고, 그사이 마커가 옆 사람(track 104)의
상자 안에 3프레임(0.14초) 들어가 그 사람이 새 발화자로 등록됐습니다.

- 빠른 재등록(마커 3회 관측)에 최소 시간 조건을 추가했습니다
  (`--reacquire-marker-min-seconds`, 기본 0.5초).
- 발화자와 같은 프레임에 보였던 track은 발화자 본인일 수 없으므로 빠른 재등록 대상에서
  제외했습니다. 이런 사람에게 마커가 보이면 일반 확인 시간(프로필 기준 2초)을 거쳐야
  발화자가 바뀝니다.

녹화 재생 결과(판단 로직만 재생, RepVGG float 모델을 CPU에서 사용):

| 녹화 | 수정 전 | 수정 후 |
| --- | --- | --- |
| `pi_unity_04` | 29.29초에 앉은 사람(104)으로 전환 | 104로 전환 없음. 29.97초에 발표자(274)로 재등록 |
| `pi_unity_03` | 51.52초에 기둥 뒤에서 나온 발표자(161) 재등록 | 51.86초에 같은 재등록 |
| `room_01` (6764프레임) | - | 모든 프레임의 판단이 동일 |
| `room_02` (1053프레임) | - | 모든 프레임의 판단이 동일 |

남은 문제: 같은 발표자가 다시 등록될 때 유사도가 0.55 미만이면 `speaker_id`가 증가하고
갤러리가 새로 만들어집니다(`pi_unity_03`, `pi_unity_04` 모두 해당). 카메라가 따라가는
대상은 맞습니다. 실제 카메라로는 이 수정을 아직 돌려 보지 않았습니다.

## 2026-10-12: 위치 기반 ID 복구가 옆 사람을 잡는 문제 (pi_unity_05)

`pi_unity_05`(Unity 30fps, 8~11명)에서 발표자의 상자가 몇 프레임 빠진 사이, 가장 가까운
상자였던 옆자리 사람이 0.12초 만에 발표자의 "새 track ID"로 복구됐습니다(5.85초, 29.43초,
58.74초). 세 번 모두 마커가 화면에 보이고 있었고, 마커가 발표자에게서 1~2초 확인된 뒤에야
돌아왔습니다. `target_recovered_new_track_id` 경로는 9월 버전부터 있던 규칙입니다.

- 발화자와 같은 프레임에 떨어져(IoU 0.1 미만) 보였던 track은 위치 기반 복구 후보에서
  제외합니다.
- 마커가 보이는 프레임에서는 그 마커를 상자 안에 가진 사람만 복구 후보가 됩니다.

녹화 재생 결과(직전 커밋과 프레임 단위 비교):

| 녹화 | 결과 |
| --- | --- |
| `pi_unity_05` | 옆 사람으로의 복구 3건이 모두 사라짐. 남은 대상 전환 7건은 프레임 확인 결과 모두 발표자 |
| `pi_unity_04`, `pi_unity_03` | 대상 전환 동일. 상태만 5프레임, 2프레임 다름 |
| `room_02` | 대상 전환 동일. 상태만 2프레임 다름 |
| `room_01` | 모든 프레임 동일 |

남은 문제: `pi_unity_05`에서 같은 발표자가 다시 등록될 때마다 `speaker_id`가 올라가고
갤러리가 새로 만들어졌습니다(5회). 실제 카메라로는 이 수정을 아직 돌려 보지 않았습니다.

## 2026-10-25: dev/next-ptz 기능 이식, Unity 시나리오 시험 준비 (브랜치 `pi-npu-features`)

되돌리기 기준점: 태그 `npu-verified-20261012` (= `1cffd95`, Pi에서 검증한 상태).

`dev/next-ptz`(`929e8d8`)의 기능을 NPU 브랜치에 옵션으로 옮겼습니다. 새 옵션은 모두 기본값이
꺼짐이라 기존 Pi 실행 명령의 동작은 그대로입니다. `room_01`, `room_02`, `pi_unity_03~05`를
재생하면 판단, 대상 상자, PTZ 명령이 모든 프레임에서 이식 전과 같습니다.

- 줌(`--zoom-control` 외), 놓친 뒤 찾기(`--lost-search`, `--zoom-lost-search`), 후보 추종
  (`--switch-follow-candidate`), 예측 추종(`--grace-predict-follow`), PTZ 옵션(`--qon-speed-map`,
  `--qon-predict-target-seconds`, `--qon-level-speeds`, `--qon-axis-mode` 등), 해제 보완
  (`--target-low-confidence`, `--occlusion-head-anchor`, `--recovery-min-area-ratio`,
  `--qon-lost-coast-seconds`), `--marker-min-pixels`.
- 같은 코드를 양쪽이 고친 곳은 NPU 쪽을 유지했습니다. 카메라 오차 계산(room_01의 지연 보상,
  축 전환 hysteresis)은 그대로이고, dev의 예측 보정은 옵션을 줄 때만 그 앞에 적용됩니다.
  **C1의 `--qon-predict-target-seconds 0.4`와 프로필의 `qon_response_delay_seconds 0.5`는 둘 다
  지연 보상이라, 이 브랜치에서 C1을 켜면 보상이 겹칩니다.** dev에서 C1을 측정할 때는 후자가 없었습니다.
- `--target-low-confidence`는 dev에서는 Ultralytics 사람 소스에만 있었습니다. NPU 경로(decoupled)에도
  연결했습니다.
- `--speaker-reid-onnx`: Pi의 ReID 모델(RepVGG-A0)을 float ONNX로 CPU에서 실행합니다(정규화는
  Model Zoo 값). PC에서도 Pi와 같은 모델·보정값(1.31, -0.537)을 씁니다.
- GPU 검출기(`--model yolo26n.pt --imgsz 640 --inference-device cuda`)와 CPU ReID를 함께 쓸 수
  있게 했습니다(ReID 준비 전에 CUDA를 초기화).
- `--disable-bystander-rules`: 비교 시험 전용. pi_unity_04/05의 옆 사람 규칙을 끕니다.

PC 스모크 실행(`pi_unity_05.mp4`, C1+줌+후보 추종 옵션 전부, 카메라는 가짜 주소):

| 검출기 / ReID | 프레임당 평균 | 검출 | ReID |
| --- | --- | --- | --- |
| CPU NCNN 416 / RepVGG ONNX | 55.0 ms | 22.1 ms | 21.5 ms |
| GPU yolo26n.pt 640 / RepVGG ONNX | 40.8 ms | 12.7 ms | 18.1 ms |
| (참고) Pi 실측 `pi_unity_05` | 39.5 ms | 15.7 ms | 5.1 ms |

### 후보 추종과 옆 사람 규칙

이 브랜치에서 후보 상자(`candidate_box`)는 옆 사람 규칙이 마커를 든 옆 사람을 2초 확인하는
동안에만 나옵니다. dev에서 후보 상자를 내던 경로(대상이 안 보이고 외형이 다르면 2초 확인)는
room_01 수정으로 없어졌기 때문입니다. 따라서 후보 추종은 규칙을 켰을 때만 동작하고, 그때
카메라를 확인 중인 옆 사람 쪽으로 돌립니다.

녹화 재생(개루프, 카메라가 실제로 움직이지 않음)에서 카메라가 옆 사람을 향한 시간:

| 녹화 | 규칙 켬, 추종 끔 | 규칙 켬, 추종 켬 | 규칙 끔(추종 무관) |
| --- | --- | --- | --- |
| `pi_unity_04` | 0초 | 0.14초 | 8.37초, 104번 등록 |
| `pi_unity_05` | 0초 | 0.05초 | 2.58초, 4·252·470번 등록 |

최종 선택은 Unity 시나리오(가로지르기, 나란히 걷기, 청중 질문)의 정답 기반 결과로 정합니다.

### Unity 시나리오 시험

- `tools/unity_series.py`: `--scenario`(여러 번 지정 가능), `--list-scenarios`, `--pipeline pc|pi`,
  `--pc-detector cpu|gpu`, `--pc-reid repvgg|yolo26n`, `--record-fps`. 시나리오 실행에서는
  `truth.json`, `actors.csv`도 `/run/{id}/`에서 받아 저장합니다(이 경로는 기존 frames.csv와 같은
  형식이라고 가정했습니다).
- `tools/scenario_metrics.py`: 정답 기반 지표와 시나리오별 성공 기준. 시나리오 이름의 낱말로
  종류를 정하고(`--criterion 이름=종류`로 지정 가능), 교대 제한 시간은 `--handover-limit`
  (기본 3초, 임시값)입니다.
- `truth.json`, `actors.csv`의 실제 예시를 아직 보지 못했습니다. 여러 형태의 키·열 이름을
  받도록 만들었고, 맞지 않으면 어떤 열이 있었는지 오류로 알려줍니다.

## 2026-10-26: 마커 소유자 기하 판정, Pi 속도 맞춤, 재획득 지표

### `--holder-geometry` (기본 꺼짐)

마커 중심이 여러 사람 상자에 들어가면 "작은 상자 우대"(면적 2배당 약 83점 감점) 대신
아래 감점을 씁니다. 상자가 하나뿐이면 기존 점수와 같습니다.

- 위치: 마커가 상자의 세로 25~50%(가슴), 가로 25~75% 밖으로 벗어난 비율 × 200점.
  가로는 full1 `walk_alongside_back`에서 추가했습니다. 뒤에 붙어 걷는 사람(C) 상자에서는
  마커가 오른쪽 끝(가로 0.91)에 있었고, 실제로 든 사람(A)에서는 0.25였습니다. 두 사람의
  세로 위치(0.30, 0.26)와 거리 비(0.91, 0.87)는 거의 같았습니다.
- 거리: (상자 높이 / 1.7 m) ÷ (마커 픽셀 / 0.093 m)의 |ln| × 60점. 화면 아래에서 잘린
  상자는 높이가 작게 나옵니다(이 판정은 화면 크기를 모릅니다).
- 보류: 서로 다른 사람(IoU 0.5 미만)인 1·2위의 점수 차가 `--holder-ambiguity-margin`(20) 미만이면
  아무에게도 마커를 주지 않습니다(`marker_holder_ambiguous`). 상자가 떨어지면 다시 확인합니다.

full1 녹화 재생(검출·ReID는 같고 카메라는 원래 실행대로 움직임) 결과, 추적기 기준:

| 시나리오 | 조건 | 기존 규칙 엉뚱한 사람(s) | 보완 규칙 |
| --- | --- | --- | --- |
| walk_alongside_back | n0 / n1 / n2 | 31.0 / 34.5 / 31.2, 첫 등록 C | 25.0 / 23.5 / 17.4, 첫 등록 A |
| walk_alongside | n0 / n1 / n2 | 0.13 / 1.52 / 0.07 | 0.13 / 0.35 / 0.07 |
| handover_adjacent, handover_then_exit | 모두 | 변화 없음 | 변화 없음(보류 0프레임) |

`walk_alongside_back`의 남은 오류는 등록이 아니라 A·C가 겹친 채 가려진 뒤의 ID 복구에서
생깁니다(마커가 안 보이는 때). 실제 카메라가 움직이는 결과는 full2의 n3에서 확인합니다.

### `--max-process-fps`

프레임을 읽은 뒤 남은 간격만큼 기다렸다 처리합니다. 따라서 처리율과 프레임→명령 지연이 둘 다
그 간격이 됩니다. Pi 실측(`pi_unity_05`) 중앙값은 21.9 fps이고, PC는 Unity 스트림 상한인
30.4 fps였습니다. `tools/unity_series.py --max-process-fps`가 모든 실행에 붙입니다.

### 재획득·교대 지연의 정의 변경

카메라 중앙에 그 사람이 있고 추적기도 그 사람을 따라갈 때부터 셉니다. full1 `exit_reenter`의
n1·n2 "0.0초"는 우연이었습니다. 카메라가 넓은 화각(42°)으로 멈춰 있던 화면 중앙으로 A가 걸어
들어왔고, 마커는 약 7 px라 읽히지 않았습니다. 추적기는 n1에서 9.84초 뒤 ReID로 A를 다시
잡았고, n2는 27초부터 끝까지 `identity_resolving`(identity_candidate_verified)에 머물렀습니다.
교대 시나리오의 값은 바뀌지 않았습니다.
