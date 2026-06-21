# MVPv3 Dashboard Integration

이 브랜치는 `feature/web-ui` 대시보드에 MVPv3 실행 결과를 연결하기 위한 1차 연동 구조입니다.

## 연동 방식

대시보드가 MVPv3 코드를 직접 실행하지 않고, MVPv3가 생성하는 런타임 출력 파일을 읽습니다.

우선순위:

1. `runtime/mvpv3_status.json`
2. `logs/mvpv3_status.json`
3. `logs/identity_supervisor.csv`
4. `runtime/identity_supervisor.csv`

환경변수로 직접 지정할 수도 있습니다.

```powershell
$env:MVPV3_CSV_PATH="C:\\path\\to\\PTZ_Speaker_Tracking-MVPv3\\logs\\identity_supervisor.csv"
python -m uvicorn src.api.app:app --reload
```

## MVPv3 실행 예시

MVPv3 프로젝트 폴더에서 실행합니다.

```powershell
$env:QON_PASS="카메라비밀번호"
python -m src.app.verify_demo --config configs/qon_mvpv3.yaml --no-window
```

MVPv3의 `configs/qon_mvpv3.yaml`에 이미 아래 설정이 있습니다.

```yaml
logging:
  csv: logs/identity_supervisor.csv
```

따라서 대시보드 프로젝트가 같은 루트에 있지 않다면 `MVPV3_CSV_PATH`만 실제 CSV 경로로 맞추면 됩니다.

## 대시보드 실행 예시

대시보드 프로젝트 폴더에서 실행합니다.

```powershell
$env:MVPV3_CSV_PATH="C:\\Users\\USER\\OneDrive\\Desktop\\PTZ_Speaker_Tracking-MVPv3\\logs\\identity_supervisor.csv"
python -m uvicorn src.api.app:app --reload
```

접속:

```text
http://127.0.0.1:8000
```

## 동작 확인

MVPv3 CSV가 감지되면 대시보드의 Integration Readiness에서 다음 값이 표시됩니다.

```text
Data Source: REAL_MVPV3_CSV
Backend Connection: CONNECTED
```

CSV가 없으면 기존 Mock 화면으로 자동 fallback됩니다.
