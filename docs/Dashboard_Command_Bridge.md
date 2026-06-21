# Dashboard Command Bridge

This integration separates two directions:

- MVPv3 -> Dashboard: status/detection data via `logs/identity_supervisor.csv`
- Dashboard -> MVPv3: control commands via `runtime/dashboard_command.json`

## Web dashboard

Run the dashboard with the command path:

```powershell
$env:DASHBOARD_COMMAND_PATH="C:\Users\USER\OneDrive\Desktop\PTZ_Speaker_Tracking\runtime\dashboard_command.json"
$env:DASHBOARD_COMMAND_ACK_PATH="C:\Users\USER\OneDrive\Desktop\PTZ_Speaker_Tracking\runtime\dashboard_command_ack.json"
python -m uvicorn src.api.app:app --reload
```

The Camera Control buttons now call FastAPI endpoints and write command JSON.

## MVPv3 worker

Copy `scripts/mvpv3_command_worker.py` into the MVPv3 project root, then run:

```powershell
$env:QON_PASS="camera_password"
python scripts/mvpv3_command_worker.py `
  --command-path "C:\Users\USER\OneDrive\Desktop\PTZ_Speaker_Tracking\runtime\dashboard_command.json" `
  --ack-path "C:\Users\USER\OneDrive\Desktop\PTZ_Speaker_Tracking\runtime\dashboard_command_ack.json" `
  --camera-url "http://camera-ip" `
  --username "admin" `
  --password-env QON_PASS
```

## Command mapping

- PTZ_UP/DOWN/LEFT/RIGHT: Qon PTZ pulse then stop
- PTZ_HOME: Qon home
- ZOOM_IN/OUT: zoom pulse then zoom stop
- TARGET_LOCK/UNLOCK: tracking hint on/off
- SESSION_START: presenter mode + auto tracking on
- SESSION_END / ZONE_LOCK_TOGGLE: zone mode + PTZ stop

