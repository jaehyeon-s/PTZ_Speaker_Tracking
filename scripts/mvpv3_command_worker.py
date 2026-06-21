"""Run inside the MVPv3 project to execute dashboard command JSON.

Example:
  $env:QON_PASS="camera_password"
  python scripts/mvpv3_command_worker.py `
    --command-path "C:\\Users\\USER\\OneDrive\\Desktop\\PTZ_Speaker_Tracking\\runtime\\dashboard_command.json" `
    --ack-path "C:\\Users\\USER\\OneDrive\\Desktop\\PTZ_Speaker_Tracking\\runtime\\dashboard_command_ack.json" `
    --camera-url "http://192.168.0.10" `
    --username "admin" `
    --password-env QON_PASS
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running from MVPv3 root without installing as a package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera.qon_control import QonTrackingControl  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def execute_command(control: QonTrackingControl, command: str, payload: dict[str, Any], pulse_seconds: float) -> str:
    speed = int(payload.get("speed", 5))

    if command == "PTZ_UP":
        control.ptz_command("up", speed, speed)
        time.sleep(pulse_seconds)
        control.stop()
        return "ptz up pulse"
    if command == "PTZ_DOWN":
        control.ptz_command("down", speed, speed)
        time.sleep(pulse_seconds)
        control.stop()
        return "ptz down pulse"
    if command == "PTZ_LEFT":
        control.ptz_command("left", speed, speed)
        time.sleep(pulse_seconds)
        control.stop()
        return "ptz left pulse"
    if command == "PTZ_RIGHT":
        control.ptz_command("right", speed, speed)
        time.sleep(pulse_seconds)
        control.stop()
        return "ptz right pulse"
    if command == "PTZ_STOP":
        control.stop()
        return "ptz stop"
    if command == "PTZ_HOME":
        control.home()
        return "ptz home"

    if command == "ZOOM_IN":
        control.zoom_in(speed)
        time.sleep(pulse_seconds)
        control.zoom_stop(speed)
        return "zoom in pulse"
    if command == "ZOOM_OUT":
        control.zoom_out(speed)
        time.sleep(pulse_seconds)
        control.zoom_stop(speed)
        return "zoom out pulse"
    if command == "ZOOM_STOP":
        control.zoom_stop(speed)
        return "zoom stop"

    if command == "TARGET_LOCK":
        control.set_tracking_hint(True)
        return "target lock hint on"
    if command == "TARGET_UNLOCK":
        control.set_tracking_hint(False)
        return "target lock hint off"

    if command in ("SESSION_START", "CAMERA_PRESENTER_MODE"):
        control.set_presenter_mode()
        control.enable_auto_tracking()
        return "presenter mode + auto tracking on"
    if command in ("SESSION_END", "CAMERA_ZONE_MODE", "ZONE_LOCK_TOGGLE"):
        control.enable_zone_tracking()
        control.stop()
        return "zone mode + ptz stop"

    return f"ignored unknown command: {command}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute dashboard command JSON for MVPv3/Qon camera")
    parser.add_argument("--command-path", required=True)
    parser.add_argument("--ack-path", required=True)
    parser.add_argument("--camera-url", required=True, help="Example: http://192.168.0.10")
    parser.add_argument("--username", default=None)
    parser.add_argument("--password-env", default="QON_PASS")
    parser.add_argument("--auth-mode", choices=("none", "basic", "digest"), default="basic")
    parser.add_argument("--poll-seconds", type=float, default=0.2)
    parser.add_argument("--pulse-seconds", type=float, default=0.25)
    args = parser.parse_args()

    command_path = Path(args.command_path)
    ack_path = Path(args.ack_path)
    password = os.environ.get(args.password_env) if args.password_env else None

    control = QonTrackingControl(args.camera_url, args.username, password, auth_mode=args.auth_mode)
    last_command_id = None

    print(f"[command-worker] watching: {command_path}")
    print(f"[command-worker] ack path : {ack_path}")

    while True:
        item = read_json(command_path)
        if item is None:
            time.sleep(args.poll_seconds)
            continue

        command_id = item.get("command_id")
        if not command_id or command_id == last_command_id:
            time.sleep(args.poll_seconds)
            continue

        command = str(item.get("command", ""))
        payload = item.get("payload") or {}

        ack = {
            "command_id": command_id,
            "sequence": item.get("sequence"),
            "command": command,
            "received_at": utc_now(),
            "status": "OK",
            "message": "",
        }

        try:
            message = execute_command(control, command, payload, args.pulse_seconds)
            ack["message"] = message
            print(f"[command-worker] OK {command}: {message}")
        except Exception as exc:
            ack["status"] = "ERROR"
            ack["message"] = repr(exc)
            print(f"[command-worker] ERROR {command}: {exc!r}")

        write_json(ack_path, ack)
        last_command_id = command_id
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
