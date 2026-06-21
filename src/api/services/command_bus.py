"""Dashboard -> MVPv3 command bridge.

The dashboard writes the latest camera/control command to a JSON file.
A separate MVPv3 worker process can read the file and execute commands
against the real Qon camera. This keeps the dashboard alive even if the
camera program fails.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_COMMAND_PATH = Path("runtime/dashboard_command.json")
DEFAULT_HISTORY_PATH = Path("runtime/dashboard_command_history.jsonl")
DEFAULT_ACK_PATH = Path("runtime/dashboard_command_ack.json")


def _path_from_env(env_name: str, default_path: Path) -> Path:
    return Path(os.environ.get(env_name, str(default_path)))


def command_path() -> Path:
    return _path_from_env("DASHBOARD_COMMAND_PATH", DEFAULT_COMMAND_PATH)


def history_path() -> Path:
    return _path_from_env("DASHBOARD_COMMAND_HISTORY_PATH", DEFAULT_HISTORY_PATH)


def ack_path() -> Path:
    return _path_from_env("DASHBOARD_COMMAND_ACK_PATH", DEFAULT_ACK_PATH)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent), suffix=".tmp") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
        tmp_name = fp.name
    os.replace(tmp_name, path)


def read_last_command() -> dict[str, Any] | None:
    path = command_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_last_ack() -> dict[str, Any] | None:
    path = ack_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_command(command: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    now = _utc_now()
    last = read_last_command() or {}
    sequence = int(last.get("sequence", 0)) + 1

    command_payload = {
        "command_id": str(uuid4()),
        "sequence": sequence,
        "command": command,
        "payload": payload or {},
        "source": "dashboard",
        "created_at": now,
        "expires_at": time.time() + 5.0,
    }

    _atomic_write_json(command_path(), command_payload)

    history_path().parent.mkdir(parents=True, exist_ok=True)
    with history_path().open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(command_payload, ensure_ascii=False) + "\n")

    return command_payload


def command_bridge_status() -> dict[str, Any]:
    last_command = read_last_command()
    last_ack = read_last_ack()
    return {
        "command_path": str(command_path()),
        "ack_path": str(ack_path()),
        "last_command": last_command,
        "last_ack": last_ack,
        "bridge_mode": "COMMAND_JSON",
    }
