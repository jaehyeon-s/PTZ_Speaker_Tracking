from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


app_state = {
    "camera_connected": True,
    "rtsp_active": True,
    "session_state": "IDLE",
    "detector": "YOLO26n (NCNN)",
    "tracker": "ByteTrack",
    "fps": 27.2,
    "target_id": "None",
    "zone_lock": "ON",
    "ptz_status": "READY",
    "bbox_enabled": True,
    "last_system_action": "NONE",
    "last_system_action_at": "-",
    "system_logs": [
        {"time": now_iso(), "level": "INFO", "message": "Dashboard initialized"},
        {"time": now_iso(), "level": "INFO", "message": "YOLO26n NCNN ready"},
        {"time": now_iso(), "level": "INFO", "message": "ByteTrack + Zone Lock ready"},
    ],
}


def add_system_log(level: str, message: str):
    logs = app_state.setdefault("system_logs", [])
    logs.append({"time": now_iso(), "level": level, "message": message})
    if len(logs) > 120:
        del logs[:-120]
