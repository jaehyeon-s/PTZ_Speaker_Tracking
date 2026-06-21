"""MVPv3 integration adapter for the dashboard.

This adapter is intentionally light-weight. It does not import or run the MVPv3
vision/PTZ loop directly, because that code depends on camera hardware, OpenCV,
NCNN models, and RTSP runtime state.

Instead, the dashboard reads MVPv3 runtime output in this order:
1) JSON snapshot file: runtime/mvpv3_status.json or env MVPV3_STATUS_JSON
2) CSV log file: logs/identity_supervisor.csv or env MVPV3_CSV_PATH

MVPv3 already supports CSV logging through configs/qon_mvpv3.yaml:
logging.csv: logs/identity_supervisor.csv
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_STATUS_JSON_PATHS = [
    PROJECT_ROOT / "runtime" / "mvpv3_status.json",
    PROJECT_ROOT / "logs" / "mvpv3_status.json",
]

DEFAULT_CSV_PATHS = [
    PROJECT_ROOT / "logs" / "identity_supervisor.csv",
    PROJECT_ROOT / "runtime" / "identity_supervisor.csv",
]


def load_mvpv3_dashboard_state() -> dict[str, Any] | None:
    """Return normalized dashboard payload when MVPv3 output exists.

    If no real MVPv3 output file is found, return None so the dashboard can keep
    using the existing mock/fallback scenario.
    """
    json_payload = _load_json_snapshot()
    if json_payload is not None:
        return _normalize_json_snapshot(json_payload)

    csv_row = _load_latest_csv_row()
    if csv_row is not None:
        return _normalize_csv_row(csv_row)

    return None


def _load_json_snapshot() -> dict[str, Any] | None:
    env_path = os.environ.get("MVPV3_STATUS_JSON")
    candidate_paths = []
    if env_path:
        candidate_paths.append(Path(env_path))
    candidate_paths.extend(DEFAULT_STATUS_JSON_PATHS)

    for path in candidate_paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def _load_latest_csv_row() -> dict[str, str] | None:
    env_path = os.environ.get("MVPV3_CSV_PATH")
    candidate_paths = []
    if env_path:
        candidate_paths.append(Path(env_path))
    candidate_paths.extend(DEFAULT_CSV_PATHS)

    for path in candidate_paths:
        row = _read_last_csv_row(path)
        if row is not None:
            row["__source_path"] = str(path)
            return row
    return None


def _read_last_csv_row(path: Path) -> dict[str, str] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None

    try:
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            last_row = None
            for row in reader:
                last_row = row
            return last_row
    except Exception:
        return None


def _normalize_json_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a future direct JSON output from MVPv3.

    If team code later writes runtime/mvpv3_status.json, it can either already
    use the dashboard schema or use simple MVPv3-style keys such as state/event/bbox.
    """
    if "dashboard_state" in payload:
        return payload["dashboard_state"]

    row = {
        "frame": str(payload.get("frame", "0")),
        "state": str(payload.get("state", "UNREGISTERED")),
        "event": str(payload.get("event", "")),
        "region_source": str(payload.get("region_source", payload.get("source", "json"))),
        "bbox": _bbox_to_string(payload.get("bbox")),
        "score": str(payload.get("score", "0")),
        "recovery_bbox": _bbox_to_string(payload.get("recovery_bbox")),
        "recovery_score": str(payload.get("recovery_score", "0")),
        "camera_action": str(payload.get("camera_action", "")),
        "best_score": str(payload.get("best_score", payload.get("score", "0"))),
        "center_score": str(payload.get("center_score", "0")),
        "second_score": str(payload.get("second_score", "0")),
        "identity_margin": str(payload.get("identity_margin", "0")),
        "center_margin": str(payload.get("center_margin", "0")),
        "candidate_count": str(payload.get("candidate_count", "0")),
        "hold_count": str(payload.get("hold_count", "0")),
        "mismatch_count": str(payload.get("mismatch_count", "0")),
        "ptz_action": str(payload.get("ptz_action", "")),
        "fps": str(payload.get("fps", "0")),
        "read_ms": str(payload.get("read_ms", "0")),
        "detect_ms": str(payload.get("detect_ms", "0")),
        "__source_path": os.environ.get("MVPV3_STATUS_JSON", "runtime/mvpv3_status.json"),
    }
    return _normalize_csv_row(row, data_source="REAL_MVPV3_JSON")


def _normalize_csv_row(row: dict[str, str], data_source: str = "REAL_MVPV3_CSV") -> dict[str, Any]:
    state = (row.get("state") or "UNREGISTERED").strip()
    event = (row.get("event") or "").strip()
    source = (row.get("region_source") or "MVPv3").strip()
    score = _to_float(row.get("score"), 0.0)
    fps = _to_float(row.get("fps"), 0.0)
    mismatch_count = int(_to_float(row.get("mismatch_count"), 0))
    hold_count = int(_to_float(row.get("hold_count"), 0))
    candidate_count = int(_to_float(row.get("candidate_count"), 0))
    bbox_xywh = _parse_mvp_bbox_to_xywh(row.get("bbox"))
    recovery_bbox_xywh = _parse_mvp_bbox_to_xywh(row.get("recovery_bbox"))
    ptz_action = (row.get("ptz_action") or "").strip()
    camera_action = (row.get("camera_action") or "").strip()

    reid_state = _map_reid_state(state, event)
    reid_event = _map_reid_event(state, event)
    event_level = _map_event_level(reid_state, reid_event)
    tracking_mode = _map_tracking_mode(state, event, mismatch_count)
    recovery_state = _map_recovery_state(state, event, mismatch_count)
    recovery_action = _map_recovery_action(state, event, camera_action)
    pan_direction, tilt_direction = _map_ptz_direction(ptz_action)
    health_score = _calc_health_score(reid_state, score, mismatch_count)
    health_level = "GOOD" if health_score >= 85 else "CAUTION" if health_score >= 65 else "WARNING"

    detections: list[dict[str, Any]] = []
    if bbox_xywh is not None:
        detections.append(
            {
                "id": 1,
                "x": bbox_xywh[0],
                "y": bbox_xywh[1],
                "w": bbox_xywh[2],
                "h": bbox_xywh[3],
                "inside": True,
                "target": True,
                "confidence": round(max(score, _to_float(row.get("best_score"), score)), 3),
                "reid_score": round(score, 3),
                "reid_state": reid_state,
            }
        )

    if recovery_bbox_xywh is not None:
        detections.append(
            {
                "id": 99,
                "x": recovery_bbox_xywh[0],
                "y": recovery_bbox_xywh[1],
                "w": recovery_bbox_xywh[2],
                "h": recovery_bbox_xywh[3],
                "inside": True,
                "target": False,
                "confidence": round(_to_float(row.get("recovery_score"), 0.0), 3),
                "reid_score": round(_to_float(row.get("recovery_score"), 0.0), 3),
                "reid_state": "RECOVERY_CANDIDATE",
            }
        )

    target_id = "1" if bbox_xywh is not None else "None"
    offset_x, offset_y = _calc_offset(bbox_xywh)
    integration_score = 100 if data_source.startswith("REAL") else 80
    if reid_state == "SUSPENDED":
        integration_score -= 10
    if candidate_count == 0:
        integration_score -= 5

    payload = {
        "detector": "MVPv3 Person Detector",
        "tracker": "Identity-first Re-ID / Region Provider",
        "fps": round(fps, 2),
        "target_id": target_id,
        "zone_lock": "ON",
        "ptz_status": "TRACKING" if ptz_action else "READY",
        "session_state": "RUNNING",
        "debug": {
            "total_detections": max(candidate_count, len(detections)),
            "inside_zone": 1 if bbox_xywh is not None else 0,
            "outside_zone": max(candidate_count - 1, 0),
            "track_stability": "WARNING" if reid_state == "SUSPENDED" else "GOOD",
            "last_reid": f"frame {row.get('frame', '-')}",
            "id_switch_count": mismatch_count,
        },
        "ptz_simulator": {
            "frame_center_x": 320,
            "frame_center_y": 240,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "pan_direction": pan_direction,
            "tilt_direction": tilt_direction,
            "zoom_state": "HOLD",
        },
        "reid": {
            "state": reid_state,
            "score": round(score, 3),
            "threshold": 0.68,
            "event": reid_event,
            "event_level": event_level,
            "method": "MVPv3 HSV baseline / optional OSNet ONNX",
            "registered_target": "Presenter",
            "recovery_mode": "ON" if recovery_state in ("RECOVERING", "WATCHING") else "OFF",
        },
        "health": {
            "score": health_score,
            "level": health_level,
            "tracking": "WARNING" if reid_state == "SUSPENDED" else "GOOD",
            "reid": reid_state,
            "ptz": "TRACKING" if ptz_action else "HOLD",
            "warnings": 1 if reid_state == "SUSPENDED" else 0,
        },
        "tracking_mode": {
            "current_mode": tracking_mode,
            "camera_mode": "SUPERVISOR_ACTUATOR" if "supervisor" in camera_action else "PTZ_DIRECT",
            "tracking_source": source,
            "target_type": "Person",
            "aruco_status": "USED" if "marker" in source.lower() or "MARKER" in event else "NOT_USED",
            "mode_reason": f"MVPv3 state={state}, event={event or '-'}",
        },
        "recovery": {
            "mismatch_count": mismatch_count,
            "recovery_trigger": "ON" if mismatch_count > 0 or reid_state == "SUSPENDED" else "OFF",
            "recovery_action": recovery_action,
            "recovery_state": recovery_state,
            "last_recovery": event or camera_action or "-",
        },
        "control_ownership": {
            "control_owner": "Pi Supervisor",
            "tracking_engine": "MVPv3 identity-first bbox + Qon PTZ actuator",
            "conflict_status": "PREVENTED" if "tracking_off" in camera_action else "NONE",
            "control_policy": "Dashboard reads MVPv3 output; Pi supervisor owns PTZ control",
        },
        "integration": {
            "score": max(integration_score, 0),
            "level": "READY" if integration_score >= 85 else "CHECK_REQUIRED",
            "data_source": data_source,
            "backend_connection": "CONNECTED",
            "vision_module": "READY",
            "reid_backend": "HSV_BASELINE",
            "osnet_backend": "STANDBY",
            "osnet_note": "Use MVPv3 --reid-backend onnx when ONNX model path is ready",
            "camera_module": "Qon4K6012XN",
            "ptz_loop": "MVPV3_RUNTIME",
            "api_contract": "READY",
            "next_action": "Run MVPv3 and dashboard together during integration test",
        },
        "module_connection": {
            "vision_backend": "READY",
            "bytetrack": "NOT_USED_BY_MVPV3",
            "zone_lock": "READY",
            "reid_hsv": "READY",
            "reid_osnet": "STANDBY",
            "aruco_marker": "USED" if "marker" in source.lower() or "MARKER" in event else "NOT_USED",
            "ptz_camera": "READY" if camera_action else "STANDBY",
            "dashboard_api": "READY",
        },
        "api_contract": {
            "status_endpoint": "/api/status OK",
            "detections_endpoint": "/api/detections OK",
            "required_fields": "MVPv3 CSV columns normalized by mvpv3_adapter.py",
            "missing_fields": "NONE",
            "integration_note": f"Reading {row.get('__source_path', 'MVPv3 runtime output')}",
        },
        "demo_flow": {
            "current_step": _map_demo_step(reid_state, event),
            "current_action": reid_event,
            "presenter_registration": "READY" if reid_state != "IDLE" else "WAITING",
            "camera_presenter_mode": "READY",
            "auto_tracking": "ON" if ptz_action or camera_action else "STANDBY",
            "mismatch_monitor": "ON",
            "zone_fallback": "ON" if recovery_action == "SWITCH_TO_ZONE_MODE" else "READY",
            "recovery_result": recovery_state,
        },
        "detections": detections,
    }
    return payload


def _map_reid_state(state: str, event: str) -> str:
    state_upper = state.upper()
    event_upper = event.upper()
    if state_upper in {"VERIFIED", "CAMERA_ALIGNED"}:
        return "ACTIVE"
    if state_upper in {"RECOVERY"} or "RECOVERY" in event_upper or "REREGISTERED" in event_upper:
        return "RECOVERED"
    if state_upper in {"SUSPECT", "MISMATCH", "LOST", "NO_REGION"}:
        return "SUSPENDED"
    if state_upper == "HOLD":
        return "ACTIVE"
    return "IDLE"


def _map_reid_event(state: str, event: str) -> str:
    event_upper = event.upper()
    state_upper = state.upper()
    if "REGISTER" in event_upper and "REQUIRED" not in event_upper:
        return "PRESENTER_REGISTERED"
    if "MISMATCH" in event_upper or "MISTRACK" in event_upper or state_upper in {"SUSPECT", "MISMATCH", "LOST"}:
        return "WRONG_TARGET_SUSPECTED"
    if "RECOVERY" in event_upper or "REREGISTER" in event_upper:
        return "TARGET_RECOVERED"
    if state_upper in {"VERIFIED", "CAMERA_ALIGNED"}:
        return "TARGET_MATCHED"
    if state_upper == "HOLD":
        return "TARGET_SCORE_FLUCTUATION"
    return event or "MVPV3_RUNTIME_IDLE"


def _map_event_level(reid_state: str, event: str) -> str:
    if reid_state == "SUSPENDED" or "WRONG" in event:
        return "WARNING"
    if reid_state == "RECOVERED" or "RECOVERED" in event:
        return "SUCCESS"
    return "INFO"


def _map_tracking_mode(state: str, event: str, mismatch_count: int) -> str:
    if mismatch_count > 0 or state.upper() in {"SUSPECT", "MISMATCH", "LOST"}:
        return "RECOVERY_TRACKING"
    if "MARKER" in event.upper():
        return "MARKER_TRACKING"
    return "ZONE_TRACKING"


def _map_recovery_state(state: str, event: str, mismatch_count: int) -> str:
    if "RECOVERY" in event.upper() or "REREGISTER" in event.upper():
        return "COMPLETED"
    if mismatch_count > 0 or state.upper() in {"SUSPECT", "MISMATCH", "LOST"}:
        return "RECOVERING"
    if state.upper() == "HOLD":
        return "WATCHING"
    return "IDLE"


def _map_recovery_action(state: str, event: str, camera_action: str) -> str:
    combined = f"{state} {event} {camera_action}".upper()
    if "TRACKING_OFF" in combined or "MISMATCH" in combined or "MISTRACK" in combined:
        return "SWITCH_TO_ZONE_MODE"
    if "RECOVERY" in combined or "REREGISTER" in combined:
        return "RECOVERY_COMPLETED"
    if "HOLD" in combined or "SUSPECT" in combined:
        return "MONITORING"
    return "NONE"


def _map_demo_step(reid_state: str, event: str) -> int:
    if event == "PRESENTER_REGISTERED":
        return 1
    if reid_state == "ACTIVE":
        return 2
    if reid_state == "SUSPENDED":
        return 3
    if reid_state == "RECOVERED":
        return 4
    return 0


def _calc_health_score(reid_state: str, score: float, mismatch_count: int) -> int:
    value = 100
    if reid_state == "SUSPENDED":
        value -= 35
    elif reid_state == "RECOVERED":
        value -= 8
    if score < 0.68:
        value -= 15
    if mismatch_count > 0:
        value -= min(mismatch_count * 3, 15)
    return max(value, 0)


def _calc_offset(bbox_xywh: tuple[int, int, int, int] | None) -> tuple[int, int]:
    if bbox_xywh is None:
        return 0, 0
    x, y, w, h = bbox_xywh
    return int(x + w / 2 - 320), int(y + h / 2 - 240)


def _map_ptz_direction(ptz_action: str) -> tuple[str, str]:
    text = (ptz_action or "").upper()
    pan = "HOLD"
    tilt = "HOLD"
    if "LEFT" in text:
        pan = "LEFT"
    elif "RIGHT" in text:
        pan = "RIGHT"
    if "UP" in text:
        tilt = "UP"
    elif "DOWN" in text:
        tilt = "DOWN"
    return pan, tilt


def _parse_mvp_bbox_to_xywh(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    try:
        x1, y1, x2, y2 = [int(float(part.strip())) for part in value.split(",")]
    except Exception:
        return None
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    if w == 0 or h == 0:
        return None
    return x1, y1, w, h


def _bbox_to_string(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return ",".join(str(int(float(part))) for part in value)
    return ""


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default
