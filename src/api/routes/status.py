from fastapi import APIRouter
from src.api.state import app_state, add_system_log
import time
import math

router = APIRouter()


def ensure_base_state():
    app_state.setdefault("detector", "YOLO26n (NCNN)")
    app_state.setdefault("tracker", "ByteTrack")
    app_state.setdefault("fps", 27.2)
    app_state.setdefault("target_id", "None")
    app_state.setdefault("zone_lock", "ON")
    app_state.setdefault("ptz_status", "READY")
    app_state.setdefault("session_state", "IDLE")
    app_state.setdefault("bbox_enabled", True)
    app_state.setdefault("system_logs", [])


def get_scenario():
    phase = int(time.time()) % 32

    if phase < 8:
        return {
            "tracking_mode": "ZONE_TRACKING",
            "camera_mode": "PRESENTER",
            "tracking_source": "Re-ID + Zone Lock",
            "target_type": "Person",
            "aruco_status": "NOT_USED",
            "mode_reason": "Professor detected inside lecture zone",
            "reid_state": "ACTIVE",
            "reid_score": 0.81,
            "reid_event": "TARGET_MATCHED",
            "reid_event_level": "INFO",
            "recovery_mode": "OFF",
            "mismatch_count": 0,
            "recovery_trigger": "OFF",
            "recovery_action": "NONE",
            "recovery_state": "IDLE",
            "last_recovery": "-",
        }
    elif phase < 16:
        return {
            "tracking_mode": "CLASS_TRACKING",
            "camera_mode": "PRESENTER",
            "tracking_source": "YOLO Class",
            "target_type": "Person/Class",
            "aruco_status": "NOT_USED",
            "mode_reason": "Class-based target tracking test",
            "reid_state": "ACTIVE",
            "reid_score": 0.74,
            "reid_event": "TARGET_SCORE_FLUCTUATION",
            "reid_event_level": "INFO",
            "recovery_mode": "OFF",
            "mismatch_count": 1,
            "recovery_trigger": "OFF",
            "recovery_action": "MONITORING",
            "recovery_state": "WATCHING",
            "last_recovery": "-",
        }
    elif phase < 24:
        return {
            "tracking_mode": "MARKER_TRACKING",
            "camera_mode": "ZONE",
            "tracking_source": "ArUco Marker",
            "target_type": "Marker",
            "aruco_status": "NOT_DETECTED",
            "mode_reason": "Marker recognition failed, fallback required",
            "reid_state": "SUSPENDED",
            "reid_score": 0.59,
            "reid_event": "WRONG_TARGET_SUSPECTED",
            "reid_event_level": "WARNING",
            "recovery_mode": "ON",
            "mismatch_count": 3,
            "recovery_trigger": "ON",
            "recovery_action": "SWITCH_TO_ZONE_MODE",
            "recovery_state": "RECOVERING",
            "last_recovery": "Zone mode fallback",
        }
    else:
        return {
            "tracking_mode": "RECOVERY_TRACKING",
            "camera_mode": "ZONE",
            "tracking_source": "Re-ID Recovery + Zone Lock",
            "target_type": "Person",
            "aruco_status": "NOT_USED",
            "mode_reason": "Target recovered after mismatch detection",
            "reid_state": "RECOVERED",
            "reid_score": 0.78,
            "reid_event": "TARGET_RECOVERED",
            "reid_event_level": "SUCCESS",
            "recovery_mode": "OFF",
            "mismatch_count": 0,
            "recovery_trigger": "OFF",
            "recovery_action": "RECOVERY_COMPLETED",
            "recovery_state": "COMPLETED",
            "last_recovery": "Target restored",
        }


def calc_health_score(scenario, track_stability):
    score = 100
    if scenario["reid_state"] == "SUSPENDED":
        score -= 35
    elif scenario["reid_state"] == "RECOVERED":
        score -= 8
    if scenario["reid_score"] < 0.70:
        score -= 20
    elif scenario["reid_score"] < 0.78:
        score -= 8
    if track_stability == "WARNING":
        score -= 15
    if app_state.get("session_state") == "IDLE":
        score -= 5
    return max(score, 0)


def build_mock_detections():
    ensure_base_state()
    t = time.time()
    offset = int(70 * math.sin(t))
    scenario = get_scenario()

    detections = [
        {
            "id": 1,
            "x": 210 + offset,
            "y": 95,
            "w": 90,
            "h": 170,
            "inside": True,
            "target": True,
            "confidence": 0.92,
            "reid_score": scenario["reid_score"],
            "reid_state": scenario["reid_state"],
        },
        {
            "id": 2,
            "x": 420 - offset,
            "y": 120,
            "w": 95,
            "h": 160,
            "inside": False,
            "target": False,
            "confidence": 0.81,
            "reid_score": 0.69,
            "reid_state": "CANDIDATE",
        },
    ]

    target = next((d for d in detections if d["target"]), None)
    app_state["target_id"] = str(target["id"]) if target else "None"

    inside_count = len([d for d in detections if d["inside"]])
    track_stability = "GOOD" if scenario["reid_state"] in ["ACTIVE", "RECOVERED"] else "WARNING"
    health_score = calc_health_score(scenario, track_stability)
    if health_score >= 85:
        health_level = "GOOD"
    elif health_score >= 65:
        health_level = "CAUTION"
    else:
        health_level = "WARNING"

    app_state["debug"] = {
        "total_detections": len(detections),
        "inside_zone": inside_count,
        "outside_zone": len(detections) - inside_count,
        "track_stability": track_stability,
        "last_reid": "2.1 sec ago",
        "id_switch_count": 0 if scenario["reid_state"] != "SUSPENDED" else 1,
    }

    app_state["reid"] = {
        "state": scenario["reid_state"],
        "score": scenario["reid_score"],
        "threshold": 0.70,
        "event": scenario["reid_event"],
        "event_level": scenario["reid_event_level"],
        "method": "HSV Histogram / OSNet-style Backend",
        "registered_target": "Professor",
        "recovery_mode": scenario["recovery_mode"],
    }

    app_state["health"] = {
        "score": health_score,
        "level": health_level,
        "tracking": track_stability,
        "reid": scenario["reid_state"],
        "ptz": app_state.get("ptz_status", "READY"),
        "warnings": 1 if scenario["reid_state"] == "SUSPENDED" else 0,
    }

    app_state["tracking_mode"] = {
        "current_mode": scenario["tracking_mode"],
        "camera_mode": scenario["camera_mode"],
        "tracking_source": scenario["tracking_source"],
        "target_type": scenario["target_type"],
        "aruco_status": scenario["aruco_status"],
        "mode_reason": scenario["mode_reason"],
    }

    app_state["recovery"] = {
        "mismatch_count": scenario["mismatch_count"],
        "recovery_trigger": scenario["recovery_trigger"],
        "recovery_action": scenario["recovery_action"],
        "recovery_state": scenario["recovery_state"],
        "last_recovery": scenario["last_recovery"],
    }

    return detections


@router.get("/api/status")
def get_status():
    build_mock_detections()
    return app_state


@router.get("/api/detections")
def get_detections():
    return build_mock_detections()


@router.get("/api/logs")
def get_logs():
    ensure_base_state()
    return {"logs": app_state.get("system_logs", [])}


@router.post("/api/zone/toggle")
def toggle_zone_lock():
    ensure_base_state()
    app_state["zone_lock"] = "OFF" if app_state["zone_lock"] == "ON" else "ON"
    add_system_log("INFO", f"Zone Lock toggled to {app_state['zone_lock']}")
    return {"status": app_state["zone_lock"]}
