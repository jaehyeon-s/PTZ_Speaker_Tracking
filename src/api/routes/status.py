from fastapi import APIRouter
from src.api.state import app_state
import time
import math

router = APIRouter()


def get_reid_scenario():
    t = time.time()
    phase = int(t) % 24

    if phase < 6:
        return {"state": "ACTIVE", "score": 0.81, "event": "TARGET_MATCHED", "recovery_mode": "OFF", "event_level": "INFO"}
    elif phase < 12:
        return {"state": "ACTIVE", "score": 0.74, "event": "TARGET_SCORE_FLUCTUATION", "recovery_mode": "OFF", "event_level": "INFO"}
    elif phase < 18:
        return {"state": "SUSPENDED", "score": 0.59, "event": "WRONG_TARGET_SUSPECTED", "recovery_mode": "ON", "event_level": "WARNING"}
    else:
        return {"state": "RECOVERED", "score": 0.78, "event": "TARGET_RECOVERED", "recovery_mode": "OFF", "event_level": "SUCCESS"}


def calc_health_score(reid_state, reid_score, track_stability, pan_direction, tilt_direction):
    score = 100

    if reid_state == "SUSPENDED":
        score -= 30
    elif reid_state == "RECOVERED":
        score -= 8

    if reid_score < 0.70:
        score -= 20
    elif reid_score < 0.78:
        score -= 8

    if track_stability == "WARNING":
        score -= 15

    if pan_direction != "HOLD" or tilt_direction != "HOLD":
        score -= 5

    return max(score, 0)


def build_mock_detections():
    t = time.time()
    offset = int(70 * math.sin(t))
    reid = get_reid_scenario()

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
            "reid_score": reid["score"],
            "reid_state": reid["state"],
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

    frame_w = 640
    frame_h = 480
    center_x = frame_w // 2
    center_y = frame_h // 2

    if target:
        target_center_x = target["x"] + target["w"] // 2
        target_center_y = target["y"] + target["h"] // 2

        offset_x = target_center_x - center_x
        offset_y = target_center_y - center_y

        pan_direction = "HOLD"
        tilt_direction = "HOLD"

        if offset_x > 40:
            pan_direction = "RIGHT"
        elif offset_x < -40:
            pan_direction = "LEFT"

        if offset_y > 40:
            tilt_direction = "DOWN"
        elif offset_y < -40:
            tilt_direction = "UP"

        app_state["target_id"] = str(target["id"])
    else:
        offset_x = 0
        offset_y = 0
        pan_direction = "HOLD"
        tilt_direction = "HOLD"
        app_state["target_id"] = "None"

    inside_count = len([d for d in detections if d["inside"]])
    track_stability = "GOOD" if reid["state"] in ["ACTIVE", "RECOVERED"] else "WARNING"
    health_score = calc_health_score(
        reid["state"],
        reid["score"],
        track_stability,
        pan_direction,
        tilt_direction
    )

    if health_score >= 85:
        health_level = "GOOD"
    elif health_score >= 65:
        health_level = "CAUTION"
    else:
        health_level = "WARNING"

    warning_count = 1 if reid["state"] == "SUSPENDED" else 0

    app_state["debug"] = {
        "total_detections": len(detections),
        "inside_zone": inside_count,
        "outside_zone": len(detections) - inside_count,
        "track_stability": track_stability,
        "last_reid": "2.1 sec ago",
        "id_switch_count": 0 if reid["state"] != "SUSPENDED" else 1,
    }

    app_state["ptz_simulator"] = {
        "frame_center_x": center_x,
        "frame_center_y": center_y,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "pan_direction": pan_direction,
        "tilt_direction": tilt_direction,
        "zoom_state": "HOLD",
    }

    app_state["reid"] = {
        "state": reid["state"],
        "score": reid["score"],
        "threshold": 0.70,
        "event": reid["event"],
        "event_level": reid["event_level"],
        "method": "Color Histogram",
        "registered_target": "Professor",
        "recovery_mode": reid["recovery_mode"],
    }

    app_state["health"] = {
        "score": health_score,
        "level": health_level,
        "tracking": track_stability,
        "reid": reid["state"],
        "ptz": "TRACKING" if pan_direction != "HOLD" or tilt_direction != "HOLD" else "HOLD",
        "warnings": warning_count,
    }

    return detections


@router.get("/api/status")
def get_status():
    build_mock_detections()
    return app_state


@router.get("/api/detections")
def get_detections():
    return build_mock_detections()


@router.post("/api/zone/toggle")
def toggle_zone_lock():
    app_state["zone_lock"] = "OFF" if app_state["zone_lock"] == "ON" else "ON"
    return {"status": app_state["zone_lock"]}
