from fastapi import APIRouter
from src.api.state import app_state
import time
import math

router = APIRouter()


def build_mock_detections():
    t = time.time()
    offset = int(70 * math.sin(t))

    # 시간에 따라 Re-ID 상태가 바뀌는 테스트용 시나리오
    phase = int(t) % 18

    if phase < 7:
        reid_state = "ACTIVE"
        reid_score = 0.81
        reid_event = "TARGET_MATCHED"
        recovery_mode = "OFF"
    elif phase < 13:
        reid_state = "ACTIVE"
        reid_score = 0.74
        reid_event = "TARGET_SCORE_FLUCTUATION"
        recovery_mode = "OFF"
    else:
        reid_state = "SUSPENDED"
        reid_score = 0.59
        reid_event = "WRONG_TARGET_SUSPECTED"
        recovery_mode = "ON"

    threshold = 0.70

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
            "reid_score": reid_score,
            "reid_state": reid_state
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
            "reid_state": "CANDIDATE"
        }
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

    app_state["debug"] = {
        "total_detections": len(detections),
        "inside_zone": inside_count,
        "outside_zone": len(detections) - inside_count,
        "track_stability": "GOOD" if reid_state == "ACTIVE" else "WARNING",
        "last_reid": "2.1 sec ago",
        "id_switch_count": 0
    }

    app_state["ptz_simulator"] = {
        "frame_center_x": center_x,
        "frame_center_y": center_y,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "pan_direction": pan_direction,
        "tilt_direction": tilt_direction,
        "zoom_state": "HOLD"
    }

    app_state["reid"] = {
        "state": reid_state,
        "score": reid_score,
        "threshold": threshold,
        "event": reid_event,
        "method": "Color Histogram",
        "registered_target": "Professor",
        "recovery_mode": recovery_mode
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
