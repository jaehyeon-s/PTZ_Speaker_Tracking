from fastapi import APIRouter
from src.api.state import app_state
import time
import math

router = APIRouter()


def build_mock_detections():
    t = time.time()
    offset = int(35 * math.sin(t))

    detections = [
        {
            "id": 1,
            "x": 90 + offset,
            "y": 70,
            "w": 90,
            "h": 170,
            "inside": True,
            "target": True,
            "confidence": 0.92
        },
        {
            "id": 2,
            "x": 260 - offset,
            "y": 105,
            "w": 95,
            "h": 160,
            "inside": False,
            "target": False,
            "confidence": 0.81
        },
        {
            "id": 4,
            "x": 430,
            "y": 95 + int(20 * math.cos(t)),
            "w": 85,
            "h": 150,
            "inside": True,
            "target": False,
            "confidence": 0.76
        }
    ]

    target = next((d for d in detections if d["target"]), None)
    app_state["target_id"] = str(target["id"]) if target else "None"

    inside_count = len([d for d in detections if d["inside"]])

    app_state["debug"] = {
        "total_detections": len(detections),
        "inside_zone": inside_count,
        "outside_zone": len(detections) - inside_count,
        "track_stability": "GOOD",
        "last_reid": "2.1 sec ago",
        "id_switch_count": 0
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
