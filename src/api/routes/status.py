from fastapi import APIRouter
from src.api.state import app_state
import time
import math

router = APIRouter()


def build_mock_detections():
    t = time.time()

    # 박스가 조금씩 좌우로 움직이게 해서 실시간처럼 보이게 구현
    offset = int(30 * math.sin(t))

    detections = [
        {
            "id": 1,
            "x": 90 + offset,
            "y": 70,
            "w": 90,
            "h": 170,
            "inside": True,
            "target": True
        },
        {
            "id": 2,
            "x": 240 - offset,
            "y": 95,
            "w": 95,
            "h": 160,
            "inside": False,
            "target": False
        }
    ]

    # target_id는 target=True인 객체 기준으로 반영
    target = next((d for d in detections if d["target"]), None)
    app_state["target_id"] = str(target["id"]) if target else "None"

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
    if app_state["zone_lock"] == "ON":
        app_state["zone_lock"] = "OFF"
    else:
        app_state["zone_lock"] = "ON"

    return {"status": app_state["zone_lock"]}
