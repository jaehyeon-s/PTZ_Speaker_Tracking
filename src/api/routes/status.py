from fastapi import APIRouter

router = APIRouter()

# 임시 더미 상태 데이터
mock_status = {
    "camera_connected": True,
    "rtsp_active": True,
    "session_state": "IDLE",
    "detector": "YOLO26n (NCNN)",
    "tracker": "ByteTrack",
    "fps": 27.2,
    "target_id": "None",
    "zone_lock": "OFF",
    "ptz_status": "Ready"
}


@router.get("/api/status")
def get_status():
    return mock_status


@router.post("/api/session/start")
def start_session():
    mock_status["session_state"] = "RUNNING"
    return {"status": "session started"}


@router.post("/api/session/end")
def end_session():
    mock_status["session_state"] = "IDLE"
    return {"status": "session ended"}
