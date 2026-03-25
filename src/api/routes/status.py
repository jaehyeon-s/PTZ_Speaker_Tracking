from fastapi import APIRouter

router = APIRouter()

@router.get("/api/status")
def get_status():
    return {
        "camera_connected": True,
        "session_state": "IDLE",
        "target_id": None,
        "fps": 0
    }
