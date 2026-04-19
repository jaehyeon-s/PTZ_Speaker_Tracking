from fastapi import APIRouter
from src.api.state import app_state

router = APIRouter()


@router.post("/api/session/start")
def start_session():
    app_state["session_state"] = "RUNNING"
    return {"status": "session started"}


@router.post("/api/session/end")
def end_session():
    app_state["session_state"] = "IDLE"
    return {"status": "session ended"}
