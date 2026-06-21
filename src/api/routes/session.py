from fastapi import APIRouter
from src.api.state import app_state
from src.api.services.command_bus import write_command

router = APIRouter()


@router.post("/api/session/start")
def start_session():
    app_state["session_state"] = "RUNNING"
    command = write_command("SESSION_START")
    return {"status": "session started", "command": command}


@router.post("/api/session/end")
def end_session():
    app_state["session_state"] = "IDLE"
    command = write_command("SESSION_END")
    return {"status": "session ended", "command": command}
