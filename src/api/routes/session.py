from fastapi import APIRouter
from src.api.state import app_state, add_system_log, now_iso

router = APIRouter()


@router.post("/api/session/start")
def start_session():
    app_state["session_state"] = "RUNNING"
    app_state["ptz_status"] = "TRACKING"
    add_system_log("INFO", "Session started")
    return {"status": "session started", "session_state": app_state["session_state"]}


@router.post("/api/session/end")
def end_session():
    app_state["session_state"] = "IDLE"
    app_state["ptz_status"] = "READY"
    app_state["target_id"] = "None"
    add_system_log("INFO", "Session ended")
    return {"status": "session ended", "session_state": app_state["session_state"]}


@router.post("/api/system/reset")
def system_reset():
    app_state["session_state"] = "IDLE"
    app_state["target_id"] = "None"
    app_state["zone_lock"] = "ON"
    app_state["ptz_status"] = "READY"
    app_state["bbox_enabled"] = True
    app_state["last_system_action"] = "SYSTEM_RESET"
    app_state["last_system_action_at"] = now_iso()
    add_system_log("WARNING", "System reset requested from dashboard")
    return {"status": "system reset requested"}


@router.post("/api/system/reboot")
def system_reboot():
    # 실제 OS 재부팅은 안전상 실행하지 않고, 대시보드/연동 모듈이 처리할 명령 상태만 기록한다.
    app_state["last_system_action"] = "REBOOT_REQUESTED"
    app_state["last_system_action_at"] = now_iso()
    add_system_log("WARNING", "Reboot requested from dashboard")
    return {"status": "reboot requested"}
