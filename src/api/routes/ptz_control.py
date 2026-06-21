from fastapi import APIRouter
from src.api.state import app_state, add_system_log

router = APIRouter()


def set_ptz_action(action: str):
    app_state["ptz_status"] = action
    add_system_log("INFO", f"PTZ action requested: {action}")
    return {"status": action}


@router.post("/api/ptz/up")
def ptz_up():
    return set_ptz_action("PTZ_UP")


@router.post("/api/ptz/down")
def ptz_down():
    return set_ptz_action("PTZ_DOWN")


@router.post("/api/ptz/left")
def ptz_left():
    return set_ptz_action("PTZ_LEFT")


@router.post("/api/ptz/right")
def ptz_right():
    return set_ptz_action("PTZ_RIGHT")


@router.post("/api/ptz/home")
def ptz_home():
    return set_ptz_action("PTZ_HOME")


@router.post("/api/ptz/zoom-in")
def zoom_in():
    return set_ptz_action("ZOOM_IN")


@router.post("/api/ptz/zoom-out")
def zoom_out():
    return set_ptz_action("ZOOM_OUT")


@router.post("/api/ptz/lock")
def lock_target():
    app_state["target_lock"] = "LOCKED"
    add_system_log("INFO", "Target lock requested")
    return {"status": "target locked"}


@router.post("/api/ptz/unlock")
def unlock_target():
    app_state["target_lock"] = "UNLOCKED"
    add_system_log("INFO", "Target unlock requested")
    return {"status": "target unlocked"}


@router.post("/api/view/bbox/toggle")
def toggle_bbox():
    app_state["bbox_enabled"] = not bool(app_state.get("bbox_enabled", True))
    add_system_log("INFO", f"B-Box overlay set to {app_state['bbox_enabled']}")
    return {"bbox_enabled": app_state["bbox_enabled"]}
