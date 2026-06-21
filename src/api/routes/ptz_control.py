from fastapi import APIRouter, HTTPException
from src.api.services.command_bus import command_bridge_status, write_command

router = APIRouter()


_DIRECTION_COMMANDS = {
    "up": "PTZ_UP",
    "down": "PTZ_DOWN",
    "left": "PTZ_LEFT",
    "right": "PTZ_RIGHT",
    "stop": "PTZ_STOP",
}

_ZOOM_COMMANDS = {
    "in": "ZOOM_IN",
    "out": "ZOOM_OUT",
    "stop": "ZOOM_STOP",
}


@router.post("/api/ptz/move/{direction}")
def move_camera(direction: str):
    direction = direction.lower()
    if direction not in _DIRECTION_COMMANDS:
        raise HTTPException(status_code=400, detail="direction must be one of up/down/left/right/stop")
    command = write_command(_DIRECTION_COMMANDS[direction], {"direction": direction, "speed": 5})
    return {"status": "queued", "command": command}


@router.post("/api/ptz/home")
def home_camera():
    command = write_command("PTZ_HOME")
    return {"status": "queued", "command": command}


@router.post("/api/ptz/zoom/{action}")
def zoom_camera(action: str):
    action = action.lower()
    if action not in _ZOOM_COMMANDS:
        raise HTTPException(status_code=400, detail="action must be one of in/out/stop")
    command = write_command(_ZOOM_COMMANDS[action], {"action": action, "speed": 4})
    return {"status": "queued", "command": command}


@router.post("/api/ptz/lock")
def lock_target():
    command = write_command("TARGET_LOCK")
    return {"status": "queued", "command": command}


@router.post("/api/ptz/unlock")
def unlock_target():
    command = write_command("TARGET_UNLOCK")
    return {"status": "queued", "command": command}


@router.post("/api/ptz/presenter-mode")
def presenter_mode():
    command = write_command("CAMERA_PRESENTER_MODE")
    return {"status": "queued", "command": command}


@router.post("/api/ptz/zone-mode")
def zone_mode():
    command = write_command("CAMERA_ZONE_MODE")
    return {"status": "queued", "command": command}


@router.get("/api/ptz/command/status")
def get_command_status():
    return command_bridge_status()
