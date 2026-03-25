from fastapi import APIRouter

router = APIRouter()

@router.post("/api/ptz/lock")
def lock_target():
    return {"status": "target locked"}

@router.post("/api/ptz/unlock")
def unlock_target():
    return {"status": "target unlocked"}
