#session.py, ptz_control.py 연결 예정
from fastapi import FastAPI
from src.api.routes.status import router as status_router
from src.api.routes.session import router as session_router
from src.api.routes.ptz_control import router as ptz_router

app = FastAPI(title="PTZ Speaker Tracking API")

app.include_router(status_router)
app.include_router(session_router)
app.include_router(ptz_router)


@app.get("/")
def home():
    return {"message": "PTZ Speaker Tracking API Server Running"}
