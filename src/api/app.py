# session.py, ptz_control.py 연결 예정
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.routes.status import router as status_router
from src.api.routes.session import router as session_router
from src.api.routes.ptz_control import router as ptz_router

app = FastAPI(title="PTZ Speaker Tracking API")

# 정적 파일(css, js)
app.mount("/static", StaticFiles(directory="src/api/dashboard/static"), name="static")

# 템플릿(html)
templates = Jinja2Templates(directory="src/api/dashboard/templates")

app.include_router(status_router)
app.include_router(session_router)
app.include_router(ptz_router)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
