import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.routes.status import router as status_router
from src.api.routes.session import router as session_router
from src.api.routes.ptz_control import router as ptz_router

app = FastAPI(title="PTZ Speaker Tracking API")

app.mount("/static", StaticFiles(directory="src/api/dashboard/static"), name="static")
templates = Jinja2Templates(directory="src/api/dashboard/templates")

app.include_router(status_router)
app.include_router(session_router)
app.include_router(ptz_router)


def is_logged_in(request: Request) -> bool:
    return request.cookies.get("dashboard_auth") == "ok"


def require_login(request: Request):
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=303)
    return None


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if is_logged_in(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": None},
    )


@app.post("/login", response_class=HTMLResponse)
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    dashboard_user = os.getenv("DASHBOARD_USER", "admin")
    dashboard_password = os.getenv("DASHBOARD_PASSWORD", "1234")

    if username == dashboard_user and password == dashboard_password:
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(
            key="dashboard_auth",
            value="ok",
            httponly=True,
            samesite="lax",
        )
        return response

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": "아이디 또는 비밀번호가 올바르지 않습니다."},
        status_code=401,
    )


@app.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("dashboard_auth")
    return response


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    redirect = require_login(request)
    if redirect:
        return redirect
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/tracking", response_class=HTMLResponse)
def tracking_detail(request: Request):
    redirect = require_login(request)
    if redirect:
        return redirect
    return templates.TemplateResponse(request=request, name="tracking.html", context={})


@app.get("/logs", response_class=HTMLResponse)
def log_page(request: Request):
    redirect = require_login(request)
    if redirect:
        return redirect
    return templates.TemplateResponse(request=request, name="logs.html", context={})
