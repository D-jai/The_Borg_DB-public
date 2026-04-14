# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Authentication Routes
=====================

Login page, password setup (first run), and logout. All HTML is
rendered via Jinja2 templates with htmx for progressive enhancement.

Depends on:
    - ctxmtg.web.auth (password + session management)
    - fastapi (router, request, response)
    - Jinja2 templates (login.html, setup.html)

Used by:
    - ctxmtg.web.app (included in the main FastAPI app)
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ctxmtg.web import auth

router = APIRouter(tags=["auth"])

# Templates reference -- set by app.py at startup.
templates: Jinja2Templates | None = None


def set_templates(t: Jinja2Templates) -> None:
    global templates
    templates = t


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Render the login page (or setup page if no password exists)."""
    if not auth.is_password_set():
        return templates.TemplateResponse(request, "setup.html")
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
async def login_submit(request: Request, password: str = Form(...)):
    """Validate credentials and set a session cookie."""
    if not auth.verify_password(password):
        return templates.TemplateResponse(
            request, "login.html", {"error": "Invalid password"},
            status_code=401,
        )
    token = auth.create_session()
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        "session", token, httponly=True, samesite="strict", max_age=auth.SESSION_MAX_AGE
    )
    return response


@router.post("/setup")
async def setup_submit(
    request: Request,
    password: str = Form(...),
    confirm: str = Form(...),
):
    """Set the admin password on first run."""
    if auth.is_password_set():
        return RedirectResponse(url="/login", status_code=303)

    if password != confirm:
        return templates.TemplateResponse(
            request, "setup.html", {"error": "Passwords do not match"},
            status_code=400,
        )
    if len(password) < 8:
        return templates.TemplateResponse(
            request, "setup.html", {"error": "Password must be at least 8 characters"},
            status_code=400,
        )

    auth.set_password(password)
    token = auth.create_session()
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        "session", token, httponly=True, samesite="strict", max_age=auth.SESSION_MAX_AGE
    )
    return response


@router.get("/logout")
async def logout(request: Request):
    """Destroy the session and redirect to login."""
    token = request.cookies.get("session")
    auth.destroy_session(token)
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session")
    return response
