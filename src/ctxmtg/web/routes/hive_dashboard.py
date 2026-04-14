# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Hive Dashboard Routes
======================

Dashboard, link management, and pull actions for the hive web UI.

Routes:
    GET  /           -- Dashboard with stats and links table
    POST /api/links  -- Create a new link
    DELETE /api/links/{id} -- Remove a link
    POST /api/links/{id}/pull -- Pull from a link's outbox
    POST /api/pull-all -- Pull from all links
    GET  /api/hive-stats -- Refreshable stats fragment
"""

from __future__ import annotations

from fastapi import APIRouter, Cookie, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ctxmtg.web import hive_auth

import structlog

logger = structlog.get_logger("ctxmtg.web.routes.hive_dashboard")

router = APIRouter(tags=["hive"])

templates: Jinja2Templates | None = None
_hive_db = None


def set_templates(t: Jinja2Templates) -> None:
    global templates
    templates = t


def set_hive_db(db) -> None:
    global _hive_db
    _hive_db = db


def get_hive_db():
    if _hive_db is None:
        raise HTTPException(status_code=503, detail="Hive DB not initialised")
    return _hive_db


def require_auth(request: Request, session: str | None = Cookie(default=None)):
    if hive_auth.validate_session(session):
        return True
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"},
        )
    raise HTTPException(status_code=401, detail="Not authenticated")


# =================================================================
# Auth routes (login, setup, logout)
# =================================================================

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if not hive_auth.is_password_set():
        return templates.TemplateResponse(request, "hive_setup.html")
    return templates.TemplateResponse(request, "hive_login.html", {"error": None})


@router.post("/login")
async def login_submit(request: Request, password: str = Form(...)):
    if not hive_auth.verify_password(password):
        return templates.TemplateResponse(
            request, "hive_login.html", {"error": "Invalid password"},
            status_code=401,
        )
    token = hive_auth.create_session()
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        "session", token, httponly=True, samesite="strict",
        max_age=hive_auth.SESSION_MAX_AGE,
    )
    return response


@router.post("/setup")
async def setup_submit(
    request: Request,
    password: str = Form(...),
    confirm: str = Form(...),
):
    if hive_auth.is_password_set():
        return RedirectResponse(url="/login", status_code=303)
    if password != confirm:
        return templates.TemplateResponse(
            request, "hive_setup.html", {"error": "Passwords do not match"},
            status_code=400,
        )
    if len(password) < 8:
        return templates.TemplateResponse(
            request, "hive_setup.html",
            {"error": "Password must be at least 8 characters"},
            status_code=400,
        )
    hive_auth.set_password(password)
    token = hive_auth.create_session()
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        "session", token, httponly=True, samesite="strict",
        max_age=hive_auth.SESSION_MAX_AGE,
    )
    return response


@router.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("session")
    hive_auth.destroy_session(token)
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session")
    return response


# =================================================================
# Dashboard
# =================================================================

@router.get("/", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def dashboard(request: Request):
    hive_db = get_hive_db()
    stats = await _get_stats(hive_db)
    links = await hive_db.get_links()
    return templates.TemplateResponse(
        request, "hive_dashboard.html",
        {"stats": stats, "links": links},
    )


@router.get("/api/hive-stats", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def refresh_stats(request: Request):
    hive_db = get_hive_db()
    stats = await _get_stats(hive_db)
    return templates.TemplateResponse(
        request, "fragments/hive_stats.html", {"stats": stats},
    )


# =================================================================
# Link CRUD
# =================================================================

@router.post("/api/links", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def create_link(
    request: Request,
    local_name: str = Form(...),
    outbox_path: str = Form(...),
    notes: str = Form(default=""),
):
    hive_db = get_hive_db()
    try:
        await hive_db.add_link(local_name, outbox_path, notes)
    except Exception as exc:
        logger.error("link_create_failed", error=str(exc))
        return HTMLResponse(
            f'<div class="alert alert-error">Failed to create link: {exc}</div>',
            status_code=400,
        )

    # Return updated links table
    links = await hive_db.get_links()
    return templates.TemplateResponse(
        request, "fragments/links_table.html", {"links": links},
    )


@router.delete("/api/links/{link_id}", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def remove_link(link_id: str):
    hive_db = get_hive_db()
    deleted = await hive_db.remove_link(link_id)
    if not deleted:
        return HTMLResponse("", status_code=404)
    return HTMLResponse("")


@router.post("/api/links/{link_id}/remove", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def remove_link_post(request: Request, link_id: str):
    """POST-based remove (htmx-friendly). Returns refreshed links table."""
    hive_db = get_hive_db()
    await hive_db.remove_link(link_id)
    links = await hive_db.get_links()
    return templates.TemplateResponse(
        request, "fragments/links_table.html", {"links": links},
    )


# =================================================================
# Pull actions
# =================================================================

@router.post("/api/links/{link_id}/pull", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def pull_from_link(request: Request, link_id: str):
    hive_db = get_hive_db()
    link = await hive_db.get_link(link_id)
    if not link:
        return HTMLResponse(
            '<div class="alert alert-error">Link not found</div>',
            status_code=404,
        )

    from ctxmtg.sync.outbox_reader import OutboxReader

    reader = OutboxReader(hive_db)
    result = await reader.pull_from_link(link)

    # Render pull result
    pull_html = templates.TemplateResponse(
        request, "fragments/pull_result.html",
        {"result": result, "link_name": link["local_name"]},
    ).body.decode()

    # Render refreshed links table with out-of-band swap
    links = await hive_db.get_links()
    links_html = templates.TemplateResponse(
        request, "fragments/links_table.html", {"links": links},
    ).body.decode()

    # Combine: main target gets pull result, OOB updates links table
    combined = (
        pull_html
        + f'<div id="links-table" hx-swap-oob="innerHTML">{links_html}</div>'
    )
    return HTMLResponse(combined)


@router.post("/api/pull-all", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def pull_all(request: Request):
    hive_db = get_hive_db()
    links = await hive_db.get_links()

    from ctxmtg.sync.outbox_reader import OutboxReader

    reader = OutboxReader(hive_db)
    total = {"batches": 0, "profiles": 0, "insights": 0, "errors": []}

    for link in links:
        result = await reader.pull_from_link(link)
        total["batches"] += result["batches"]
        total["profiles"] += result["profiles"]
        total["insights"] += result["insights"]
        total["errors"].extend(result["errors"])

    pull_html = templates.TemplateResponse(
        request, "fragments/pull_result.html",
        {"result": total, "link_name": "all links"},
    ).body.decode()

    refreshed_links = await hive_db.get_links()
    links_html = templates.TemplateResponse(
        request, "fragments/links_table.html", {"links": refreshed_links},
    ).body.decode()

    combined = (
        pull_html
        + f'<div id="links-table" hx-swap-oob="innerHTML">{links_html}</div>'
    )
    return HTMLResponse(combined)


# =================================================================
# Helpers
# =================================================================

async def _get_stats(hive_db) -> dict:
    counts = await hive_db.get_record_counts()
    links = await hive_db.get_links()
    return {
        "profiles": counts.get("hive_entity_profiles", 0),
        "insights": counts.get("hive_insights", 0),
        "native_insights": counts.get("hive_native_insights", 0),
        "linked_locals": len(links),
    }
