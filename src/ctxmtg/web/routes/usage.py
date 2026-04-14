# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
LLM Usage Stats Routes
=======================

Dashboard page showing LLM API usage: per-day totals, per-model
breakdown, and per-stage (role) distribution.  All data comes from
the llm_usage SQLite table populated by the UsageTracker.

Depends on:
    - ctxmtg.llm.usage_tracker (UsageTracker)
    - ctxmtg.web.deps (auth, store access)

Used by:
    - ctxmtg.web.app (included as a router)
"""

from __future__ import annotations

from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ctxmtg.web.deps import get_sql_store, require_auth

logger = structlog.get_logger("ctxmtg.web.routes.usage")

router = APIRouter(tags=["usage"])
templates: Jinja2Templates | None = None


def set_templates(t: Jinja2Templates) -> None:
    global templates
    templates = t


def _db_path_from_store(sql_store) -> str:
    """Extract the database file path from the SQLiteStore."""
    return sql_store._db_path


@router.get("/usage", response_class=HTMLResponse)
async def usage_page(
    request: Request,
    _auth=Depends(require_auth),
    sql_store=Depends(get_sql_store),
):
    """Render the LLM usage stats page."""
    from ctxmtg.llm.usage_tracker import UsageTracker

    tracker = UsageTracker(db_path=_db_path_from_store(sql_store))

    daily = await tracker.get_daily_stats(days=30)
    models = await tracker.get_model_stats(days=30)
    stages = await tracker.get_stage_stats(days=30)
    daily_by_model = await tracker.get_daily_by_model(days=30)

    totals = {
        "calls": sum(d.get("calls", 0) for d in daily),
        "prompt_tokens": sum(d.get("prompt_tokens", 0) for d in daily),
        "completion_tokens": sum(d.get("completion_tokens", 0) for d in daily),
        "total_tokens": sum(d.get("total_tokens", 0) for d in daily),
        "errors": sum(d.get("errors", 0) for d in daily),
    }

    assert templates is not None
    return templates.TemplateResponse(
        request=request,
        name="usage.html",
        context={
            "daily": daily,
            "models": models,
            "stages": stages,
            "daily_by_model": daily_by_model,
            "totals": totals,
        },
    )


@router.get("/usage/stats", response_class=HTMLResponse)
async def usage_stats_fragment(
    request: Request,
    days: int = 30,
    _auth=Depends(require_auth),
    sql_store=Depends(get_sql_store),
):
    """htmx fragment: refreshable usage stats tables."""
    from ctxmtg.llm.usage_tracker import UsageTracker

    tracker = UsageTracker(db_path=_db_path_from_store(sql_store))

    daily = await tracker.get_daily_stats(days=days)
    models = await tracker.get_model_stats(days=days)
    stages = await tracker.get_stage_stats(days=days)

    totals = {
        "calls": sum(d.get("calls", 0) for d in daily),
        "total_tokens": sum(d.get("total_tokens", 0) for d in daily),
        "errors": sum(d.get("errors", 0) for d in daily),
    }

    assert templates is not None
    return templates.TemplateResponse(
        request=request,
        name="fragments/usage_stats.html",
        context={
            "daily": daily,
            "models": models,
            "stages": stages,
            "totals": totals,
        },
    )
