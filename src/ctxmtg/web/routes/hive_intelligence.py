# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Hive Intelligence Routes
=========================

Browse entity profiles and insights in the hive.

Routes:
    GET /profiles  -- Entity profiles browser
    GET /insights  -- Insights browser (local + native)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ctxmtg.web.routes.hive_dashboard import get_hive_db, require_auth

router = APIRouter(tags=["hive-intelligence"])

templates: Jinja2Templates | None = None


def set_templates(t: Jinja2Templates) -> None:
    global templates
    templates = t


@router.get("/profiles", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def profiles_page(request: Request):
    hive_db = get_hive_db()
    profiles = await hive_db.get_all_entity_profiles()
    return templates.TemplateResponse(
        request, "hive_profiles.html", {"profiles": profiles},
    )


@router.get("/insights", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def insights_page(request: Request):
    hive_db = get_hive_db()
    local_insights = await hive_db.get_insights(limit=200)
    native_insights = await hive_db.get_native_insights(limit=200)
    return templates.TemplateResponse(
        request, "hive_insights.html",
        {"local_insights": local_insights, "native_insights": native_insights},
    )
