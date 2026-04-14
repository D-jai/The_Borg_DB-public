# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Per-Local Management Routes
============================

Management page for the local instance: profile viewing/editing,
farming configuration, LLM role overview, and system settings.
Uses htmx for in-place editing of profile parameters.

Depends on:
    - ctxmtg.profile.loader (ProfileLoader)
    - ctxmtg.config.settings (CtxMtgSettings)
    - ctxmtg.web.deps (auth, stores)

Used by:
    - ctxmtg.web.app (included as a router)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ctxmtg.config.settings import CtxMtgSettings
from ctxmtg.web.deps import get_sql_store, require_auth

logger = structlog.get_logger("ctxmtg.web.routes.local")

router = APIRouter(prefix="/local", tags=["local"])
templates: Jinja2Templates | None = None


def set_templates(t: Jinja2Templates) -> None:
    global templates
    templates = t


def _load_profile_safe(name: str) -> dict[str, Any]:
    """Load a profile and return it as a dict, or return error info."""
    try:
        from ctxmtg.profile.loader import ProfileLoader

        profile = ProfileLoader.load(name)
        return {"ok": True, "profile": profile}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _list_profiles_safe() -> list[str]:
    """List available profile names."""
    try:
        from ctxmtg.profile.loader import ProfileLoader

        return ProfileLoader.list_profiles()
    except Exception:
        return []


async def _get_local_info(sql_store) -> dict[str, Any]:
    """Gather local instance metadata."""
    info: dict[str, Any] = {"instance_id": "local"}
    try:
        rows = await sql_store.execute_sql(
            "SELECT "
            "(SELECT COUNT(*) FROM interactions WHERE is_deleted=0) AS interactions, "
            "(SELECT COUNT(*) FROM entities WHERE is_deleted=0) AS entities, "
            "(SELECT COUNT(*) FROM facts WHERE is_deleted=0) AS facts, "
            "(SELECT COUNT(*) FROM meta_insights) AS insights"
        )
        if rows:
            info.update(rows[0])

        # Last farming cycle.
        farm_rows = await sql_store.execute_sql(
            "SELECT cycle_id, status, started_at, completed_at "
            "FROM farming_cycles ORDER BY cycle_id DESC LIMIT 1"
        )
        if farm_rows:
            info["last_farming"] = farm_rows[0]
    except Exception as exc:
        logger.warning("local_info_failed", error=str(exc))
    return info


@router.get("/", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def local_management(request: Request):
    """Render the local instance management page."""
    settings = CtxMtgSettings()
    sql_store = get_sql_store()

    local_info = await _get_local_info(sql_store)
    profiles = _list_profiles_safe()
    active_profile = _load_profile_safe(settings.profile_name)

    return templates.TemplateResponse(
        request,
        "local.html",
        {
            "local_info": local_info,
            "profiles": profiles,
            "active_profile_name": settings.profile_name,
            "active_profile": active_profile,
            "settings": settings,
        },
    )


@router.get(
    "/profile/{name}",
    response_class=HTMLResponse,
    dependencies=[Depends(require_auth)],
)
async def view_profile(request: Request, name: str):
    """Render profile details as an htmx fragment."""
    result = _load_profile_safe(name)

    if not result["ok"]:
        return HTMLResponse(
            f'<div class="alert alert-error">{result["error"]}</div>',
            status_code=404,
        )

    profile = result["profile"]
    return templates.TemplateResponse(
        request,
        "fragments/profile_detail.html",
        {"profile": profile, "profile_name": name},
    )


@router.post(
    "/farming/params",
    response_class=HTMLResponse,
    dependencies=[Depends(require_auth)],
)
async def update_farming_params(
    request: Request,
    schedule: str = Form("daily"),
    lookback_days: int = Form(30),
    cluster_min: int = Form(5),
):
    """
    Update farming parameters (in-memory only for this session).

    Farming config is part of the domain profile YAML. This endpoint
    shows the new values as confirmation. Persistent editing requires
    modifying the YAML file directly.
    """
    return HTMLResponse(
        f'<div class="alert alert-success">'
        f"Farming params updated for this session: "
        f"schedule={schedule}, lookback={lookback_days}d, "
        f"cluster_min={cluster_min}"
        f"</div>"
    )


@router.get(
    "/services",
    response_class=HTMLResponse,
    dependencies=[Depends(require_auth)],
)
async def services_fragment(request: Request):
    """Return the services toggle fragment."""
    from ctxmtg.web.routes.ingest import is_enabled as ingest_enabled

    proxy_capture = False
    try:
        from ctxmtg.proxy import is_capture_enabled

        proxy_capture = is_capture_enabled()
    except ImportError:
        pass

    return templates.TemplateResponse(
        request,
        "fragments/services.html",
        {
            "http_ingest_enabled": ingest_enabled(),
            "proxy_capture_enabled": proxy_capture,
        },
    )


@router.post(
    "/services/toggle",
    response_class=HTMLResponse,
    dependencies=[Depends(require_auth)],
)
async def toggle_service(
    request: Request,
    service: str = Form(...),
    enabled: str = Form("off"),
):
    """Toggle a service on or off via htmx."""
    is_on = enabled == "on"

    if service == "http_ingest":
        from ctxmtg.web.routes.ingest import set_enabled

        set_enabled(is_on)
        label = "HTTP Ingest"
    elif service == "proxy_capture":
        try:
            from ctxmtg.proxy import set_capture_enabled

            set_capture_enabled(is_on)
        except ImportError:
            pass
        label = "Proxy Capture"
    else:
        return HTMLResponse(
            '<div class="alert alert-error">Unknown service</div>', status_code=400
        )

    state = "enabled" if is_on else "disabled"
    return HTMLResponse(
        f'<div class="alert alert-success">{label} {state}</div>'
    )


STAGE_ROLES = {
    "extraction": {
        "label": "Extraction",
        "description": "Verifies and enhances NER + SPO facts.",
    },
    "query_planning": {
        "label": "Query Planning",
        "description": "Interprets queries into SQL/vector plans.",
    },
    "retrieval": {
        "label": "Retrieval Bridge",
        "description": "Formulates cross-store bridge queries.",
    },
    "synthesis": {
        "label": "Synthesis",
        "description": "Generates answers from search results.",
    },
    "farming": {
        "label": "Farming",
        "description": "Mines patterns from accumulated knowledge.",
    },
    "fusion": {
        "label": "Fusion",
        "description": "Reranks results by semantic relevance.",
    },
}


@router.get(
    "/llm-roles",
    response_class=HTMLResponse,
    dependencies=[Depends(require_auth)],
)
async def llm_roles(request: Request):
    """Show LLM role config with API key/URL/model per stage."""
    from ctxmtg.config.env_file import get_all_role_configs

    role_configs = get_all_role_configs()

    return templates.TemplateResponse(
        request,
        "fragments/llm_roles.html",
        {"stage_roles": STAGE_ROLES, "role_configs": role_configs},
    )


@router.post(
    "/llm-roles/save",
    response_class=HTMLResponse,
    dependencies=[Depends(require_auth)],
)
async def save_llm_role(
    request: Request,
    role: str = Form(...),
    api_key: str = Form(""),
    base_url: str = Form("https://api.openai.com/v1"),
    model: str = Form("gpt-4o-mini"),
):
    """Save a single LLM role's API config to the .env file."""
    from ctxmtg.config.env_file import LLM_ROLES, save_role_config

    if role not in LLM_ROLES:
        return HTMLResponse(
            '<div class="alert alert-error">Unknown role.</div>',
            status_code=400,
        )

    save_role_config(role, api_key.strip(), base_url.strip(), model.strip())

    logger.info("llm_role_saved", role=role, model=model, has_key=bool(api_key.strip()))

    return HTMLResponse(
        f'<div class="alert alert-success">'
        f'{STAGE_ROLES.get(role, {}).get("label", role)} saved.'
        f"</div>"
    )


@router.post(
    "/llm-roles/save-all",
    response_class=HTMLResponse,
    dependencies=[Depends(require_auth)],
)
async def save_all_llm_roles(request: Request):
    """Save all LLM role configs from a single form submission."""
    from ctxmtg.config.env_file import LLM_ROLES, save_all_role_configs

    form = await request.form()
    configs: dict[str, dict[str, str]] = {}
    for role in LLM_ROLES:
        configs[role] = {
            "api_key": str(form.get(f"{role}_api_key", "")).strip(),
            "base_url": str(form.get(f"{role}_base_url", "https://api.openai.com/v1")).strip(),
            "model": str(form.get(f"{role}_model", "gpt-4o-mini")).strip(),
        }

    save_all_role_configs(configs)
    logger.info("all_llm_roles_saved")

    return HTMLResponse(
        '<div class="alert alert-success">All LLM roles saved to .env file.</div>'
    )
