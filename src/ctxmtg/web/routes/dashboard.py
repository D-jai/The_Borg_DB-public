# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Dashboard Routes
================

Main dashboard page showing system overview: record counts, hive status,
recent farming cycles, and action buttons for push/farm operations.
All dynamic fragments are htmx-powered for partial page updates.

Depends on:
    - ctxmtg.web.deps (store access, auth)
    - ctxmtg.storage.sqlite (SQLiteStore)
    - ctxmtg.sync.hive_db (HiveDatabase)

Used by:
    - ctxmtg.web.app (included as a router)
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ctxmtg.web.deps import get_hive_db, get_sql_store, get_vector_store, require_auth

logger = structlog.get_logger("ctxmtg.web.routes.dashboard")

router = APIRouter(tags=["dashboard"])
templates: Jinja2Templates | None = None


def set_templates(t: Jinja2Templates) -> None:
    global templates
    templates = t


async def _get_local_stats(sql_store) -> dict[str, Any]:
    """Gather local store statistics."""
    stats: dict[str, Any] = {
        "interactions": 0,
        "entities": 0,
        "facts": 0,
        "insights": 0,
    }
    try:
        rows = await sql_store.execute_sql(
            "SELECT "
            "(SELECT COUNT(*) FROM interactions WHERE is_deleted=0) AS interactions, "
            "(SELECT COUNT(*) FROM entities WHERE is_deleted=0) AS entities, "
            "(SELECT COUNT(*) FROM facts WHERE is_deleted=0) AS facts, "
            "(SELECT COUNT(*) FROM meta_insights) AS insights"
        )
        if rows:
            stats.update(rows[0])
    except Exception as exc:
        logger.warning("local_stats_failed", error=str(exc))
    return stats


async def _get_hive_stats(hive_db) -> dict[str, Any] | None:
    """Gather hive statistics if available."""
    if hive_db is None:
        return None
    try:
        counts = await hive_db.get_record_counts()
        return counts
    except Exception as exc:
        logger.warning("hive_stats_failed", error=str(exc))
        return None


async def _get_recent_farming(sql_store, limit: int = 5) -> list[dict[str, Any]]:
    """Fetch recent farming cycle summaries."""
    try:
        rows = await sql_store.execute_sql(
            "SELECT cycle_id, status, trigger, stages_done, "
            "started_at, completed_at "
            "FROM farming_cycles "
            "ORDER BY cycle_id DESC LIMIT :limit",
            {"limit": limit},
        )
        return rows
    except Exception:
        return []


async def _get_vector_count(vector_store) -> int:
    """Get vector store record count."""
    try:
        return await vector_store.count()
    except Exception:
        return 0


@router.get("/", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def dashboard(request: Request):
    """Render the main dashboard page."""
    sql_store = get_sql_store()
    vector_store = get_vector_store()
    hive_db = get_hive_db()

    local_stats = await _get_local_stats(sql_store)
    hive_stats = await _get_hive_stats(hive_db)
    farming_cycles = await _get_recent_farming(sql_store)
    vector_count = await _get_vector_count(vector_store)

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "local": local_stats,
            "hive": hive_stats,
            "vectors": vector_count,
            "farming_cycles": farming_cycles,
        },
    )


@router.post("/api/farm/run", dependencies=[Depends(require_auth)])
async def trigger_farm(request: Request):
    """
    Trigger a farming cycle via htmx. Returns an HTML fragment
    with the result summary.
    """
    sql_store = get_sql_store()
    vector_store = get_vector_store()

    try:
        from ctxmtg.farming import FarmingPipeline, create_default_stages

        # Load the farming LLM provider (same pattern as cli.py farm_run).
        # ORIGINAL CODE (disabled 2026-04-07): LLM was not passed to stages,
        # causing all 17 stages to degrade to math-only operations.
        # for stage in create_default_stages():
        llm = None
        try:
            from ctxmtg.llm.factory import get_best_provider
            llm = get_best_provider("farming", "extraction")
            logger.info("farm_llm_loaded", model=llm.get_model_name() if llm else "None")
        except Exception as llm_exc:
            logger.warning("farm_llm_load_failed", error=str(llm_exc))

        pipeline = FarmingPipeline(sql_store, vector_store)
        for stage in create_default_stages(llm=llm):
            pipeline.register_stage(stage)

        result = await pipeline.run_cycle(trigger="web_dashboard")

        return HTMLResponse(
            f'<div class="alert alert-success">'
            f'Farming cycle #{result["cycle_id"]} complete: '
            f'{result["stages_succeeded"]}/{result["stages_run"]} stages, '
            f'{result["insights_produced"]} insights, '
            f'{result["duration_ms"]:.0f}ms'
            f"</div>"
        )
    except Exception as exc:
        logger.error("farm_trigger_failed", error=str(exc))
        return HTMLResponse(
            f'<div class="alert alert-error">Farming failed: {exc}</div>',
            status_code=500,
        )


@router.post("/api/hive/push", dependencies=[Depends(require_auth)])
async def trigger_push(request: Request):
    """
    Write intelligence to local outbox via htmx.  Returns an HTML
    fragment with counts.

    2026-04-08: Redesigned from direct hive DB push to outbox pattern.
    """
    sql_store = get_sql_store()

    try:
        from pathlib import Path

        from ctxmtg.config.settings import CtxMtgSettings
        from ctxmtg.sync.hive_sync import HiveSyncWorker
        from ctxmtg.sync.outbox_writer import OutboxWriter

        settings = CtxMtgSettings()
        outbox_path = Path(settings.hive.outbox_path).expanduser()
        instance_name = settings.hive.instance_name

        writer = OutboxWriter(
            outbox_path=outbox_path,
            instance_name=instance_name,
        )
        worker = HiveSyncWorker(
            local_store=sql_store,
            outbox_writer=writer,
        )
        counts = await worker.sync()

        summaries = counts.get("summaries", 0)
        insights = counts.get("insights", 0)
        manifest = counts.get("manifest")

        if not summaries and not insights:
            return HTMLResponse(
                '<div class="alert alert-info">'
                "No new intelligence to push to outbox."
                "</div>"
            )

        return HTMLResponse(
            f'<div class="alert alert-success">'
            f"Written to outbox: "
            f"{summaries} summaries, {insights} insights"
            f"</div>"
        )
    except Exception as exc:
        logger.error("outbox_push_failed", error=str(exc))
        return HTMLResponse(
            f'<div class="alert alert-error">Outbox push failed: {exc}</div>',
            status_code=500,
        )

    # ORIGINAL trigger_push (disabled 2026-04-08): direct hive DB push
    # hive_db = get_hive_db()
    # if hive_db is None:
    #     return HTMLResponse(
    #         '<div class="alert alert-error">Hive not configured</div>',
    #         status_code=400,
    #     )
    # try:
    #     from ctxmtg.sync.hive_sync import HiveSyncWorker
    #     worker = HiveSyncWorker(local_store=sql_store, hive_db=hive_db)
    #     counts = await worker.sync()
    #     total = sum(counts.values())
    #     return HTMLResponse(
    #         f'<div class="alert alert-success">'
    #         f"Pushed {total} records to hive "
    #         f'(interactions: {counts.get("interactions", 0)}, '
    #         f'entities: {counts.get("entities", 0)}, '
    #         f'facts: {counts.get("facts", 0)})'
    #         f"</div>"
    #     )
    # except Exception as exc:
    #     logger.error("hive_push_failed", error=str(exc))
    #     return HTMLResponse(
    #         f'<div class="alert alert-error">Hive push failed: {exc}</div>',
    #         status_code=500,
    #     )


@router.get("/api/stats", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def refresh_stats(request: Request):
    """htmx endpoint: return updated stat cards as an HTML fragment."""
    sql_store = get_sql_store()
    vector_store = get_vector_store()
    hive_db = get_hive_db()

    local_stats = await _get_local_stats(sql_store)
    hive_stats = await _get_hive_stats(hive_db)
    vector_count = await _get_vector_count(vector_store)

    return templates.TemplateResponse(
        request,
        "fragments/stats.html",
        {"local": local_stats, "hive": hive_stats, "vectors": vector_count},
    )
