# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
FastAPI Application Factory
============================

Creates and configures the FastAPI application for the ctxmtg web
command center. Binds to 127.0.0.1 only. Uses Jinja2 templates
with htmx for a lightweight reactive UI.

Lifecycle:
    1. Startup: initialise SQLite, LanceDB, and optional hive stores.
    2. Serve: route requests through auth middleware + route modules.
    3. Shutdown: close all database connections.

Depends on:
    - fastapi (web framework)
    - uvicorn (ASGI server)
    - jinja2 (template rendering)
    - ctxmtg.config.settings (CtxMtgSettings)
    - ctxmtg.web.deps (dependency injection)
    - ctxmtg.web.routes.* (route modules)

Used by:
    - ctxmtg.cli (`ctxmtg serve` command)
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ctxmtg.config.settings import CtxMtgSettings

logger = structlog.get_logger("ctxmtg.web.app")

# Resolve the templates directory relative to this file.
_TEMPLATE_DIR = Path(__file__).parent / "templates"


def create_app(settings: CtxMtgSettings | None = None) -> FastAPI:
    """
    Build and return a configured FastAPI application.

    Args:
        settings: Optional settings override (defaults to env/config).

    Returns:
        A FastAPI app ready to be served with uvicorn.
    """
    if settings is None:
        settings = CtxMtgSettings()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialise stores on startup, close on shutdown."""
        from ctxmtg.storage.lancedb_store import LanceDBStore
        from ctxmtg.storage.sqlite import SQLiteStore
        from ctxmtg.web.deps import set_stores

        db_path = Path(settings.db_path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        vec_path = Path(settings.vector_path).expanduser()
        vec_path.mkdir(parents=True, exist_ok=True)

        sql_store = SQLiteStore(db_path=str(db_path))
        vector_store = LanceDBStore(db_path=str(vec_path))
        await sql_store.initialize()
        await vector_store.initialize()

        # Optional hive database.
        hive_db = None
        hive_path = Path(settings.hive.local_db_path).expanduser()
        if hive_path.exists():
            try:
                from ctxmtg.sync.hive_db import HiveDatabase

                hive_db = HiveDatabase(
                    mode=settings.hive.mode,
                    local_db_path=str(hive_path),
                )
                await hive_db.initialize()
            except Exception as exc:
                logger.warning("hive_init_failed", error=str(exc))

        set_stores(sql_store, vector_store, hive_db)
        logger.info("stores_initialised", db=str(db_path), vectors=str(vec_path))

        yield

        # Shutdown: close stores.
        await sql_store.close()
        await vector_store.close()
        if hive_db is not None:
            with contextlib.suppress(Exception):
                await hive_db.close()
        logger.info("stores_closed")

    app = FastAPI(
        title="The_Borg_DB Command Center",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    # Jinja2 templates.
    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

    # Register route modules.
    from ctxmtg.web.routes import (
        auth_routes, completions, dashboard, entities, ingest, local, usage,
    )

    auth_routes.set_templates(templates)
    dashboard.set_templates(templates)
    local.set_templates(templates)
    usage.set_templates(templates)
    entities.set_templates(templates)

    app.include_router(auth_routes.router)
    app.include_router(dashboard.router)
    app.include_router(local.router)
    app.include_router(completions.router)
    app.include_router(ingest.router)
    app.include_router(usage.router)
    app.include_router(entities.router)

    # Apply initial enabled/disabled state from settings.
    ingest.set_enabled(settings.http_ingest_enabled)

    # Store templates on app state for route modules added later.
    app.state.templates = templates

    return app


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    """
    Start the uvicorn server. Called by `ctxmtg serve`.

    Always binds to 127.0.0.1 -- ignores any non-localhost host
    to prevent accidental network exposure.
    """
    import uvicorn

    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning("forcing_localhost", requested=host)
        host = "127.0.0.1"

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")
