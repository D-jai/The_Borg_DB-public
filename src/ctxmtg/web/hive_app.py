# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Hive FastAPI Application Factory
==================================

Creates the FastAPI application for the Hive Command Center.
This is a SEPARATE app from the local Command Center -- different
process, different port, different auth.

The hive app only needs access to hive.db (no SQLiteStore, no
LanceDBStore, no vector store).  It manages links to local instances
and pulls intelligence from their outbox directories.

Lifecycle:
    1. Startup: initialise HiveDatabase.
    2. Serve: route requests through hive auth + route modules.
    3. Shutdown: close HiveDatabase.

Used by:
    - ctxmtg.cli (``ctxmtg hive serve`` command)
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from ctxmtg.config.settings import CtxMtgSettings

logger = structlog.get_logger("ctxmtg.web.hive_app")

_TEMPLATE_DIR = Path(__file__).parent / "hive_templates"


def create_hive_app(settings: CtxMtgSettings | None = None) -> FastAPI:
    """
    Build and return the Hive FastAPI application.

    Args:
        settings: Optional settings override (defaults to env/config).

    Returns:
        A FastAPI app ready to be served with uvicorn.
    """
    if settings is None:
        settings = CtxMtgSettings()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialise HiveDatabase on startup, close on shutdown."""
        from ctxmtg.sync.hive_db import HiveDatabase

        hive_path = Path(settings.hive.local_db_path).expanduser()
        hive_path.parent.mkdir(parents=True, exist_ok=True)

        hive_db = HiveDatabase(
            mode=settings.hive.mode,
            local_db_path=str(hive_path),
        )
        await hive_db.initialize()

        # Wire up the hive_db to routes
        from ctxmtg.web.routes import hive_dashboard
        hive_dashboard.set_hive_db(hive_db)

        logger.info("hive_app_started", db_path=str(hive_path))

        yield

        with contextlib.suppress(Exception):
            await hive_db.close()
        logger.info("hive_app_stopped")

    app = FastAPI(
        title="The_Borg_DB Hive",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    # Jinja2 templates for hive UI
    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

    # Register hive route modules
    from ctxmtg.web.routes import hive_dashboard, hive_intelligence

    hive_dashboard.set_templates(templates)
    hive_intelligence.set_templates(templates)

    app.include_router(hive_dashboard.router)
    app.include_router(hive_intelligence.router)

    app.state.templates = templates

    return app


def run_hive_server(
    host: str = "127.0.0.1",
    port: int = 8081,
    settings: CtxMtgSettings | None = None,
) -> None:
    """
    Start the hive web server with uvicorn.

    Args:
        host:     Bind address (default localhost only).
        port:     Port number (default 8081).
        settings: Optional settings override.
    """
    import uvicorn

    app = create_hive_app(settings)
    uvicorn.run(app, host=host, port=port, log_level="info")
