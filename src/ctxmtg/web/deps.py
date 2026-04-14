# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
FastAPI Dependencies
====================

Provides shared dependency-injection callables for the web routes.
Manages store lifecycle (initialised once at startup, closed on
shutdown) and session validation.

Depends on:
    - ctxmtg.config.settings (CtxMtgSettings)
    - ctxmtg.storage.sqlite (SQLiteStore)
    - ctxmtg.storage.lancedb_store (LanceDBStore)
    - ctxmtg.sync.hive_db (HiveDatabase)
    - ctxmtg.web.auth (session validation)

Used by:
    - ctxmtg.web.routes.* (all route modules)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from fastapi import Cookie, HTTPException, Request, status

from ctxmtg.web.auth import validate_session

logger = structlog.get_logger("ctxmtg.web.deps")

# Singleton store references, populated by lifespan handler in app.py.
_stores: dict[str, Any] = {}


def set_stores(
    sql_store: Any,
    vector_store: Any,
    hive_db: Any | None = None,
) -> None:
    """Called once at startup to register the shared store instances."""
    _stores["sql"] = sql_store
    _stores["vector"] = vector_store
    _stores["hive"] = hive_db


def get_sql_store():
    """Return the shared SQLiteStore instance."""
    store = _stores.get("sql")
    if store is None:
        raise HTTPException(status_code=503, detail="SQL store not initialised")
    return store


def get_vector_store():
    """Return the shared LanceDBStore instance."""
    store = _stores.get("vector")
    if store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialised")
    return store


def get_hive_db():
    """Return the shared HiveDatabase instance (may be None)."""
    return _stores.get("hive")


def require_auth(request: Request, session: str | None = Cookie(default=None)):
    """
    Dependency that enforces session authentication.

    Raises 303 redirect to /login for browser requests,
    or 401 for API requests.
    """
    if validate_session(session):
        return True

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        from fastapi.responses import RedirectResponse

        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"},
        )
    raise HTTPException(status_code=401, detail="Not authenticated")
