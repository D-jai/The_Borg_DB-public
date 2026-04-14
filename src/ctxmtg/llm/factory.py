# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
LLM Provider Factory
=====================

Creates APIProvider instances for each pipeline role using config
from the ~/.ctxmtg/.env file. Returns None if the role has no API
key configured, allowing callers to degrade gracefully.

Depends on:
    - ctxmtg.config.env_file (reads per-role config)
    - ctxmtg.llm.api_provider (APIProvider)
    - ctxmtg.llm.usage_tracker (UsageTracker, optional)
    - ctxmtg.interfaces.llm (LLMProvider ABC)

Used by:
    - ctxmtg.cli (query, farm run, ingest commands)
    - ctxmtg.web.app (lifespan provider init)
"""

from __future__ import annotations

import structlog

from ctxmtg.config.env_file import LLM_ROLES, get_role_config
from ctxmtg.interfaces.llm import LLMProvider

logger = structlog.get_logger("ctxmtg.llm.factory")


def create_provider(
    role: str,
    db_path: str | None = None,
) -> LLMProvider | None:
    """Create an APIProvider for a pipeline role, or None if not configured.

    Args:
        role: One of the 6 pipeline roles (extraction, query_planning,
              retrieval, synthesis, farming, fusion).
        db_path: Optional path to the SQLite database for usage tracking.

    Returns:
        An APIProvider instance if the role has an API key configured,
        or None if the role is unconfigured.
    """
    if role not in LLM_ROLES:
        logger.warning("unknown_llm_role", role=role)
        return None

    cfg = get_role_config(role)
    api_key = cfg.get("api_key", "").strip()

    if not api_key:
        return None

    base_url = cfg.get("base_url", "https://api.openai.com/v1").strip()
    model = cfg.get("model", "gpt-4o-mini").strip()

    if not base_url:
        base_url = "https://api.openai.com/v1"
    if not model:
        model = "gpt-4o-mini"

    tracker = None
    if db_path:
        try:
            from ctxmtg.llm.usage_tracker import UsageTracker
            tracker = UsageTracker(db_path=db_path)
        except Exception:
            pass

    from ctxmtg.llm.api_provider import APIProvider

    provider = APIProvider(
        api_key=api_key,
        base_url=base_url,
        model=model,
        usage_tracker=tracker,
    )
    provider._stage = role

    logger.info(
        "llm_provider_created",
        role=role,
        model=model,
        base_url=base_url[:40],
    )
    return provider


def create_providers(
    db_path: str | None = None,
) -> dict[str, LLMProvider | None]:
    """Create providers for all 6 roles.

    Returns a dict mapping role name to APIProvider (or None).
    """
    return {role: create_provider(role, db_path) for role in LLM_ROLES}


def get_best_provider(
    *roles: str,
    db_path: str | None = None,
) -> LLMProvider | None:
    """Return the first configured provider from the given role priority list.

    Useful when a component can use any of several roles (e.g., the
    query executor tries synthesis first, then falls back to extraction).
    """
    for role in roles:
        provider = create_provider(role, db_path)
        if provider is not None:
            return provider
    return None
