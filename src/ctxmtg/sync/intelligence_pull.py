# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Intelligence Pull Worker (Phase 5 Rewrite)
============================================

Pulls hive-native intelligence (merged entity profiles and hive
native insights) back to the local instance for use in extraction
enrichment and query synthesis.

Phase 5 changes:
    - Pulls from hive_entity_profiles (merged cross-stream profiles)
      instead of distiller_summaries
    - Pulls hive_native_insights (latent relationships, correlated
      patterns, coverage gaps)
    - Caches both in local_intelligence_cache

Depends on:
    - json (parse JSON columns)
    - structlog (structured logging)
    - ctxmtg.interfaces.storage (SQLStore for local cache)
    - ctxmtg.sync.hive_db (HiveDatabase for hive access)

Used by:
    - ctxmtg.extraction.pipeline (reads cached hints)
    - ctxmtg.cli (``ctxmtg intelligence pull`` command)
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from ctxmtg.interfaces.storage import SQLStore
from ctxmtg.sync.hive_db import HiveDatabase

logger = structlog.get_logger("ctxmtg.sync.intelligence_pull")

UPSERT_LOCAL_CACHE_SQL = """\
INSERT OR REPLACE INTO local_intelligence_cache
    (entity_name, entity_type, summary, top_co_entities,
     top_predicates, relevance_score, source_instances, fetched_at)
VALUES
    (:entity_name, :entity_type, :summary, :top_co_entities,
     :top_predicates, :relevance_score, :source_instances, :fetched_at)
"""

ALL_HINTS_SQL = """\
SELECT entity_name, entity_type, summary, top_co_entities,
       top_predicates, relevance_score, source_instances
FROM local_intelligence_cache
ORDER BY relevance_score DESC
LIMIT :limit
"""


class IntelligencePullWorker:
    """
    Pulls hive intelligence and caches it locally.

    Reads hive_entity_profiles (merged cross-stream intelligence) and
    caches them in local_intelligence_cache for use by the extraction
    pipeline and query engine.

    Usage:
        worker = IntelligencePullWorker(local_store, hive_db, "laptop")
        cached_count = await worker.pull()
        hints = await worker.get_hints_for_entities(["Alice", "Bob"])
    """

    def __init__(
        self,
        local_store: SQLStore,
        hive_db: HiveDatabase,
        instance_id: str = "local",
    ) -> None:
        self._local_store = local_store
        self._hive_db = hive_db
        self._instance_id = instance_id

    async def pull(self) -> int:
        """
        Pull entity profiles from the hive and cache locally.

        Fetches hive_entity_profiles, filters out profiles sourced
        exclusively from this instance, and caches qualifying profiles
        in local_intelligence_cache.

        Returns:
            The number of hints cached locally.
        """
        logger.info(
            "intelligence_pull_start",
            instance_id=self._instance_id,
        )

        # Fetch merged entity profiles from hive
        try:
            profiles = await self._hive_db.get_entity_profiles(
                min_score=0.0, limit=200
            )
        except Exception as exc:
            logger.warning(
                "intelligence_pull_hive_query_failed",
                error_code="CTXMTG-SYN-004",
                error=str(exc),
                instance_id=self._instance_id,
            )
            return 0

        if not profiles:
            logger.info(
                "intelligence_pull_empty",
                instance_id=self._instance_id,
            )
            return 0

        # Filter out self-sourced profiles
        filtered = []
        for profile in profiles:
            sources = _parse_json_list(profile.get("source_streams", "[]"))
            if not sources or any(
                s != self._instance_id for s in sources
            ):
                filtered.append(profile)

        logger.debug(
            "intelligence_pull_filtered",
            total_profiles=len(profiles),
            after_filter=len(filtered),
        )

        # Cache locally
        cached_count = 0
        now_iso = _now_iso()

        for profile in filtered:
            try:
                await self._local_store.execute_sql(
                    UPSERT_LOCAL_CACHE_SQL,
                    {
                        "entity_name": profile["entity_name"],
                        "entity_type": profile.get("entity_type", "other"),
                        "summary": profile.get("merged_summary", ""),
                        "top_co_entities": profile.get("top_co_entities", "[]"),
                        "top_predicates": profile.get("top_predicates", "[]"),
                        "relevance_score": profile.get("cross_stream_score", 0.0),
                        "source_instances": profile.get("source_streams", "[]"),
                        "fetched_at": now_iso,
                    },
                )
                cached_count += 1
            except Exception as exc:
                logger.warning(
                    "intelligence_pull_upsert_failed",
                    error_code="CTXMTG-SYN-004",
                    error=str(exc),
                )

        logger.info(
            "intelligence_pull_complete",
            cached=cached_count,
            instance_id=self._instance_id,
        )

        return cached_count

    async def get_hints_for_entities(
        self, entity_names: list[str]
    ) -> dict[str, dict]:
        """Retrieve cached intelligence hints for specific entities."""
        if not entity_names:
            return {}

        named_params = {f"e{i}": name for i, name in enumerate(entity_names)}
        named_placeholders = ", ".join(f":e{i}" for i in range(len(entity_names)))
        sql = (
            f"SELECT entity_name, entity_type, summary, top_co_entities, "
            f"top_predicates, relevance_score "
            f"FROM local_intelligence_cache "
            f"WHERE entity_name IN ({named_placeholders})"
        )

        try:
            rows = await self._local_store.execute_sql(sql, named_params)
        except Exception as exc:
            logger.warning(
                "intelligence_hints_query_failed",
                error_code="CTXMTG-SYN-004",
                error=str(exc),
            )
            return {}

        result: dict[str, dict] = {}
        for row in rows:
            result[row["entity_name"]] = {
                "summary": row["summary"],
                "top_co_entities": _parse_json_list(row.get("top_co_entities", "[]")),
                "top_predicates": _parse_json_list(row.get("top_predicates", "[]")),
                "relevance_score": row.get("relevance_score", 0.0),
            }

        return result

    async def get_all_hints(self, limit: int = 200) -> list[dict]:
        """Retrieve all cached intelligence hints, ordered by relevance."""
        try:
            rows = await self._local_store.execute_sql(
                ALL_HINTS_SQL, {"limit": limit}
            )
        except Exception as exc:
            logger.warning(
                "intelligence_all_hints_query_failed",
                error_code="CTXMTG-SYN-004",
                error=str(exc),
            )
            return []

        hints: list[dict] = []
        for row in rows:
            hints.append({
                "entity_name": row["entity_name"],
                "entity_type": row["entity_type"],
                "summary": row["summary"],
                "top_co_entities": _parse_json_list(row.get("top_co_entities", "[]")),
                "top_predicates": _parse_json_list(row.get("top_predicates", "[]")),
                "relevance_score": row.get("relevance_score", 0.0),
                "source_instances": _parse_json_list(row.get("source_instances", "[]")),
            })

        return hints


def _parse_json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
