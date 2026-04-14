# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Background Hive Query Runner (P5-11)
=====================================

After the local query returns (foreground, no latency hit), this
runner executes the same query against the hive stores and logs
the result to ``hive_answer.json`` in the same evaluation folder.

The runner instantiates a second QueryExecutor with hive stores.
Since the executor is store-agnostic, this works transparently.

Depends on:
    - ctxmtg.query.executor (QueryExecutor -- store-agnostic)
    - ctxmtg.query.evaluation (log_hive_answer)
    - ctxmtg.sync.hive_db (HiveDatabase for hive SQL access)

Used by:
    - ctxmtg.cli (query command triggers background hive query)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from ctxmtg.models.query import RetrievalMode

logger = structlog.get_logger("ctxmtg.query.hive_runner")


async def run_hive_query(
    query: str,
    mode: RetrievalMode,
    top_k: int,
    hive_db_path: str,
    eval_folder: Path,
    profile_name: str = "general",
) -> dict[str, Any] | None:
    """
    Run the same query against the hive and log the result.

    This is designed to be called after the local query has returned,
    so there's no latency impact on the user's foreground query.

    The hive stores merged entity profiles and insights.  We query
    them by creating a temporary local SQL store pointed at the hive
    DB file.  Since the hive schema differs from the local schema,
    we use a limited query path (SQL-only, no vector search).

    Args:
        query: The user's question.
        mode: The retrieval mode used for the local query.
        top_k: Number of results.
        hive_db_path: Path to the hive.db file.
        eval_folder: The evaluation folder to log to.
        profile_name: Domain profile name.

    Returns:
        The hive query results dict, or None if hive is unavailable.
    """
    try:
        from ctxmtg.query.evaluation import log_hive_answer
        from ctxmtg.sync.hive_db import HiveDatabase

        hive = HiveDatabase(mode="local", local_db_path=hive_db_path)
        await hive.initialize()

        try:
            # Query hive entity profiles for relevant entities
            profiles = await hive.get_all_entity_profiles()
            insights = await hive.get_insights(limit=50)
            native_insights = await hive.get_native_insights(limit=50)

            # Build a pseudo-QueryResult from hive intelligence
            from ctxmtg.models.query import QueryResult, SearchResult
            import time

            start = time.monotonic()

            # Search entity profiles for query-relevant matches
            query_lower = query.lower()
            matching_profiles = []
            for p in profiles:
                name = p.get("entity_name", "").lower()
                summary = p.get("merged_summary", "").lower()
                if (
                    any(word in name for word in query_lower.split())
                    or any(word in summary for word in query_lower.split() if len(word) > 3)
                ):
                    matching_profiles.append(p)

            # Convert matching profiles to SearchResult format
            results = []
            for i, p in enumerate(matching_profiles[:top_k]):
                results.append(SearchResult(
                    id=f"hive-profile-{p['entity_name']}",
                    source_store="hive",
                    content=(
                        f"[{p.get('entity_type', '?')}] {p['entity_name']}: "
                        f"{p.get('merged_summary', '')}"
                    ),
                    score=p.get("cross_stream_score", 0.0),
                    metadata={
                        "entity_type": p.get("entity_type", "other"),
                        "stream_count": p.get("stream_count", 1),
                        "total_mentions": p.get("total_mentions", 0),
                    },
                ))

            # Also include relevant insights as results
            for ins in insights[:5]:
                title = ins.get("title", "").lower()
                if any(word in title for word in query_lower.split() if len(word) > 3):
                    results.append(SearchResult(
                        id=ins.get("id", "unknown"),
                        source_store="hive_insight",
                        content=f"[{ins.get('insight_type', '?')}] {ins.get('title', '')}: {ins.get('description', '')}",
                        score=ins.get("confidence", 0.5),
                        metadata={"source_instance": ins.get("source_instance", "?")},
                    ))

            latency_ms = (time.monotonic() - start) * 1000

            hive_result = QueryResult(
                query=query,
                mode=RetrievalMode.PARALLEL,
                results=results,
                total_results=len(results),
                sql_results_count=len(matching_profiles),
                vector_results_count=0,
                synthesis=None,
                latency_ms=latency_ms,
            )

            log_hive_answer(hive_result, eval_folder)

            logger.info(
                "hive_query_complete",
                query=query[:80],
                results=len(results),
                latency_ms=round(latency_ms, 2),
            )

            return {"results": len(results), "latency_ms": round(latency_ms, 2)}

        finally:
            await hive.close()

    except Exception as exc:
        logger.warning(
            "hive_query_failed",
            error=str(exc),
            query=query[:80],
        )
        return None
