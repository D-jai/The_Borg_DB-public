# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Query Quality Logger
====================

Records every query's retrieval metrics into the query_quality_log
table so that the self-learning feedback loop (FeedbackLoopStage)
can detect poor-quality patterns and adjust the system.

Each log row captures:
    - The raw query text and retrieval mode
    - Result IDs serialised as a JSON array
    - Per-store result counts (SQL vs vector)
    - End-to-end latency in milliseconds
    - A refinement flag (did the user re-phrase within 60 s?)

The refinement flag is the primary implicit-negative-feedback signal.
When a user immediately re-phrases a query, it strongly suggests the
original results were unsatisfactory.  The FeedbackLoopStage reads
these flags to generate "gap" insights consumed by the Completionist
maintenance agent.

Depends on:
    - json (result_ids serialisation)
    - uuid (UUIDv4 generation for log row IDs)
    - structlog (structured JSON logging)
    - ctxmtg.interfaces.storage (SQLStore for database writes)

Used by:
    - ctxmtg.query.executor (logs every query after execution)
    - ctxmtg.farming.feedback_loop (reads quality signals)
    - tests/test_query/test_quality_logger.py
"""

from __future__ import annotations

import json
import uuid

import structlog

from ctxmtg.interfaces.storage import SQLStore

# ---------------------------------------------------------------
# Module-level logger -- structured JSON, no PII in log events.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.quality_logger")


class QueryQualityLogger:
    """
    Logs query execution metrics to the query_quality_log table.

    The logger is injected with a SQLStore instance and provides
    three operations:

    1. log_query()         -- INSERT a new quality record after
                              every query execution.
    2. mark_refinement()   -- UPDATE the refined_within_60s flag
                              when the user re-phrases quickly.
    3. get_recent_queries() -- SELECT the most recent log entries
                               (used by the feedback loop stage).

    Each log row is identified by a UUIDv4 so that the executor can
    correlate a refinement with its predecessor query.

    Usage:
        qlogger = QueryQualityLogger(sql_store)
        qid = await qlogger.log_query(
            query_text="What did Alice propose?",
            mode="parallel",
            result_ids=["r-1", "r-2"],
            sql_result_count=5,
            vector_result_count=3,
            latency_ms=42.7,
        )
        # User immediately re-phrases → mark the original as refined
        await qlogger.mark_refinement(qid)
    """

    def __init__(self, sql_store: SQLStore) -> None:
        """
        Bind the logger to a SQL store.

        Args:
            sql_store: The SQL store that owns the
                       query_quality_log table.
        """
        self._sql_store = sql_store

    # =============================================================
    # log_query -- INSERT a new quality record
    # =============================================================

    async def log_query(
        self,
        query_text: str,
        mode: str,
        result_ids: list[str],
        sql_result_count: int,
        vector_result_count: int,
        latency_ms: float,
    ) -> str:
        """
        Record a query's retrieval metrics in the quality log.

        Generates a fresh UUIDv4 for the row, serialises result_ids
        to a JSON array string, and INSERTs into query_quality_log.
        The created_at timestamp is set by the SQLite DEFAULT.

        Args:
            query_text:          The raw user query string.
            mode:                Retrieval mode used (e.g. "parallel").
            result_ids:          IDs of the returned results.
            sql_result_count:    Number of SQL-store results.
            vector_result_count: Number of vector-store results.
            latency_ms:          Total query latency in milliseconds.

        Returns:
            The UUIDv4 string assigned to this log row.
        """
        row_id = str(uuid.uuid4())

        # Serialise the result IDs list into a JSON string so it
        # fits the TEXT column in query_quality_log.
        result_ids_json = json.dumps(result_ids)

        await self._sql_store.execute_sql(
            "INSERT INTO query_quality_log "
            "(id, query_text, mode, result_ids, sql_result_count, "
            "vector_result_count, latency_ms) "
            "VALUES (:id, :query_text, :mode, :result_ids, "
            ":sql_count, :vec_count, :latency_ms)",
            {
                "id": row_id,
                "query_text": query_text,
                "mode": mode,
                "result_ids": result_ids_json,
                "sql_count": sql_result_count,
                "vec_count": vector_result_count,
                "latency_ms": latency_ms,
            },
        )

        logger.info(
            "query_quality_logged",
            query_id=row_id,
            mode=mode,
            sql_count=sql_result_count,
            vector_count=vector_result_count,
            latency_ms=round(latency_ms, 2),
        )

        return row_id

    # =============================================================
    # mark_refinement -- flag a query as "user refined quickly"
    # =============================================================

    async def mark_refinement(self, previous_query_id: str) -> None:
        """
        Flag a previous query as refined within 60 seconds.

        This is the primary implicit-negative-feedback signal.  When
        the user re-phrases a query shortly after, the original query
        likely returned poor results.  The FeedbackLoopStage harvests
        these flags to generate "gap" insights.

        Args:
            previous_query_id: The UUIDv4 of the original query row.
        """
        await self._sql_store.execute_sql(
            "UPDATE query_quality_log "
            "SET refined_within_60s = 1 "
            "WHERE id = :id",
            {"id": previous_query_id},
        )

        logger.info(
            "query_refinement_marked",
            query_id=previous_query_id,
        )

    # =============================================================
    # get_recent_queries -- SELECT recent quality log entries
    # =============================================================

    async def get_recent_queries(self, limit: int = 50) -> list[dict]:
        """
        Fetch the most recent query quality log entries.

        Returns rows ordered newest-first, up to *limit* rows.
        Each row is a plain dict (column_name → value) as returned
        by SQLStore.execute_sql().

        Args:
            limit: Maximum number of rows to return (default 50).

        Returns:
            A list of dicts representing query_quality_log rows.
        """
        rows = await self._sql_store.execute_sql(
            "SELECT * FROM query_quality_log "
            "ORDER BY created_at DESC "
            "LIMIT :limit",
            {"limit": limit},
        )

        logger.debug(
            "recent_queries_fetched",
            count=len(rows),
            limit=limit,
        )

        return rows
