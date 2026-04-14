# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
LLM Usage Tracker
==================

Records every LLM generate() call to the llm_usage SQLite table for
cost monitoring and analytics.  Each row captures model name, pipeline
stage (role), token counts, latency, and success/failure status.

The tracker is designed to be injected into LLM providers as a callback.
It accepts an aiosqlite connection (or path) and provides both
recording and querying methods.

Depends on:
    - aiosqlite (async SQLite writes)
    - ctxmtg.storage.schema (llm_usage table, v5 migration)

Used by:
    - ctxmtg.llm.provider (records usage after each generate call)
    - ctxmtg.web.routes.usage (reads stats for the usage dashboard)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
import structlog

logger = structlog.get_logger("ctxmtg.llm.usage_tracker")


@dataclass
class UsageRecord:
    """One LLM API call record."""

    model_name: str
    stage: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error_message: str | None = None


class UsageTracker:
    """
    Records and queries LLM usage data in the llm_usage table.

    Call record() after each LLM generate() invocation.
    Call get_daily_stats() / get_model_stats() for dashboard queries.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)

    async def record(self, rec: UsageRecord) -> None:
        """Insert a single usage record."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """INSERT INTO llm_usage
                       (model_name, stage, prompt_tokens, completion_tokens,
                        total_tokens, latency_ms, success, error_message)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        rec.model_name,
                        rec.stage,
                        rec.prompt_tokens,
                        rec.completion_tokens,
                        rec.total_tokens,
                        rec.latency_ms,
                        1 if rec.success else 0,
                        rec.error_message,
                    ),
                )
                await db.commit()
        except Exception:
            logger.warning("usage_record_failed", model=rec.model_name)

    async def get_daily_stats(self, days: int = 30) -> list[dict]:
        """Per-day totals: calls, tokens, avg latency."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT date(created_at) AS day,
                          COUNT(*) AS calls,
                          SUM(prompt_tokens) AS prompt_tokens,
                          SUM(completion_tokens) AS completion_tokens,
                          SUM(total_tokens) AS total_tokens,
                          ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                          SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS errors
                   FROM llm_usage
                   WHERE created_at >= date('now', ?)
                   GROUP BY day
                   ORDER BY day DESC""",
                (f"-{days} days",),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def get_model_stats(self, days: int = 30) -> list[dict]:
        """Per-model totals over the given window."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT model_name,
                          COUNT(*) AS calls,
                          SUM(prompt_tokens) AS prompt_tokens,
                          SUM(completion_tokens) AS completion_tokens,
                          SUM(total_tokens) AS total_tokens,
                          ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                          SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS errors
                   FROM llm_usage
                   WHERE created_at >= date('now', ?)
                   GROUP BY model_name
                   ORDER BY total_tokens DESC""",
                (f"-{days} days",),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def get_stage_stats(self, days: int = 30) -> list[dict]:
        """Per-stage (role) totals over the given window."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT stage,
                          COUNT(*) AS calls,
                          SUM(total_tokens) AS total_tokens,
                          ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                          SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS errors
                   FROM llm_usage
                   WHERE created_at >= date('now', ?)
                   GROUP BY stage
                   ORDER BY total_tokens DESC""",
                (f"-{days} days",),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def get_daily_by_model(self, days: int = 30) -> list[dict]:
        """Per-day, per-model breakdown for charting."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT date(created_at) AS day,
                          model_name,
                          COUNT(*) AS calls,
                          SUM(total_tokens) AS total_tokens
                   FROM llm_usage
                   WHERE created_at >= date('now', ?)
                   GROUP BY day, model_name
                   ORDER BY day DESC, total_tokens DESC""",
                (f"-{days} days",),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
