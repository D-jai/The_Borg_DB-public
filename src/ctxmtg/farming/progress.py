# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Farming Stage Progress Tracker
================================

Tracks per-stage offsets so farming stages progressively scan through
all entities/facts instead of always processing the same top N.

Uses the farming_progress table:
    stage TEXT PRIMARY KEY
    last_offset INTEGER
    total_processed INTEGER
    updated_at TEXT

Added 2026-04-07 to fix: all stages had hardcoded LIMIT with no OFFSET,
causing repeated processing of the same top records every cycle.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger("ctxmtg.farming.progress")


def _run_async(coro):
    """Bridge sync farming stages to async store methods."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


async def get_offset(sql_store, stage: str) -> int:
    """Get the current offset for a stage. Returns 0 if no record exists."""
    rows = await sql_store.execute_sql(
        "SELECT last_offset FROM farming_progress WHERE stage = :stage",
        {"stage": stage},
    )
    return rows[0]["last_offset"] if rows else 0


async def update_offset(sql_store, stage: str, new_offset: int, batch_size: int) -> None:
    """Update the offset for a stage after processing a batch."""
    db = sql_store._ensure_db()
    await db.execute(
        """INSERT INTO farming_progress (stage, last_offset, total_processed, updated_at)
           VALUES (:stage, :offset, :batch, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
           ON CONFLICT(stage) DO UPDATE SET
               last_offset = :offset,
               total_processed = total_processed + :batch,
               updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')""",
        {"stage": stage, "offset": new_offset, "batch": batch_size},
    )
    await db.commit()


async def get_offset_with_wrap(sql_store, stage: str, total_count: int, batch_size: int) -> int:
    """Get offset, wrapping to 0 if it exceeds total count."""
    offset = await get_offset(sql_store, stage)
    if offset >= total_count:
        offset = 0
        # Reset the offset in the table
        db = sql_store._ensure_db()
        await db.execute(
            "UPDATE farming_progress SET last_offset = 0 WHERE stage = :stage",
            {"stage": stage},
        )
        await db.commit()
        logger.info("progress_wrapped", stage=stage, total=total_count)
    return offset
