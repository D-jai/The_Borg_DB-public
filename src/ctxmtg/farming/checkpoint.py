# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
SQLite Checkpoint Store
========================

Implements the CheckpointStore protocol using the farming_checkpoints
table's state_blob column.  Stages use this to save and load partial
progress (e.g., a half-fitted KMeans model) for crash recovery and
cross-cycle resumption.

Uses pickle for serialisation.  The state_blob is only used for crash
recovery -- never sent over the network or read by untrusted code.

Depends on:
    - pickle (state serialisation)
    - asyncio (bridge sync CheckpointStore protocol to async SQLStore)
    - ctxmtg.interfaces.storage (SQLStore for database access)

Used by:
    - ctxmtg.farming.pipeline (creates one per stage per cycle)
"""

from __future__ import annotations

import asyncio
import pickle
from typing import Any

import structlog

from ctxmtg.interfaces.storage import SQLStore

# ---------------------------------------------------------------
# Module-level logger -- logs checkpoint save/load events.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.checkpoint")


def _run_async(coro: Any) -> Any:
    """
    Run an async coroutine from synchronous code.

    The CheckpointStore protocol methods are synchronous (because
    FarmingStage.run() is sync), but SQLStore is async.  This helper
    bridges the gap using the same pattern as ingestion/worker.py.
    """
    try:
        asyncio.get_running_loop()
        # Already in an async context -- use a thread to avoid deadlock
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop -- safe to use asyncio.run()
        return asyncio.run(coro)


class SQLiteCheckpointStore:
    """
    CheckpointStore backed by the farming_checkpoints table.

    Each instance is scoped to one (cycle_id, stage) pair.  The
    pipeline orchestrator creates a fresh instance for each stage
    execution, passing it into FarmingContext.

    State is serialised with pickle and stored in the state_blob
    BLOB column.  Loading returns None if no checkpoint exists or
    if the blob is empty.

    Usage:
        cp = SQLiteCheckpointStore(sql_store, cycle_id=42, stage="clustering")
        cp.save({"centroids": [...], "processed": 7500})
        state = cp.load()  # -> {"centroids": [...], "processed": 7500}
    """

    def __init__(self, sql_store: SQLStore, cycle_id: int, stage: str) -> None:
        """
        Bind the checkpoint to a specific cycle and stage.

        Args:
            sql_store:  The SQL store for database access.
            cycle_id:   The farming cycle this checkpoint belongs to.
            stage:      The stage name (e.g., "clustering").
        """
        self._sql_store = sql_store
        self._cycle_id = cycle_id
        self._stage = stage

    def save(self, state: Any) -> None:
        """
        Pickle the state and persist it to the state_blob column.

        Overwrites any previously saved state for this (cycle, stage).
        """
        blob = pickle.dumps(state)
        _run_async(
            self._sql_store.execute_sql(
                "UPDATE farming_checkpoints "
                "SET state_blob = :blob, items_processed = :items "
                "WHERE cycle_id = :cid AND stage = :stage",
                {
                    "blob": blob,
                    "items": state.get("items_processed", 0) if isinstance(state, dict) else 0,
                    "cid": self._cycle_id,
                    "stage": self._stage,
                },
            )
        )
        logger.debug(
            "checkpoint_saved",
            cycle_id=self._cycle_id,
            stage=self._stage,
            blob_bytes=len(blob),
        )

    def load(self) -> Any | None:
        """
        Load and unpickle the saved state.

        Returns None if no checkpoint exists or if the blob is empty.
        """
        rows = _run_async(
            self._sql_store.execute_sql(
                "SELECT state_blob FROM farming_checkpoints "
                "WHERE cycle_id = :cid AND stage = :stage",
                {"cid": self._cycle_id, "stage": self._stage},
            )
        )
        # No checkpoint row found
        if not rows:
            return None

        blob = rows[0].get("state_blob")

        # Blob column is NULL or empty
        if not blob:
            return None

        try:
            return pickle.loads(blob)
        except Exception as exc:
            logger.warning(
                "checkpoint_load_failed",
                error_code="CTXMTG-FRM-010",
                cycle_id=self._cycle_id,
                stage=self._stage,
                error=str(exc),
            )
            return None
