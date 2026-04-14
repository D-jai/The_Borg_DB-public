# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Hive Pull Worker (Hive Side)
=============================

This module implements the hive-side pull worker that reads unsynced
records from local instance databases.  The hive is the active party --
it opens the local SQLite file read-only and copies new records.

This replaces the push model (where instances push to the hive) with
a pull model (where the hive pulls from instances).  Benefits:
    - Hive controls its own schedule (no pause/resume flag needed)
    - Hive tracks its own high-water mark (hive_pull_progress table)
    - Local instances stay simple (just ingest and store)
    - For local mode: hive reads local SQLite directly (read-only)

See plans/06-phase3-brainstorm-notes.md for the design rationale.

Depends on:
    - aiosqlite (async SQLite driver for reading local DB)
    - ctxmtg.sync.hive_db (HiveDatabase for writing to hive)
    - ctxmtg.exceptions (SyncError for error reporting)

Used by:
    - ctxmtg.farming.pipeline (suspends/resumes pull during farming)
    - ctxmtg.cli (future: ``ctxmtg hive pull`` command)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import structlog

from ctxmtg.exceptions import SyncError
from ctxmtg.sync.hive_db import HiveDatabase

# ---------------------------------------------------------------
# Module-level logger -- structured JSON output, no PII in logs.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.sync.hive_pull")

# ---------------------------------------------------------------
# Batch limits per pull cycle.
# Keep these in sync with hive_sync.py push-side limits so the
# two models process comparable volumes per cycle.
# ---------------------------------------------------------------
INTERACTION_BATCH_LIMIT = 100
ENTITY_BATCH_LIMIT = 500
FACT_BATCH_LIMIT = 500


class HivePullWorker:
    """
    Hive-side worker that pulls unsynced records from local instances.

    The hive opens the local SQLite database in read-only mode and
    queries for records created after the last pull timestamp.  New
    records are inserted into the hive via HiveDatabase.push_records().

    Progress is tracked in the hive's hive_pull_progress table, keyed
    by table_name.  Each pull cycle updates the high-water mark.

    The worker can be suspended/resumed by the farming pipeline
    orchestrator to avoid write contention during farming windows.

    Usage:
        worker = HivePullWorker(
            hive_db=hive_database,
            local_db_path="/path/to/local/knowledge.db",
            instance_name="laptop",
        )
        counts = await worker.pull()
        status = await worker.get_status()
    """

    def __init__(
        self,
        hive_db: HiveDatabase,
        local_db_path: str,
        instance_name: str = "local",
    ) -> None:
        """
        Prepare the pull worker.

        Args:
            hive_db:        The HiveDatabase instance to write into.
            local_db_path:  Absolute path to the local instance's SQLite file.
                            Opened read-only -- the local DB is never modified.
            instance_name:  Human-readable name for this instance.
        """
        self._hive_db = hive_db
        self._local_db_path = local_db_path
        self._instance_name = instance_name
        self._suspended = False

    # =================================================================
    # Suspend / resume (called by farming pipeline)
    # =================================================================

    @property
    def suspended(self) -> bool:
        """Whether the worker is currently suspended (farming window)."""
        return self._suspended

    def suspend(self) -> None:
        """Suspend pull operations (called by farming pipeline before farming)."""
        self._suspended = True
        logger.info("hive_pull_suspended", instance=self._instance_name)

    def resume(self) -> None:
        """Resume pull operations (called by farming pipeline after farming)."""
        self._suspended = False
        logger.info("hive_pull_resumed", instance=self._instance_name)

    # =================================================================
    # Main pull cycle
    # =================================================================

    async def pull(self) -> dict[str, int]:
        """
        Pull new records from the local instance to the hive.

        Steps:
        1. Check if suspended (return immediately if so)
        2. Read hive_pull_progress for last_pulled_at per table
        3. Open local DB read-only
        4. For each table (interactions, entities, facts):
           a. SELECT * WHERE created_at > :last_pulled_at LIMIT :batch
           b. Push to hive via HiveDatabase.push_records()
           c. Update hive_pull_progress with new high-water mark

        Returns:
            Dict with counts: {"interactions": n, "entities": n, "facts": n}

        Raises:
            SyncError: If the pull fails at any step.
        """
        # If suspended (farming window active), skip this cycle
        if self._suspended:
            logger.info("hive_pull_skipped", reason="suspended")
            return {"interactions": 0, "entities": 0, "facts": 0}

        try:
            # -------------------------------------------------------
            # Step 1: Read current progress from hive
            # -------------------------------------------------------
            progress = await self._get_progress()

            # -------------------------------------------------------
            # Step 2: Open local database read-only
            # -------------------------------------------------------
            local_db = await aiosqlite.connect(
                f"file:{self._local_db_path}?mode=ro",
                uri=True,
            )
            local_db.row_factory = aiosqlite.Row

            try:
                # ---------------------------------------------------
                # Step 3: Pull each table since last high-water mark
                # ---------------------------------------------------
                interaction_rows = await self._pull_table(
                    local_db,
                    "interactions",
                    progress.get("interactions", {}).get(
                        "last_pulled_at", "1970-01-01T00:00:00Z"
                    ),
                    INTERACTION_BATCH_LIMIT,
                )

                entity_rows = await self._pull_table(
                    local_db,
                    "entities",
                    progress.get("entities", {}).get(
                        "last_pulled_at", "1970-01-01T00:00:00Z"
                    ),
                    ENTITY_BATCH_LIMIT,
                )

                fact_rows = await self._pull_table(
                    local_db,
                    "facts",
                    progress.get("facts", {}).get(
                        "last_pulled_at", "1970-01-01T00:00:00Z"
                    ),
                    FACT_BATCH_LIMIT,
                )
            finally:
                await local_db.close()

            # If nothing new, return zeros
            if not interaction_rows and not entity_rows and not fact_rows:
                logger.info("hive_pull_noop", reason="no_new_records")
                return {"interactions": 0, "entities": 0, "facts": 0}

            # -------------------------------------------------------
            # Step 4: Push collected records to hive
            # -------------------------------------------------------
            counts = await self._hive_db.push_records(
                interactions=interaction_rows,
                entities=entity_rows,
                facts=fact_rows,
            )

            # -------------------------------------------------------
            # Step 5: Update progress high-water marks in hive
            # -------------------------------------------------------
            now_iso = datetime.now(timezone.utc).isoformat()

            if interaction_rows:
                last_ts = interaction_rows[-1].get("created_at", now_iso)
                await self._update_progress(
                    "interactions", last_ts, len(interaction_rows)
                )
            if entity_rows:
                last_ts = entity_rows[-1].get("created_at", now_iso)
                await self._update_progress(
                    "entities", last_ts, len(entity_rows)
                )
            if fact_rows:
                last_ts = fact_rows[-1].get("created_at", now_iso)
                await self._update_progress(
                    "facts", last_ts, len(fact_rows)
                )

            logger.info(
                "hive_pull_complete",
                instance=self._instance_name,
                interactions=counts.get("interactions", 0),
                entities=counts.get("entities", 0),
                facts=counts.get("facts", 0),
            )
            return counts

        except SyncError:
            raise
        except Exception as exc:
            logger.error(
                "hive_pull_failed",
                error_code="CTXMTG-SYN-004",
                error=str(exc),
            )
            raise SyncError(
                f"Hive pull failed: {exc}",
                error_code="CTXMTG-SYN-004",
            ) from exc

    # =================================================================
    # Status reporting
    # =================================================================

    async def get_status(self) -> dict[str, Any]:
        """
        Return pull status including progress per table and suspended state.

        Returns:
            Dict with instance_name, local_db_path, suspended flag,
            and per-table progress (last_pulled_at, records_pulled).
        """
        progress = await self._get_progress()
        return {
            "instance_name": self._instance_name,
            "local_db_path": self._local_db_path,
            "suspended": self._suspended,
            "progress": progress,
        }

    # =================================================================
    # Private helpers
    # =================================================================

    async def _get_progress(self) -> dict[str, dict[str, Any]]:
        """
        Read hive_pull_progress from hive DB.

        Returns a dict keyed by table_name, each containing
        last_pulled_at, last_row_id, and records_pulled.
        Returns empty dict if the progress table doesn't exist yet.
        """
        try:
            rows = await self._hive_db.execute_sql(
                "SELECT table_name, last_pulled_at, last_row_id, records_pulled "
                "FROM hive_pull_progress",
                {},
            )
            return {
                r["table_name"]: {
                    "last_pulled_at": r["last_pulled_at"],
                    "last_row_id": r["last_row_id"],
                    "records_pulled": r["records_pulled"],
                }
                for r in rows
            }
        except Exception:
            # Table might not exist yet on first run -- that's fine,
            # we just return empty progress so everything gets pulled.
            return {}

    async def _pull_table(
        self,
        local_db: aiosqlite.Connection,
        table: str,
        last_pulled_at: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Pull new records from a local table since last_pulled_at.

        Opens the local DB read-only and selects rows newer than
        the high-water mark, ordered by created_at ASC.

        Args:
            local_db:       The read-only aiosqlite connection.
            table:          Table name (interactions, entities, or facts).
            last_pulled_at: ISO timestamp -- only pull rows after this.
            limit:          Maximum number of rows to pull.

        Returns:
            List of row dicts, ordered by created_at ASC.
        """
        cursor = await local_db.execute(
            f"SELECT * FROM {table} WHERE created_at > ? "  # noqa: S608
            f"ORDER BY created_at ASC LIMIT ?",
            (last_pulled_at, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def _update_progress(
        self,
        table_name: str,
        last_pulled_at: str,
        records_count: int,
    ) -> None:
        """
        Update hive_pull_progress with new high-water mark.

        Uses INSERT OR REPLACE to upsert the progress record.
        The records_pulled counter accumulates across pull cycles
        (COALESCE adds to existing count).

        Args:
            table_name:     Which table's progress to update.
            last_pulled_at: New high-water mark timestamp.
            records_count:  Number of records pulled in this cycle.
        """
        await self._hive_db.execute_sql(
            "INSERT OR REPLACE INTO hive_pull_progress "
            "(table_name, last_pulled_at, records_pulled) "
            "VALUES (:table, :ts, "
            "COALESCE("
            "(SELECT records_pulled FROM hive_pull_progress "
            "WHERE table_name = :table), 0) + :cnt)",
            {"table": table_name, "ts": last_pulled_at, "cnt": records_count},
        )
