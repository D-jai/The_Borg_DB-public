# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Hive Sync Worker (Instance Side) -- Outbox Push
=================================================

2026-04-08: Redesigned to use outbox pattern.  The local writes
JSON manifests to its own outbox directory instead of directly
opening the hive database.  The hive pulls from the outbox via
user-configured "links" in the hive UI.

The sync algorithm (outbox model):
    1. Read high-water marks from local outbox_progress table.
    2. Query local distiller_summaries WHERE cycle_id > last_synced_cycle.
    3. Query local meta_insights WHERE created_at > last_synced_at.
    4. Write a JSON manifest to the outbox directory.
    5. Update outbox_progress in the local DB.

The local has NO knowledge of where the hive is.  It just writes
to its outbox.  The hive discovers locals via "links" that point
to each local's outbox path.

Sync is user-triggered: ``ctxmtg hive push``.  No polling, no daemon.
Incremental via high-water marks stored locally.

ORIGINAL (pre-2026-04-08): Pushed directly to hive.db via
HiveDatabase.push_intelligence().  See commented-out code below.

Depends on:
    - ctxmtg.interfaces.storage (SQLStore for local store queries)
    - ctxmtg.sync.outbox_writer (OutboxWriter for manifest files)
    - ctxmtg.exceptions (SyncError for error reporting)

Used by:
    - ctxmtg.cli (``ctxmtg hive push`` command)
    - ctxmtg.web.routes.dashboard (Push to Hive button)
"""

from __future__ import annotations

# ORIGINAL imports (disabled 2026-04-08): direct hive DB push
# import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from ctxmtg.exceptions import SyncError
from ctxmtg.sync.outbox_writer import OutboxWriter

logger = structlog.get_logger("ctxmtg.sync.hive_sync")

DISTILLER_BATCH_LIMIT = 500
INSIGHT_BATCH_LIMIT = 200


class HiveSyncWorker:
    """
    Instance-side sync worker: writes intelligence to the local outbox.

    2026-04-08: Redesigned from direct hive DB push to outbox pattern.
    Reads distiller_summaries and meta_insights from the local store
    and writes them as a JSON manifest to the outbox directory.

    The local tracks its own high-water marks in the outbox_progress
    table (no dependency on hive).

    Usage:
        writer = OutboxWriter(outbox_path, instance_name)
        worker = HiveSyncWorker(local_store, writer)
        counts = await worker.sync()
    """

    # ORIGINAL constructor (disabled 2026-04-08): took hive_db
    # def __init__(
    #     self,
    #     local_store: Any,
    #     hive_db: HiveDatabase,
    #     instance_name: str = "local",
    # ) -> None:
    #     self._local_store = local_store
    #     self._hive_db = hive_db
    #     self._instance_name = instance_name

    def __init__(
        self,
        local_store: Any,
        outbox_writer: OutboxWriter,
    ) -> None:
        self._local_store = local_store
        self._outbox_writer = outbox_writer

    async def sync(self) -> dict[str, Any]:
        """
        Run one outbox sync cycle.

        Reads high-water marks from local outbox_progress, fetches
        new intelligence, writes a manifest to the outbox, and
        updates outbox_progress.

        Returns:
            Dict with counts and manifest path:
            {"summaries": n, "insights": n, "manifest": str}
        """
        try:
            db = self._local_store._ensure_db()

            # ----------------------------------------------------------
            # Step 1: Read high-water marks from local outbox_progress
            # ----------------------------------------------------------
            last_cycle = 0
            last_insight_at = "1970-01-01T00:00:00Z"

            try:
                cursor = await db.execute(
                    "SELECT last_synced_cycle FROM outbox_progress "
                    "WHERE table_name = 'distiller_summaries'"
                )
                row = await cursor.fetchone()
                if row:
                    last_cycle = row[0] or 0
            except Exception:
                pass  # Table may not exist yet

            try:
                cursor = await db.execute(
                    "SELECT last_synced_at FROM outbox_progress "
                    "WHERE table_name = 'meta_insights'"
                )
                row = await cursor.fetchone()
                if row and row[0]:
                    last_insight_at = row[0]
            except Exception:
                pass  # Table may not exist yet

            # ----------------------------------------------------------
            # Step 2: Fetch new intelligence from local DB
            # ----------------------------------------------------------
            try:
                summaries = await self._fetch_distiller_summaries(
                    db, last_cycle, DISTILLER_BATCH_LIMIT
                )
            except Exception:
                summaries = []
            try:
                insights = await self._fetch_meta_insights(
                    db, last_insight_at, INSIGHT_BATCH_LIMIT
                )
            except Exception:
                insights = []

            if not summaries and not insights:
                logger.info("outbox_sync_noop", reason="no_new_intelligence")
                return {"summaries": 0, "insights": 0, "manifest": None}

            # ----------------------------------------------------------
            # Step 2b: Collect local metadata snapshot
            # ----------------------------------------------------------
            local_metadata = await self._collect_local_metadata(db)

            # ----------------------------------------------------------
            # Step 3: Write manifest to outbox
            # ----------------------------------------------------------
            high_water_marks = {
                "distiller_summaries": {"last_cycle": last_cycle},
                "meta_insights": {"last_synced_at": last_insight_at},
            }

            manifest_path = self._outbox_writer.write_manifest(
                summaries=summaries,
                insights=insights,
                high_water_marks=high_water_marks,
                local_metadata=local_metadata,
            )

            # ----------------------------------------------------------
            # Step 4: Update local outbox_progress
            # ----------------------------------------------------------
            now_iso = datetime.now(timezone.utc).isoformat()
            batch_id = manifest_path.stem  # filename without .json

            if summaries:
                max_cycle = max(s.get("cycle_id", 0) for s in summaries)
                await db.execute(
                    "INSERT OR REPLACE INTO outbox_progress "
                    "(table_name, last_synced_cycle, last_synced_at, "
                    "records_sent, last_batch_id) "
                    "VALUES ('distiller_summaries', :cycle, :at, "
                    "COALESCE((SELECT records_sent FROM outbox_progress "
                    "WHERE table_name = 'distiller_summaries'), 0) + :cnt, "
                    ":batch_id)",
                    {
                        "cycle": max_cycle,
                        "at": now_iso,
                        "cnt": len(summaries),
                        "batch_id": batch_id,
                    },
                )

            if insights:
                max_created = max(
                    i.get("created_at", "") for i in insights
                )
                await db.execute(
                    "INSERT OR REPLACE INTO outbox_progress "
                    "(table_name, last_synced_cycle, last_synced_at, "
                    "records_sent, last_batch_id) "
                    "VALUES ('meta_insights', 0, :at, "
                    "COALESCE((SELECT records_sent FROM outbox_progress "
                    "WHERE table_name = 'meta_insights'), 0) + :cnt, "
                    ":batch_id)",
                    {
                        "at": max_created,
                        "cnt": len(insights),
                        "batch_id": batch_id,
                    },
                )

            await db.commit()

            logger.info(
                "outbox_sync_complete",
                instance=self._outbox_writer.instance_name,
                summaries=len(summaries),
                insights=len(insights),
                manifest=str(manifest_path),
            )

            return {
                "summaries": len(summaries),
                "insights": len(insights),
                "manifest": str(manifest_path),
            }

        except SyncError:
            raise
        except Exception as exc:
            logger.error(
                "outbox_sync_failed",
                error_code="CTXMTG-SYN-003",
                error=str(exc),
            )
            raise SyncError(
                f"Outbox sync failed: {exc}",
                error_code="CTXMTG-SYN-003",
            ) from exc

    # ORIGINAL sync() (disabled 2026-04-08): direct push to hive DB
    # async def sync(self, max_retries: int = 3) -> dict[str, int]:
    #     try:
    #         progress = await self._hive_db.get_sync_progress(
    #             self._instance_name
    #         )
    #         last_cycle = (
    #             progress.get("distiller_summaries", {})
    #             .get("last_synced_cycle", 0)
    #         )
    #         last_insight_at = (
    #             progress.get("meta_insights", {})
    #             .get("last_synced_at", "1970-01-01T00:00:00Z")
    #         )
    #         db = self._local_store._ensure_db()
    #         try:
    #             summaries = await self._fetch_distiller_summaries(
    #                 db, last_cycle, DISTILLER_BATCH_LIMIT
    #             )
    #         except Exception:
    #             summaries = []
    #         try:
    #             insights = await self._fetch_meta_insights(
    #                 db, last_insight_at, INSIGHT_BATCH_LIMIT
    #             )
    #         except Exception:
    #             insights = []
    #         if not summaries and not insights:
    #             logger.info("hive_sync_noop", reason="no_new_intelligence")
    #             return {"profiles": 0, "insights": 0}
    #         counts = await self._push_with_retry(
    #             summaries, insights, max_retries
    #         )
    #         if counts is None:
    #             return {"profiles": 0, "insights": 0}
    #         if summaries:
    #             max_cycle = max(s.get("cycle_id", 0) for s in summaries)
    #             await self._hive_db.update_sync_progress(
    #                 self._instance_name,
    #                 "distiller_summaries",
    #                 max_cycle,
    #                 len(summaries),
    #             )
    #         if insights:
    #             max_created = max(
    #                 i.get("created_at", "") for i in insights
    #             )
    #             await self._hive_db.update_sync_progress(
    #                 self._instance_name,
    #                 "meta_insights",
    #                 0,
    #                 len(insights),
    #             )
    #         logger.info(
    #             "hive_sync_complete",
    #             instance=self._instance_name,
    #             profiles=counts.get("profiles", 0),
    #             insights=counts.get("insights", 0),
    #         )
    #         return counts
    #     except SyncError:
    #         raise
    #     except Exception as exc:
    #         logger.error(
    #             "hive_sync_cycle_failed",
    #             error_code="CTXMTG-SYN-003",
    #             error=str(exc),
    #         )
    #         raise SyncError(
    #             f"Hive sync cycle failed: {exc}",
    #             error_code="CTXMTG-SYN-003",
    #         ) from exc

    # ORIGINAL get_status() (disabled 2026-04-08): queried hive DB
    # async def get_status(self) -> dict[str, Any]:
    #     try:
    #         db = self._local_store._ensure_db()
    #         try:
    #             cursor = await db.execute(
    #                 "SELECT COUNT(*) FROM distiller_summaries"
    #             )
    #             row = await cursor.fetchone()
    #             local_summaries = row[0] if row else 0
    #         except Exception:
    #             local_summaries = 0
    #         try:
    #             cursor = await db.execute(
    #                 "SELECT COUNT(*) FROM meta_insights"
    #             )
    #             row = await cursor.fetchone()
    #             local_insights = row[0] if row else 0
    #         except Exception:
    #             local_insights = 0
    #         hive_counts = await self._hive_db.get_record_counts()
    #         hive_status = await self._hive_db.get_status()
    #         return {
    #             "instance_name": self._instance_name,
    #             "local_summaries": local_summaries,
    #             "local_insights": local_insights,
    #             "hive_record_counts": hive_counts,
    #             "source_instances": hive_status.get("source_instances", []),
    #         }
    #     except Exception as exc:
    #         logger.error(
    #             "hive_sync_status_failed",
    #             error_code="CTXMTG-SYN-001",
    #             error=str(exc),
    #         )
    #         raise SyncError(
    #             f"Failed to get sync status: {exc}",
    #             error_code="CTXMTG-SYN-001",
    #         ) from exc

    # =================================================================
    # Private helpers (fetch methods unchanged -- still read from local)
    # =================================================================

    @staticmethod
    async def _fetch_distiller_summaries(
        db: Any,
        last_cycle: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch distiller summaries newer than last_cycle."""
        cursor = await db.execute(
            "SELECT * FROM distiller_summaries "
            "WHERE cycle_id > :last_cycle "
            "ORDER BY cycle_id ASC LIMIT :limit",
            {"last_cycle": last_cycle, "limit": limit},
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    async def _fetch_meta_insights(
        db: Any,
        last_synced_at: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch meta insights created after last_synced_at."""
        cursor = await db.execute(
            "SELECT * FROM meta_insights "
            "WHERE created_at > :last_at "
            "ORDER BY created_at ASC LIMIT :limit",
            {"last_at": last_synced_at, "limit": limit},
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def _collect_local_metadata(self, db: Any) -> dict[str, Any]:
        """
        Collect a snapshot of this local instance's state.

        Included in every manifest so the hive knows what it's
        pulling from: schema version, table sizes, profile config,
        farming state, platform info, LLM config, outbox history.
        """
        import os
        import platform
        import sys

        metadata: dict[str, Any] = {}

        # -- Identity & versions ----------------------------------------
        metadata["instance_name"] = self._outbox_writer.instance_name
        metadata["collected_at"] = datetime.now(timezone.utc).isoformat()

        try:
            row = await (await db.execute("PRAGMA user_version")).fetchone()
            metadata["schema_version"] = row[0] if row else None
        except Exception:
            metadata["schema_version"] = None

        # Software version (from package if available)
        try:
            from importlib.metadata import version as pkg_version
            metadata["software_version"] = pkg_version("ctxmtg")
        except Exception:
            metadata["software_version"] = "dev"

        # -- Database state ---------------------------------------------
        table_counts: dict[str, int] = {}
        for table in [
            "interactions", "entities", "facts",
            "meta_insights", "distiller_summaries",
            "farming_cycles", "embeddings_metadata",
        ]:
            try:
                row = await (await db.execute(
                    f"SELECT COUNT(*) FROM {table}"  # noqa: S608
                )).fetchone()
                table_counts[table] = row[0] if row else 0
            except Exception:
                table_counts[table] = -1  # table doesn't exist
        metadata["table_counts"] = table_counts

        # All table names in the DB
        try:
            rows = await (await db.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' ORDER BY name"
            )).fetchall()
            metadata["table_names"] = [r[0] for r in rows]
        except Exception:
            metadata["table_names"] = []

        # DB file size
        try:
            db_path = self._local_store._db_path
            if db_path and os.path.exists(db_path):
                metadata["db_size_bytes"] = os.path.getsize(db_path)
        except Exception:
            pass

        # -- Farming state ----------------------------------------------
        try:
            row = await (await db.execute(
                "SELECT MAX(cycle_id) as last_cycle, "
                "COUNT(*) as total_cycles, "
                "MAX(started_at) as last_farmed_at "
                "FROM farming_cycles"
            )).fetchone()
            metadata["farming"] = {
                "last_cycle_id": row[0] if row else None,
                "total_cycles": row[1] if row else 0,
                "last_farmed_at": row[2] if row else None,
            }
        except Exception:
            metadata["farming"] = {}

        # -- Profile & entity types -------------------------------------
        try:
            from ctxmtg.config.settings import CtxMtgSettings
            settings = CtxMtgSettings()
            metadata["profile"] = {
                "name": settings.profile_name,
            }

            # Load profile to get entity types, description, version, tags
            try:
                from ctxmtg.profile.loader import ProfileLoader
                profile = ProfileLoader().load(settings.profile_name)
                if hasattr(profile, "ner") and hasattr(profile.ner, "entity_types"):
                    metadata["profile"]["entity_types"] = list(
                        profile.ner.entity_types
                    )
                if hasattr(profile, "description"):
                    metadata["profile"]["description"] = profile.description
                if hasattr(profile, "version"):
                    metadata["profile"]["version"] = profile.version
                if hasattr(profile, "tags"):
                    metadata["profile"]["tags"] = profile.tags
            except Exception:
                pass

            # Embedding model
            metadata["embedding"] = {
                "model": settings.embedding_model,
            }

            # LLM config (which roles are configured, model names)
            try:
                llm_roles: dict[str, str] = {}
                for role_name in [
                    "extraction", "query_planning", "retrieval",
                    "synthesis", "farming", "fusion",
                ]:
                    role = getattr(settings.llm, role_name, None)
                    if role and role.model:
                        llm_roles[role_name] = role.model
                metadata["llm_roles"] = llm_roles
            except Exception:
                metadata["llm_roles"] = {}

        except Exception:
            metadata["profile"] = {}
            metadata["embedding"] = {}
            metadata["llm_roles"] = {}

        # -- Vector store -----------------------------------------------
        try:
            metadata["embedding"]["total_vectors"] = table_counts.get(
                "embeddings_metadata", 0
            )
        except Exception:
            pass

        # -- Platform ---------------------------------------------------
        metadata["platform"] = {
            "python_version": sys.version.split()[0],
            "os": platform.system(),
            "os_version": platform.release(),
            "architecture": platform.machine(),
            "platform_detail": platform.platform(),
        }

        # -- Outbox history ---------------------------------------------
        try:
            rows = await (await db.execute(
                "SELECT table_name, records_sent, last_batch_id "
                "FROM outbox_progress"
            )).fetchall()
            metadata["outbox_history"] = {
                r[0]: {"records_sent": r[1], "last_batch_id": r[2]}
                for r in rows
            }
        except Exception:
            metadata["outbox_history"] = {}

        return metadata

    # ORIGINAL _push_with_retry (disabled 2026-04-08): direct hive push
    # async def _push_with_retry(
    #     self,
    #     summaries: list[dict[str, Any]],
    #     insights: list[dict[str, Any]],
    #     max_retries: int,
    # ) -> dict[str, int] | None:
    #     for attempt in range(max_retries):
    #         try:
    #             result = await self._hive_db.push_intelligence(
    #                 distiller_summaries=summaries,
    #                 meta_insights=insights,
    #                 local_name=self._instance_name,
    #             )
    #             if attempt > 0:
    #                 logger.info(
    #                     "hive_push_retry_success",
    #                     attempt=attempt + 1,
    #                 )
    #             return result
    #         except Exception as exc:
    #             logger.warning(
    #                 "hive_push_retry_failed",
    #                 error_code="CTXMTG-SYN-003",
    #                 attempt=attempt + 1,
    #                 max_retries=max_retries,
    #                 error=str(exc),
    #             )
    #             if attempt < max_retries - 1:
    #                 await asyncio.sleep(2 ** attempt)
    #     logger.error(
    #         "hive_push_exhausted",
    #         error_code="CTXMTG-SYN-003",
    #         max_retries=max_retries,
    #     )
    #     return None
