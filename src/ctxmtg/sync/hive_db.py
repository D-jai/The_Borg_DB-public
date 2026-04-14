# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Hive Database Connection Manager
=================================

This module manages the hive database -- the central intelligence
aggregator that merges distilled knowledge from one or more ctxmtg
instances.  The hive is always a *separate* database from the local
store, even on single-instance deployments.

Phase 5 redesign: the hive is an intelligence aggregator, not a
data mirror.  It receives distiller summaries and meta insights
from locals, merges them into unified entity profiles, and mines
the collective for patterns no single local can see.

What syncs TO the hive (intelligence layer):
    - Entities (name, type, tags, confidence) -- identity layer
    - Distiller summaries -- per-entity intelligence from local farming
    - Meta insights -- patterns discovered by local farming
    - Interaction metadata (id, source_type, title, created_at -- NO content)

What stays LOCAL (raw data):
    - Full interaction content (transcripts, emails, chat logs)
    - Raw SPO facts
    - Source spans

The hive schema uses four tables:
    - hive_entity_profiles: merged entity view across all locals
    - hive_insights: collective insight pool from all locals
    - hive_native_insights: patterns only visible at aggregate level
    - hive_sync_progress: high-water marks per local per table

Depends on:
    - aiosqlite (async SQLite driver for local mode)
    - ctxmtg.storage.schema (PRAGMAS for connection settings)
    - ctxmtg.exceptions (SyncError for error reporting)

Used by:
    - ctxmtg.sync.hive_sync (pushes intelligence here)
    - ctxmtg.sync.intelligence_merger (merges entity profiles)
    - ctxmtg.sync.hive_farming (runs hive-native farming)
    - ctxmtg.cli (hive commands)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import structlog

from ctxmtg.exceptions import SyncError
from ctxmtg.storage.schema import PRAGMAS

logger = structlog.get_logger("ctxmtg.sync.hive_db")

# =====================================================================
# Hive Intelligence Schema (Phase 5 redesign)
# =====================================================================

HIVE_CREATE_ENTITY_PROFILES = """\
CREATE TABLE IF NOT EXISTS hive_entity_profiles (
    entity_name       TEXT PRIMARY KEY,
    entity_type       TEXT NOT NULL,
    merged_summary    TEXT NOT NULL,
    top_predicates    TEXT NOT NULL DEFAULT '[]',
    top_co_entities   TEXT NOT NULL DEFAULT '[]',
    total_mentions    INTEGER NOT NULL DEFAULT 0,
    source_streams    TEXT NOT NULL DEFAULT '[]',
    stream_count      INTEGER NOT NULL DEFAULT 1,
    cross_stream_score REAL NOT NULL DEFAULT 0.0,
    merged_tags       TEXT NOT NULL DEFAULT '{}',
    last_updated      TEXT NOT NULL,
    archived_on       TEXT DEFAULT '[]'
);
"""

HIVE_CREATE_INSIGHTS = """\
CREATE TABLE IF NOT EXISTS hive_insights (
    id              TEXT PRIMARY KEY,
    insight_type    TEXT NOT NULL,
    title           TEXT NOT NULL,
    description     TEXT,
    confidence      REAL NOT NULL DEFAULT 1.0,
    parameters      TEXT DEFAULT '{}',
    entity_names    TEXT DEFAULT '[]',
    source_instance TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    correlation_id  TEXT
);
"""

HIVE_CREATE_NATIVE_INSIGHTS = """\
CREATE TABLE IF NOT EXISTS hive_native_insights (
    id              TEXT PRIMARY KEY,
    insight_type    TEXT NOT NULL,
    title           TEXT NOT NULL,
    description     TEXT,
    confidence      REAL NOT NULL DEFAULT 1.0,
    parameters      TEXT DEFAULT '{}',
    entity_names    TEXT DEFAULT '[]',
    created_at      TEXT NOT NULL
);
"""

HIVE_CREATE_SYNC_PROGRESS = """\
CREATE TABLE IF NOT EXISTS hive_sync_progress (
    local_name          TEXT NOT NULL,
    table_name          TEXT NOT NULL,
    last_synced_cycle   INTEGER DEFAULT 0,
    last_synced_at      TEXT,
    records_synced      INTEGER DEFAULT 0,
    PRIMARY KEY (local_name, table_name)
);
"""

HIVE_CREATE_INDEXES = """\
CREATE INDEX IF NOT EXISTS idx_hive_profiles_type
    ON hive_entity_profiles(entity_type);
CREATE INDEX IF NOT EXISTS idx_hive_profiles_score
    ON hive_entity_profiles(cross_stream_score DESC);
CREATE INDEX IF NOT EXISTS idx_hive_insights_type
    ON hive_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_hive_insights_source
    ON hive_insights(source_instance);
CREATE INDEX IF NOT EXISTS idx_hive_native_type
    ON hive_native_insights(insight_type);
"""

# 2026-04-08: Links table -- hive-side registry of known local instances.
# Each link = (user-chosen name, path to that local's outbox).
# Locals don't know about the hive; hive discovers locals via links.
HIVE_CREATE_LINKS = """\
CREATE TABLE IF NOT EXISTS hive_links (
    link_id        TEXT PRIMARY KEY,
    local_name     TEXT NOT NULL UNIQUE,
    outbox_path    TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    last_pulled_at TEXT,
    total_batches  INTEGER DEFAULT 0,
    notes          TEXT DEFAULT ''
);
"""

HIVE_ALL_DDL: list[str] = [
    HIVE_CREATE_ENTITY_PROFILES,
    HIVE_CREATE_INSIGHTS,
    HIVE_CREATE_NATIVE_INSIGHTS,
    HIVE_CREATE_SYNC_PROGRESS,
    HIVE_CREATE_LINKS,
    HIVE_CREATE_INDEXES,
]


class HiveDatabase:
    """
    Hive database connection manager (intelligence aggregator model).

    Opens a SQLite database file for the hive.  The hive schema stores
    merged entity profiles, collective insights from all locals, and
    hive-native intelligence discovered at the aggregate level.

    Usage:
        hive = HiveDatabase(local_db_path="/path/to/hive.db")
        await hive.initialize()
        await hive.push_intelligence(profiles, insights, "laptop")
        await hive.close()
    """

    def __init__(
        self,
        mode: str = "local",
        local_db_path: str | None = None,
        local_vector_path: str | None = None,
    ) -> None:
        self._mode = mode
        self._local_db_path = local_db_path
        self._local_vector_path = local_vector_path
        self._db: aiosqlite.Connection | None = None

    @property
    def mode(self) -> str:
        return self._mode

    async def initialize(self) -> None:
        """Create hive tables if they don't exist."""
        if not self._local_db_path:
            logger.error("hive_db_no_path", error_code="CTXMTG-SYN-001")
            raise SyncError(
                "local_db_path is required for local hive mode",
                error_code="CTXMTG-SYN-001",
            )

        try:
            self._db = await aiosqlite.connect(self._local_db_path)
            self._db.row_factory = aiosqlite.Row

            for pragma in PRAGMAS.strip().splitlines():
                pragma = pragma.strip()
                if pragma:
                    await self._db.execute(pragma)

            full_script = "\n".join(HIVE_ALL_DDL)
            await self._db.executescript(full_script)
            await self._db.commit()

            logger.info(
                "hive_db_initialized",
                mode=self._mode,
                db_path=self._local_db_path,
            )
        except Exception as exc:
            logger.error(
                "hive_db_init_failed",
                error_code="CTXMTG-SYN-001",
                db_path=self._local_db_path,
                error=str(exc),
            )
            raise SyncError(
                f"Failed to initialise hive database at "
                f"{self._local_db_path}: {exc}",
                error_code="CTXMTG-SYN-001",
            ) from exc

    async def close(self) -> None:
        """Close the hive database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.info("hive_db_closed", mode=self._mode)

    def _ensure_db(self) -> aiosqlite.Connection:
        """Return the active connection or raise SyncError."""
        if self._db is None:
            logger.error("hive_db_not_initialized", error_code="CTXMTG-SYN-001")
            raise SyncError(
                "HiveDatabase is not initialised. Call initialize() first.",
                error_code="CTXMTG-SYN-001",
            )
        return self._db

    # =================================================================
    # Push intelligence from a local instance
    # =================================================================

    async def push_intelligence(
        self,
        distiller_summaries: list[dict[str, Any]],
        meta_insights: list[dict[str, Any]],
        local_name: str,
    ) -> dict[str, int]:
        """
        Push distilled intelligence from a local instance to the hive.

        Receives distiller summaries and meta insights from a local
        and upserts them into the hive's intelligence tables.

        Args:
            distiller_summaries: Distiller summary dicts from the local.
            meta_insights: Meta insight dicts from the local.
            local_name: Name of the source local instance.

        Returns:
            Dict with counts: {"profiles": n, "insights": n}
        """
        db = self._ensure_db()
        counts = {"profiles": 0, "insights": 0}

        try:
            await db.execute("BEGIN IMMEDIATE")

            now_iso = datetime.now(timezone.utc).isoformat()

            for summary in distiller_summaries:
                entity_name = summary["entity_name"]

                # Check if profile already exists
                cursor = await db.execute(
                    "SELECT source_streams, total_mentions FROM hive_entity_profiles "
                    "WHERE entity_name = :name",
                    {"name": entity_name},
                )
                existing = await cursor.fetchone()

                if existing:
                    # Merge into existing profile
                    existing_streams = json.loads(existing["source_streams"])
                    if local_name not in existing_streams:
                        existing_streams.append(local_name)

                    new_mentions = (
                        existing["total_mentions"]
                        + summary.get("total_mentions", 1)
                    )

                    await db.execute(
                        "UPDATE hive_entity_profiles SET "
                        "merged_summary = :summary, "
                        "top_predicates = :preds, "
                        "top_co_entities = :co_ents, "
                        "total_mentions = :mentions, "
                        "source_streams = :streams, "
                        "stream_count = :stream_count, "
                        "merged_tags = :tags, "
                        "last_updated = :updated "
                        "WHERE entity_name = :name",
                        {
                            "summary": summary.get("summary", ""),
                            "preds": summary.get("top_predicates", "[]"),
                            "co_ents": summary.get("top_co_entities", "[]"),
                            "mentions": new_mentions,
                            "streams": json.dumps(existing_streams),
                            "stream_count": len(existing_streams),
                            "tags": summary.get("merged_tags", "{}"),
                            "updated": now_iso,
                            "name": entity_name,
                        },
                    )
                else:
                    # Insert new profile
                    streams = [local_name]
                    await db.execute(
                        "INSERT INTO hive_entity_profiles "
                        "(entity_name, entity_type, merged_summary, "
                        "top_predicates, top_co_entities, total_mentions, "
                        "source_streams, stream_count, cross_stream_score, "
                        "merged_tags, last_updated) "
                        "VALUES (:name, :type, :summary, :preds, :co_ents, "
                        ":mentions, :streams, :stream_count, :score, "
                        ":tags, :updated)",
                        {
                            "name": entity_name,
                            "type": summary.get("entity_type", "other"),
                            "summary": summary.get("summary", ""),
                            "preds": summary.get("top_predicates", "[]"),
                            "co_ents": summary.get("top_co_entities", "[]"),
                            "mentions": summary.get("total_mentions", 1),
                            "streams": json.dumps(streams),
                            "stream_count": 1,
                            "score": 0.0,
                            "tags": summary.get("merged_tags", "{}"),
                            "updated": now_iso,
                        },
                    )
                counts["profiles"] += 1

            for insight in meta_insights:
                cursor = await db.execute(
                    "INSERT OR IGNORE INTO hive_insights "
                    "(id, insight_type, title, description, confidence, "
                    "parameters, entity_names, source_instance, created_at) "
                    "VALUES (:id, :type, :title, :desc, :conf, "
                    ":params, :names, :source, :created)",
                    {
                        "id": insight["id"],
                        "type": insight.get("insight_type", "meta"),
                        "title": insight.get("title", ""),
                        "desc": insight.get("description"),
                        "conf": insight.get("confidence", 1.0),
                        "params": insight.get("parameters", "{}"),
                        "names": insight.get("entity_names", "[]"),
                        "source": local_name,
                        "created": insight.get("created_at", now_iso),
                    },
                )
                counts["insights"] += cursor.rowcount

            await db.commit()

            logger.info(
                "hive_intelligence_pushed",
                local=local_name,
                profiles=counts["profiles"],
                insights=counts["insights"],
            )
        except Exception as exc:
            await db.rollback()
            logger.error(
                "hive_push_failed",
                error_code="CTXMTG-SYN-003",
                error=str(exc),
            )
            raise SyncError(
                f"Failed to push intelligence to hive: {exc}",
                error_code="CTXMTG-SYN-003",
            ) from exc

        return counts

    # =================================================================
    # Read operations for hive farming and intelligence pull
    # =================================================================

    async def get_entity_profiles(
        self,
        min_score: float = 0.0,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return entity profiles above a minimum cross-stream score."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM hive_entity_profiles "
            "WHERE cross_stream_score >= :min_score "
            "ORDER BY cross_stream_score DESC LIMIT :limit",
            {"min_score": min_score, "limit": limit},
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def get_all_entity_profiles(self) -> list[dict[str, Any]]:
        """Return all entity profiles."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM hive_entity_profiles ORDER BY entity_name"
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def get_insights(
        self,
        source_instance: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return hive insights, optionally filtered by source."""
        db = self._ensure_db()
        if source_instance:
            cursor = await db.execute(
                "SELECT * FROM hive_insights "
                "WHERE source_instance = :source "
                "ORDER BY created_at DESC LIMIT :limit",
                {"source": source_instance, "limit": limit},
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM hive_insights "
                "ORDER BY created_at DESC LIMIT :limit",
                {"limit": limit},
            )
        return [dict(row) for row in await cursor.fetchall()]

    async def get_native_insights(
        self,
        insight_type: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return hive-native insights, optionally filtered by type."""
        db = self._ensure_db()
        if insight_type:
            cursor = await db.execute(
                "SELECT * FROM hive_native_insights "
                "WHERE insight_type = :type "
                "ORDER BY created_at DESC LIMIT :limit",
                {"type": insight_type, "limit": limit},
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM hive_native_insights "
                "ORDER BY created_at DESC LIMIT :limit",
                {"limit": limit},
            )
        return [dict(row) for row in await cursor.fetchall()]

    async def insert_native_insight(
        self,
        insight: dict[str, Any],
    ) -> None:
        """Insert a hive-native insight (from hive farming stages)."""
        db = self._ensure_db()
        await db.execute(
            "INSERT OR IGNORE INTO hive_native_insights "
            "(id, insight_type, title, description, confidence, "
            "parameters, entity_names, created_at) "
            "VALUES (:id, :type, :title, :desc, :conf, "
            ":params, :names, :created)",
            {
                "id": insight["id"],
                "type": insight["insight_type"],
                "title": insight["title"],
                "desc": insight.get("description"),
                "conf": insight.get("confidence", 1.0),
                "params": insight.get("parameters", "{}"),
                "names": insight.get("entity_names", "[]"),
                "created": insight.get(
                    "created_at",
                    datetime.now(timezone.utc).isoformat(),
                ),
            },
        )
        await db.commit()

    async def update_entity_profile(
        self,
        entity_name: str,
        updates: dict[str, Any],
    ) -> None:
        """Update specific fields on an entity profile."""
        db = self._ensure_db()
        set_parts = []
        params: dict[str, Any] = {"name": entity_name}
        for key, value in updates.items():
            set_parts.append(f"{key} = :{key}")
            params[key] = value

        if not set_parts:
            return

        sql = (
            f"UPDATE hive_entity_profiles SET {', '.join(set_parts)} "
            f"WHERE entity_name = :name"
        )
        await db.execute(sql, params)
        await db.commit()

    async def set_correlation_id(
        self,
        insight_id: str,
        correlation_id: str,
    ) -> None:
        """Set correlation_id on a hive insight."""
        db = self._ensure_db()
        await db.execute(
            "UPDATE hive_insights SET correlation_id = :cid WHERE id = :iid",
            {"cid": correlation_id, "iid": insight_id},
        )
        await db.commit()

    # =================================================================
    # Sync progress tracking
    # =================================================================

    async def get_sync_progress(
        self,
        local_name: str,
    ) -> dict[str, dict[str, Any]]:
        """Return sync progress for a local instance."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT table_name, last_synced_cycle, last_synced_at, "
            "records_synced FROM hive_sync_progress "
            "WHERE local_name = :local",
            {"local": local_name},
        )
        rows = await cursor.fetchall()
        return {
            r["table_name"]: {
                "last_synced_cycle": r["last_synced_cycle"],
                "last_synced_at": r["last_synced_at"],
                "records_synced": r["records_synced"],
            }
            for r in rows
        }

    async def update_sync_progress(
        self,
        local_name: str,
        table_name: str,
        last_synced_cycle: int,
        records_synced: int,
    ) -> None:
        """Update sync progress high-water mark for a local/table pair."""
        db = self._ensure_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "INSERT OR REPLACE INTO hive_sync_progress "
            "(local_name, table_name, last_synced_cycle, last_synced_at, "
            "records_synced) "
            "VALUES (:local, :table, :cycle, :ts, "
            "COALESCE("
            "(SELECT records_synced FROM hive_sync_progress "
            "WHERE local_name = :local AND table_name = :table), 0) + :cnt)",
            {
                "local": local_name,
                "table": table_name,
                "cycle": last_synced_cycle,
                "ts": now_iso,
                "cnt": records_synced,
            },
        )
        await db.commit()

    # =================================================================
    # Status reporting
    # =================================================================

    async def get_record_counts(self) -> dict[str, int]:
        """Return counts of records in the hive intelligence tables."""
        db = self._ensure_db()
        counts: dict[str, int] = {}
        for table in ("hive_entity_profiles", "hive_insights", "hive_native_insights"):
            cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            row = await cursor.fetchone()
            counts[table] = row[0] if row else 0
        return counts

    async def get_status(self) -> dict[str, Any]:
        """Return a comprehensive hive status report."""
        counts = await self.get_record_counts()
        db = self._ensure_db()

        # Get distinct source streams
        cursor = await db.execute(
            "SELECT DISTINCT source_instance FROM hive_insights"
        )
        sources = [row[0] for row in await cursor.fetchall()]

        return {
            "mode": self._mode,
            "record_counts": counts,
            "source_instances": sources,
        }

    # =================================================================
    # Link management (2026-04-08: outbox pattern)
    # =================================================================

    async def add_link(
        self,
        local_name: str,
        outbox_path: str,
        notes: str = "",
    ) -> dict[str, Any]:
        """
        Register a new local instance link in the hive.

        Args:
            local_name:  User-chosen name for this local (e.g., "Tickets").
            outbox_path: Filesystem path to the local's outbox directory.
            notes:       Optional description or notes about this local.

        Returns:
            Dict with the created link's fields.
        """
        from uuid import uuid4

        db = self._ensure_db()
        link_id = str(uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        await db.execute(
            "INSERT INTO hive_links "
            "(link_id, local_name, outbox_path, created_at, notes) "
            "VALUES (:id, :name, :path, :created, :notes)",
            {
                "id": link_id,
                "name": local_name,
                "path": outbox_path,
                "created": now_iso,
                "notes": notes,
            },
        )
        await db.commit()

        logger.info(
            "hive_link_added",
            link_id=link_id,
            local_name=local_name,
            outbox_path=outbox_path,
        )

        return {
            "link_id": link_id,
            "local_name": local_name,
            "outbox_path": outbox_path,
            "created_at": now_iso,
            "last_pulled_at": None,
            "total_batches": 0,
            "notes": notes,
        }

    async def remove_link(self, link_id: str) -> bool:
        """
        Remove a local instance link from the hive.

        Args:
            link_id: The UUID of the link to remove.

        Returns:
            True if a link was deleted, False if not found.
        """
        db = self._ensure_db()
        cursor = await db.execute(
            "DELETE FROM hive_links WHERE link_id = :id",
            {"id": link_id},
        )
        await db.commit()
        deleted = cursor.rowcount > 0

        if deleted:
            logger.info("hive_link_removed", link_id=link_id)
        return deleted

    async def get_links(self) -> list[dict[str, Any]]:
        """Return all registered local instance links."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM hive_links ORDER BY local_name ASC"
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def get_link(self, link_id: str) -> dict[str, Any] | None:
        """Return a single link by ID, or None if not found."""
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM hive_links WHERE link_id = :id",
            {"id": link_id},
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def update_link_pulled(
        self,
        link_id: str,
        batches_processed: int,
    ) -> None:
        """
        Update a link's pull stats after a successful pull.

        Args:
            link_id:           The link that was pulled.
            batches_processed: Number of manifest files processed.
        """
        db = self._ensure_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "UPDATE hive_links SET "
            "last_pulled_at = :ts, "
            "total_batches = total_batches + :cnt "
            "WHERE link_id = :id",
            {"ts": now_iso, "cnt": batches_processed, "id": link_id},
        )
        await db.commit()

    # =================================================================
    # Execute arbitrary SQL (used by maintenance and hive farming)
    # =================================================================

    async def execute_sql(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a SQL query against the hive database."""
        db = self._ensure_db()
        try:
            cursor = await db.execute(sql, params or {})
            rows = await cursor.fetchall()
            if sql.strip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
                await db.commit()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.error(
                "hive_sql_failed",
                error_code="CTXMTG-SYN-001",
                error=str(exc),
            )
            raise SyncError(
                f"Hive SQL execution failed: {exc}",
                error_code="CTXMTG-SYN-001",
            ) from exc
