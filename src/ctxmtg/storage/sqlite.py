# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
SQLite Storage Layer
====================

This module provides the SQLite implementation of the SQL store.
It handles all structured data: interactions (meetings, conversations),
entities (people, places, topics), facts (who said what about whom),
and metadata that links SQL records to their vector embeddings.

In the system architecture, this is the "left side" of the dual-store
design. The "right side" is the vector store (LanceDB). Together they
answer hybrid queries -- SQL handles precise lookups ("how many meetings
this week?") while vectors handle semantic search ("discussions about
scaling").

Key design decisions (references to research documents):
    - aiosqlite for async I/O so the query server event loop is never
      blocked by database access.
    - WAL mode + PRAGMA busy_timeout = 5000 for concurrent-reader /
      single-writer safety (countermeasures 2.2).
    - BEGIN IMMEDIATE for write transactions to fail fast on contention
      instead of silently retrying (countermeasures 2.2).
    - All entity name queries use COLLATE NOCASE for case-insensitive
      matching (plan spec P1-04).
    - Entity tags are stored as JSON and queryable via json_extract()
      for SQL-level filtering (plan spec P1-04).
    - FTS5 virtual table for full-text keyword search over interactions
      (research/round-2/03-unified-schema-design.md § 5.3).

Depends on:
    - aiosqlite (async SQLite driver)
    - ctxmtg.interfaces.storage (the abstract contract this implements)
    - ctxmtg.models.interaction (data models stored in the database)
    - ctxmtg.models.farming (FarmingInsight model)
    - ctxmtg.storage.schema (DDL constants and migration functions)
    - ctxmtg.exceptions (StorageError for error reporting)

Used by:
    - ctxmtg.ingestion.worker (writes extracted data here)
    - ctxmtg.query.executor (reads data from here during queries)
    - ctxmtg.farming.pipeline (reads accumulated data for pattern mining)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import structlog

from ctxmtg.exceptions import StorageError
from ctxmtg.interfaces.storage import SQLStore
from ctxmtg.models.farming import FarmingInsight
from ctxmtg.models.interaction import (
    EmbeddingMetadata,
    Entity,
    EntityType,
    Fact,
    IntakeAction,
    Interaction,
    SourceType,
)
from ctxmtg.storage.schema import apply_pragmas, apply_schema, migrate

# ---------------------------------------------------------------
# Module-level logger -- structured JSON output, no PII in logs.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.storage.sqlite")


class SQLiteStore(SQLStore):
    """
    SQLite-based storage for structured knowledge data.

    This is the workhorse of the system on edge devices (Pi, laptop).
    SQLite is a file-based database -- no server to install, no
    configuration needed. It stores all the structured information
    the system extracts from your interactions.

    Why SQLite and not PostgreSQL on edge?
    SQLite is 600 KB, runs in-process, and handles up to ~500 K records
    before needing optimisation. PostgreSQL is for the server tier.
    (See research/round-1/02-dual-store-architecture.md for the comparison.)

    Usage:
        store = SQLiteStore(db_path="/path/to/knowledge.db")
        await store.initialize()  # Creates tables if they don't exist
        await store.store_interaction(interaction)
        entities = await store.get_entities(entity_type="person")
        await store.close()
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """
        Prepare the store (does NOT open the connection yet).

        The actual connection is created lazily by initialize().
        Passing ":memory:" gives an ephemeral in-memory database
        that is perfect for unit tests.

        Args:
            db_path: Filesystem path to the SQLite file, or ":memory:".
        """
        self._db_path = db_path

        # The aiosqlite connection — set by initialize(), cleared by close().
        self._db: aiosqlite.Connection | None = None

        # An asyncio lock that serialises write transactions.
        # SQLite supports only one writer at a time; this lock prevents
        # two coroutines from issuing BEGIN IMMEDIATE concurrently.
        self._write_lock = asyncio.Lock()

    # =================================================================
    # Lifecycle
    # =================================================================

    async def initialize(self) -> None:
        """
        Open the database connection, apply pragmas, create schema.

        Steps:
        1. Open an aiosqlite connection to the configured path.
        2. Enable WAL mode, foreign keys, and busy_timeout.
        3. Run all CREATE TABLE / INDEX / TRIGGER statements.
        4. Apply any pending schema migrations.
        """
        try:
            # Open the connection
            self._db = await aiosqlite.connect(self._db_path)

            # Return rows as sqlite3.Row so we can access columns by name
            self._db.row_factory = aiosqlite.Row

            # Apply connection-scoped pragmas (WAL, FK, busy_timeout)
            await apply_pragmas(self._db)

            # Create tables and indexes if this is a fresh database
            await apply_schema(self._db)

            # Apply any pending migrations (v1 → v2, v2 → v3, …)
            await migrate(self._db)

            logger.info("sqlite_initialized", db_path=self._db_path)
        except StorageError:
            raise
        except Exception as exc:
            logger.error(
                "sqlite_init_failed",
                error_code="CTXMTG-STG-001",
                db_path=self._db_path,
                error=str(exc),
            )
            raise StorageError(
                f"Failed to initialise SQLite at {self._db_path}: {exc}",
                error_code="CTXMTG-STG-001",
            ) from exc

    async def close(self) -> None:
        """
        Close the database connection cleanly.

        Flushes any pending WAL pages and releases the file handle.
        After calling close() the store cannot be used until
        initialize() is called again.
        """
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.info("sqlite_closed", db_path=self._db_path)

    # -----------------------------------------------------------------
    # Internal helper: ensure the connection is open.
    # Raises StorageError immediately if someone forgot to initialise.
    # -----------------------------------------------------------------
    def _ensure_db(self) -> aiosqlite.Connection:
        """Return the active connection or raise StorageError."""
        if self._db is None:
            logger.error(
                "sqlite_not_initialized",
                error_code="CTXMTG-STG-001",
                db_path=self._db_path,
            )
            raise StorageError(
                "SQLiteStore is not initialised. Call initialize() first.",
                error_code="CTXMTG-STG-001",
            )
        return self._db

    # =================================================================
    # Interaction CRUD
    # =================================================================

    async def store_interaction(self, interaction: Interaction) -> str:
        """
        Store an interaction. Returns the interaction ID.

        Uses INSERT OR REPLACE so that re-ingesting the same content
        (same deterministic ID) updates the existing row rather than
        failing on a UNIQUE constraint.

        Args:
            interaction: The Interaction object to persist.

        Returns:
            The interaction's ID string.
        """
        db = self._ensure_db()

        # Serialise list/dict fields to JSON strings for TEXT columns
        participants_json = json.dumps(interaction.participants)
        metadata_json = json.dumps(interaction.metadata)
        created_at_str = interaction.created_at.isoformat()
        updated_at_str = (
            interaction.updated_at.isoformat() if interaction.updated_at else created_at_str
        )

        sql = """
            INSERT OR REPLACE INTO interactions
                (id, source_type, source_id, title, content, participants,
                 metadata, source_instance, intake_action,
                 created_at, updated_at, is_deleted)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """
        params = (
            interaction.id,
            interaction.source_type.value,
            interaction.source_id,
            interaction.title,
            interaction.content,
            participants_json,
            metadata_json,
            interaction.source_instance,
            interaction.intake_action.value,
            created_at_str,
            updated_at_str,
        )

        async with self._write_lock:
            try:
                await db.execute("BEGIN IMMEDIATE")
                await db.execute(sql, params)
                await db.commit()
            except Exception as exc:
                await db.rollback()
                logger.error(
                    "interaction_store_failed",
                    error_code="CTXMTG-STG-014",
                    interaction_id=interaction.id,
                    error=str(exc),
                )
                raise StorageError(
                    f"Failed to store interaction {interaction.id}: {exc}",
                    error_code="CTXMTG-STG-014",
                ) from exc

        logger.info("interaction_stored", interaction_id=interaction.id)
        return interaction.id

    async def get_interaction(self, interaction_id: str) -> Interaction | None:
        """
        Retrieve an interaction by ID. Returns None if not found.

        Args:
            interaction_id: The unique ID of the interaction.

        Returns:
            The Interaction model, or None.
        """
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM interactions WHERE id = ? AND is_deleted = 0",
            (interaction_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_interaction(row)

    async def list_interactions(
        self,
        source_type: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Interaction]:
        """
        List interactions with optional filters.

        Filters are ANDed together. Results come back in reverse
        chronological order (most recent first).
        """
        db = self._ensure_db()

        # Build the WHERE clause dynamically from the supplied filters
        clauses: list[str] = ["is_deleted = 0"]
        params: list[Any] = []

        if source_type is not None:
            clauses.append("source_type = ?")
            params.append(source_type)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since)
        if until is not None:
            clauses.append("created_at <= ?")
            params.append(until)

        where = " AND ".join(clauses)
        sql = (
            f"SELECT * FROM interactions WHERE {where} "
            f"ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        return [self._row_to_interaction(r) for r in rows]

    # =================================================================
    # Entity CRUD
    # =================================================================

    async def store_entities(self, entities: list[Entity]) -> int:
        """
        Batch-store entities. Returns count of inserted (non-duplicate) entities.

        All inserts happen inside a single BEGIN IMMEDIATE transaction
        so that SQLite writes them in one disk sync (much faster than
        one transaction per row — see countermeasures 2.2).
        Duplicates (same ID) are silently skipped via INSERT OR IGNORE.
        """
        if not entities:
            return 0

        db = self._ensure_db()
        inserted = 0

        sql = """
            INSERT OR IGNORE INTO entities
                (id, interaction_id, name, entity_type, aliases, confidence,
                 provenance, context, tags, source_instance, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        async with self._write_lock:
            try:
                await db.execute("BEGIN IMMEDIATE")
                for entity in entities:
                    created_at_str = (
                        entity.created_at.isoformat()
                        if entity.created_at
                        else datetime.now(timezone.utc).isoformat()
                    )
                    params = (
                        entity.id,
                        entity.interaction_id,
                        entity.name,
                        entity.entity_type.value,
                        json.dumps(entity.aliases),
                        entity.confidence,
                        entity.provenance,
                        json.dumps(entity.context),
                        json.dumps(entity.tags),
                        entity.source_instance,
                        created_at_str,
                    )
                    cursor = await db.execute(sql, params)
                    # rowcount == 1 when a row was actually inserted,
                    # 0 when INSERT OR IGNORE skipped a duplicate
                    inserted += cursor.rowcount
                await db.commit()
            except Exception as exc:
                await db.rollback()
                logger.error(
                    "entities_store_failed",
                    error_code="CTXMTG-STG-015",
                    entity_count=len(entities),
                    error=str(exc),
                )
                raise StorageError(
                    f"Failed to store entities batch ({len(entities)} entities): {exc}",
                    error_code="CTXMTG-STG-015",
                ) from exc

        logger.info("entities_stored", count=inserted, total=len(entities))
        return inserted

    async def get_entities(
        self,
        interaction_id: str | None = None,
        entity_type: str | None = None,
        name_like: str | None = None,
        limit: int = 100,
    ) -> list[Entity]:
        """
        Query entities with optional filters.

        All name matching uses COLLATE NOCASE so "alice", "Alice",
        and "ALICE" are treated identically (plan spec P1-04).
        """
        db = self._ensure_db()

        clauses: list[str] = []
        params: list[Any] = []

        if interaction_id is not None:
            clauses.append("interaction_id = ?")
            params.append(interaction_id)
        if entity_type is not None:
            clauses.append("entity_type = ?")
            params.append(entity_type)
        if name_like is not None:
            # Wrap in wildcards for partial matching and enforce NOCASE
            clauses.append("name LIKE ? COLLATE NOCASE")
            params.append(f"%{name_like}%")

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM entities{where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        return [self._row_to_entity(r) for r in rows]

    # =================================================================
    # Fact CRUD
    # =================================================================

    async def store_facts(self, facts: list[Fact]) -> int:
        """
        Batch-store facts inside a single transaction.

        Duplicates (same ID) are skipped via INSERT OR IGNORE.
        Returns the number of newly inserted facts.
        """
        if not facts:
            return 0

        db = self._ensure_db()
        inserted = 0

        sql = """
            INSERT OR IGNORE INTO facts
                (id, interaction_id, subject_entity_id, predicate,
                 object_entity_id, object_literal, confidence,
                 source_span, source_instance, created_at, superseded_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        async with self._write_lock:
            try:
                await db.execute("BEGIN IMMEDIATE")
                for fact in facts:
                    created_at_str = (
                        fact.created_at.isoformat()
                        if fact.created_at
                        else datetime.now(timezone.utc).isoformat()
                    )
                    params = (
                        fact.id,
                        fact.interaction_id,
                        fact.subject_entity_id,
                        fact.predicate,
                        fact.object_entity_id,
                        fact.object_literal,
                        fact.confidence,
                        fact.source_span,
                        fact.source_instance,
                        created_at_str,
                        fact.superseded_by,
                    )
                    cursor = await db.execute(sql, params)
                    inserted += cursor.rowcount
                await db.commit()
            except Exception as exc:
                await db.rollback()
                logger.error(
                    "facts_store_failed",
                    error_code="CTXMTG-STG-002",
                    fact_count=len(facts),
                    error=str(exc),
                )
                raise StorageError(
                    f"Failed to store facts batch ({len(facts)} facts): {exc}",
                    error_code="CTXMTG-STG-002",
                ) from exc

        logger.info("facts_stored", count=inserted, total=len(facts))
        return inserted

    async def get_facts(
        self,
        interaction_id: str | None = None,
        subject_entity_id: str | None = None,
        predicate: str | None = None,
        limit: int = 100,
    ) -> list[Fact]:
        """
        Query facts with optional filters (ANDed together).
        """
        db = self._ensure_db()

        clauses: list[str] = []
        params: list[Any] = []

        if interaction_id is not None:
            clauses.append("interaction_id = ?")
            params.append(interaction_id)
        if subject_entity_id is not None:
            clauses.append("subject_entity_id = ?")
            params.append(subject_entity_id)
        if predicate is not None:
            clauses.append("predicate = ?")
            params.append(predicate)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM facts{where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        return [self._row_to_fact(r) for r in rows]

    # =================================================================
    # Embedding Metadata
    # =================================================================

    async def store_embedding_metadata(self, metadata: list[EmbeddingMetadata]) -> int:
        """
        Batch-store embedding metadata records (INSERT OR IGNORE).
        """
        if not metadata:
            return 0

        db = self._ensure_db()
        inserted = 0

        sql = """
            INSERT OR IGNORE INTO embeddings_metadata
                (id, source_table, source_id, chunk_index,
                 chunk_start, chunk_end, model_name,
                 model_version, dimensions, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        async with self._write_lock:
            try:
                await db.execute("BEGIN IMMEDIATE")
                for m in metadata:
                    created_at_str = (
                        m.created_at.isoformat()
                        if m.created_at
                        else datetime.now(timezone.utc).isoformat()
                    )
                    params = (
                        m.id,
                        m.source_table,
                        m.source_id,
                        m.chunk_index,
                        m.chunk_start,
                        m.chunk_end,
                        m.model_name,
                        m.model_version,
                        m.dimensions,
                        created_at_str,
                    )
                    cursor = await db.execute(sql, params)
                    inserted += cursor.rowcount
                await db.commit()
            except Exception as exc:
                await db.rollback()
                logger.error(
                    "embedding_metadata_store_failed",
                    error_code="CTXMTG-STG-010",
                    record_count=len(metadata),
                    error=str(exc),
                )
                raise StorageError(
                    f"Failed to store embedding metadata ({len(metadata)} records): {exc}",
                    error_code="CTXMTG-STG-010",
                ) from exc

        logger.info("embedding_metadata_stored", count=inserted)
        return inserted

    # =================================================================
    # Farming Insights
    # =================================================================

    async def store_insight(self, insight: FarmingInsight) -> str:
        """
        Store a farming insight (INSERT OR REPLACE). Returns the insight ID.
        """
        db = self._ensure_db()

        sql = """
            INSERT OR REPLACE INTO meta_insights
                (id, insight_type, title, description, evidence,
                 confidence, parameters, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        created_at_str = (
            insight.created_at.isoformat()
            if insight.created_at
            else datetime.now(timezone.utc).isoformat()
        )
        expires_at_str = insight.expires_at.isoformat() if insight.expires_at else None

        params = (
            insight.id,
            insight.insight_type,
            insight.title,
            insight.description,
            json.dumps(insight.evidence),
            insight.confidence,
            json.dumps(insight.parameters),
            created_at_str,
            expires_at_str,
        )

        async with self._write_lock:
            try:
                await db.execute("BEGIN IMMEDIATE")
                await db.execute(sql, params)
                await db.commit()
            except Exception as exc:
                await db.rollback()
                logger.error(
                    "insight_store_failed",
                    error_code="CTXMTG-STG-013",
                    insight_id=insight.id,
                    error=str(exc),
                )
                raise StorageError(
                    f"Failed to store insight {insight.id}: {exc}",
                    error_code="CTXMTG-STG-013",
                ) from exc

        logger.info("insight_stored", insight_id=insight.id)
        return insight.id

    async def get_insights(
        self,
        insight_type: str | None = None,
        limit: int = 50,
    ) -> list[FarmingInsight]:
        """
        Query farming insights, optionally filtered by type.
        """
        db = self._ensure_db()

        clauses: list[str] = []
        params: list[Any] = []

        if insight_type is not None:
            clauses.append("insight_type = ?")
            params.append(insight_type)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM meta_insights{where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        return [self._row_to_insight(r) for r in rows]

    # =================================================================
    # Raw SQL Execution (for the query engine)
    # =================================================================

    async def execute_sql(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a raw SQL query and return rows as dicts.

        The query engine generates SQL from user questions and executes
        it through this method. Uses named-parameter binding (:name)
        to prevent SQL injection.

        Args:
            sql:    SQL string with optional named placeholders.
            params: Dict mapping placeholder names to values.

        Returns:
            List of dicts (column_name → value) for each result row.
        """
        db = self._ensure_db()
        try:
            cursor = await db.execute(sql, params or {})
            rows = await cursor.fetchall()
            # Convert sqlite3.Row objects to plain dicts
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.error(
                "sql_execution_failed",
                error_code="CTXMTG-STG-003",
                error=str(exc),
            )
            raise StorageError(
                f"SQL execution failed: {exc}",
                error_code="CTXMTG-STG-003",
            ) from exc

    # =================================================================
    # Full-Text Search
    # =================================================================

    async def search_fts(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Full-text search across interactions using FTS5.

        Searches the interactions_fts virtual table using the porter-
        stemmed index. Returns rows ordered by BM25 relevance rank
        (lower rank = more relevant in SQLite FTS5).

        Args:
            query: The keyword search string (FTS5 query syntax).
            limit: Maximum number of results.

        Returns:
            List of dicts with interaction id, title, content snippet,
            and relevance rank.
        """
        db = self._ensure_db()

        # Join the FTS table back to interactions via rowid to get
        # the full row including the primary-key id column.
        sql = """
            SELECT i.id, i.title, i.content, i.source_type,
                   i.created_at, rank
            FROM interactions_fts fts
            JOIN interactions i ON i.rowid = fts.rowid
            WHERE interactions_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        try:
            cursor = await db.execute(sql, (query, limit))
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.error(
                "fts_search_failed",
                error_code="CTXMTG-STG-003",
                error=str(exc),
            )
            raise StorageError(
                f"FTS search failed: {exc}",
                error_code="CTXMTG-STG-003",
            ) from exc

    # =================================================================
    # Row → Model helpers
    # =================================================================
    # These private methods convert sqlite3.Row objects (dict-like) into
    # the Pydantic models defined in ctxmtg.models. JSON TEXT columns
    # are deserialised, and ISO-8601 strings are parsed to datetime.

    @staticmethod
    def _row_to_interaction(row: aiosqlite.Row) -> Interaction:
        """Convert a database row to an Interaction model."""
        return Interaction(
            id=row["id"],
            source_type=SourceType(row["source_type"]),
            source_id=row["source_id"],
            title=row["title"],
            content=row["content"],
            participants=json.loads(row["participants"]),
            metadata=json.loads(row["metadata"]),
            source_instance=row["source_instance"],
            intake_action=IntakeAction(row["intake_action"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
            hive_synced_at=(
                datetime.fromisoformat(row["hive_synced_at"])
                if row["hive_synced_at"]
                else None
            ),
        )

    @staticmethod
    def _row_to_entity(row: aiosqlite.Row) -> Entity:
        """Convert a database row to an Entity model."""
        return Entity(
            id=row["id"],
            interaction_id=row["interaction_id"],
            name=row["name"],
            entity_type=EntityType(row["entity_type"]),
            aliases=json.loads(row["aliases"]),
            confidence=row["confidence"],
            provenance=row["provenance"],
            context=json.loads(row["context"]) if row["context"] else {},
            tags=json.loads(row["tags"]) if row["tags"] else {},
            source_instance=row["source_instance"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            hive_synced_at=(
                datetime.fromisoformat(row["hive_synced_at"])
                if row["hive_synced_at"]
                else None
            ),
        )

    @staticmethod
    def _row_to_fact(row: aiosqlite.Row) -> Fact:
        """Convert a database row to a Fact model."""
        return Fact(
            id=row["id"],
            interaction_id=row["interaction_id"],
            subject_entity_id=row["subject_entity_id"],
            predicate=row["predicate"],
            object_entity_id=row["object_entity_id"],
            object_literal=row["object_literal"],
            confidence=row["confidence"],
            source_span=row["source_span"],
            source_instance=row["source_instance"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            superseded_by=row["superseded_by"],
            hive_synced_at=(
                datetime.fromisoformat(row["hive_synced_at"])
                if row["hive_synced_at"]
                else None
            ),
        )

    @staticmethod
    def _row_to_insight(row: aiosqlite.Row) -> FarmingInsight:
        """Convert a database row to a FarmingInsight model."""
        return FarmingInsight(
            id=row["id"],
            insight_type=row["insight_type"],
            title=row["title"],
            description=row["description"],
            evidence=json.loads(row["evidence"]) if row["evidence"] else [],
            confidence=row["confidence"],
            parameters=json.loads(row["parameters"]) if row["parameters"] else {},
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            expires_at=(
                datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None
            ),
        )
