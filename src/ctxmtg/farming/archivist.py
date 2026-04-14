# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Archivist Maintenance Stage
=============================

Identifies cold (old, low-confidence, unreferenced) entities and
archives them to a separate ``archive.db`` SQLite file.  This frees
the main database from rarely-accessed data and improves query
performance.

Cold entity criteria (all must hold):
    1. ``created_at`` is older than ``archive_age_days`` (default 180).
    2. ``confidence`` is below ``archive_confidence_threshold`` (default 0.5).
    3. Entity is NOT referenced by any *recent* ``meta_insights.entity_ids``
       row (scoped to insights created within ``archive_age_days``).

After identification, the stage:
    - Logs to ``maintenance_archivist`` (action='identified_cold').
    - If ``archive_db_path`` is configured, copies candidate entities
      and their facts to ``archive.db`` then deletes them from the
      live database (action='archived').
    - Returns a single :class:`FarmingInsight` with type ``archive``
      summarising the results.

Depends on:
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- reserved, unused)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async bridge)
    - aiosqlite (archive database access)
    - structlog (structured logging)
    - uuid (unique IDs for maintenance log entries)
    - json (serialisation of log details)

Used by:
    - ctxmtg.farming.pipeline (registered as maintenance stage 13)
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import aiosqlite
import structlog

from ctxmtg.farming.checkpoint import _run_async
from ctxmtg.farming.progress import get_offset_with_wrap, update_offset
from ctxmtg.interfaces.farming import FarmingContext, FarmingStage
from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.models.farming import FarmingInsight

# ---------------------------------------------------------------
# Module-level logger -- logs archival identification events.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.archivist")

# ---------------------------------------------------------------
# Schema for the archive database.  Mirrors the live database
# tables but without foreign-key constraints (archived entities
# may reference interactions that are still in the live DB).
# ---------------------------------------------------------------
ARCHIVE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,
    interaction_id  TEXT NOT NULL,
    name            TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    aliases         TEXT DEFAULT '[]',
    confidence      REAL NOT NULL DEFAULT 1.0,
    provenance      TEXT,
    context         TEXT DEFAULT '{}',
    tags            TEXT DEFAULT '{}',
    source_instance TEXT NOT NULL DEFAULT 'local',
    created_at      TEXT,
    hive_synced_at  TEXT,
    archived_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS facts (
    id                  TEXT PRIMARY KEY,
    interaction_id      TEXT NOT NULL,
    subject_entity_id   TEXT NOT NULL,
    predicate           TEXT NOT NULL,
    object_entity_id    TEXT,
    object_literal      TEXT,
    confidence          REAL NOT NULL DEFAULT 1.0,
    source_span         TEXT,
    source_instance     TEXT NOT NULL DEFAULT 'local',
    created_at          TEXT,
    superseded_by       TEXT,
    hive_synced_at      TEXT,
    archived_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""


class ArchivistStage(FarmingStage):
    """
    Maintenance stage that identifies cold entities for archival.

    Cold entities are old, low-confidence records that no active
    meta-insight references.  Identifying them is the first step
    toward moving them to a separate archive database -- freeing
    the main store from rarely-accessed data and improving query
    performance.

    This also serves as the natural cleanup for garbage entities
    that spaCy's NER extracts incorrectly -- timestamps ("09:15"),
    version numbers ("6.2"), prices ("99.99"), and sentence
    fragments.  These entities enter the DB but never accumulate
    facts, co-occurrences, or meta-insight references.  Over
    farming cycles they remain low-confidence islands in the graph,
    and the archivist's cold-entity criteria will eventually flag
    them.  No special pre-extraction filter is needed.

    Phase 3 scope: identify + log only.  Full archival (ATTACH
    DATABASE + INSERT INTO archive.entities + DELETE FROM entities)
    is planned for a future phase.

    Usage:
        archivist = ArchivistStage(archive_age_days=180)
        insights = archivist.run(sql_store, vector_store, context)
    """

    def __init__(
        self,
        archive_age_days: int = 180,
        archive_query_days: int = 90,
        archive_confidence_threshold: float = 0.5,
        archive_db_path: str | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Configure the archivist.

        Args:
            archive_age_days: Minimum age (in days) for an entity to
                be considered cold.  Entities created more recently
                than this threshold are always excluded.
            archive_query_days: Reserved for future use -- minimum
                days since last query access to consider an entity
                cold.  Not yet implemented.
            archive_confidence_threshold: Maximum confidence for a
                cold entity.  Entities with confidence >= this value
                are not considered cold (they are still trustworthy).
            archive_db_path: Path to the archive database.  When set,
                entities and their facts are copied to this database
                then deleted from the live store.  When None, the
                archivist only identifies and logs candidates.
            llm: Optional LLM provider.  Reserved for future
                 intelligence-assisted archival decisions.
        """
        self._archive_age_days = archive_age_days
        self._archive_query_days = archive_query_days
        self._archive_confidence_threshold = archive_confidence_threshold
        self._archive_db_path = archive_db_path
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface -- stage name for logging/checkpointing.
    # -----------------------------------------------------------------
    def get_name(self) -> str:
        """Return the stage name used for logging and checkpointing."""
        return "archivist"

    # =================================================================
    # Main entry point
    # =================================================================
    def run(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        """
        Identify cold entities eligible for archival.

        Steps:
        1. SEED: Query for cold entities -- old, low-confidence, and
           not referenced by any active meta_insights row.
        2. Log the identified candidate IDs to maintenance_archivist.
        3. Return a FarmingInsight summarising the candidates.

        The actual data movement (DELETE from main, INSERT into
        archive.db) is deferred to a future phase.

        Args:
            sql_store:    SQL store to query entities and insights.
            vector_store: Vector store (unused by archivist).
            context:      Farming context with cycle_id and budget.

        Returns:
            List containing zero or one FarmingInsight objects.
            Empty list if no cold entities are found.
        """
        return _run_async(self._run_impl(sql_store, context))

    # =================================================================
    # Async implementation
    # =================================================================
    async def _run_impl(
        self,
        sql_store: SQLStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        """
        Async implementation of the archival identification logic.

        Separated from run() so we can ``await`` on sql_store methods.
        The sync run() method bridges to this via _run_async().
        """
        # ---------------------------------------------------------
        # STEP 1: SEED -- find cold entities.
        #
        # Criteria:
        #   a) created_at older than archive_age_days
        #   b) confidence below archive_confidence_threshold
        #   c) NOT referenced in any meta_insights.entity_ids
        #
        # The sub-query ``json_each(entity_ids)`` unpacks each
        # insight's entity_ids JSON array so we can exclude entities
        # that are still actively referenced.
        # ---------------------------------------------------------
        age_param = f"-{self._archive_age_days} days"

        # ORIGINAL (disabled 2026-04-07): Hardcoded LIMIT with no OFFSET.
        # seed_sql = (
        #     "SELECT e.id, e.interaction_id, e.name, e.entity_type, "
        #     "e.confidence, e.created_at, e.aliases, e.provenance, "
        #     "e.context, e.tags, e.source_instance, e.hive_synced_at "
        #     "FROM entities e "
        #     "WHERE e.created_at < DATE('now', :age_param) "
        #     "AND e.confidence < :conf_threshold "
        #     "AND e.id NOT IN ("
        #     "  SELECT DISTINCT json_each.value "
        #     "  FROM meta_insights, json_each(entity_ids) "
        #     "  WHERE entity_ids != '[]'"
        #     "  AND meta_insights.created_at > DATE('now', :age_param)"
        #     ") "
        #     "ORDER BY e.created_at ASC "
        #     "LIMIT 50"
        # )

        batch_size = 50

        # Get total count for offset wrapping
        total_rows = await sql_store.execute_sql(
            "SELECT COUNT(*) as cnt FROM entities e "
            "WHERE e.created_at < DATE('now', :age_param) "
            "AND e.confidence < :conf_threshold "
            "AND e.id NOT IN ("
            "  SELECT DISTINCT json_each.value "
            "  FROM meta_insights, json_each(entity_ids) "
            "  WHERE entity_ids != '[]'"
            "  AND meta_insights.created_at > DATE('now', :age_param)"
            ")",
            {
                "age_param": age_param,
                "conf_threshold": self._archive_confidence_threshold,
            },
        )
        total_count = total_rows[0]["cnt"] if total_rows else 0

        offset = await get_offset_with_wrap(sql_store, "archivist", total_count, batch_size)

        seed_sql = (
            "SELECT e.id, e.interaction_id, e.name, e.entity_type, "
            "e.confidence, e.created_at, e.aliases, e.provenance, "
            "e.context, e.tags, e.source_instance, e.hive_synced_at "
            "FROM entities e "
            "WHERE e.created_at < DATE('now', :age_param) "
            "AND e.confidence < :conf_threshold "
            "AND e.id NOT IN ("
            "  SELECT DISTINCT json_each.value "
            "  FROM meta_insights, json_each(entity_ids) "
            "  WHERE entity_ids != '[]'"
            "  AND meta_insights.created_at > DATE('now', :age_param)"
            ") "
            "ORDER BY e.created_at ASC "
            "LIMIT 50 OFFSET :offset"
        )
        candidates = await sql_store.execute_sql(
            seed_sql,
            {
                "age_param": age_param,
                "conf_threshold": self._archive_confidence_threshold,
                "offset": offset,
            },
        )

        # ---------------------------------------------------------
        # STEP 1b: GARBAGE BYPASS -- find entities marked by the
        # Rationalizer (confidence <= 0.1) regardless of age.
        # These are garbage entities that should be archived
        # immediately without waiting for the age threshold.
        # ---------------------------------------------------------
        garbage_rows = await sql_store.execute_sql(
            "SELECT e.id, e.interaction_id, e.name, e.entity_type, "
            "e.confidence, e.created_at, e.aliases, e.provenance, "
            "e.context, e.tags, e.source_instance, e.hive_synced_at "
            "FROM entities e "
            "WHERE e.confidence <= 0.1 "
            "AND e.id NOT IN ("
            "  SELECT DISTINCT json_each.value "
            "  FROM meta_insights, json_each(entity_ids) "
            "  WHERE entity_ids != '[]'"
            "  AND meta_insights.created_at > DATE('now', :age_param)"
            ") "
            "ORDER BY e.confidence ASC, e.created_at ASC "
            "LIMIT 50",
            {"age_param": age_param},
        )

        # Merge garbage candidates with cold candidates (dedup by id)
        seen_ids = {row["id"] for row in candidates}
        for row in garbage_rows:
            if row["id"] not in seen_ids:
                candidates.append(row)
                seen_ids.add(row["id"])

        garbage_bypass_count = len(garbage_rows)

        candidate_count = len(candidates)
        candidate_ids = [row["id"] for row in candidates]

        logger.info(
            "archivist_seed_complete",
            cold_entity_count=candidate_count - garbage_bypass_count,
            garbage_bypass_count=garbage_bypass_count,
            total_candidates=candidate_count,
            archive_age_days=self._archive_age_days,
            confidence_threshold=self._archive_confidence_threshold,
        )

        # ---------------------------------------------------------
        # No cold entities → nothing to do.
        # ---------------------------------------------------------
        if candidate_count == 0:
            logger.info(
                "archivist_no_candidates",
                cycle_id=context.cycle_id,
            )
            return []

        # ---------------------------------------------------------
        # STEP 2: Log to maintenance_archivist table.
        # Records the set of identified cold entity IDs for audit.
        # ---------------------------------------------------------
        log_id = str(uuid4())
        detail = f"{candidate_count} entities eligible for archival"
        await sql_store.execute_sql(
            "INSERT INTO maintenance_archivist "
            "(id, cycle_id, action, target_ids, detail) "
            "VALUES (:id, :cycle, 'identified_cold', :targets, :detail)",
            {
                "id": log_id,
                "cycle": context.cycle_id,
                "targets": json.dumps(candidate_ids),
                "detail": detail,
            },
        )

        # Commit the log entry
        db = sql_store._ensure_db()  # type: ignore[attr-defined]
        await db.commit()

        logger.info(
            "archivist_logged_candidates",
            candidate_count=candidate_count,
            cycle_id=context.cycle_id,
        )

        # ---------------------------------------------------------
        # STEP 3: Archive and delete if archive_db_path is configured.
        #
        # Copies candidate entities and their facts to archive.db,
        # then deletes them from the live database.  Facts that
        # reference the archived entity (via subject_entity_id) are
        # archived and deleted as well.
        # ---------------------------------------------------------
        archived_count = 0
        archived_facts = 0
        if self._archive_db_path is not None:
            archived_count, archived_facts = await self._archive_and_delete(
                sql_store=sql_store,
                candidates=candidates,
                cycle_id=context.cycle_id,
            )

        # ---------------------------------------------------------
        # STEP 4: Build a FarmingInsight summarising the results.
        # ---------------------------------------------------------
        if archived_count > 0:
            title = (
                f"{archived_count} entities archived "
                f"({archived_facts} facts)"
            )
            description = (
                f"Archived {archived_count} cold entities and "
                f"{archived_facts} associated facts to "
                f"{self._archive_db_path}."
            )
        else:
            title = f"{candidate_count} entities identified for archival"
            description = (
                f"Found {candidate_count} cold entities older than "
                f"{self._archive_age_days} days with confidence below "
                f"{self._archive_confidence_threshold}."
            )
            if self._archive_db_path is None:
                description += (
                    "  Set archive_db_path to enable actual archival."
                )

        insight = FarmingInsight(
            id=str(uuid4()),
            insight_type="archive",
            title=title,
            description=description,
            evidence=candidate_ids,
            confidence=1.0,
            parameters={
                "candidate_count": candidate_count,
                "archived_count": archived_count,
                "archived_facts": archived_facts,
                "archive_age_days": self._archive_age_days,
                "confidence_threshold": self._archive_confidence_threshold,
                "candidate_ids": candidate_ids,
            },
        )

        await update_offset(sql_store, "archivist", offset + batch_size, candidate_count)

        logger.info(
            "archivist_complete",
            candidates_identified=candidate_count,
            archived=archived_count,
            archived_facts=archived_facts,
            cycle_id=context.cycle_id,
        )

        return [insight]

    async def _archive_and_delete(
        self,
        sql_store: SQLStore,
        candidates: list[dict],
        cycle_id: int,
    ) -> tuple[int, int]:
        """
        Copy candidate entities and their facts to archive.db, then
        delete them from the live database.

        Creates the archive database and schema on first use.

        Args:
            sql_store:  The live SQL store.
            candidates: Rows from the seed query (each has id, name, etc.).
            cycle_id:   Current farming cycle ID (for audit logging).

        Returns:
            A tuple of (entities_archived, facts_archived).
        """
        archive_path = Path(self._archive_db_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        # Open (or create) the archive database and ensure schema.
        async with aiosqlite.connect(str(archive_path)) as archive_db:
            archive_db.row_factory = aiosqlite.Row
            for statement in ARCHIVE_SCHEMA.strip().split(";"):
                stmt = statement.strip()
                if stmt:
                    await archive_db.execute(stmt)
            await archive_db.commit()

            entities_archived = 0
            facts_archived = 0
            db = sql_store._ensure_db()  # type: ignore[attr-defined]

            for candidate in candidates:
                eid = candidate["id"]

                # Copy entity to archive
                await archive_db.execute(
                    "INSERT OR IGNORE INTO entities "
                    "(id, interaction_id, name, entity_type, aliases, "
                    "confidence, provenance, context, tags, "
                    "source_instance, created_at, hive_synced_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        candidate["id"],
                        candidate.get("interaction_id", ""),
                        candidate["name"],
                        candidate["entity_type"],
                        candidate.get("aliases", "[]"),
                        candidate["confidence"],
                        candidate.get("provenance"),
                        candidate.get("context", "{}"),
                        candidate.get("tags", "{}"),
                        candidate.get("source_instance", "local"),
                        candidate["created_at"],
                        candidate.get("hive_synced_at"),
                    ),
                )
                entities_archived += 1

                # Fetch and copy associated facts to archive
                fact_rows = await sql_store.execute_sql(
                    "SELECT * FROM facts WHERE subject_entity_id = :eid",
                    {"eid": eid},
                )
                for fact in fact_rows:
                    await archive_db.execute(
                        "INSERT OR IGNORE INTO facts "
                        "(id, interaction_id, subject_entity_id, "
                        "predicate, object_entity_id, object_literal, "
                        "confidence, source_span, source_instance, "
                        "created_at, superseded_by, hive_synced_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            fact["id"],
                            fact["interaction_id"],
                            fact["subject_entity_id"],
                            fact["predicate"],
                            fact.get("object_entity_id"),
                            fact.get("object_literal"),
                            fact["confidence"],
                            fact.get("source_span"),
                            fact.get("source_instance", "local"),
                            fact["created_at"],
                            fact.get("superseded_by"),
                            fact.get("hive_synced_at"),
                        ),
                    )
                    facts_archived += 1

                # Delete facts from live DB first (FK constraint)
                await sql_store.execute_sql(
                    "DELETE FROM facts WHERE subject_entity_id = :eid",
                    {"eid": eid},
                )

                # Nullify object_entity_id references to this entity
                # and set a placeholder object_literal to satisfy the
                # CHECK(object_entity_id IS NOT NULL OR object_literal IS NOT NULL)
                # constraint that would otherwise fire via ON DELETE SET NULL.
                await sql_store.execute_sql(
                    "UPDATE facts SET object_literal = '[archived]' "
                    "WHERE object_entity_id = :eid AND object_literal IS NULL",
                    {"eid": eid},
                )

                # Delete entity from live DB
                await sql_store.execute_sql(
                    "DELETE FROM entities WHERE id = :eid",
                    {"eid": eid},
                )

            await archive_db.commit()
            await db.commit()

            # Log the archival action
            log_id = str(uuid4())
            await sql_store.execute_sql(
                "INSERT INTO maintenance_archivist "
                "(id, cycle_id, action, target_ids, detail) "
                "VALUES (:id, :cycle, 'archived', :targets, :detail)",
                {
                    "id": log_id,
                    "cycle": cycle_id,
                    "targets": json.dumps([c["id"] for c in candidates]),
                    "detail": (
                        f"Archived {entities_archived} entities and "
                        f"{facts_archived} facts to {self._archive_db_path}"
                    ),
                },
            )
            await db.commit()

            logger.info(
                "archivist_archive_complete",
                entities_archived=entities_archived,
                facts_archived=facts_archived,
                archive_path=str(archive_path),
            )

        return (entities_archived, facts_archived)
