# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Consolidator Maintenance Stage
===============================

Merges duplicate facts in the knowledge store. When the same
subject + predicate + object_literal triple appears multiple times
(from different interactions), the consolidator picks the highest-
confidence copy as canonical and marks the rest as superseded.

CRITICAL INVARIANT: Fact content (subject, predicate, object,
source_span) is NEVER modified.  Only the superseded_by column
may be set on non-canonical duplicates.

The consolidator also populates the entity_interactions junction
table for the canonical entity so downstream queries can quickly
look up all interactions involving a given entity.

All actions are logged to the maintenance_consolidator table for
audit and debugging.

Depends on:
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- reserved for Tier 2+ entity similarity)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async bridge)
    - structlog (structured logging)
    - uuid (unique IDs for maintenance log entries)
    - json (serialisation of log details)

Used by:
    - ctxmtg.farming.pipeline (registered as maintenance stage 8)
"""

from __future__ import annotations

import json
from uuid import uuid4

import structlog

from ctxmtg.farming.checkpoint import _run_async
from ctxmtg.farming.progress import get_offset_with_wrap, update_offset
from ctxmtg.interfaces.farming import FarmingContext, FarmingStage
from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.models.farming import FarmingInsight

# ---------------------------------------------------------------
# Module-level logger -- logs consolidation events.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.consolidator")


class ConsolidatorStage(FarmingStage):
    """
    Maintenance stage that merges duplicate facts.

    Duplicate facts arise when the same triple (subject + predicate +
    object_literal) is extracted from multiple interactions.  The
    consolidator groups these duplicates, picks the highest-confidence
    fact as canonical, and sets superseded_by on the rest.

    The duplicate_threshold controls how many copies must exist before
    a group is considered worth merging (default 3).  The
    entity_similarity_threshold is reserved for future Tier 2+ work
    where an LLM could identify near-duplicate entities.

    Usage:
        consolidator = ConsolidatorStage(duplicate_threshold=3)
        insights = consolidator.run(sql_store, vector_store, context)
    """

    def __init__(
        self,
        duplicate_threshold: int = 3,
        entity_similarity_threshold: float = 0.85,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Configure the consolidator.

        Args:
            duplicate_threshold: Minimum number of identical facts in a
                group before merging is triggered.  Lower values merge
                more aggressively; higher values are more conservative.
            entity_similarity_threshold: Reserved for Tier 2+ entity
                similarity matching via LLM embeddings.  Not used in
                the current (Tier 0-1) implementation.
            llm: Optional LLM provider for future entity similarity.
                 Currently unused -- set to None.
        """
        self._duplicate_threshold = duplicate_threshold
        self._entity_similarity_threshold = entity_similarity_threshold
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface -- stage name for logging/checkpointing.
    # -----------------------------------------------------------------
    def get_name(self) -> str:
        """Return the stage name used for logging and checkpointing."""
        return "consolidator"

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
        Find and merge duplicate fact groups.

        Steps:
        1. SEED: Query for duplicate fact groups (same subject +
           predicate + object_literal, count >= threshold).
        2. For each group: pick the highest-confidence fact as
           canonical, mark the rest as superseded, log to
           maintenance_consolidator, and populate entity_interactions.
        3. Return a FarmingInsight per merge group.

        Args:
            sql_store:    SQL store to read/write facts.
            vector_store: Vector store (unused by consolidator).
            context:      Farming context with cycle_id and budget.

        Returns:
            List of FarmingInsight objects describing each merge.
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
        Async implementation of the consolidation logic.

        Separated from run() so we can use await on sql_store methods.
        The sync run() method bridges to this via _run_async().
        """
        # ---------------------------------------------------------
        # STEP 1: SEED -- find duplicate fact groups.
        # Groups facts by (subject_entity_id, predicate, object_literal)
        # and returns groups with count >= threshold.
        # ---------------------------------------------------------
        # ORIGINAL (disabled 2026-04-07): Hardcoded LIMIT with no OFFSET.
        # seed_sql = (
        #     "SELECT subject_entity_id, predicate, object_literal, "
        #     "COUNT(*) as cnt, GROUP_CONCAT(id) as fact_ids, "
        #     "MAX(confidence) as max_conf "
        #     "FROM facts "
        #     "WHERE superseded_by IS NULL "
        #     "GROUP BY subject_entity_id, predicate, object_literal "
        #     "HAVING cnt >= :threshold "
        #     "ORDER BY cnt DESC "
        #     "LIMIT 50"
        # )

        batch_size = 50

        # Get total count for offset wrapping
        total_rows = await sql_store.execute_sql(
            "SELECT COUNT(*) as cnt FROM ("
            "SELECT subject_entity_id, predicate, object_literal "
            "FROM facts "
            "WHERE superseded_by IS NULL "
            "GROUP BY subject_entity_id, predicate, object_literal "
            "HAVING COUNT(*) >= :threshold"
            ")",
            {"threshold": self._duplicate_threshold},
        )
        total_count = total_rows[0]["cnt"] if total_rows else 0

        offset = await get_offset_with_wrap(sql_store, "consolidator", total_count, batch_size)

        seed_sql = (
            "SELECT subject_entity_id, predicate, object_literal, "
            "COUNT(*) as cnt, GROUP_CONCAT(id) as fact_ids, "
            "MAX(confidence) as max_conf "
            "FROM facts "
            "WHERE superseded_by IS NULL "
            "GROUP BY subject_entity_id, predicate, object_literal "
            "HAVING cnt >= :threshold "
            "ORDER BY cnt DESC "
            "LIMIT 50 OFFSET :offset"
        )
        groups = await sql_store.execute_sql(
            seed_sql, {"threshold": self._duplicate_threshold, "offset": offset}
        )

        logger.info(
            "consolidator_seed_complete",
            duplicate_groups=len(groups),
            threshold=self._duplicate_threshold,
        )

        # Collect one FarmingInsight per merged group
        insights: list[FarmingInsight] = []

        # ---------------------------------------------------------
        # STEP 2: Process each duplicate group.
        # ---------------------------------------------------------
        for group in groups:
            # Parse the comma-separated fact IDs from GROUP_CONCAT
            fact_ids = group["fact_ids"].split(",")
            cnt = group["cnt"]
            predicate = group["predicate"]
            max_conf = group["max_conf"]

            # Find the canonical fact (highest confidence).
            # Query all facts in the group to find the one with max
            # confidence.  If multiple share the same confidence,
            # the first one (by rowid / insertion order) wins.
            placeholders = ",".join(f"'{fid}'" for fid in fact_ids)
            fact_rows = await sql_store.execute_sql(
                f"SELECT id, confidence FROM facts "
                f"WHERE id IN ({placeholders}) "
                f"ORDER BY confidence DESC, rowid ASC "
                f"LIMIT 1"
            )

            # Safety: skip if we somehow get no rows
            if not fact_rows:
                continue  # pragma: no cover

            canonical_id = fact_rows[0]["id"]

            # IDs of non-canonical duplicates to supersede
            merged_ids = [fid for fid in fact_ids if fid != canonical_id]

            # -------------------------------------------------------
            # STEP 2a: Set superseded_by on non-canonical facts.
            # CRITICAL: only superseded_by is modified; content is
            # never touched.
            # -------------------------------------------------------
            for dup_id in merged_ids:
                await sql_store.execute_sql(
                    "UPDATE facts SET superseded_by = :canonical "
                    "WHERE id = :dup_id",
                    {"canonical": canonical_id, "dup_id": dup_id},
                )

            # Commit after processing the group
            db = sql_store._ensure_db()  # type: ignore[attr-defined]
            await db.commit()

            # -------------------------------------------------------
            # STEP 2b: Log to maintenance_consolidator table.
            # -------------------------------------------------------
            log_id = str(uuid4())
            detail = json.dumps({
                "predicate": predicate,
                "duplicate_count": cnt,
                "merged_ids": merged_ids,
                "max_confidence": max_conf,
            })
            await sql_store.execute_sql(
                "INSERT INTO maintenance_consolidator "
                "(id, cycle_id, action, target_ids, canonical_id, detail) "
                "VALUES (:id, :cycle, 'merge_facts', :targets, :canonical, :detail)",
                {
                    "id": log_id,
                    "cycle": context.cycle_id,
                    "targets": json.dumps(merged_ids),
                    "canonical": canonical_id,
                    "detail": detail,
                },
            )
            await db.commit()

            # -------------------------------------------------------
            # STEP 2c: Populate entity_interactions junction table.
            # Gather interaction_ids from all merged facts and link
            # the canonical entity to those interactions.
            # -------------------------------------------------------
            all_fact_ids = [canonical_id] + merged_ids
            ph = ",".join(f"'{fid}'" for fid in all_fact_ids)
            interaction_rows = await sql_store.execute_sql(
                f"SELECT DISTINCT interaction_id FROM facts "
                f"WHERE id IN ({ph})"
            )

            # Get the subject entity ID for the junction table
            subject_entity_id = group["subject_entity_id"]
            for irow in interaction_rows:
                # INSERT OR IGNORE avoids duplicates in the junction
                await sql_store.execute_sql(
                    "INSERT OR IGNORE INTO entity_interactions "
                    "(entity_id, interaction_id) "
                    "VALUES (:eid, :iid)",
                    {
                        "eid": subject_entity_id,
                        "iid": irow["interaction_id"],
                    },
                )
            await db.commit()

            logger.info(
                "consolidator_merged_group",
                canonical_id=canonical_id,
                merged_count=len(merged_ids),
                predicate=predicate,
            )

            # -------------------------------------------------------
            # STEP 3: Build a FarmingInsight for this merge.
            # -------------------------------------------------------
            insight = FarmingInsight(
                id=str(uuid4()),
                insight_type="consolidation",
                title=f"Merged {cnt} duplicate facts for {predicate}",
                description=(
                    f"Consolidated {cnt} duplicate facts with predicate "
                    f"'{predicate}' into canonical fact {canonical_id}."
                ),
                evidence=all_fact_ids,
                confidence=max_conf,
                parameters={
                    "duplicate_count": cnt,
                    "canonical_id": canonical_id,
                    "merged_ids": merged_ids,
                },
            )
            insights.append(insight)

        await update_offset(sql_store, "consolidator", offset + batch_size, len(groups))

        logger.info(
            "consolidator_complete",
            groups_merged=len(insights),
            cycle_id=context.cycle_id,
        )
        return insights
