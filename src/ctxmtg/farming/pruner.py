# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Pruner Maintenance Stage
=========================

Detects and supersedes stale facts that have been replaced by newer
information.  For example, if "deadline_is March 15" was extracted
from one meeting and "deadline_is April 1" was extracted from a later
meeting, the pruner marks the older fact as superseded by the newer
one.

CRITICAL INVARIANT: Fact content (subject, predicate, object,
source_span) is NEVER modified.  Only the superseded_by column
may be set on the older fact.

The pruner uses a list of "auto-supersede predicates" -- predicates
for which a newer value automatically replaces the older one without
requiring LLM verification.  For predicates NOT in this list, the
pruner currently skips them (Tier 2+ would use an LLM to decide).

All actions are logged to the maintenance_pruner table for audit
and debugging.

Depends on:
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- reserved for Tier 2+ verification)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async bridge)
    - structlog (structured logging)
    - uuid (unique IDs for maintenance log entries)
    - json (serialisation of log details)

Used by:
    - ctxmtg.farming.pipeline (registered as maintenance stage 9)
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
# Module-level logger -- logs pruning events.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.pruner")

# ---------------------------------------------------------------
# Default auto-supersede predicates.  These are predicates where
# a newer value always replaces an older one (temporal semantics).
# For other predicates, LLM verification would be needed (Tier 2+).
# ---------------------------------------------------------------
DEFAULT_AUTO_SUPERSEDE_PREDICATES: list[str] = [
    "deadline_is",
    "status_is",
    "role_is",
    "leads",
    "reports_to",
]


class PrunerStage(FarmingStage):
    """
    Maintenance stage that supersedes stale facts.

    When the same subject + predicate combination has multiple facts
    with different object values and different timestamps, the newer
    fact supersedes the older one (for auto-supersede predicates).

    For example:
    - "Alice deadline_is March 15" (created 2024-01-01)
    - "Alice deadline_is April 1"  (created 2024-02-01)
    → The older "March 15" fact gets superseded_by pointing to the
      newer "April 1" fact.

    Usage:
        pruner = PrunerStage()
        insights = pruner.run(sql_store, vector_store, context)
    """

    def __init__(
        self,
        auto_supersede_predicates: list[str] | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Configure the pruner.

        Args:
            auto_supersede_predicates: Predicates for which a newer
                value automatically replaces the older one without
                LLM verification.  Defaults to the standard list
                (deadline_is, status_is, role_is, leads, reports_to).
            llm: Optional LLM provider for future Tier 2+ verification
                 of non-auto predicates.  Currently unused.
        """
        # Use the default list if none is provided
        self._auto_predicates = (
            auto_supersede_predicates
            if auto_supersede_predicates is not None
            else list(DEFAULT_AUTO_SUPERSEDE_PREDICATES)
        )
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface -- stage name for logging/checkpointing.
    # -----------------------------------------------------------------
    def get_name(self) -> str:
        """Return the stage name used for logging and checkpointing."""
        return "pruner"

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
        Find and supersede stale facts.

        Steps:
        1. SEED: Query for fact pairs where the same subject + predicate
           has different objects at different timestamps.
        2. For each pair: if the predicate is auto-supersede, mark the
           older fact as superseded by the newer one.
        3. Log each action to maintenance_pruner.
        4. Return a FarmingInsight per supersession.

        Args:
            sql_store:    SQL store to read/write facts.
            vector_store: Vector store (unused by pruner).
            context:      Farming context with cycle_id and budget.

        Returns:
            List of FarmingInsight objects describing each supersession.
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
        Async implementation of the pruning logic.

        Separated from run() so we can use await on sql_store methods.
        The sync run() method bridges to this via _run_async().
        """
        # ---------------------------------------------------------
        # STEP 1: SEED -- find potentially superseded fact pairs.
        # Joins facts to itself on (subject_entity_id, predicate)
        # where the two facts have different objects and timestamps.
        # The older fact (f1) is the candidate for supersession.
        # ---------------------------------------------------------
        # ORIGINAL (disabled 2026-04-07): Hardcoded LIMIT with no OFFSET.
        # seed_sql = (
        #     "SELECT f1.id as old_id, f2.id as new_id, "
        #     "f1.predicate, "
        #     "f1.object_literal as old_value, "
        #     "f2.object_literal as new_value, "
        #     "f1.created_at as old_date, "
        #     "f2.created_at as new_date "
        #     "FROM facts f1 "
        #     "JOIN facts f2 "
        #     "  ON f1.subject_entity_id = f2.subject_entity_id "
        #     "  AND f1.predicate = f2.predicate "
        #     "  AND f1.id != f2.id "
        #     "WHERE f1.superseded_by IS NULL "
        #     "  AND f2.superseded_by IS NULL "
        #     "  AND f1.created_at < f2.created_at "
        #     "  AND (f1.object_literal != f2.object_literal "
        #     "       OR f1.object_entity_id != f2.object_entity_id) "
        #     "LIMIT 50"
        # )

        batch_size = 50

        # Get total count for offset wrapping
        total_rows = await sql_store.execute_sql(
            "SELECT COUNT(*) as cnt FROM facts f1 "
            "JOIN facts f2 "
            "  ON f1.subject_entity_id = f2.subject_entity_id "
            "  AND f1.predicate = f2.predicate "
            "  AND f1.id != f2.id "
            "WHERE f1.superseded_by IS NULL "
            "  AND f2.superseded_by IS NULL "
            "  AND f1.created_at < f2.created_at "
            "  AND (f1.object_literal != f2.object_literal "
            "       OR f1.object_entity_id != f2.object_entity_id)",
        )
        total_count = total_rows[0]["cnt"] if total_rows else 0

        offset = await get_offset_with_wrap(sql_store, "pruner", total_count, batch_size)

        seed_sql = (
            "SELECT f1.id as old_id, f2.id as new_id, "
            "f1.predicate, "
            "f1.object_literal as old_value, "
            "f2.object_literal as new_value, "
            "f1.created_at as old_date, "
            "f2.created_at as new_date "
            "FROM facts f1 "
            "JOIN facts f2 "
            "  ON f1.subject_entity_id = f2.subject_entity_id "
            "  AND f1.predicate = f2.predicate "
            "  AND f1.id != f2.id "
            "WHERE f1.superseded_by IS NULL "
            "  AND f2.superseded_by IS NULL "
            "  AND f1.created_at < f2.created_at "
            "  AND (f1.object_literal != f2.object_literal "
            "       OR f1.object_entity_id != f2.object_entity_id) "
            "LIMIT 50 OFFSET :offset"
        )
        pairs = await sql_store.execute_sql(seed_sql, {"offset": offset})

        logger.info(
            "pruner_seed_complete",
            candidate_pairs=len(pairs),
            auto_predicates=self._auto_predicates,
        )

        # Collect one FarmingInsight per supersession
        insights: list[FarmingInsight] = []

        # ---------------------------------------------------------
        # STEP 2: Process each candidate pair.
        # ---------------------------------------------------------
        for pair in pairs:
            predicate = pair["predicate"]
            old_id = pair["old_id"]
            new_id = pair["new_id"]
            old_value = pair["old_value"]
            new_value = pair["new_value"]

            # -------------------------------------------------------
            # Only auto-supersede for predicates in the allow-list.
            # Non-auto predicates would need LLM verification (Tier 2+)
            # -- skip them for now.
            # -------------------------------------------------------
            if predicate not in self._auto_predicates:
                logger.debug(
                    "pruner_skipped_non_auto",
                    predicate=predicate,
                    old_id=old_id,
                    new_id=new_id,
                )
                continue

            # -------------------------------------------------------
            # STEP 2a: Set superseded_by on the older fact.
            # CRITICAL: only superseded_by is modified; content is
            # never touched.
            # -------------------------------------------------------
            await sql_store.execute_sql(
                "UPDATE facts SET superseded_by = :new_id "
                "WHERE id = :old_id",
                {"new_id": new_id, "old_id": old_id},
            )

            # Commit after each supersession
            db = sql_store._ensure_db()  # type: ignore[attr-defined]
            await db.commit()

            # -------------------------------------------------------
            # STEP 2b: Log to maintenance_pruner table.
            # -------------------------------------------------------
            log_id = str(uuid4())
            detail = json.dumps({
                "predicate": predicate,
                "old_value": old_value,
                "new_value": new_value,
                "old_date": pair["old_date"],
                "new_date": pair["new_date"],
            })
            await sql_store.execute_sql(
                "INSERT INTO maintenance_pruner "
                "(id, cycle_id, action, target_ids, canonical_id, detail) "
                "VALUES (:id, :cycle, 'supersede', :targets, :canonical, :detail)",
                {
                    "id": log_id,
                    "cycle": context.cycle_id,
                    "targets": json.dumps([old_id]),
                    "canonical": new_id,
                    "detail": detail,
                },
            )
            await db.commit()

            logger.info(
                "pruner_superseded_fact",
                old_id=old_id,
                new_id=new_id,
                predicate=predicate,
                old_value=old_value,
                new_value=new_value,
            )

            # -------------------------------------------------------
            # STEP 3: Build a FarmingInsight for this supersession.
            # -------------------------------------------------------
            insight = FarmingInsight(
                id=str(uuid4()),
                insight_type="supersession",
                title=f"'{old_value}' superseded by '{new_value}' for {predicate}",
                description=(
                    f"Fact {old_id} ('{old_value}') was superseded by "
                    f"fact {new_id} ('{new_value}') for predicate "
                    f"'{predicate}'."
                ),
                evidence=[old_id, new_id],
                confidence=1.0,
                parameters={
                    "old_id": old_id,
                    "new_id": new_id,
                    "predicate": predicate,
                    "old_value": old_value,
                    "new_value": new_value,
                },
            )
            insights.append(insight)

        await update_offset(sql_store, "pruner", offset + batch_size, len(pairs))

        logger.info(
            "pruner_complete",
            facts_superseded=len(insights),
            cycle_id=context.cycle_id,
        )
        return insights
