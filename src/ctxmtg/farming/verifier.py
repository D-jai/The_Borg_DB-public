# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Verifier Maintenance Stage
===========================

Verifies high-importance facts by checking for recent corroborating
evidence.  For each important fact (based on its predicate), the
verifier looks for recent facts about the same entity and predicate
within a configurable time window.

CRITICAL INVARIANT: Fact content (subject, predicate, object,
source_span) is NEVER modified.  Only ``facts.confidence`` may be
adjusted.  This is the sole column the verifier touches.

Verification outcomes:
    - CONFIRMED: A recent fact with the same entity + predicate AND
      the same object_literal exists.  Confidence is boosted by
      ``confidence_boost`` (capped at 1.0).
    - CONTRADICTED: A recent fact with the same entity + predicate
      exists but with a DIFFERENT object_literal.  Confidence is
      reduced by ``contradiction_penalty``.
    - UNVERIFIED: No recent facts found for the same entity +
      predicate.  Confidence decays by ``confidence_decay``.

All actions are logged to the maintenance_verifier table for audit
and debugging.

Depends on:
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- reserved for Tier 2+)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async bridge)
    - structlog (structured logging)
    - uuid (unique IDs for maintenance log entries)
    - json (serialisation of log details)

Used by:
    - ctxmtg.farming.pipeline (registered as maintenance stage 10)
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
# Module-level logger -- logs verification events.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.verifier")

# ---------------------------------------------------------------
# Default importance predicates.  These are fact predicates that
# carry high business value and should be periodically verified
# for freshness.  Facts with other predicates are ignored by the
# verifier (they are less sensitive to staleness).
# ---------------------------------------------------------------
DEFAULT_IMPORTANCE_PREDICATES: list[str] = [
    "leads",
    "decided",
    "deadline_is",
    "committed_to",
    "responsible_for",
    "status_is",
    "reports_to",
]


class VerifierStage(FarmingStage):
    """
    Maintenance stage that verifies high-importance facts.

    The verifier scans facts whose predicate is in the importance
    list and checks whether recent corroborating evidence exists.
    Confirmed facts get a confidence boost; unverified or
    contradicted facts have their confidence reduced.

    CRITICAL: Only ``facts.confidence`` is ever updated.  Fact
    content (subject, predicate, object, source_span) is NEVER
    modified by this stage.

    Usage:
        verifier = VerifierStage(verification_window_days=90)
        insights = verifier.run(sql_store, vector_store, context)
    """

    def __init__(
        self,
        verification_window_days: int = 90,
        importance_predicates: list[str] | None = None,
        confidence_boost: float = 0.05,
        confidence_decay: float = 0.1,
        contradiction_penalty: float = 0.3,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Configure the verifier.

        Args:
            verification_window_days: How far back (in days) to look
                for recent corroborating evidence.  Default 90 days.
            importance_predicates: Which fact predicates to verify.
                Defaults to the standard list (leads, decided,
                deadline_is, committed_to, responsible_for,
                status_is, reports_to).
            confidence_boost: How much to increase confidence when a
                fact is confirmed by recent evidence (default 0.05).
            confidence_decay: How much to decrease confidence when no
                recent evidence is found (default 0.1).
            contradiction_penalty: How much to decrease confidence
                when recent evidence contradicts the fact (default 0.3).
            llm: Optional LLM provider for future Tier 2+ semantic
                 verification.  Currently unused.
        """
        self._window_days = verification_window_days
        # Use the default list if none is provided
        self._predicates = (
            importance_predicates
            if importance_predicates is not None
            else list(DEFAULT_IMPORTANCE_PREDICATES)
        )
        self._boost = confidence_boost
        self._decay = confidence_decay
        self._penalty = contradiction_penalty
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface -- stage name for logging/checkpointing.
    # -----------------------------------------------------------------
    def get_name(self) -> str:
        """Return the stage name used for logging and checkpointing."""
        return "verifier"

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
        Verify high-importance facts and adjust their confidence.

        Steps:
        1. SEED: Find facts with high-importance predicates.
        2. For each fact, check for recent corroborating evidence.
        3. Categorize as confirmed / contradicted / unverified.
        4. Update confidence (ONLY confidence -- never content).
        5. Log to maintenance_verifier.
        6. Return a summary FarmingInsight.

        Args:
            sql_store:    SQL store to read/write facts.
            vector_store: Vector store (unused by verifier).
            context:      Farming context with cycle_id and budget.

        Returns:
            List containing a single FarmingInsight summarising
            the verification results.
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
        Async implementation of the verification logic.

        Separated from run() so we can use await on sql_store methods.
        The sync run() method bridges to this via _run_async().
        """
        # ---------------------------------------------------------
        # STEP 1: SEED -- find high-importance facts to verify.
        # Build placeholders dynamically for the predicate list so
        # we can use parameterised queries (no SQL injection risk).
        # ---------------------------------------------------------
        placeholders = ", ".join(
            f":p{i}" for i in range(len(self._predicates))
        )
        # Build the parameter dict for the predicate list
        pred_params: dict[str, str] = {
            f"p{i}": pred for i, pred in enumerate(self._predicates)
        }

        # ORIGINAL (disabled 2026-04-07): Hardcoded LIMIT with no OFFSET.
        # seed_sql = (
        #     "SELECT f.id, f.subject_entity_id, f.predicate, "
        #     "f.object_literal, f.confidence, f.created_at, "
        #     "e.name as entity_name "
        #     "FROM facts f "
        #     "JOIN entities e ON f.subject_entity_id = e.id "
        #     f"WHERE f.predicate IN ({placeholders}) "
        #     "AND f.superseded_by IS NULL "
        #     "ORDER BY f.created_at ASC "
        #     "LIMIT 50"
        # )

        batch_size = 50

        # Get total count for offset wrapping
        total_rows = await sql_store.execute_sql(
            "SELECT COUNT(*) as cnt FROM facts f "
            "JOIN entities e ON f.subject_entity_id = e.id "
            f"WHERE f.predicate IN ({placeholders}) "
            "AND f.superseded_by IS NULL",
            pred_params,
        )
        total_count = total_rows[0]["cnt"] if total_rows else 0

        offset = await get_offset_with_wrap(sql_store, "verifier", total_count, batch_size)

        seed_sql = (
            "SELECT f.id, f.subject_entity_id, f.predicate, "
            "f.object_literal, f.confidence, f.created_at, "
            "e.name as entity_name "
            "FROM facts f "
            "JOIN entities e ON f.subject_entity_id = e.id "
            f"WHERE f.predicate IN ({placeholders}) "
            "AND f.superseded_by IS NULL "
            "ORDER BY f.created_at ASC "
            "LIMIT 50 OFFSET :offset"
        )
        pred_params["offset"] = offset
        facts = await sql_store.execute_sql(seed_sql, pred_params)

        logger.info(
            "verifier_seed_complete",
            facts_to_verify=len(facts),
            predicates=self._predicates,
        )

        # ---------------------------------------------------------
        # Counters for the summary insight
        # ---------------------------------------------------------
        confirmed = 0
        contradicted = 0
        unverified = 0

        # Build the window parameter string (e.g. "-90 days")
        window = f"-{self._window_days} days"

        # ---------------------------------------------------------
        # STEP 2-4: Process each fact.
        # ---------------------------------------------------------
        for fact in facts:
            fid = fact["id"]
            entity_name = fact["entity_name"]
            predicate = fact["predicate"]
            object_literal = fact["object_literal"]
            current_conf = fact["confidence"]

            # ---------------------------------------------------
            # STEP 2: Check for recent evidence with same entity
            # and predicate (within the verification window).
            # ---------------------------------------------------
            evidence_sql = (
                "SELECT COUNT(*) as recent_count "
                "FROM facts f2 "
                "JOIN entities e2 ON f2.subject_entity_id = e2.id "
                "WHERE e2.name = :entity_name "
                "AND f2.predicate = :predicate "
                "AND f2.created_at > DATE('now', :window) "
                "AND f2.superseded_by IS NULL"
            )
            evidence_rows = await sql_store.execute_sql(
                evidence_sql,
                {
                    "entity_name": entity_name,
                    "predicate": predicate,
                    "window": window,
                },
            )
            recent_count = evidence_rows[0]["recent_count"]

            # ---------------------------------------------------
            # STEP 3: Categorize the fact based on evidence.
            # If recent evidence exists, check whether the object
            # literal matches (confirmed) or differs (contradicted).
            # ---------------------------------------------------
            if recent_count > 0:
                # Check if any recent fact has the same object_literal
                # to distinguish confirmation from contradiction.
                same_object_sql = (
                    "SELECT COUNT(*) as same_count "
                    "FROM facts f2 "
                    "JOIN entities e2 ON f2.subject_entity_id = e2.id "
                    "WHERE e2.name = :entity_name "
                    "AND f2.predicate = :predicate "
                    "AND f2.object_literal = :object_literal "
                    "AND f2.created_at > DATE('now', :window) "
                    "AND f2.superseded_by IS NULL"
                )
                same_rows = await sql_store.execute_sql(
                    same_object_sql,
                    {
                        "entity_name": entity_name,
                        "predicate": predicate,
                        "object_literal": object_literal,
                        "window": window,
                    },
                )
                same_count = same_rows[0]["same_count"]

                if same_count > 0:
                    # CONFIRMED: Recent evidence with same object
                    action = "confirmed"
                    new_conf = min(current_conf + self._boost, 1.0)
                    confirmed += 1
                else:
                    # CONTRADICTED: Recent evidence with different object
                    action = "contradicted"
                    new_conf = max(current_conf - self._penalty, 0.0)
                    contradicted += 1
            else:
                # UNVERIFIED: No recent evidence at all
                action = "unverified"
                new_conf = max(current_conf - self._decay, 0.0)
                unverified += 1

            # ---------------------------------------------------
            # STEP 4: Update confidence via execute_sql.
            # CRITICAL: Only confidence is modified.  Fact content
            # (subject, predicate, object, source_span) is NEVER
            # touched by the verifier.
            # ---------------------------------------------------
            await sql_store.execute_sql(
                "UPDATE facts SET confidence = :new_conf WHERE id = :fid",
                {"new_conf": new_conf, "fid": fid},
            )

            # Commit after each update
            db = sql_store._ensure_db()  # type: ignore[attr-defined]
            await db.commit()

            # ---------------------------------------------------
            # STEP 5: Log to maintenance_verifier table.
            # Records the action, target fact ID, and detail JSON.
            # ---------------------------------------------------
            log_id = str(uuid4())
            detail = json.dumps({
                "predicate": predicate,
                "entity_name": entity_name,
                "object_literal": object_literal,
                "old_confidence": current_conf,
                "new_confidence": new_conf,
                "recent_count": recent_count,
            })
            await sql_store.execute_sql(
                "INSERT INTO maintenance_verifier "
                "(id, cycle_id, action, target_ids, detail) "
                "VALUES (:id, :cycle, :action, :targets, :detail)",
                {
                    "id": log_id,
                    "cycle": context.cycle_id,
                    "action": action,
                    "targets": json.dumps([fid]),
                    "detail": detail,
                },
            )
            await db.commit()

            logger.info(
                "verifier_fact_processed",
                fact_id=fid,
                action=action,
                old_confidence=current_conf,
                new_confidence=new_conf,
                recent_count=recent_count,
            )

        # ---------------------------------------------------------
        # STEP 5b: Propagate fact confidence to entity confidence.
        #
        # For every entity that had at least one fact verified in
        # this cycle, recompute entities.confidence as the average
        # confidence of its active (non-superseded) facts.  Entities
        # with no remaining active facts get confidence 0.0.
        #
        # This closes the lifecycle gap: entities.confidence was
        # previously frozen at 1.0 from ingestion and never updated
        # by any farming stage.
        # ---------------------------------------------------------
        affected_entity_ids = list({
            fact["subject_entity_id"] for fact in facts
        })
        entities_updated = 0
        for eid in affected_entity_ids:
            avg_rows = await sql_store.execute_sql(
                "SELECT AVG(confidence) as avg_conf "
                "FROM facts "
                "WHERE subject_entity_id = :eid "
                "AND superseded_by IS NULL",
                {"eid": eid},
            )
            avg_conf = avg_rows[0]["avg_conf"] if avg_rows[0]["avg_conf"] is not None else 0.0
            await sql_store.execute_sql(
                "UPDATE entities SET confidence = :conf WHERE id = :eid",
                {"conf": avg_conf, "eid": eid},
            )
            entities_updated += 1

        if entities_updated > 0:
            db = sql_store._ensure_db()  # type: ignore[attr-defined]
            await db.commit()
            logger.info(
                "verifier_entity_confidence_propagated",
                entities_updated=entities_updated,
            )

        await update_offset(sql_store, "verifier", offset + batch_size, len(facts))

        # ---------------------------------------------------------
        # STEP 6: Return a summary FarmingInsight.
        # The insight summarises how many facts were confirmed,
        # unverified, and contradicted during this cycle.
        # ---------------------------------------------------------
        total = confirmed + unverified + contradicted

        logger.info(
            "verifier_complete",
            total_verified=total,
            confirmed=confirmed,
            unverified=unverified,
            contradicted=contradicted,
            cycle_id=context.cycle_id,
        )

        insight = FarmingInsight(
            id=str(uuid4()),
            insight_type="verification",
            title=(
                f"Verified {total} facts: {confirmed} confirmed, "
                f"{unverified} unverified, {contradicted} contradicted"
            ),
            description=(
                f"Verification pass over {total} high-importance facts. "
                f"{confirmed} confirmed by recent evidence, "
                f"{unverified} had no recent evidence (confidence decayed), "
                f"{contradicted} contradicted by newer information."
            ),
            confidence=1.0,
            parameters={
                "total": total,
                "confirmed": confirmed,
                "unverified": unverified,
                "contradicted": contradicted,
                "window_days": self._window_days,
                "entities_confidence_updated": entities_updated,
            },
        )

        return [insight]
