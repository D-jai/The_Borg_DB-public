# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Linker Maintenance Stage
=========================

Discovers implicit relationships between entities that frequently
co-occur in the same interactions but have no explicit fact
connecting them.  When two entities appear together in enough
interactions (meeting the Jaccard similarity threshold), the linker
creates a NEW "co_occurs_with" fact to materialise the relationship.

CRITICAL INVARIANT: The linker NEVER modifies existing facts.
It only creates NEW facts with provenance "linker:co-occurrence:v1"
to track that the relationship was inferred (not directly extracted).

The Jaccard similarity score is computed as:
    J(A, B) = |A ∩ B| / |A ∪ B|
where A and B are the sets of interactions mentioning each entity.
A high Jaccard score means the two entities almost always appear
together, suggesting a strong implicit relationship.

Before creating a fact, the linker checks that no explicit fact
already connects the pair (in either direction).  Pairs that are
already linked are skipped.

All actions are logged to the maintenance_linker table for audit
and debugging.

Depends on:
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- reserved for Tier 2+ relationship typing)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async bridge)
    - structlog (structured logging)
    - uuid (unique IDs for facts and maintenance log entries)
    - json (serialisation of log details)

Used by:
    - ctxmtg.farming.pipeline (registered as maintenance stage 11)
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
# Module-level logger -- logs linker events.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.linker")


class LinkerStage(FarmingStage):
    """
    Maintenance stage that discovers implicit entity relationships.

    When two entities appear together in many interactions but have
    no explicit fact connecting them, the linker creates a new
    "co_occurs_with" fact to materialise the implied relationship.

    The linker uses Jaccard similarity to measure co-occurrence
    strength.  Only pairs exceeding the co_occurrence_threshold
    and meeting the min_shared_interactions count are considered.

    Created facts carry:
    - predicate: "co_occurs_with"
    - confidence: inferred_confidence (default 0.6)
    - source_span: provenance string "linker:co-occurrence:v1"
    - interaction_id: first shared interaction (for FK validity)

    Usage:
        linker = LinkerStage(co_occurrence_threshold=0.3)
        insights = linker.run(sql_store, vector_store, context)
    """

    def __init__(
        self,
        co_occurrence_threshold: float = 0.3,
        min_shared_interactions: int = 3,
        inferred_confidence: float = 0.6,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Configure the linker.

        Args:
            co_occurrence_threshold: Minimum Jaccard similarity score
                for a pair to be considered strongly co-occurring.
                Range 0.0-1.0.  Higher values require stronger overlap.
            min_shared_interactions: Minimum number of shared interactions
                before computing Jaccard.  Filters out noise from
                pairs that only co-occur once or twice.
            inferred_confidence: Confidence score assigned to newly
                created co_occurs_with facts.  Lower than extraction-
                derived facts (typically 0.8-1.0) because these are
                inferred, not directly stated.
            llm: Optional LLM provider for future Tier 2+ relationship
                 type inference.  Currently unused -- set to None.
        """
        self._co_occurrence_threshold = co_occurrence_threshold
        self._min_shared_interactions = min_shared_interactions
        self._inferred_confidence = inferred_confidence
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface -- stage name for logging/checkpointing.
    # -----------------------------------------------------------------
    def get_name(self) -> str:
        """Return the stage name used for logging and checkpointing."""
        return "linker"

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
        Find co-occurring entity pairs and create relationship facts.

        Steps:
        1. SEED: Query for entity pairs with shared interactions above
           the min_shared_interactions threshold.
        2. Compute Jaccard similarity for each pair.
        3. Filter pairs by co_occurrence_threshold.
        4. Check each pair for existing explicit facts.
        5. Create new co_occurs_with facts for unconnected pairs.
        6. Log to maintenance_linker.
        7. Return a FarmingInsight summarising discoveries.

        Args:
            sql_store:    SQL store to read entities/facts and write new facts.
            vector_store: Vector store (unused by linker).
            context:      Farming context with cycle_id and budget.

        Returns:
            List of FarmingInsight objects describing new relationships.
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
        Async implementation of the co-occurrence linking logic.

        Separated from run() so we can use await on sql_store methods.
        The sync run() method bridges to this via _run_async().
        """
        # ---------------------------------------------------------
        # STEP 1: SEED -- find entity pairs with high co-occurrence.
        # Joins entities to themselves on interaction_id (same
        # interaction) with a.id < b.id to avoid duplicate pairs
        # and self-joins.  Groups by (name_a, name_b) and counts
        # the distinct shared interactions.
        # ---------------------------------------------------------
        # ORIGINAL (disabled 2026-04-07): Hardcoded LIMIT with no OFFSET.
        # seed_sql = (
        #     "SELECT a.name as entity_a, b.name as entity_b, "
        #     "COUNT(DISTINCT a.interaction_id) as shared "
        #     "FROM entities a "
        #     "JOIN entities b "
        #     "  ON a.interaction_id = b.interaction_id "
        #     "  AND a.id < b.id "
        #     "GROUP BY a.name, b.name "
        #     "HAVING shared >= :min_shared "
        #     "ORDER BY shared DESC "
        #     "LIMIT 30"
        # )

        batch_size = 30

        # Get total count for offset wrapping
        total_rows = await sql_store.execute_sql(
            "SELECT COUNT(*) as cnt FROM ("
            "SELECT a.name, b.name "
            "FROM entities a "
            "JOIN entities b "
            "  ON a.interaction_id = b.interaction_id "
            "  AND a.id < b.id "
            "GROUP BY a.name, b.name "
            "HAVING COUNT(DISTINCT a.interaction_id) >= :min_shared"
            ")",
            {"min_shared": self._min_shared_interactions},
        )
        total_count = total_rows[0]["cnt"] if total_rows else 0

        offset = await get_offset_with_wrap(sql_store, "linker", total_count, batch_size)

        seed_sql = (
            "SELECT a.name as entity_a, b.name as entity_b, "
            "COUNT(DISTINCT a.interaction_id) as shared "
            "FROM entities a "
            "JOIN entities b "
            "  ON a.interaction_id = b.interaction_id "
            "  AND a.id < b.id "
            "GROUP BY a.name, b.name "
            "HAVING shared >= :min_shared "
            "ORDER BY shared DESC "
            "LIMIT 30 OFFSET :offset"
        )
        pairs = await sql_store.execute_sql(
            seed_sql, {"min_shared": self._min_shared_interactions, "offset": offset}
        )

        logger.info(
            "linker_seed_complete",
            candidate_pairs=len(pairs),
            min_shared=self._min_shared_interactions,
        )

        # Track newly created relationships
        created_links: list[dict] = []

        # ---------------------------------------------------------
        # STEP 2-5: Process each candidate pair.
        # ---------------------------------------------------------
        for pair in pairs:
            entity_a = pair["entity_a"]
            entity_b = pair["entity_b"]
            shared = pair["shared"]

            # -------------------------------------------------------
            # STEP 2: Compute Jaccard similarity.
            # J(A,B) = |A ∩ B| / |A ∪ B|
            #         = shared / (count_a + count_b - shared)
            # -------------------------------------------------------
            count_a_rows = await sql_store.execute_sql(
                "SELECT COUNT(DISTINCT interaction_id) as cnt "
                "FROM entities WHERE name = :name",
                {"name": entity_a},
            )
            count_b_rows = await sql_store.execute_sql(
                "SELECT COUNT(DISTINCT interaction_id) as cnt "
                "FROM entities WHERE name = :name",
                {"name": entity_b},
            )

            count_a = count_a_rows[0]["cnt"] if count_a_rows else 0
            count_b = count_b_rows[0]["cnt"] if count_b_rows else 0

            # Compute the union size (avoid division by zero)
            union_size = count_a + count_b - shared
            if union_size <= 0:
                continue  # pragma: no cover -- safety guard

            jaccard = shared / union_size

            # -------------------------------------------------------
            # STEP 3: Filter by co_occurrence_threshold.
            # Pairs below the threshold are not strongly enough
            # correlated to warrant a co_occurs_with fact.
            # -------------------------------------------------------
            if jaccard < self._co_occurrence_threshold:
                logger.debug(
                    "linker_below_threshold",
                    entity_a=entity_a,
                    entity_b=entity_b,
                    jaccard=jaccard,
                    threshold=self._co_occurrence_threshold,
                )
                continue

            # -------------------------------------------------------
            # STEP 4: Check for existing explicit facts connecting
            # this pair.  We check both directions (a→b and b→a)
            # to avoid creating redundant co-occurrence facts.
            # -------------------------------------------------------
            existing_rows = await sql_store.execute_sql(
                "SELECT COUNT(*) as cnt FROM facts f "
                "JOIN entities e1 ON f.subject_entity_id = e1.id "
                "JOIN entities e2 ON f.object_entity_id = e2.id "
                "WHERE (e1.name = :name_a AND e2.name = :name_b) "
                "   OR (e1.name = :name_b AND e2.name = :name_a)",
                {"name_a": entity_a, "name_b": entity_b},
            )

            existing_count = existing_rows[0]["cnt"] if existing_rows else 0

            if existing_count > 0:
                logger.debug(
                    "linker_already_connected",
                    entity_a=entity_a,
                    entity_b=entity_b,
                    existing_facts=existing_count,
                )
                continue

            # -------------------------------------------------------
            # STEP 5: Create a NEW co_occurs_with fact.
            # We need a valid subject_entity_id and interaction_id
            # from one of the shared interactions.  Look up the
            # actual entity row for entity_a in a shared interaction.
            # -------------------------------------------------------
            entity_row = await sql_store.execute_sql(
                "SELECT e.id as entity_id, e.interaction_id "
                "FROM entities e "
                "WHERE e.name = :name "
                "LIMIT 1",
                {"name": entity_a},
            )

            # Safety: skip if we can't find the entity row
            if not entity_row:
                continue  # pragma: no cover

            subject_entity_id = entity_row[0]["entity_id"]
            interaction_id = entity_row[0]["interaction_id"]

            # Generate a unique ID for the new fact
            fact_id = str(uuid4())

            # INSERT the new co_occurs_with fact with provenance tag.
            # source_span carries the provenance string so it's clear
            # this fact was inferred, not directly extracted.
            await sql_store.execute_sql(
                "INSERT INTO facts "
                "(id, interaction_id, subject_entity_id, predicate, "
                "object_literal, confidence, source_span) "
                "VALUES (:id, :iid, :eid, 'co_occurs_with', "
                ":obj, :conf, 'linker:co-occurrence:v1')",
                {
                    "id": fact_id,
                    "iid": interaction_id,
                    "eid": subject_entity_id,
                    "obj": entity_b,
                    "conf": self._inferred_confidence,
                },
            )

            # Commit after each fact creation
            db = sql_store._ensure_db()  # type: ignore[attr-defined]
            await db.commit()

            logger.info(
                "linker_created_fact",
                fact_id=fact_id,
                entity_a=entity_a,
                entity_b=entity_b,
                jaccard=jaccard,
                shared=shared,
            )

            # Track the link for logging and insight generation
            created_links.append({
                "fact_id": fact_id,
                "entity_a": entity_a,
                "entity_b": entity_b,
                "jaccard": round(jaccard, 4),
                "shared_interactions": shared,
            })

        # ---------------------------------------------------------
        # STEP 6: Log to maintenance_linker table.
        # One log entry summarising all links created this cycle.
        # ---------------------------------------------------------
        if created_links:
            db = sql_store._ensure_db()  # type: ignore[attr-defined]

            log_id = str(uuid4())
            # Target IDs = list of newly created fact IDs
            target_ids = [link["fact_id"] for link in created_links]
            detail = json.dumps({
                "links_created": len(created_links),
                "links": created_links,
            })

            await sql_store.execute_sql(
                "INSERT INTO maintenance_linker "
                "(id, cycle_id, action, target_ids, detail) "
                "VALUES (:id, :cycle, 'co_occurrence_link', :targets, :detail)",
                {
                    "id": log_id,
                    "cycle": context.cycle_id,
                    "targets": json.dumps(target_ids),
                    "detail": detail,
                },
            )
            await db.commit()

        await update_offset(sql_store, "linker", offset + batch_size, len(pairs))

        logger.info(
            "linker_complete",
            links_created=len(created_links),
            cycle_id=context.cycle_id,
        )

        # ---------------------------------------------------------
        # STEP 7: Return a FarmingInsight summarising discoveries.
        # Returns empty list if no new links were created.
        # ---------------------------------------------------------
        if not created_links:
            return []

        # Build a single insight summarising all new links
        pair_descriptions = [
            f"{link['entity_a']} ↔ {link['entity_b']}"
            for link in created_links
        ]
        insight = FarmingInsight(
            id=str(uuid4()),
            insight_type="relationship",
            title=f"Discovered {len(created_links)} implicit relationships",
            description=(
                f"Created {len(created_links)} co_occurs_with facts for "
                f"entity pairs that frequently appear together: "
                f"{', '.join(pair_descriptions)}."
            ),
            evidence=[link["fact_id"] for link in created_links],
            confidence=self._inferred_confidence,
            parameters={
                "links_created": len(created_links),
                "threshold": self._co_occurrence_threshold,
                "min_shared": self._min_shared_interactions,
                "pairs": [
                    {"a": l["entity_a"], "b": l["entity_b"], "jaccard": l["jaccard"]}
                    for l in created_links
                ],
            },
        )
        return [insight]
