# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Completionist Maintenance Stage
=================================

Identifies "knowledge gaps" -- entities that appear frequently across
interactions but have very few associated facts.  A high mention-to-
fact ratio suggests the system has seen the entity many times but
failed to extract structured knowledge about it.

The completionist does NOT modify existing facts.  It creates NEW
gap-identification metadata (entity tag enrichment) and logs the
gaps it found.  A future enhancement (Approach A, llm_reextract=True)
could re-run LLM extraction on the source interactions, but the
Phase 3 implementation focuses on Approach B: context enrichment.

Approach B works on all tiers (0-3) because it only writes JSON
tags to existing entity rows and logs the gaps for downstream use.
The gap information can then be used by query enrichment, insight
generation, and human-review workflows.

Additionally, the completionist reads "gap" insights from the
meta_insights table (produced by other farming stages) to
supplement its own entity-frequency analysis.

All actions are logged to the maintenance_completionist table for
audit and debugging.

Depends on:
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- reserved for Tier 2+ re-extraction)
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
# Module-level logger -- logs completionist events.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.completionist")


class CompletionistStage(FarmingStage):
    """
    Maintenance stage that identifies knowledge gaps.

    A "knowledge gap" is an entity that co-occurs in many interactions
    (high mention count) but has very few associated facts (low fact
    count).  This ratio suggests the NER pipeline recognised the
    entity but the extraction pipeline did not capture structured
    relationships about it.

    The completionist:
    1. Finds entities with mention_count >= mention_to_fact_ratio
       and fewer than 2 associated facts.
    2. Reads existing "gap" meta_insights for supplementary info.
    3. Enriches gap entities by updating their tags with gap metadata.
    4. Logs the identified gaps to maintenance_completionist.
    5. Returns a FarmingInsight summarising the discovered gaps.

    The llm_reextract flag is reserved for a future Tier 2+
    enhancement (Approach A) that would re-run LLM extraction on
    the source interactions to fill the gaps.  Phase 3 uses only
    Approach B (context enrichment).

    Usage:
        completionist = CompletionistStage(mention_to_fact_ratio=5)
        insights = completionist.run(sql_store, vector_store, context)
    """

    def __init__(
        self,
        mention_to_fact_ratio: int = 5,
        max_reextractions_per_cycle: int = 10,
        llm_reextract: bool = False,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Configure the completionist.

        Args:
            mention_to_fact_ratio: Minimum number of distinct interactions
                an entity must appear in before it qualifies as a gap
                candidate.  Lower values flag more entities; higher
                values focus on the most-mentioned gaps.
            max_reextractions_per_cycle: Maximum entities to re-extract
                per farming cycle.  Reserved for Approach A (llm_reextract).
                Not used in the current Phase 3 implementation.
            llm_reextract: If True, attempt LLM-driven re-extraction on
                the source interactions for gap entities.  Currently
                disabled (Phase 3 uses Approach B only).
            llm: Optional LLM provider for Approach A re-extraction.
                 Currently unused -- set to None.
        """
        self._mention_to_fact_ratio = mention_to_fact_ratio
        self._max_reextractions = max_reextractions_per_cycle
        self._llm_reextract = llm_reextract
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface -- stage name for logging/checkpointing.
    # -----------------------------------------------------------------
    def get_name(self) -> str:
        """Return the stage name used for logging and checkpointing."""
        return "completionist"

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
        Find entities with knowledge gaps and enrich their context.

        Steps:
        1. SEED: Query for entities with high mention count but low
           fact count (mention_count >= ratio, fact_count < 2).
        2. Read gap insights from meta_insights for supplementary data.
        3. Enrich gap entities with tag metadata.
        4. Log to maintenance_completionist.
        5. Return a summary FarmingInsight.

        Args:
            sql_store:    SQL store to read entities/facts and write tags.
            vector_store: Vector store (unused by completionist).
            context:      Farming context with cycle_id and budget.

        Returns:
            List of FarmingInsight objects describing identified gaps.
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
        Async implementation of the gap-identification logic.

        Separated from run() so we can use await on sql_store methods.
        The sync run() method bridges to this via _run_async().
        """
        # ---------------------------------------------------------
        # STEP 1: SEED -- find entities with high mention count.
        # Groups entities by (name, entity_type) and counts the
        # distinct interactions each entity appears in.  Only
        # entities with mention_count >= ratio are considered.
        # ---------------------------------------------------------
        # ORIGINAL (disabled 2026-04-07): Hardcoded LIMIT with no OFFSET.
        # mention_sql = (
        #     "SELECT name, entity_type, "
        #     "COUNT(DISTINCT interaction_id) as mention_count "
        #     "FROM entities "
        #     "GROUP BY name, entity_type "
        #     "HAVING mention_count >= :ratio "
        #     "ORDER BY mention_count DESC "
        #     "LIMIT 20"
        # )

        mention_batch_size = 20

        # Get total count for offset wrapping
        mention_total_rows = await sql_store.execute_sql(
            "SELECT COUNT(*) as cnt FROM ("
            "SELECT name, entity_type "
            "FROM entities "
            "GROUP BY name, entity_type "
            "HAVING COUNT(DISTINCT interaction_id) >= :ratio"
            ")",
            {"ratio": self._mention_to_fact_ratio},
        )
        mention_total_count = mention_total_rows[0]["cnt"] if mention_total_rows else 0

        mention_offset = await get_offset_with_wrap(
            sql_store, "completionist_mentions", mention_total_count, mention_batch_size
        )

        mention_sql = (
            "SELECT name, entity_type, "
            "COUNT(DISTINCT interaction_id) as mention_count "
            "FROM entities "
            "GROUP BY name, entity_type "
            "HAVING mention_count >= :ratio "
            "ORDER BY mention_count DESC "
            "LIMIT 20 OFFSET :offset"
        )
        high_mention_entities = await sql_store.execute_sql(
            mention_sql, {"ratio": self._mention_to_fact_ratio, "offset": mention_offset}
        )

        logger.info(
            "completionist_seed_mentions",
            high_mention_count=len(high_mention_entities),
            ratio_threshold=self._mention_to_fact_ratio,
        )

        # ---------------------------------------------------------
        # STEP 1b: For each high-mention entity, count its facts.
        # Only entities with fewer than 2 facts qualify as gaps.
        # ---------------------------------------------------------
        gap_entities: list[dict] = []

        for entity in high_mention_entities:
            # Count facts referencing this entity (as subject or object)
            fact_count_rows = await sql_store.execute_sql(
                "SELECT COUNT(*) as fact_count FROM facts f "
                "JOIN entities e "
                "  ON f.subject_entity_id = e.id "
                "  OR f.object_entity_id = e.id "
                "WHERE e.name = :name",
                {"name": entity["name"]},
            )

            fact_count = fact_count_rows[0]["fact_count"] if fact_count_rows else 0

            # Filter: only entities with fewer than 2 facts are gaps
            if fact_count < 2:
                gap_entities.append({
                    "name": entity["name"],
                    "entity_type": entity["entity_type"],
                    "mention_count": entity["mention_count"],
                    "fact_count": fact_count,
                })

        logger.info(
            "completionist_gaps_found",
            gap_count=len(gap_entities),
        )

        # ---------------------------------------------------------
        # STEP 2: Read existing "gap" meta_insights for extra context.
        # These may have been produced by other farming stages
        # (e.g., insight generator detecting sparse subgraphs).
        # ---------------------------------------------------------
        # ORIGINAL (disabled 2026-04-07): Hardcoded LIMIT with no OFFSET.
        # gap_insights_rows = await sql_store.execute_sql(
        #     "SELECT * FROM meta_insights "
        #     "WHERE insight_type = 'gap' "
        #     "ORDER BY created_at DESC "
        #     "LIMIT 10"
        # )

        gap_batch_size = 10

        gap_total_rows = await sql_store.execute_sql(
            "SELECT COUNT(*) as cnt FROM meta_insights "
            "WHERE insight_type = 'gap'",
        )
        gap_total_count = gap_total_rows[0]["cnt"] if gap_total_rows else 0

        gap_offset = await get_offset_with_wrap(
            sql_store, "completionist_gaps", gap_total_count, gap_batch_size
        )

        gap_insights_rows = await sql_store.execute_sql(
            "SELECT * FROM meta_insights "
            "WHERE insight_type = 'gap' "
            "ORDER BY created_at DESC "
            "LIMIT 10 OFFSET :offset",
            {"offset": gap_offset},
        )

        logger.info(
            "completionist_meta_insights_read",
            existing_gap_insights=len(gap_insights_rows),
        )

        # If there are no gaps found, return empty
        if not gap_entities:
            logger.info(
                "completionist_no_gaps",
                cycle_id=context.cycle_id,
            )
            return []

        # ---------------------------------------------------------
        # STEP 3: Enrich gap entities with tag metadata.
        # Updates the entity's tags JSON with gap-related info so
        # downstream queries and enrichers can leverage it.
        # This is Approach B (context enrichment) -- works on all
        # tiers without an LLM.
        # ---------------------------------------------------------
        db = sql_store._ensure_db()  # type: ignore[attr-defined]

        for gap in gap_entities:
            # Build the gap tag payload -- this marks the entity as
            # having a knowledge gap and records cycle metadata.
            gap_tag = json.dumps({
                "gap_identified_cycle": context.cycle_id,
                "mention_count": gap["mention_count"],
                "fact_count": gap["fact_count"],
            })

            # Update the tags column on ALL entity rows matching
            # this name.  Uses json_set to add without clobbering
            # existing tags.
            await sql_store.execute_sql(
                "UPDATE entities "
                "SET tags = json_set(COALESCE(tags, '{}'), "
                "    '$.knowledge_gap', :gap_tag) "
                "WHERE name = :name",
                {"gap_tag": gap_tag, "name": gap["name"]},
            )

        await db.commit()

        # ---------------------------------------------------------
        # STEP 4: Log to maintenance_completionist table.
        # Records the gap-identification action with the list of
        # entity names flagged and a detail summary.
        # ---------------------------------------------------------
        gap_names = [g["name"] for g in gap_entities]
        log_id = str(uuid4())
        detail = json.dumps({
            "gap_entities": [
                {
                    "name": g["name"],
                    "entity_type": g["entity_type"],
                    "mention_count": g["mention_count"],
                    "fact_count": g["fact_count"],
                }
                for g in gap_entities
            ],
            "meta_insights_read": len(gap_insights_rows),
        })

        await sql_store.execute_sql(
            "INSERT INTO maintenance_completionist "
            "(id, cycle_id, action, target_ids, detail) "
            "VALUES (:id, :cycle, 'gap_identified', :targets, :detail)",
            {
                "id": log_id,
                "cycle": context.cycle_id,
                "targets": json.dumps(gap_names),
                "detail": detail,
            },
        )
        await db.commit()

        await update_offset(sql_store, "completionist_mentions", mention_offset + mention_batch_size, len(high_mention_entities))
        await update_offset(sql_store, "completionist_gaps", gap_offset + gap_batch_size, len(gap_insights_rows))

        logger.info(
            "completionist_logged",
            gap_count=len(gap_entities),
            cycle_id=context.cycle_id,
        )

        # ---------------------------------------------------------
        # STEP 5: Return a summary FarmingInsight.
        # One insight summarising all identified gaps.
        # ---------------------------------------------------------
        insight = FarmingInsight(
            id=str(uuid4()),
            insight_type="gap",
            title=f"{len(gap_entities)} knowledge gaps identified",
            description=(
                f"Found {len(gap_entities)} entities with high mention "
                f"counts but fewer than 2 associated facts: "
                f"{', '.join(gap_names)}."
            ),
            evidence=gap_names,
            confidence=0.8,
            parameters={
                "gap_count": len(gap_entities),
                "gap_entities": gap_names,
                "mention_threshold": self._mention_to_fact_ratio,
            },
        )

        logger.info(
            "completionist_complete",
            gaps_identified=len(gap_entities),
            cycle_id=context.cycle_id,
        )
        return [insight]
