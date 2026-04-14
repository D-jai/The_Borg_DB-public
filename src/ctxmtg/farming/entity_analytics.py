# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Entity Analytics Stage (Intelligence Stage 1)
================================================

Analyses accumulated entity data to discover co-occurrence patterns --
pairs of entities that appear together in multiple interactions.  For
example, "Alice" and "Bob" might co-occur in 12 out of 15 meetings,
suggesting a strong working relationship.

This is the first of seven intelligence stages in the farming pipeline.
It runs a lightweight SQL-based analysis (no LLM needed at this tier)
and emits ``FarmingInsight`` objects of type ``"relationship"`` for
each co-occurring pair that exceeds the configurable threshold.

Algorithm overview:
    1. SEED: Query the entities table for per-entity frequency counts
       (top 100 by mention volume) and co-occurrence pairs (entities
       that share at least ``min_co_occurrence`` interactions).
    2. For each qualifying pair, compute a Jaccard similarity index:
       ``shared / (count_a + count_b - shared)`` to normalise for
       overall frequency.
    3. Emit one ``FarmingInsight`` per pair with the raw counts,
       Jaccard index, and a deterministic insight ID so the pipeline's
       deduplication can handle re-runs gracefully.

Depends on:
    - structlog (structured logging)
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- optional, unused tier 0-1)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async bridge)

Used by:
    - ctxmtg.farming.pipeline (registered as the first intelligence stage)
"""

from __future__ import annotations

from typing import Any

import structlog

from ctxmtg.farming.checkpoint import _run_async
from ctxmtg.interfaces.farming import FarmingContext, FarmingStage
from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.models.farming import FarmingInsight

# ---------------------------------------------------------------
# Module-level logger -- structured JSON output, no PII in logs.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.entity_analytics")

# ---------------------------------------------------------------
# SQL queries -- kept as module constants for readability and to
# make them easy to unit-test independently if needed.
# ---------------------------------------------------------------

# ORIGINAL (disabled 2026-04-07): Hardcoded LIMIT 100/50 with no OFFSET.
# Always processed the same top entities every cycle.
# ENTITY_FREQUENCY_SQL = "... LIMIT 100"
# CO_OCCURRENCE_SQL = "... LIMIT 50"

# Progressive scan: uses LIMIT + OFFSET to cycle through all entities.
ENTITY_FREQUENCY_SQL = """\
SELECT name, entity_type, COUNT(DISTINCT interaction_id) as mention_count
FROM entities
GROUP BY name, entity_type
ORDER BY mention_count DESC
LIMIT 100 OFFSET :offset
"""

# Co-occurrence pairs with OFFSET for progressive scanning.
CO_OCCURRENCE_SQL = """\
SELECT a.name as entity_a, b.name as entity_b,
       COUNT(DISTINCT a.interaction_id) as shared_count
FROM entities a
JOIN entities b
    ON a.interaction_id = b.interaction_id AND a.id < b.id
GROUP BY a.name, b.name
HAVING shared_count >= :min_co
ORDER BY shared_count DESC
LIMIT 50 OFFSET :pair_offset
"""


class EntityAnalyticsStage(FarmingStage):
    """
    Farming stage 1: entity frequency and co-occurrence analysis.

    Discovers which entities appear together across interactions and
    emits ``"relationship"`` insights for pairs that exceed the
    minimum co-occurrence threshold.  The threshold is configurable
    via ``min_co_occurrence`` (default 3) -- lower values find more
    relationships but with weaker evidence.

    The optional ``llm`` parameter is accepted for API consistency
    with higher-tier stages but is not used in this stage's current
    implementation.  Future iterations may use it for natural-language
    summaries of the discovered relationships.

    Usage:
        stage = EntityAnalyticsStage(min_co_occurrence=3)
        insights = stage.run(sql_store, vector_store, context)
    """

    def __init__(
        self,
        min_co_occurrence: int = 3,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Configure the entity analytics stage.

        Args:
            min_co_occurrence: Minimum number of shared interactions
                for a co-occurrence pair to be reported.  Default 3.
            llm: Optional LLM provider for future narrative generation.
                 Currently unused (tier 0-1 does not require an LLM).
        """
        self._min_co = min_co_occurrence
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface
    # -----------------------------------------------------------------

    def get_name(self) -> str:
        """Return the canonical stage name for logging/checkpointing."""
        return "entity_analytics"

    def run(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        """
        Execute entity analytics: frequency + co-occurrence analysis.

        Steps:
            1. Query entity frequency counts from the SQL store.
            2. Query co-occurrence pairs above the threshold.
            3. For each pair, compute Jaccard similarity and build
               a FarmingInsight with deterministic ID.

        This method is synchronous (per the FarmingStage contract)
        but calls async store methods via ``_run_async()``.

        Args:
            sql_store:    SQL store to read entity data from.
            vector_store: Vector store (unused by this stage).
            context:      Farming context with cycle ID and budget.

        Returns:
            List of FarmingInsight objects (type ``"relationship"``).
            Empty list if no qualifying co-occurrence pairs exist.
        """
        logger.info(
            "entity_analytics_start",
            cycle_id=context.cycle_id,
            min_co_occurrence=self._min_co,
        )

        # ----------------------------------------------------------
        # Step 1: Get progressive offset and query entity frequencies.
        # Each cycle processes the NEXT batch of entities, not the same top N.
        # ----------------------------------------------------------
        from ctxmtg.farming.progress import get_offset_with_wrap, update_offset

        # Get total entity count for wrap-around
        total_rows = _run_async(
            sql_store.execute_sql("SELECT COUNT(DISTINCT name) as cnt FROM entities", {})
        )
        total_entities = total_rows[0]["cnt"] if total_rows else 0

        entity_offset = _run_async(
            get_offset_with_wrap(sql_store, "entity_analytics", total_entities, 100)
        )

        freq_rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(ENTITY_FREQUENCY_SQL, {"offset": entity_offset})
        )

        # Map entity name → total distinct interaction count
        freq_map: dict[str, int] = {
            row["name"]: row["mention_count"]
            for row in freq_rows
        }

        logger.debug(
            "entity_frequencies_loaded",
            entity_count=len(freq_map),
            offset=entity_offset,
        )

        # ----------------------------------------------------------
        # Step 2: Query co-occurrence pairs with progressive offset.
        # ----------------------------------------------------------
        total_pair_rows = _run_async(
            sql_store.execute_sql(
                "SELECT COUNT(*) as cnt FROM ("
                "SELECT a.name, b.name FROM entities a "
                "JOIN entities b ON a.interaction_id = b.interaction_id AND a.id < b.id "
                "GROUP BY a.name, b.name HAVING COUNT(DISTINCT a.interaction_id) >= :min_co"
                ")", {"min_co": self._min_co}
            )
        )
        total_pairs = total_pair_rows[0]["cnt"] if total_pair_rows else 0

        pair_offset = _run_async(
            get_offset_with_wrap(sql_store, "entity_analytics_pairs", total_pairs, 50)
        )

        co_rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(
                CO_OCCURRENCE_SQL,
                {"min_co": self._min_co, "pair_offset": pair_offset},
            )
        )

        logger.debug(
            "co_occurrence_pairs_loaded",
            pair_count=len(co_rows),
        )

        # ----------------------------------------------------------
        # Step 3: Build insights for each qualifying pair.
        # ----------------------------------------------------------
        insights: list[FarmingInsight] = []

        for row in co_rows:
            entity_a: str = row["entity_a"]
            entity_b: str = row["entity_b"]
            shared_count: int = row["shared_count"]

            # Look up individual mention counts for Jaccard denominator.
            # Default to shared_count if the entity wasn't in the top 100
            # frequency list (unlikely but defensive).
            count_a = freq_map.get(entity_a, shared_count)
            count_b = freq_map.get(entity_b, shared_count)

            # Jaccard index: |A ∩ B| / |A ∪ B|
            # Union = count_a + count_b - shared (inclusion-exclusion).
            union = count_a + count_b - shared_count
            jaccard = shared_count / union if union > 0 else 0.0

            # Confidence: scales linearly from 0.0 to 1.0 as shared_count
            # goes from 0 to 10.  Capped at 1.0 for strong co-occurrences.
            confidence = min(shared_count / 10.0, 1.0)

            # Deterministic ID so re-running the same cycle overwrites
            # rather than duplicates the insight.
            insight_id = (
                # ORIGINAL (disabled 2026-04-07): included cycle_id, causing duplicates
                # f"co-occur-{entity_a}-{entity_b}-{context.cycle_id}"
                f"co-occur-{entity_a}-{entity_b}"
            )

            insight = FarmingInsight(
                id=insight_id,
                insight_type="relationship",
                title=(
                    f"{entity_a} and {entity_b} co-occur "
                    f"in {shared_count} interactions"
                ),
                confidence=confidence,
                parameters={
                    "entity_a": entity_a,
                    "entity_b": entity_b,
                    "shared_count": shared_count,
                    "jaccard": round(jaccard, 4),
                },
                # No canonical entity IDs available at this stage
                entity_ids=[],
            )
            insights.append(insight)

        # Advance offsets for next cycle
        _run_async(update_offset(sql_store, "entity_analytics", entity_offset + 100, len(freq_rows)))
        _run_async(update_offset(sql_store, "entity_analytics_pairs", pair_offset + 50, len(co_rows)))

        logger.info(
            "entity_analytics_complete",
            cycle_id=context.cycle_id,
            insights_produced=len(insights),
            entity_offset=entity_offset,
            pair_offset=pair_offset,
        )

        return insights
