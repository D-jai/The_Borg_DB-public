# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Causal Miner Stage (Intelligence Stage 14)
=============================================

Discovers causal relationships between predicates by analysing
time-lagged co-occurrence patterns in the facts table.  When
predicate A consistently precedes predicate B for the same entity
within a configurable time window, this stage infers a potential
causal link: "When A happens, B tends to follow."

This is a statistical heuristic, not a formal causal inference
algorithm.  It identifies correlations with temporal ordering
(A before B) and filters by reliability (how often B follows A
vs how often B happens on its own).

Algorithm overview:
    1. SEED: Query time-lagged predicate co-occurrences from the
       facts table.  Two facts are co-occurring if they share the
       same ``subject_entity_id``, the second was created after the
       first, and the time gap is within ``window_days``.  Only
       pairs with at least ``min_occurrences`` co-occurrences are
       kept (top 30 by frequency).
    2. CROSS-REFERENCE: For each candidate pair (A, B), count the
       total occurrences of B (regardless of A) to compute a
       reliability ratio: ``occurrences / total_b``.  This measures
       how much A predicts B beyond B's base rate.
    3. FILTER: Discard pairs where reliability ≤ 0.2 (B happens
       too often without A for the relationship to be meaningful).
    4. EMIT: Create a ``FarmingInsight`` of type ``"causal"`` for
       each surviving pair, with deterministic ID for deduplication.

Depends on:
    - structlog (structured logging)
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- optional, unused)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async)

Used by:
    - ctxmtg.farming.pipeline (registered as intelligence stage 14)
"""

from __future__ import annotations

from typing import Any

import structlog

from ctxmtg.farming.checkpoint import _run_async
from ctxmtg.farming.progress import get_offset_with_wrap, update_offset
from ctxmtg.interfaces.farming import FarmingContext, FarmingStage
from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.models.farming import FarmingInsight

# ---------------------------------------------------------------
# Module-level logger -- structured JSON output, no PII in logs.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.causal_miner")

# ---------------------------------------------------------------
# SQL queries -- kept as module constants for readability and to
# make them easy to unit-test independently if needed.
# ---------------------------------------------------------------

# Time-lagged predicate co-occurrence query.
# Finds pairs of predicates (event_a, event_b) that occur for the
# same subject entity, where event_b follows event_a within the
# specified time window.  Only active (non-superseded) facts are
# considered.  Results are grouped by predicate pair, filtered by
# minimum occurrence count, and sorted by frequency (top 30).
# ORIGINAL (disabled 2026-04-07): Hardcoded LIMIT with no OFFSET.
# CO_OCCURRENCE_SQL = """\
# SELECT f1.predicate AS event_a, f2.predicate AS event_b,
#        COUNT(*) AS occurrences,
#        AVG(julianday(f2.created_at) - julianday(f1.created_at)) AS avg_lag_days
# FROM facts f1
# JOIN facts f2 ON f1.subject_entity_id = f2.subject_entity_id
#              AND f2.created_at > f1.created_at
#              AND julianday(f2.created_at) - julianday(f1.created_at) < :window
# WHERE f1.superseded_by IS NULL AND f2.superseded_by IS NULL
# GROUP BY f1.predicate, f2.predicate
# HAVING occurrences >= :min_occ
# ORDER BY occurrences DESC
# LIMIT 30
# """

CO_OCCURRENCE_SQL = """\
SELECT f1.predicate AS event_a, f2.predicate AS event_b,
       COUNT(*) AS occurrences,
       AVG(julianday(f2.created_at) - julianday(f1.created_at)) AS avg_lag_days
FROM facts f1
JOIN facts f2 ON f1.subject_entity_id = f2.subject_entity_id
             AND f2.created_at > f1.created_at
             AND julianday(f2.created_at) - julianday(f1.created_at) < :window
WHERE f1.superseded_by IS NULL AND f2.superseded_by IS NULL
  AND f1.source_span NOT LIKE 'linker:%'
  AND f2.source_span NOT LIKE 'linker:%'
GROUP BY f1.predicate, f2.predicate
HAVING occurrences >= :min_occ
ORDER BY occurrences DESC
LIMIT 30 OFFSET :offset
"""

CO_OCCURRENCE_COUNT_SQL = """\
SELECT COUNT(*) AS cnt FROM (
    SELECT f1.predicate, f2.predicate
    FROM facts f1
    JOIN facts f2 ON f1.subject_entity_id = f2.subject_entity_id
                 AND f2.created_at > f1.created_at
                 AND julianday(f2.created_at) - julianday(f1.created_at) < :window
    WHERE f1.superseded_by IS NULL AND f2.superseded_by IS NULL
      AND f1.source_span NOT LIKE 'linker:%'
      AND f2.source_span NOT LIKE 'linker:%'
    GROUP BY f1.predicate, f2.predicate
    HAVING COUNT(*) >= :min_occ
)
"""

# Base-rate query: count total occurrences of a given predicate
# across all active (non-superseded) facts.  Used to compute the
# reliability ratio for each candidate pair.
BASE_RATE_SQL = """\
SELECT COUNT(*) as total_b
FROM facts
WHERE predicate = :pred_b AND superseded_by IS NULL
"""

# ---------------------------------------------------------------
# Reliability threshold: discard pairs where B doesn't follow A
# often enough relative to B's overall occurrence rate.
# A ratio of 0.2 means "at least 20% of all B events are
# preceded by A within the time window".
# ---------------------------------------------------------------
RELIABILITY_THRESHOLD = 0.2


class CausalMinerStage(FarmingStage):
    """
    Farming stage 14: time-lagged causal relationship discovery.

    Analyses temporal patterns in the facts table to discover
    predicate pairs where one consistently precedes the other.
    For example, if "raised_concerns" is often followed by
    "timeline_extended" within 14 days for the same entity,
    this stage will emit a causal insight.

    The optional ``llm`` parameter is accepted for API consistency
    with higher-tier stages but is not used in this stage's current
    implementation.  Future iterations may use it for natural-language
    summaries of the discovered causal relationships.

    Usage:
        stage = CausalMinerStage(window_days=14, min_occurrences=3)
        insights = stage.run(sql_store, vector_store, context)
    """

    def __init__(
        self,
        window_days: int = 14,
        min_occurrences: int = 3,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Configure the causal miner stage.

        Args:
            window_days: Maximum number of days between event A and
                event B for the pair to be counted as co-occurring.
                Default 14 (two weeks).
            min_occurrences: Minimum number of times the (A, B) pair
                must co-occur before being considered a candidate.
                Default 3.
            llm: Optional LLM provider for future narrative generation.
                 Currently unused (pure SQL-based analysis).
        """
        self._window_days = window_days
        self._min_occurrences = min_occurrences
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface
    # -----------------------------------------------------------------

    def get_name(self) -> str:
        """Return the canonical stage name for logging/checkpointing."""
        return "causal_miner"

    def run(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        """
        Execute causal mining: time-lagged predicate co-occurrence.

        Steps:
            1. SEED: Query time-lagged co-occurrence pairs from the
               facts table (same subject entity, B after A, within
               window_days).
            2. CROSS-REFERENCE: For each candidate pair, compute
               reliability = occurrences / total_b (base rate of B).
            3. FILTER: Keep only pairs with reliability > 0.2.
            4. EMIT: Build a FarmingInsight per surviving pair.

        This method is synchronous (per the FarmingStage contract)
        but calls async store methods via ``_run_async()``.

        Args:
            sql_store:    SQL store to read fact data from.
            vector_store: Vector store (unused by this stage).
            context:      Farming context with cycle ID and budget.

        Returns:
            List of FarmingInsight objects (type ``"causal"``).
            Empty list if no qualifying causal pairs exist.
        """
        logger.info(
            "causal_miner_start",
            cycle_id=context.cycle_id,
            window_days=self._window_days,
            min_occurrences=self._min_occurrences,
        )

        # ----------------------------------------------------------
        # Step 1 (SEED): Query time-lagged predicate co-occurrences.
        # This finds all (A, B) pairs where predicate B follows
        # predicate A for the same entity within the time window.
        # ----------------------------------------------------------
        batch_size = 30

        # Get total count for offset wrapping
        total_rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(
                CO_OCCURRENCE_COUNT_SQL,
                {
                    "window": self._window_days,
                    "min_occ": self._min_occurrences,
                },
            )
        )
        total_count = total_rows[0]["cnt"] if total_rows else 0

        offset = _run_async(get_offset_with_wrap(sql_store, "causal_miner", total_count, batch_size))

        co_rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(
                CO_OCCURRENCE_SQL,
                {
                    "window": self._window_days,
                    "min_occ": self._min_occurrences,
                    "offset": offset,
                },
            )
        )

        logger.debug(
            "causal_candidates_loaded",
            candidate_count=len(co_rows),
        )

        # ----------------------------------------------------------
        # Steps 2-3 (CROSS-REFERENCE + FILTER): For each candidate,
        # compute reliability and discard weak relationships.
        # ----------------------------------------------------------
        insights: list[FarmingInsight] = []

        for row in co_rows:
            event_a: str = row["event_a"]
            event_b: str = row["event_b"]
            occurrences: int = row["occurrences"]
            avg_lag_days: float = row["avg_lag_days"]

            # Query the base rate of event B (total occurrences
            # regardless of whether A preceded it).
            base_rows: list[dict[str, Any]] = _run_async(
                sql_store.execute_sql(
                    BASE_RATE_SQL,
                    {"pred_b": event_b},
                )
            )

            # Extract total_b; default to 0 if no rows returned
            # (shouldn't happen since B appeared in co-occurrences).
            total_b: int = (
                base_rows[0]["total_b"] if base_rows else 0
            )

            # Compute reliability: how often B follows A vs how
            # often B happens overall.  A ratio of 1.0 means B
            # *always* follows A; 0.0 means the co-occurrence is
            # purely coincidental.
            reliability: float = (
                occurrences / total_b if total_b > 0 else 0.0
            )

            # Filter out weak relationships (reliability ≤ 0.2).
            if reliability <= RELIABILITY_THRESHOLD:
                logger.debug(
                    "causal_pair_filtered",
                    event_a=event_a,
                    event_b=event_b,
                    reliability=round(reliability, 4),
                )
                continue

            # ----------------------------------------------------------
            # Step 4 (EMIT): Build a FarmingInsight for each qualifying
            # causal pair with a deterministic ID for deduplication.
            # ----------------------------------------------------------

            # Deterministic ID: ensures re-running the same cycle
            # overwrites rather than duplicates the insight.
            insight_id = (
                # ORIGINAL: f"causal-{event_a}-{event_b}-{context.cycle_id}"
                f"causal-{event_a}-{event_b}"
            )

            # Confidence is capped at 1.0 (reliability can exceed
            # 1.0 if B always co-occurs with A but also happens
            # independently -- though that's rare with this formula).
            confidence = min(reliability, 1.0)

            insight = FarmingInsight(
                id=insight_id,
                insight_type="causal",
                title=(
                    f"When '{event_a}' occurs, '{event_b}' tends "
                    f"to follow ({occurrences}x observed)"
                ),
                confidence=confidence,
                parameters={
                    "event_a": event_a,
                    "event_b": event_b,
                    "occurrences": occurrences,
                    "avg_lag_days": round(avg_lag_days, 4),
                    "reliability": round(reliability, 4),
                },
                # No canonical entity IDs at this stage
                entity_ids=[],
            )
            insights.append(insight)

        _run_async(update_offset(sql_store, "causal_miner", offset + batch_size, len(co_rows)))

        logger.info(
            "causal_miner_complete",
            cycle_id=context.cycle_id,
            insights_produced=len(insights),
        )

        return insights
