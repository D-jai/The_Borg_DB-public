# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Insight Generator Stage (Intelligence Stage 6)
=================================================

Generates meta-insights by comparing the current farming cycle's
discoveries with those from the previous cycle.  This produces a
"summary of summaries" -- a higher-order view that tracks how the
knowledge base is evolving over time.

For example, if this cycle produced 5 new "relationship" insights
that didn't exist last cycle, and 2 "trend" insights disappeared,
the meta-insight would capture that shift: "5 new relationships
emerged, 2 trends faded."

This is the sixth of seven intelligence stages in the farming
pipeline.  It performs pure SQL-based comparison (no LLM needed at
this tier) and emits a single ``FarmingInsight`` of type ``"meta"``
summarising the cycle-over-cycle changes.

Algorithm overview:
    1. Query current cycle's insights from meta_insights (last 1 day).
    2. Query previous cycle's insights from meta_insights (1-7 days ago).
    3. Compare: count insight types in each set, identify new and
       disappeared types.
    4. Emit one ``"meta"`` FarmingInsight summarising the comparison
       with counts and type breakdowns in the parameters dict.
    5. Return an empty list if there is no comparison data (first
       cycle or no prior insights).

Depends on:
    - structlog (structured logging)
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- optional, unused tier 0-1)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async bridge)

Used by:
    - ctxmtg.farming.pipeline (registered as the sixth intelligence stage)
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
logger = structlog.get_logger("ctxmtg.farming.insight_generator")

# ---------------------------------------------------------------
# SQL queries: fetch insights from the current cycle (last day)
# and the previous cycle window (1-7 days ago).  The time windows
# are chosen to align with typical daily farming schedules.
# ---------------------------------------------------------------

# Current cycle: insights created in the last 24 hours.
# These represent the latest discoveries from this farming run.
CURRENT_CYCLE_SQL = """\
SELECT * FROM meta_insights
WHERE created_at > DATE('now', '-1 day')
ORDER BY created_at DESC
LIMIT 100
"""

# Previous cycle: insights created between 1 and 7 days ago.
# This window captures the most recent prior cycle's output
# for comparison, with enough breadth to handle irregular
# farming schedules.
PREVIOUS_CYCLE_SQL = """\
SELECT * FROM meta_insights
WHERE created_at <= DATE('now', '-1 day')
  AND created_at > DATE('now', '-7 days')
ORDER BY created_at DESC
LIMIT 100
"""


def _count_by_type(rows: list[dict[str, Any]]) -> dict[str, int]:
    """
    Count the number of insights per insight_type.

    Iterates through SQL result rows and tallies each distinct
    insight_type value.  Used for both current and previous cycle
    data to enable type-level comparison.

    Args:
        rows: List of row dicts from the meta_insights table.
              Each must have an ``insight_type`` key.

    Returns:
        Dict mapping insight_type strings to their occurrence count.
        E.g., {"relationship": 5, "trend": 3, "cluster": 2}
    """
    counts: dict[str, int] = {}
    for row in rows:
        itype = row.get("insight_type", "unknown")
        counts[itype] = counts.get(itype, 0) + 1
    return counts


class InsightGeneratorStage(FarmingStage):
    """
    Farming stage 6: meta-insight generation from cycle comparison.

    Compares the current cycle's insights with the previous cycle's
    insights to produce a higher-order "meta" insight that tracks
    knowledge evolution.  This enables the system to detect shifts
    in the knowledge landscape -- for example, new relationships
    emerging or old trends fading.

    The stage reads directly from the meta_insights table (which
    stores output from all prior farming stages) and writes a
    single ``"meta"`` insight summarising the delta.

    The optional ``llm`` parameter is accepted for API consistency
    but is not used in this stage's current implementation.  Future
    iterations may use it for natural-language summaries.

    Usage:
        stage = InsightGeneratorStage()
        insights = stage.run(sql_store, vector_store, context)
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Configure the insight generator stage.

        Args:
            llm: Optional LLM provider for future narrative generation.
                 Currently unused (tier 0-1 does not require an LLM).
        """
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface
    # -----------------------------------------------------------------

    def get_name(self) -> str:
        """Return the canonical stage name for logging/checkpointing."""
        return "insight_generation"

    def run(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        """
        Execute meta-insight generation: compare current vs. previous cycle.

        Steps:
            1. Query current cycle's insights (last 1 day).
            2. Query previous cycle's insights (1-7 days ago).
            3. Count insight types in each set and compute deltas.
            4. Build a single ``"meta"`` FarmingInsight summarising
               the comparison.

        This method is synchronous (per the FarmingStage contract)
        but calls async store methods via ``_run_async()``.

        Args:
            sql_store:    SQL store to read meta_insights from.
            vector_store: Vector store (unused by this stage).
            context:      Farming context with cycle ID and budget.

        Returns:
            List containing one ``"meta"`` FarmingInsight, or an
            empty list if there is no comparison data available.
        """
        logger.info(
            "insight_generation_start",
            cycle_id=context.cycle_id,
        )

        # ----------------------------------------------------------
        # Step 1: Query current cycle's insights (last 24 hours).
        # ----------------------------------------------------------
        current_rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(CURRENT_CYCLE_SQL, {})
        )

        logger.debug(
            "current_cycle_loaded",
            row_count=len(current_rows),
        )

        # ----------------------------------------------------------
        # Step 2: Query previous cycle's insights (1-7 days ago).
        # ----------------------------------------------------------
        previous_rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(PREVIOUS_CYCLE_SQL, {})
        )

        logger.debug(
            "previous_cycle_loaded",
            row_count=len(previous_rows),
        )

        # ----------------------------------------------------------
        # Step 3: Count insight types and compute deltas.
        # ----------------------------------------------------------
        current_types = _count_by_type(current_rows)
        previous_types = _count_by_type(previous_rows)

        # Identify insight types that are new (in current but not
        # in previous) or disappeared (in previous but not in current).
        all_types = set(current_types.keys()) | set(previous_types.keys())

        # New types: present in current cycle but absent from previous.
        new_types = {
            t: current_types[t]
            for t in current_types
            if t not in previous_types
        }

        # Disappeared types: present in previous but absent from current.
        disappeared_types = {
            t: previous_types[t]
            for t in previous_types
            if t not in current_types
        }

        # Total new insight count for the summary
        new_insight_count = len(current_rows)

        logger.debug(
            "cycle_comparison",
            current_count=len(current_rows),
            previous_count=len(previous_rows),
            new_types=len(new_types),
            disappeared_types=len(disappeared_types),
        )

        # ----------------------------------------------------------
        # Step 4: Build the meta-insight.
        # We always produce one meta-insight per cycle to track the
        # evolution of the knowledge base, even if counts are zero.
        # ----------------------------------------------------------

        # Build a human-readable title summarising the comparison.
        title_parts: list[str] = [
            f"Cycle {context.cycle_id}:",
            f"{new_insight_count} insights",
        ]
        if new_types:
            title_parts.append(
                f"({len(new_types)} new type{'s' if len(new_types) != 1 else ''})"
            )
        if disappeared_types:
            title_parts.append(
                f"({len(disappeared_types)} faded type{'s' if len(disappeared_types) != 1 else ''})"
            )

        title = " ".join(title_parts)

        # Confidence: proportional to the amount of data available.
        # More data → higher confidence in the comparison.
        total_rows = len(current_rows) + len(previous_rows)
        confidence = min(total_rows / 50.0, 1.0)
        # Floor at 0.1 to avoid near-zero confidence
        confidence = max(confidence, 0.1)

        # Deterministic ID for deduplication across cycles.
        insight_id = f"meta-cycle-{context.cycle_id}"

        # Build the parameters dict with all comparison details.
        # This provides structured data for downstream consumers
        # (e.g., dashboards, reports, or future LLM summarisation).
        parameters: dict[str, Any] = {
            "new_insights": new_insight_count,
            "previous_insights": len(previous_rows),
            "insight_types": {
                t: current_types.get(t, 0) for t in all_types
            },
            "new_types": new_types,
            "disappeared_types": disappeared_types,
            "cycle_id": context.cycle_id,
        }

        meta_insight = FarmingInsight(
            id=insight_id,
            insight_type="meta",
            title=title,
            confidence=confidence,
            parameters=parameters,
            # No canonical entity IDs for meta-insights
            entity_ids=[],
        )

        logger.info(
            "insight_generation_complete",
            cycle_id=context.cycle_id,
            insights_produced=1,
            new_insights=new_insight_count,
        )

        return [meta_insight]
