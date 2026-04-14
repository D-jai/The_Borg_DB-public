# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Self-Learning Feedback Loop Stage
==================================

Implements the FeedbackLoopStage -- a farming stage that reads the
query_quality_log table and generates "gap" insights whenever it
detects poor retrieval quality signals.

Two signal types are mined:

1. **Zero-result queries**: queries where both sql_result_count and
   vector_result_count are zero.  These indicate knowledge gaps --
   the user asked about something the system has no data on.

2. **Refinement queries**: queries where refined_within_60s = 1.
   These are implicit negative feedback -- the user immediately
   re-phrased, suggesting the original results were bad.

Each signal group is aggregated into a FarmingInsight of type "gap".
These gap insights are consumed by the Completionist maintenance
agent in future farming cycles to prioritise what data to acquire
or improve.

Depends on:
    - uuid (insight IDs)
    - structlog (structured logging)
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- optional, unused here)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async bridge)

Used by:
    - ctxmtg.farming.pipeline (registers this as a stage)
    - tests/test_farming/test_feedback_loop.py
"""

from __future__ import annotations

import uuid

import structlog

from ctxmtg.farming.checkpoint import _run_async
from ctxmtg.interfaces.farming import FarmingContext, FarmingStage
from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.models.farming import FarmingInsight

# ---------------------------------------------------------------
# Module-level logger -- logs feedback-loop stage activity.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.feedback_loop")


class FeedbackLoopStage(FarmingStage):
    """
    Farming stage that mines query_quality_log for gap signals.

    The feedback loop is the self-learning backbone of the system.
    It turns implicit user feedback (zero results, quick refinements)
    into actionable "gap" insights that other maintenance agents
    (especially the Completionist) can act on.

    The stage is lightweight and does not require an LLM -- it runs
    pure SQL aggregation queries.  An optional LLM provider is
    accepted for interface consistency but is not used.

    Usage:
        stage = FeedbackLoopStage()
        insights = stage.run(sql_store, vector_store, context)
        for insight in insights:
            await sql_store.store_insight(insight)
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        """
        Initialise the feedback loop stage.

        Args:
            llm: Optional LLM provider (accepted for interface
                 consistency with other stages, but not used).
        """
        # LLM is unused but kept for API parity with other stages
        self._llm = llm

    # =============================================================
    # FarmingStage ABC implementation
    # =============================================================

    def get_name(self) -> str:
        """Return the stage name for logging and checkpointing."""
        return "feedback_loop"

    def run(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        """
        Mine query_quality_log for gap signals and produce insights.

        Steps:
        1. Query for zero-result queries (both stores returned nothing).
        2. Query for refined queries (user re-phrased within 60 s).
        3. For each non-empty signal group, create a "gap" insight
           summarising the count and sample query texts.

        Uses _run_async() from checkpoint.py to bridge the synchronous
        FarmingStage.run() contract with the async SQLStore methods.

        Args:
            sql_store:    SQL store with the query_quality_log table.
            vector_store: Vector store (unused -- signals come from SQL).
            context:      Farming context with cycle ID and budget.

        Returns:
            A list of FarmingInsight objects (type="gap").  Empty if
            no quality signals are detected in the log.
        """
        insights: list[FarmingInsight] = []

        logger.info(
            "feedback_loop_started",
            cycle_id=context.cycle_id,
        )

        # ---------------------------------------------------------
        # Signal 1: Zero-result queries
        # Queries where both SQL and vector stores returned nothing.
        # These represent knowledge gaps in the system.
        # ---------------------------------------------------------
        zero_result_rows = _run_async(
            sql_store.execute_sql(
                "SELECT id, query_text "
                "FROM query_quality_log "
                "WHERE sql_result_count = 0 "
                "AND vector_result_count = 0"
            )
        )

        if zero_result_rows:
            # Collect up to 3 sample query texts for the insight
            sample_queries = [
                row["query_text"] for row in zero_result_rows[:3]
            ]
            count = len(zero_result_rows)

            insights.append(
                FarmingInsight(
                    id=str(uuid.uuid4()),
                    insight_type="gap",
                    title=f"{count} queries returned zero results",
                    description=(
                        f"Detected {count} queries where both the SQL "
                        f"and vector stores returned no results.  This "
                        f"indicates knowledge gaps that the Completionist "
                        f"should address."
                    ),
                    confidence=0.9,
                    parameters={
                        "zero_result_count": count,
                        "sample_queries": sample_queries,
                    },
                )
            )

            logger.info(
                "zero_result_gap_detected",
                count=count,
                sample_queries=sample_queries,
            )

        # ---------------------------------------------------------
        # Signal 2: Refined queries (implicit negative feedback)
        # Queries that the user re-phrased within 60 seconds,
        # suggesting the original results were unsatisfactory.
        # ---------------------------------------------------------
        refined_rows = _run_async(
            sql_store.execute_sql(
                "SELECT id, query_text "
                "FROM query_quality_log "
                "WHERE refined_within_60s = 1"
            )
        )

        if refined_rows:
            # Collect up to 3 sample query texts for the insight
            sample_queries = [
                row["query_text"] for row in refined_rows[:3]
            ]
            count = len(refined_rows)

            insights.append(
                FarmingInsight(
                    id=str(uuid.uuid4()),
                    insight_type="gap",
                    title=(
                        f"{count} queries were refined "
                        f"(implicit negative feedback)"
                    ),
                    description=(
                        f"Detected {count} queries that were refined "
                        f"within 60 seconds.  Quick refinement is a "
                        f"strong signal of poor result quality."
                    ),
                    confidence=0.8,
                    parameters={
                        "refinement_count": count,
                        "sample_queries": sample_queries,
                    },
                )
            )

            logger.info(
                "refinement_gap_detected",
                count=count,
                sample_queries=sample_queries,
            )

        logger.info(
            "feedback_loop_completed",
            cycle_id=context.cycle_id,
            insights_produced=len(insights),
        )

        return insights
