# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Calibrator Stage
=================

Evaluates the accuracy of historical trend predictions by checking
whether previously detected trends continued, weakened, or reversed.
This provides a feedback signal for tuning the farming pipeline's
confidence in its own predictions.

The calibrator queries trend insights older than 7 days from the
meta_insights table, then checks each entity's current mention
frequency to see if the predicted direction held.  It produces a
single ``FarmingInsight`` of type ``"meta"`` summarising the
prediction accuracy.

Algorithm overview:
    1. Query historical trend insights (created > 7 days ago).
    2. For each trend, parse the entity name and slope direction
       from the title and parameters.
    3. Query the entity's recent mention frequency to assess
       whether the trend persisted, flattened, or reversed.
    4. Compute overall accuracy: confirmed / total_checked.
    5. Return a meta-insight with the accuracy score.

Depends on:
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- reserved for Tier 2+)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async bridge)
    - structlog (structured logging)
    - json (deserialisation of parameters column)
    - uuid (unique IDs for insights)

Used by:
    - ctxmtg.farming.pipeline (registered as the calibrator stage)
"""

from __future__ import annotations

import json
from uuid import uuid4

import structlog

from ctxmtg.farming.checkpoint import _run_async
from ctxmtg.interfaces.farming import FarmingContext, FarmingStage
from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.models.farming import FarmingInsight

# ---------------------------------------------------------------
# Module-level logger -- logs calibration events.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.calibrator")


class CalibratorStage(FarmingStage):
    """
    Calibration stage that evaluates trend prediction accuracy.

    The calibrator acts as the pipeline's self-assessment mechanism.
    It reviews historical trend insights to see whether the predicted
    direction (rising or declining) actually held over subsequent
    days.  The resulting accuracy score can inform confidence
    adjustments in future farming cycles.

    Scoring categories:
        - **confirmed**: The trend continued in the predicted direction.
        - **weakened**: The trend flattened (entity still appears but
          without a clear directional signal).
        - **falsified**: The trend reversed (rising became declining,
          or vice versa).

    Usage:
        calibrator = CalibratorStage()
        insights = calibrator.run(sql_store, vector_store, context)
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        """
        Configure the calibrator.

        Args:
            llm: Optional LLM provider for future Tier 2+ semantic
                 calibration.  Currently unused.
        """
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface -- stage name for logging/checkpointing.
    # -----------------------------------------------------------------
    def get_name(self) -> str:
        """Return the stage name used for logging and checkpointing."""
        return "calibrator"

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
        Evaluate historical trend prediction accuracy.

        Steps:
        1. Query historical trend insights (> 7 days old).
        2. For each trend, parse entity name and slope direction.
        3. Query current entity frequency to check persistence.
        4. Score: confirmed / weakened / falsified.
        5. Compute overall accuracy and return a meta-insight.

        Args:
            sql_store:    SQL store to read meta_insights and entities.
            vector_store: Vector store (unused by calibrator).
            context:      Farming context with cycle_id and budget.

        Returns:
            List containing a single FarmingInsight with the
            calibration results.
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
        Async implementation of the calibration logic.

        Separated from run() so we can use await on sql_store methods.
        The sync run() method bridges to this via _run_async().
        """
        # ---------------------------------------------------------
        # STEP 1: Query historical trend insights.
        # Only look at trends created more than 7 days ago, giving
        # enough time for the predicted direction to play out.
        # ---------------------------------------------------------
        trend_sql = (
            "SELECT * FROM meta_insights "
            "WHERE insight_type = 'trend' "
            "AND created_at < DATE('now', '-7 days') "
            "ORDER BY created_at DESC "
            "LIMIT 20"
        )
        trends = await sql_store.execute_sql(trend_sql)

        logger.info(
            "calibrator_seed_complete",
            historical_trends=len(trends),
        )

        # ---------------------------------------------------------
        # Counters for the summary
        # ---------------------------------------------------------
        confirmed_count = 0
        weakened_count = 0
        falsified_count = 0
        total_checked = 0

        # ---------------------------------------------------------
        # STEP 2-3: Process each historical trend.
        # ---------------------------------------------------------
        for trend in trends:
            # Parse the parameters JSON to get the slope
            params = trend.get("parameters", "{}")
            if isinstance(params, str):
                # The parameters column is stored as JSON text
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    # Skip trends with malformed parameters
                    continue
            # Safety: params must be a dict with a slope key
            if not isinstance(params, dict) or "slope" not in params:
                continue

            original_slope = params["slope"]

            # Parse entity name from the title.
            # Trend titles follow the format:
            #   "{entity_name} {direction} ({slope}/day)"
            # We extract the entity name by splitting on " rising"
            # or " declining".
            title = trend.get("title", "")
            entity_name = _parse_entity_name(title)
            if not entity_name:
                # Could not parse entity name -- skip this trend
                continue

            # Determine the original predicted direction
            original_direction = "rising" if original_slope > 0 else "declining"

            # ---------------------------------------------------
            # STEP 3: Query current entity mention frequency.
            # Count mentions in the last 7 days vs. the 7 days
            # before that to detect the current trend direction.
            # ---------------------------------------------------
            recent_sql = (
                "SELECT COUNT(*) as cnt FROM entities "
                "WHERE name = :name "
                "AND created_at > DATE('now', '-7 days')"
            )
            recent_rows = await sql_store.execute_sql(
                recent_sql, {"name": entity_name}
            )
            recent_count = recent_rows[0]["cnt"]

            # Count mentions in the prior 7-14 day window
            prior_sql = (
                "SELECT COUNT(*) as cnt FROM entities "
                "WHERE name = :name "
                "AND created_at > DATE('now', '-14 days') "
                "AND created_at <= DATE('now', '-7 days')"
            )
            prior_rows = await sql_store.execute_sql(
                prior_sql, {"name": entity_name}
            )
            prior_count = prior_rows[0]["cnt"]

            # ---------------------------------------------------
            # STEP 4: Score the prediction.
            # Compare current vs. prior counts to determine the
            # actual direction, then compare against the prediction.
            # ---------------------------------------------------
            total_checked += 1

            if recent_count > prior_count:
                actual_direction = "rising"
            elif recent_count < prior_count:
                actual_direction = "declining"
            else:
                actual_direction = "flat"

            # Categorize the outcome
            if actual_direction == original_direction:
                # Trend continued as predicted
                confirmed_count += 1
            elif actual_direction == "flat":
                # Trend flattened -- weakened but not reversed
                weakened_count += 1
            else:
                # Trend reversed direction
                falsified_count += 1

            logger.debug(
                "calibrator_trend_checked",
                entity_name=entity_name,
                original_direction=original_direction,
                actual_direction=actual_direction,
                recent_count=recent_count,
                prior_count=prior_count,
            )

        # ---------------------------------------------------------
        # STEP 5: Compute accuracy and return a meta-insight.
        # Accuracy is the ratio of confirmed predictions to total.
        # ---------------------------------------------------------
        accuracy = (
            confirmed_count / total_checked
            if total_checked > 0
            else 0.0
        )

        logger.info(
            "calibrator_complete",
            accuracy=accuracy,
            total_checked=total_checked,
            confirmed=confirmed_count,
            weakened=weakened_count,
            falsified=falsified_count,
            cycle_id=context.cycle_id,
        )

        insight = FarmingInsight(
            id=str(uuid4()),
            insight_type="meta",
            title=f"Calibration: {accuracy:.0%} trend prediction accuracy",
            description=(
                f"Evaluated {total_checked} historical trend predictions. "
                f"{confirmed_count} confirmed, {weakened_count} weakened, "
                f"{falsified_count} falsified."
            ),
            confidence=1.0,
            parameters={
                "trend_accuracy": accuracy,
                "total_checked": total_checked,
                "confirmed": confirmed_count,
                "weakened": weakened_count,
                "falsified": falsified_count,
            },
        )

        return [insight]


# =====================================================================
# Helper: Parse entity name from a trend insight title.
# =====================================================================


def _parse_entity_name(title: str) -> str | None:
    """
    Extract the entity name from a trend insight title.

    Trend titles follow the format produced by TrendDetectionStage:
        "{entity_name} rising (+0.60/day)"
        "{entity_name} declining (-0.30/day)"

    We split on " rising " or " declining " and take the first part
    as the entity name.  Returns None if neither keyword is found.

    Args:
        title: The trend insight title string.

    Returns:
        The entity name, or None if parsing fails.
    """
    # Try " rising" first, then " declining"
    for direction_keyword in (" rising", " declining"):
        idx = title.find(direction_keyword)
        if idx > 0:
            return title[:idx]
    return None
