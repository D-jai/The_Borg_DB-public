# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Trend Detection Stage (Intelligence Stage 2)
===============================================

Detects temporal trends in entity mention frequency -- entities whose
daily mention count is rising, declining, or spiking anomalously.
For example, "OAuth2" might show a steady increase over the past week,
suggesting growing focus on authentication topics.

This is the second of seven intelligence stages.  It uses simple
linear regression (via numpy when available, or a two-point fallback)
to estimate the slope of each entity's daily mention curve within a
configurable time window.

Algorithm overview:
    1. SEED: Query per-entity, per-day mention counts within the
       sliding window (default 7 days).
    2. For entities with enough data points (>= ``min_observations``),
       fit a linear slope to the time series.
    3. If numpy is available, use ``numpy.polyfit(degree=1)`` for a
       proper least-squares fit.  Otherwise, fall back to a simple
       ``(last - first) / days`` two-point slope.
    4. Flag anomalies when the latest count exceeds ``mean + 2 * std``.
    5. Emit ``FarmingInsight`` objects of type ``"trend"`` for entities
       whose absolute slope exceeds 0.1 mentions/day.

Depends on:
    - structlog (structured logging)
    - numpy (optional -- graceful degradation to two-point slope)
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- optional, unused tier 0-1)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async bridge)

Used by:
    - ctxmtg.farming.pipeline (registered as the second intelligence stage)
"""

from __future__ import annotations

import math
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
logger = structlog.get_logger("ctxmtg.farming.trend_detection")

# ---------------------------------------------------------------
# Optional numpy import.  If numpy is not installed (e.g., on a
# resource-constrained edge device), we fall back to a simple
# two-point slope estimation.  The degradation is graceful --
# results are less accurate but the stage still runs.
# ---------------------------------------------------------------
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# ---------------------------------------------------------------
# SQL query: per-entity, per-day mention counts within the window.
# Uses DATE() for day-level granularity and parameterised window.
# ---------------------------------------------------------------
DAILY_COUNTS_SQL = """\
SELECT name, DATE(created_at) as day, COUNT(*) as daily_count
FROM entities
WHERE created_at > DATE('now', :window)
GROUP BY name, DATE(created_at)
ORDER BY name, day
"""

# ---------------------------------------------------------------
# Minimum absolute slope to consider a trend "significant".
# Entities with flatter curves are quietly ignored.
# ---------------------------------------------------------------
SLOPE_THRESHOLD = 0.1


def _compute_slope_numpy(counts: list[float]) -> float:
    """
    Compute linear regression slope using numpy polyfit.

    Uses a degree-1 polynomial fit (y = slope * x + intercept).
    The x-axis is simply the integer index [0, 1, 2, ...] of
    the daily data points.

    Args:
        counts: Daily mention counts as floats.

    Returns:
        The slope (mentions per day).
    """
    xs = np.arange(len(counts), dtype=float)
    # polyfit returns [slope, intercept] for degree 1
    coeffs = np.polyfit(xs, counts, 1)
    return float(coeffs[0])


def _compute_slope_simple(counts: list[float]) -> float:
    """
    Fallback slope estimation: (last - first) / (N - 1).

    Used when numpy is not available.  Less accurate than a
    least-squares fit because it ignores intermediate points,
    but sufficient for basic trend detection.

    Args:
        counts: Daily mention counts (must have >= 2 elements).

    Returns:
        The approximate slope (mentions per day).
    """
    # Number of intervals between first and last data point
    n_intervals = len(counts) - 1
    if n_intervals <= 0:
        return 0.0
    return (counts[-1] - counts[0]) / n_intervals


def _compute_stats(counts: list[float]) -> tuple[float, float]:
    """
    Compute mean and standard deviation of a list of counts.

    Uses the population standard deviation (not sample) because we
    have the full observation window, not a sample from a larger set.

    Args:
        counts: Daily mention counts.

    Returns:
        (mean, std) tuple.  std is 0.0 if only one data point.
    """
    n = len(counts)
    if n == 0:
        return 0.0, 0.0

    mean = sum(counts) / n

    if n == 1:
        return mean, 0.0

    # Population variance: sum of squared deviations / N
    variance = sum((c - mean) ** 2 for c in counts) / n
    std = math.sqrt(variance)
    return mean, std


class TrendDetectionStage(FarmingStage):
    """
    Farming stage 2: temporal trend detection over entity mentions.

    Analyses the frequency of entity mentions per day within a
    sliding window and identifies entities with rising, declining,
    or anomalous activity.

    Produces ``"trend"`` insights for entities whose mention-count
    slope exceeds ±0.1 per day.  Also flags anomalies where the
    most recent day's count is more than 2 standard deviations
    above the mean.

    The optional ``llm`` parameter is accepted for API consistency
    but not used in the current implementation.

    Usage:
        stage = TrendDetectionStage(window_days=7, min_observations=3)
        insights = stage.run(sql_store, vector_store, context)
    """

    def __init__(
        self,
        window_days: int = 7,
        min_observations: int = 3,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Configure the trend detection stage.

        Args:
            window_days: Number of days to look back for trend data.
                Default 7 (one week).
            min_observations: Minimum number of distinct days with
                data required before computing a trend.  Default 3.
            llm: Optional LLM provider (unused in current tier).
        """
        self._window_days = window_days
        self._min_obs = min_observations
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface
    # -----------------------------------------------------------------

    def get_name(self) -> str:
        """Return the canonical stage name for logging/checkpointing."""
        return "trend_detection"

    def run(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        """
        Execute trend detection over entity mention time series.

        Steps:
            1. Query daily entity mention counts within the window.
            2. Group results by entity name.
            3. For each entity with sufficient data points, compute
               slope and check for anomalies.
            4. Emit insights for entities with significant trends.

        This method is synchronous (per the FarmingStage contract)
        but calls async store methods via ``_run_async()``.

        Args:
            sql_store:    SQL store to read entity data from.
            vector_store: Vector store (unused by this stage).
            context:      Farming context with cycle ID and budget.

        Returns:
            List of FarmingInsight objects (type ``"trend"``).
            Empty list if no significant trends are detected.
        """
        logger.info(
            "trend_detection_start",
            cycle_id=context.cycle_id,
            window_days=self._window_days,
            min_observations=self._min_obs,
            has_numpy=_HAS_NUMPY,
        )

        # ----------------------------------------------------------
        # Step 1: Query per-entity, per-day counts within the window.
        # ----------------------------------------------------------
        rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(
                DAILY_COUNTS_SQL,
                {"window": f"-{self._window_days} days"},
            )
        )

        logger.debug(
            "daily_counts_loaded",
            row_count=len(rows),
        )

        # ----------------------------------------------------------
        # Step 2: Group rows by entity name.
        # Each entity gets a list of daily counts in chronological order.
        # ----------------------------------------------------------
        entity_series: dict[str, list[float]] = {}

        for row in rows:
            name: str = row["name"]
            daily_count: float = float(row["daily_count"])

            if name not in entity_series:
                entity_series[name] = []
            entity_series[name].append(daily_count)

        logger.debug(
            "entities_grouped",
            entity_count=len(entity_series),
        )

        # ----------------------------------------------------------
        # Step 3 & 4: Compute slope and detect anomalies.
        # ----------------------------------------------------------
        insights: list[FarmingInsight] = []

        for entity_name, counts in entity_series.items():
            # Skip entities without enough data points
            if len(counts) < self._min_obs:
                continue

            # Compute slope: numpy least-squares or simple fallback
            if _HAS_NUMPY:
                slope = _compute_slope_numpy(counts)
            else:
                slope = _compute_slope_simple(counts)

            # Only report trends with meaningful slope magnitude
            if abs(slope) <= SLOPE_THRESHOLD:
                continue

            # Anomaly detection: latest count > mean + 2*std
            mean, std = _compute_stats(counts)
            is_anomaly = counts[-1] > mean + 2 * std if std > 0 else False

            # Confidence: scales with the number of observations
            # and the magnitude of the slope.  More data points and
            # steeper slopes yield higher confidence.
            confidence = min(
                (len(counts) / 10.0) * min(abs(slope), 1.0),
                1.0,
            )
            # Floor at 0.1 so we never report a near-zero confidence
            confidence = max(confidence, 0.1)

            # Direction string for the title
            direction = "rising" if slope > 0 else "declining"

            # Deterministic ID for deduplication across cycles
            # ORIGINAL: f"trend-{entity_name}-{context.cycle_id}"
            insight_id = f"trend-{entity_name}"

            insight = FarmingInsight(
                id=insight_id,
                insight_type="trend",
                title=f"{entity_name} {direction} ({slope:+.2f}/day)",
                confidence=confidence,
                parameters={
                    "slope": round(slope, 4),
                    "window_days": self._window_days,
                    "data_points": len(counts),
                    "is_anomaly": is_anomaly,
                },
                # No canonical entity IDs at this stage
                entity_ids=[],
            )
            insights.append(insight)

        logger.info(
            "trend_detection_complete",
            cycle_id=context.cycle_id,
            insights_produced=len(insights),
        )

        return insights
