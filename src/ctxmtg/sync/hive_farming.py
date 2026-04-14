# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Hive Farming Pipeline Orchestrator
====================================

Runs the 3-stage hive farming pipeline over the merged collective
intelligence.  These stages discover patterns only visible at the
aggregate level (cross-stream scoring, latent relationships, insight
correlation).

Triggered by: ``ctxmtg hive farm``

Stages:
    1. CrossStreamStage -- recompute scores + find coverage gaps
    2. LatentDiscoveryStage -- 2-hop co-entity graph analysis
    3. InsightCorrelationStage -- cross-local insight matching

Depends on:
    - ctxmtg.sync.stages (the three stage implementations)
    - ctxmtg.sync.hive_db (HiveDatabase for data access)
"""

from __future__ import annotations

from typing import Any

import structlog

from ctxmtg.sync.hive_db import HiveDatabase
from ctxmtg.sync.stages.cross_stream import CrossStreamStage
from ctxmtg.sync.stages.insight_correlation import InsightCorrelationStage
from ctxmtg.sync.stages.latent_discovery import LatentDiscoveryStage

logger = structlog.get_logger("ctxmtg.sync.hive_farming")


class HiveFarmingPipeline:
    """
    Orchestrates the 3-stage hive farming pipeline.

    Usage:
        pipeline = HiveFarmingPipeline(hive_db)
        result = await pipeline.run()
    """

    def __init__(self, hive_db: HiveDatabase) -> None:
        self._hive_db = hive_db

    async def run(self) -> dict[str, Any]:
        """
        Run all 3 hive farming stages in order.

        Returns:
            Dict with per-stage results and overall status.
        """
        results: dict[str, Any] = {
            "stages_run": 0,
            "stages_succeeded": 0,
            "stages_failed": 0,
            "stage_results": {},
        }

        stages = [
            ("cross_stream", CrossStreamStage(self._hive_db)),
            ("latent_discovery", LatentDiscoveryStage(self._hive_db)),
            ("insight_correlation", InsightCorrelationStage(self._hive_db)),
        ]

        for name, stage in stages:
            results["stages_run"] += 1
            try:
                stage_result = await stage.run()
                results["stage_results"][name] = stage_result
                results["stages_succeeded"] += 1
                logger.info(
                    f"hive_farming_stage_done",
                    stage=name,
                    result=stage_result,
                )
            except Exception as exc:
                results["stages_failed"] += 1
                results["stage_results"][name] = {"error": str(exc)}
                logger.error(
                    f"hive_farming_stage_failed",
                    stage=name,
                    error=str(exc),
                )

        results["status"] = (
            "completed" if results["stages_failed"] == 0 else "partial"
        )

        logger.info(
            "hive_farming_complete",
            status=results["status"],
            stages_run=results["stages_run"],
            stages_succeeded=results["stages_succeeded"],
        )

        return results
