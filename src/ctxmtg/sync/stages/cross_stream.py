# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Cross-Stream Scoring + Coverage Gap Analysis (Hive Farming Stage 1)
=====================================================================

Recomputes cross_stream_score for all hive entity profiles and
identifies coverage gaps -- entities with high relevance but data
from only one local.

Coverage gaps are emitted as hive_native_insights with type
"coverage_gap".  These tell the user "entity X is important on
your laptop but not seen on your phone -- you might want to
ingest related data there."

Depends on:
    - ctxmtg.sync.intelligence_merger (IntelligenceMerger for scoring)
    - ctxmtg.sync.hive_db (HiveDatabase for insight storage)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import structlog

from ctxmtg.sync.hive_db import HiveDatabase
from ctxmtg.sync.intelligence_merger import IntelligenceMerger

logger = structlog.get_logger("ctxmtg.sync.stages.cross_stream")


class CrossStreamStage:
    """
    Stage 1: cross-stream scoring and coverage gap identification.

    Usage:
        stage = CrossStreamStage(hive_db)
        result = await stage.run()
    """

    def __init__(
        self,
        hive_db: HiveDatabase,
        min_mentions_for_gap: int = 5,
    ) -> None:
        self._hive_db = hive_db
        self._merger = IntelligenceMerger(hive_db)
        self._min_mentions_for_gap = min_mentions_for_gap

    async def run(self) -> dict[str, Any]:
        """
        Run cross-stream scoring and emit coverage gap insights.

        Returns:
            Dict with counts: {"profiles_scored": n, "gaps_found": n}
        """
        # Recompute cross-stream scores
        merge_result = await self._merger.merge()

        # Find coverage gaps
        gaps = await self._merger.find_coverage_gaps(
            min_mentions=self._min_mentions_for_gap
        )

        now_iso = datetime.now(timezone.utc).isoformat()

        for gap in gaps:
            entity_name = gap["entity_name"]
            streams = json.loads(gap.get("source_streams", "[]"))
            insight_id = f"gap-{entity_name}-{now_iso[:10]}"

            await self._hive_db.insert_native_insight({
                "id": insight_id,
                "insight_type": "coverage_gap",
                "title": f"Coverage gap: {entity_name}",
                "description": (
                    f"Entity '{entity_name}' has {gap.get('total_mentions', 0)} "
                    f"mentions but is only seen on {', '.join(streams)}. "
                    f"Consider ingesting related data from other locals."
                ),
                "confidence": 0.8,
                "entity_names": json.dumps([entity_name]),
                "created_at": now_iso,
            })

        logger.info(
            "cross_stream_stage_complete",
            profiles_scored=merge_result.get("profiles_updated", 0),
            gaps_found=len(gaps),
        )

        return {
            "profiles_scored": merge_result.get("profiles_updated", 0),
            "gaps_found": len(gaps),
        }
