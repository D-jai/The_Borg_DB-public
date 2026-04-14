# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Intelligence Merger (Hive Side)
================================

Merges distilled intelligence from multiple locals into unified
entity profiles on the hive.  This replaces the old ContextEnricher
which operated on raw data mirrors.

The merger runs as part of ``ctxmtg hive push`` (automatic after
push) or can be triggered separately.  It reads hive_entity_profiles
and recomputes cross-stream scores based on the number of distinct
locals that contributed data for each entity.

Cross-stream scoring formula:
    cross_stream_score = relevance * log(1 + stream_count)

Entities seen from multiple streams (locals) are more significant
than entities seen from just one.

Depends on:
    - math (log for scoring)
    - json (parse JSON columns)
    - structlog (structured logging)
    - ctxmtg.sync.hive_db (HiveDatabase for read/write)

Used by:
    - ctxmtg.sync.hive_farming (as a pre-step before hive farming)
    - ctxmtg.cli (``ctxmtg hive push`` triggers merge automatically)
"""

from __future__ import annotations

import json
import math
from typing import Any

import structlog

from ctxmtg.sync.hive_db import HiveDatabase

logger = structlog.get_logger("ctxmtg.sync.intelligence_merger")


class IntelligenceMerger:
    """
    Merges entity profiles across locals and recomputes scores.

    After intelligence is pushed to the hive from one or more locals,
    the merger recomputes cross_stream_score for all entity profiles.

    Usage:
        merger = IntelligenceMerger(hive_db)
        result = await merger.merge()
    """

    def __init__(self, hive_db: HiveDatabase) -> None:
        self._hive_db = hive_db

    async def merge(self) -> dict[str, int]:
        """
        Recompute cross-stream scores for all entity profiles.

        Returns:
            Dict with counts: {"profiles_updated": n, "multi_stream": n}
        """
        profiles = await self._hive_db.get_all_entity_profiles()

        if not profiles:
            logger.info("intelligence_merger_noop", reason="no_profiles")
            return {"profiles_updated": 0, "multi_stream": 0}

        updated = 0
        multi_stream = 0

        for profile in profiles:
            stream_count = profile.get("stream_count", 1)
            total_mentions = profile.get("total_mentions", 0)

            # Cross-stream score: relevance * log(1 + stream_count)
            base_relevance = math.log(1 + total_mentions)
            score = base_relevance * math.log(1 + stream_count)

            if stream_count > 1:
                multi_stream += 1

            current_score = profile.get("cross_stream_score", 0.0)
            if abs(score - current_score) > 0.001:
                await self._hive_db.update_entity_profile(
                    profile["entity_name"],
                    {"cross_stream_score": round(score, 4)},
                )
                updated += 1

        logger.info(
            "intelligence_merger_complete",
            profiles_updated=updated,
            multi_stream=multi_stream,
            total=len(profiles),
        )

        return {"profiles_updated": updated, "multi_stream": multi_stream}

    async def find_coverage_gaps(
        self,
        min_mentions: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Find entities with high mentions but only one source stream.

        These are coverage gaps: entities important to one local but
        not seen elsewhere.  Useful for hive farming analysis.

        Returns:
            List of entity profile dicts that are coverage gaps.
        """
        profiles = await self._hive_db.get_all_entity_profiles()
        gaps = []

        for profile in profiles:
            if (
                profile.get("stream_count", 1) == 1
                and profile.get("total_mentions", 0) >= min_mentions
            ):
                gaps.append(profile)

        return gaps
