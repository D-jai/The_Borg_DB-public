# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Insight Correlation (Hive Farming Stage 3)
============================================

Reads hive_insights from all locals and finds correlated insights:
pairs of insights from different locals that share entity names and
were created within a temporal window.

Correlated insights are flagged with a shared correlation_id and
emitted as hive_native_insights with type "correlated_pattern".

Entity name overlap is computed via Jaccard similarity on the
entity_names JSON arrays.  Temporal proximity is measured by the
absolute difference in created_at timestamps.

Depends on:
    - json (parse entity_names JSON)
    - datetime (compute temporal proximity)
    - ctxmtg.sync.hive_db (HiveDatabase for insight access)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import structlog

from ctxmtg.sync.hive_db import HiveDatabase

logger = structlog.get_logger("ctxmtg.sync.stages.insight_correlation")

CORRELATION_JACCARD_THRESHOLD = 0.3
CORRELATION_TIME_WINDOW_DAYS = 30


class InsightCorrelationStage:
    """
    Stage 3: find correlated insights across locals.

    Usage:
        stage = InsightCorrelationStage(hive_db)
        result = await stage.run()
    """

    def __init__(
        self,
        hive_db: HiveDatabase,
        jaccard_threshold: float = CORRELATION_JACCARD_THRESHOLD,
        time_window_days: int = CORRELATION_TIME_WINDOW_DAYS,
    ) -> None:
        self._hive_db = hive_db
        self._jaccard_threshold = jaccard_threshold
        self._time_window_days = time_window_days

    async def run(self) -> dict[str, Any]:
        """
        Find correlated insights from different locals.

        Returns:
            Dict: {"correlations_found": n, "insights_analyzed": n}
        """
        all_insights = await self._hive_db.get_insights(limit=500)

        if len(all_insights) < 2:
            return {"correlations_found": 0, "insights_analyzed": len(all_insights)}

        # Group by source_instance
        by_source: dict[str, list[dict]] = {}
        for ins in all_insights:
            source = ins.get("source_instance", "unknown")
            by_source.setdefault(source, []).append(ins)

        sources = list(by_source.keys())
        if len(sources) < 2:
            logger.info(
                "insight_correlation_skip",
                reason="single_source",
                source=sources[0] if sources else "none",
            )
            return {"correlations_found": 0, "insights_analyzed": len(all_insights)}

        # Compare insights across sources
        correlations = 0
        now_iso = datetime.now(timezone.utc).isoformat()
        seen_pairs: set[frozenset[str]] = set()

        for i, src_a in enumerate(sources):
            for src_b in sources[i + 1:]:
                for ins_a in by_source[src_a]:
                    for ins_b in by_source[src_b]:
                        pair = frozenset({ins_a["id"], ins_b["id"]})
                        if pair in seen_pairs:
                            continue

                        # Skip already-correlated pairs
                        if (
                            ins_a.get("correlation_id")
                            and ins_a["correlation_id"] == ins_b.get("correlation_id")
                        ):
                            continue

                        names_a = _parse_entity_names(ins_a)
                        names_b = _parse_entity_names(ins_b)

                        jaccard = _jaccard(names_a, names_b)
                        if jaccard < self._jaccard_threshold:
                            continue

                        days_apart = _days_apart(
                            ins_a.get("created_at", ""),
                            ins_b.get("created_at", ""),
                        )
                        if days_apart is None or days_apart > self._time_window_days:
                            continue

                        seen_pairs.add(pair)

                        # Link them with a correlation_id
                        corr_id = f"corr-{ins_a['id']}-{ins_b['id']}"
                        await self._hive_db.set_correlation_id(
                            ins_a["id"], corr_id
                        )
                        await self._hive_db.set_correlation_id(
                            ins_b["id"], corr_id
                        )

                        # Emit a native insight
                        shared = names_a & names_b
                        await self._hive_db.insert_native_insight({
                            "id": corr_id,
                            "insight_type": "correlated_pattern",
                            "title": (
                                f"Correlated: {ins_a.get('title', '?')} "
                                f"<-> {ins_b.get('title', '?')}"
                            ),
                            "description": (
                                f"Insights from {src_a} and {src_b} share "
                                f"entities {', '.join(sorted(shared))} "
                                f"(Jaccard={jaccard:.2f}, "
                                f"days_apart={days_apart:.0f})."
                            ),
                            "confidence": min(
                                ins_a.get("confidence", 1.0),
                                ins_b.get("confidence", 1.0),
                            ),
                            "entity_names": json.dumps(sorted(shared)),
                            "created_at": now_iso,
                        })
                        correlations += 1

        logger.info(
            "insight_correlation_complete",
            correlations_found=correlations,
            insights_analyzed=len(all_insights),
        )

        return {
            "correlations_found": correlations,
            "insights_analyzed": len(all_insights),
        }


def _parse_entity_names(insight: dict) -> set[str]:
    """Extract entity names from an insight's entity_names JSON or title."""
    raw = insight.get("entity_names", "[]")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                return set(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
    elif isinstance(raw, list):
        return set(raw)

    # Fallback: extract capitalised words from title
    title = insight.get("title", "")
    words = set()
    for word in title.split():
        if word and word[0].isupper() and len(word) > 1:
            words.add(word.rstrip(",.;:"))
    return words


def _jaccard(a: set, b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union) if union else 0.0


def _days_apart(ts_a: str, ts_b: str) -> float | None:
    """Compute absolute days between two ISO timestamps."""
    try:
        dt_a = datetime.fromisoformat(ts_a.replace("Z", "+00:00"))
        dt_b = datetime.fromisoformat(ts_b.replace("Z", "+00:00"))
        return abs((dt_a - dt_b).total_seconds()) / 86400
    except (ValueError, TypeError):
        return None
