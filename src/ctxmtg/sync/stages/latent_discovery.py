# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Latent Relationship Discovery (Hive Farming Stage 2)
=====================================================

Builds a co-entity adjacency graph from hive entity profiles and
finds entities that are 2-hops apart: they share a common co-entity
but never appear together on any single local's co-entity list.

These latent relationships are invisible to individual locals but
emerge when the hive combines data from multiple streams.  Emitted
as hive_native_insights with type "latent_relationship".

Depends on:
    - json (parse co-entity JSON arrays)
    - ctxmtg.sync.hive_db (HiveDatabase for profile/insight access)
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import structlog

from ctxmtg.sync.hive_db import HiveDatabase

logger = structlog.get_logger("ctxmtg.sync.stages.latent_discovery")


class LatentDiscoveryStage:
    """
    Stage 2: latent relationship discovery via 2-hop co-entity graph.

    Usage:
        stage = LatentDiscoveryStage(hive_db)
        result = await stage.run()
    """

    def __init__(
        self,
        hive_db: HiveDatabase,
        max_relationships: int = 50,
    ) -> None:
        self._hive_db = hive_db
        self._max_relationships = max_relationships

    async def run(self) -> dict[str, Any]:
        """
        Build co-entity graph and find 2-hop latent relationships.

        Returns:
            Dict: {"relationships_found": n, "entities_analyzed": n}
        """
        profiles = await self._hive_db.get_all_entity_profiles()

        if not profiles:
            return {"relationships_found": 0, "entities_analyzed": 0}

        # Build adjacency: entity_name -> set of co-entity names
        adjacency: dict[str, set[str]] = defaultdict(set)
        profile_names = {p["entity_name"] for p in profiles}

        for profile in profiles:
            name = profile["entity_name"]
            co_entities = _parse_json_list(profile.get("top_co_entities", "[]"))
            for co in co_entities:
                if co in profile_names:
                    adjacency[name].add(co)
                    adjacency[co].add(name)

        # Find 2-hop pairs: A-B connected, B-C connected, A-C NOT connected
        found: list[tuple[str, str, str]] = []
        seen_pairs: set[frozenset[str]] = set()

        for entity_a, neighbors_a in adjacency.items():
            for bridge in neighbors_a:
                neighbors_bridge = adjacency.get(bridge, set())
                for entity_c in neighbors_bridge:
                    if entity_c == entity_a:
                        continue
                    if entity_c in neighbors_a:
                        continue  # Already directly connected

                    pair = frozenset({entity_a, entity_c})
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    found.append((entity_a, entity_c, bridge))
                    if len(found) >= self._max_relationships:
                        break
                if len(found) >= self._max_relationships:
                    break
            if len(found) >= self._max_relationships:
                break

        # Emit insights
        now_iso = datetime.now(timezone.utc).isoformat()

        for entity_a, entity_c, bridge in found:
            insight_id = f"latent-{entity_a}-{entity_c}-{now_iso[:10]}"
            await self._hive_db.insert_native_insight({
                "id": insight_id,
                "insight_type": "latent_relationship",
                "title": f"Latent link: {entity_a} <-> {entity_c}",
                "description": (
                    f"'{entity_a}' and '{entity_c}' are not directly "
                    f"co-occurring but share a common co-entity "
                    f"'{bridge}'.  This relationship is only visible "
                    f"at the hive level."
                ),
                "confidence": 0.6,
                "parameters": json.dumps({"bridge_entity": bridge}),
                "entity_names": json.dumps([entity_a, entity_c, bridge]),
                "created_at": now_iso,
            })

        logger.info(
            "latent_discovery_complete",
            entities_analyzed=len(adjacency),
            relationships_found=len(found),
        )

        return {
            "relationships_found": len(found),
            "entities_analyzed": len(adjacency),
        }


def _parse_json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
