# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Query Autocomplete Engine
=========================

Provides query completion suggestions powered by distilled entity
intelligence stored in ``local_intelligence_cache`` (hive-sourced) or
``distiller_summaries`` (locally farmed).

The engine scans the last word fragment in a partial query, matches it
against entity names using a case-insensitive prefix search, and
generates meaningful natural-language suggestions using each entity's
``top_predicates`` and ``top_co_entities``.

Performance:
    - Target: sub-millisecond response for ~200 cached rows.
    - SQLite LIKE with a prefix pattern uses the index on entity_name
      (case-insensitive via COLLATE NOCASE) so the scan is fast.
    - All work is done in a single SQL round-trip per call.

Graceful degradation:
    - If ``local_intelligence_cache`` is missing (no hive), falls back
      to ``distiller_summaries`` (local farming output).
    - If neither table exists (fresh install, no farming done yet),
      returns an empty list — no crash, no error.

Depends on:
    - json (parse top_predicates and top_co_entities JSON arrays)
    - structlog (structured logging)
    - ctxmtg.interfaces.storage (SQLStore ABC for database access)

Used by:
    - ctxmtg.cli (``ctxmtg suggest`` command)
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from ctxmtg.interfaces.storage import SQLStore

# ---------------------------------------------------------------
# Module-level logger — structured JSON output.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.autocomplete")

# ---------------------------------------------------------------
# SQL: prefix-match entities in local_intelligence_cache.
# Uses LIKE with a trailing wildcard for prefix matching.
# COLLATE NOCASE makes the match case-insensitive.
# Ordered by relevance_score DESC so the most important entities
# come first.  Capped at 10 to keep the candidate pool small.
# ---------------------------------------------------------------
CACHE_PREFIX_SQL = """\
SELECT entity_name, entity_type, top_predicates,
       top_co_entities, relevance_score
FROM local_intelligence_cache
WHERE entity_name LIKE :prefix COLLATE NOCASE
ORDER BY relevance_score DESC
LIMIT 10
"""

# ---------------------------------------------------------------
# SQL: same prefix match but against distiller_summaries (fallback).
# ---------------------------------------------------------------
DISTILLER_PREFIX_SQL = """\
SELECT entity_name, entity_type, top_predicates,
       top_co_entities, relevance_score
FROM distiller_summaries
WHERE entity_name LIKE :prefix COLLATE NOCASE
ORDER BY relevance_score DESC
LIMIT 10
"""

# ---------------------------------------------------------------
# SQL: top entities from local_intelligence_cache for browsing.
# Returns the highest-relevance entities with their summaries.
# ---------------------------------------------------------------
CACHE_TOP_ENTITIES_SQL = """\
SELECT entity_name, entity_type, top_predicates,
       top_co_entities, relevance_score, summary
FROM local_intelligence_cache
ORDER BY relevance_score DESC
LIMIT :limit
"""

# ---------------------------------------------------------------
# SQL: same top-entities query for distiller_summaries (fallback).
# ---------------------------------------------------------------
DISTILLER_TOP_ENTITIES_SQL = """\
SELECT entity_name, entity_type, top_predicates,
       top_co_entities, relevance_score, summary
FROM distiller_summaries
ORDER BY relevance_score DESC
LIMIT :limit
"""


class AutocompleteEngine:
    """
    Suggest query completions from accumulated entity intelligence.

    The engine reads distilled entity summaries (from the hive-pulled
    local cache or the locally farmed distiller table) and generates
    natural-language query suggestions based on entity names, their
    predicates, and co-entity relationships.

    Usage:
        engine = AutocompleteEngine(sql_store)
        suggestions = await engine.suggest("What did Al")
        top = await engine.get_top_entities(limit=10)
    """

    def __init__(self, sql_store: SQLStore) -> None:
        """
        Initialise the autocomplete engine.

        Args:
            sql_store: The local SQL store containing the
                       ``local_intelligence_cache`` and/or
                       ``distiller_summaries`` tables.
        """
        self._sql_store = sql_store

    # =================================================================
    # suggest: generate query completions from a partial query string
    # =================================================================

    async def suggest(
        self, partial_query: str, max_suggestions: int = 5
    ) -> list[str]:
        """
        Generate query suggestions from a partial user query.

        Extracts the last word fragment as a prefix, matches it
        against cached entity names, and builds natural-language
        suggestions using each entity's predicates and co-entities.

        Args:
            partial_query:   The partial text the user has typed so far.
                             The last whitespace-delimited token is used
                             as the entity prefix for matching.
            max_suggestions: Maximum number of suggestions to return.
                             Defaults to 5.

        Returns:
            A list of suggested query strings, ranked by entity
            relevance.  Empty list if no matches or no tables exist.
        """
        # Guard: empty or whitespace-only input → no suggestions.
        # (Browse mode is handled separately by the CLI.)
        stripped = partial_query.strip()
        if not stripped:
            return []

        # Extract the last word as the entity-name prefix for matching.
        # Example: "What did Al" → prefix = "Al"
        prefix = stripped.split()[-1]

        # Build the LIKE pattern: prefix + wildcard for trailing chars.
        like_pattern = f"{prefix}%"

        # Try local_intelligence_cache first (hive-sourced hints),
        # then fall back to distiller_summaries (local farming output).
        rows = await self._query_with_fallback(
            primary_sql=CACHE_PREFIX_SQL,
            fallback_sql=DISTILLER_PREFIX_SQL,
            params={"prefix": like_pattern},
        )

        # No matching entities → empty suggestions.
        if not rows:
            return []

        # Generate suggestions from matched entities.
        suggestions: list[str] = []
        for row in rows:
            entity_name = row["entity_name"]
            entity_type = row.get("entity_type", "other")
            predicates = _parse_json_list(row.get("top_predicates", "[]"))
            co_entities = _parse_json_list(row.get("top_co_entities", "[]"))

            # Build type-specific suggestions using predicates.
            if entity_type == "person":
                # Person entities: ask about their actions.
                for pred in predicates[:2]:
                    suggestions.append(f"What did {entity_name} {pred}?")
                # Default fallback if no predicates available.
                if not predicates:
                    suggestions.append(f"What did {entity_name} do?")
            else:
                # Tool / topic / other entities: general inquiry.
                suggestions.append(f"Tell me about {entity_name}")

            # Comparison suggestions if co-entities are available.
            for co_ent in co_entities[:1]:
                suggestions.append(f"Compare {entity_name} and {co_ent}")

        # Deduplicate while preserving order (entities are already
        # sorted by relevance from the SQL query).
        seen: set[str] = set()
        unique: list[str] = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        # Return the top max_suggestions results.
        return unique[:max_suggestions]

    # =================================================================
    # get_top_entities: browse the most relevant entities
    # =================================================================

    async def get_top_entities(self, limit: int = 10) -> list[dict]:
        """
        Retrieve the top entities by relevance for browsing.

        Returns a list of dicts with entity info (name, type,
        predicates, co-entities, relevance score, summary).  Useful
        for the ``ctxmtg suggest --browse`` command.

        Args:
            limit: Maximum number of entities to return.  Default 10.

        Returns:
            List of entity dicts ordered by relevance_score descending.
            Empty list if no intelligence tables exist yet.
        """
        # Try local_intelligence_cache first, fall back to distiller.
        rows = await self._query_with_fallback(
            primary_sql=CACHE_TOP_ENTITIES_SQL,
            fallback_sql=DISTILLER_TOP_ENTITIES_SQL,
            params={"limit": limit},
        )

        # Convert rows to clean dicts with parsed JSON fields.
        entities: list[dict] = []
        for row in rows:
            entities.append({
                "entity_name": row["entity_name"],
                "entity_type": row.get("entity_type", "other"),
                "top_predicates": _parse_json_list(
                    row.get("top_predicates", "[]")
                ),
                "top_co_entities": _parse_json_list(
                    row.get("top_co_entities", "[]")
                ),
                "relevance_score": row.get("relevance_score", 0.0),
                "summary": row.get("summary", ""),
            })

        return entities

    # =================================================================
    # Internal: query with automatic table fallback
    # =================================================================

    async def _query_with_fallback(
        self,
        primary_sql: str,
        fallback_sql: str,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Execute the primary SQL query; if the table doesn't exist or
        returns no results, try the fallback query.

        Handles both "table missing" (fresh install) and "table empty"
        (no farming done yet) scenarios gracefully.

        Args:
            primary_sql:  SQL to try first (local_intelligence_cache).
            fallback_sql: SQL to try second (distiller_summaries).
            params:       Named parameters for both queries.

        Returns:
            List of row dicts from whichever query succeeded, or
            an empty list if both fail / return nothing.
        """
        # Attempt the primary query (local_intelligence_cache).
        try:
            rows = await self._sql_store.execute_sql(primary_sql, params)
            if rows:
                return rows
        except Exception:
            # Table probably doesn't exist — try fallback silently.
            pass

        # Attempt the fallback query (distiller_summaries).
        try:
            rows = await self._sql_store.execute_sql(fallback_sql, params)
            return rows
        except Exception:
            # Neither table exists — fresh install with no farming.
            logger.debug("autocomplete_no_tables", msg="No intelligence tables found")
            return []


# =====================================================================
# Private helpers
# =====================================================================


def _parse_json_list(value: Any) -> list:
    """
    Safely parse a JSON-encoded list from a TEXT column.

    Returns an empty list on failure (malformed JSON, wrong type,
    None).  Defensive — bad data should not crash autocomplete.

    Args:
        value: A JSON string, already-parsed list, or None.

    Returns:
        A Python list, or empty list on failure.
    """
    # Already a list (e.g. if the driver auto-deserialised).
    if isinstance(value, list):
        return value
    # None or empty string → empty list.
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
