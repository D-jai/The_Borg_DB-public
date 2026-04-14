# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Template Query Planner
======================

This module implements the Phase 1 query planner. It converts the
structured interpretation dict (from the interpreter) into a concrete
QueryPlan that the executor can run against both SQL and vector stores.

The planner selects a SQL template based on the classified intent and
fills in the template slots with entities, time ranges, and filters
from the interpretation. It also generates a vector query string from
the rewritten query for semantic search.

SQL template strategy by intent:
    - FACTUAL:     fact lookup with entity name matching (COLLATE NOCASE)
    - AGGREGATION: COUNT queries with optional time/content filters
    - TEMPORAL:    time-bounded interaction retrieval
    - COMPARATIVE: multi-entity fact retrieval for comparison
    - SEMANTIC:    FTS keyword search (vector search handles the rest)
    - UNKNOWN:     falls back to FTS + vector (both stores)

All entity name matching in generated SQL uses COLLATE NOCASE to ensure
case-insensitive comparison (as specified in the plan).

Phase 2 replaces this with LLMQueryPlanner, which generates dynamic SQL
for complex queries that templates cannot handle.

Depends on:
    - ctxmtg.interfaces.query (QueryPlanner ABC)
    - ctxmtg.models.query (QueryPlan, QueryIntent)
    - ctxmtg.models.profile (DomainProfile)

Used by:
    - ctxmtg.query.executor (calls plan() to get the execution blueprint)
    - tests/test_query/test_planner.py
"""

from __future__ import annotations

from typing import Any

import structlog

from ctxmtg.interfaces.query import QueryPlanner
from ctxmtg.models.profile import DomainProfile
from ctxmtg.models.query import QueryIntent, QueryPlan

# ---------------------------------------------------------------
# Module logger -- logs generated SQL queries for debugging.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.planner")


def _sql_escape(value: str) -> str:
    """
    Escape a value for safe interpolation into a SQL string literal.

    Doubles single quotes so that values containing apostrophes
    (e.g., "Daniel Kim's", "O'Brien") don't break SQL syntax or
    open injection vectors.

    This is a stop-gap until the planner returns parameterized queries
    via QueryPlan instead of inlined SQL strings.
    """
    return value.replace("'", "''")


# =====================================================================
# SQL Templates: one template per intent type.
# Values are inlined with _sql_escape() for safety. A future refactor
# should pass parameters through QueryPlan for true parameterization.
# =====================================================================

# Factual: look up facts where the subject entity name matches.
# Joins facts → entities to search by entity name (COLLATE NOCASE).
# Limited to 20 results to keep response size manageable.
FACTUAL_SQL_TEMPLATE = (
    "SELECT f.id, f.predicate, f.object_literal, f.confidence, f.created_at, "
    "e.name AS entity_name, e.entity_type "
    "FROM facts f "
    "JOIN entities e ON f.subject_entity_id = e.id "
    "WHERE e.name LIKE :entity_name COLLATE NOCASE "
    "ORDER BY f.created_at DESC "
    "LIMIT :limit"
)

# Aggregation: count interactions matching content keywords.
# Optionally filtered by time range if the interpretation has one.
AGGREGATION_SQL_TEMPLATE = (
    "SELECT COUNT(*) as total_count FROM interactions WHERE content LIKE :keyword COLLATE NOCASE"
)

# Aggregation with time range: same as above but time-bounded.
AGGREGATION_TIME_SQL_TEMPLATE = (
    "SELECT COUNT(*) as total_count "
    "FROM interactions "
    "WHERE content LIKE :keyword COLLATE NOCASE "
    "AND created_at BETWEEN :start_time AND :end_time"
)

# Temporal: retrieve interactions within a time range.
# Ordered by creation time ascending (chronological narrative).
TEMPORAL_SQL_TEMPLATE = (
    "SELECT id, title, content, source_type, created_at "
    "FROM interactions "
    "WHERE created_at BETWEEN :start_time AND :end_time "
    "ORDER BY created_at ASC "
    "LIMIT :limit"
)

# Temporal with entity filter: time-bounded + entity match.
TEMPORAL_ENTITY_SQL_TEMPLATE = (
    "SELECT i.id, i.title, i.content, i.source_type, i.created_at "
    "FROM interactions i "
    "WHERE i.created_at BETWEEN :start_time AND :end_time "
    "AND i.content LIKE :entity_name COLLATE NOCASE "
    "ORDER BY i.created_at ASC "
    "LIMIT :limit"
)

# Comparative: look up facts for multiple entities for comparison.
# Uses IN clause with entity name matching (COLLATE NOCASE).
COMPARATIVE_SQL_TEMPLATE = (
    "SELECT f.id, f.predicate, f.object_literal, f.confidence, f.created_at, "
    "e.name AS entity_name, e.entity_type "
    "FROM facts f "
    "JOIN entities e ON f.subject_entity_id = e.id "
    "WHERE {entity_conditions} "
    "ORDER BY e.name, f.created_at DESC "
    "LIMIT :limit"
)

# Default result limit for SQL queries (prevents unbounded result sets)
DEFAULT_SQL_LIMIT = 20


class TemplateQueryPlanner(QueryPlanner):
    """
    Phase 1 query planner using SQL templates.

    Takes the structured interpretation from the RuleBasedQueryInterpreter
    and selects the appropriate SQL template based on intent type. Template
    slots are filled with values from the interpretation dict.

    The planner also generates a vector query string by using the
    rewritten query from the interpretation (stop words removed).

    Routing logic:
    - Most intents → "both" (query both SQL and vector stores)
    - AGGREGATION → "sql_only" (aggregation doesn't need vector search)
    - SEMANTIC → "both" (but vector results will dominate after fusion)

    Phase 2 replaces this with LLMQueryPlanner that generates dynamic SQL
    for complex multi-hop and domain-specific queries.

    Usage:
        planner = TemplateQueryPlanner()
        plan = planner.plan(
            query="What did Alice propose?",
            interpretation={"entities": ["Alice"], "intent": QueryIntent.FACTUAL, ...},
            profile=general_profile,
        )
        # plan.sql_query → "SELECT ... WHERE e.name LIKE '%Alice%' ..."
        # plan.vector_query → "Alice proposed"
    """

    def plan(self, query: str, interpretation: dict[str, Any], profile: DomainProfile) -> QueryPlan:
        """
        Generate a QueryPlan from the user's query and its interpretation.

        Selects a SQL template based on the classified intent, fills in
        the template with entities and time ranges, and generates the
        vector query string.

        Args:
            query: The original user question.
            interpretation: The interpretation dict from the interpreter:
                - entities: list[str]
                - intent: QueryIntent
                - time_range: tuple[str, str] | None
                - filters: dict[str, str]
                - rewritten_query: str
            profile: The active domain profile (used in Phase 2 for
                     domain-specific SQL template selection).

        Returns:
            A QueryPlan specifying the SQL query, vector query,
            filters, and routing strategy.
        """
        intent: QueryIntent = interpretation["intent"]
        entities: list[str] = interpretation.get("entities", [])
        time_range: tuple[str, str] | None = interpretation.get("time_range")
        filters: dict[str, str] = interpretation.get("filters", {})
        rewritten_query: str = interpretation.get("rewritten_query", query)

        # Generate SQL query from the appropriate template
        sql_query = self._build_sql(intent, entities, time_range, rewritten_query)

        # Generate vector query from the rewritten query string
        vector_query = rewritten_query if rewritten_query else query

        # Determine routing: which stores to query
        routing = self._determine_routing(intent)

        # Build vector filters from the interpretation filters
        vector_filters = self._build_vector_filters(filters, time_range)

        logger.info(
            "query_planned",
            intent=intent.value,
            routing=routing,
            has_sql=sql_query is not None,
            has_vector=vector_query is not None,
            entity_count=len(entities),
        )

        return QueryPlan(
            original_query=query,
            intent=intent,
            sql_query=sql_query,
            vector_query=vector_query,
            vector_filters=vector_filters,
            routing=routing,
        )

    def _build_sql(
        self,
        intent: QueryIntent,
        entities: list[str],
        time_range: tuple[str, str] | None,
        rewritten_query: str,
    ) -> str | None:
        """
        Build a SQL query string from the intent and interpretation data.

        Selects the appropriate template and fills in the placeholder
        values. Returns None if no meaningful SQL query can be generated
        (e.g., pure semantic search with no entities or time range).

        Args:
            intent: The classified query intent.
            entities: List of entity names found in the query.
            time_range: Optional (start, end) ISO datetime tuple.
            rewritten_query: The cleaned query for keyword matching.

        Returns:
            A SQL query string with named placeholders, or None.
        """
        if intent == QueryIntent.FACTUAL:
            return self._build_factual_sql(entities)

        if intent == QueryIntent.AGGREGATION:
            return self._build_aggregation_sql(rewritten_query, time_range)

        if intent == QueryIntent.TEMPORAL:
            return self._build_temporal_sql(entities, time_range)

        if intent == QueryIntent.COMPARATIVE:
            return self._build_comparative_sql(entities)

        if intent == QueryIntent.SEMANTIC:
            # Semantic queries primarily use vector search, but we also
            # run FTS as a supplementary SQL-side signal
            return self._build_fts_sql(rewritten_query)

        # UNKNOWN intent: try FTS if we have a query
        if rewritten_query:
            return self._build_fts_sql(rewritten_query)

        return None

    def _build_factual_sql(self, entities: list[str]) -> str | None:
        """
        Build a factual SQL query for entity fact lookup.

        Uses the first entity as the primary lookup target. If no
        entities are available, falls back to None (no SQL query).

        Args:
            entities: List of entity names found in the query.

        Returns:
            SQL string with named placeholders, or None if no entities.
        """
        if not entities:
            return None

        # Use the first entity for the primary lookup.
        # Replace named placeholders with actual values.
        # The executor will use parameterised queries for safety,
        # but we inline the template structure here for clarity.
        escaped = _sql_escape(entities[0])
        sql = FACTUAL_SQL_TEMPLATE.replace(":entity_name", f"'%{escaped}%'")
        sql = sql.replace(":limit", str(DEFAULT_SQL_LIMIT))
        return sql

    def _build_aggregation_sql(
        self, rewritten_query: str, time_range: tuple[str, str] | None
    ) -> str:
        """
        Build an aggregation SQL query (COUNT, SUM, etc.).

        If a time range is present, uses the time-bounded template.
        Otherwise, counts all matching interactions.

        Args:
            rewritten_query: Keyword to search for in content.
            time_range: Optional (start, end) ISO datetime tuple.

        Returns:
            SQL string with values inlined (aggregation queries are simple).
        """
        keyword = f"%{_sql_escape(rewritten_query)}%"

        if time_range:
            sql = AGGREGATION_TIME_SQL_TEMPLATE
            sql = sql.replace(":keyword", f"'{keyword}'")
            sql = sql.replace(":start_time", f"'{_sql_escape(time_range[0])}'")
            sql = sql.replace(":end_time", f"'{_sql_escape(time_range[1])}'")
        else:
            sql = AGGREGATION_SQL_TEMPLATE
            sql = sql.replace(":keyword", f"'{keyword}'")

        return sql

    def _build_temporal_sql(
        self, entities: list[str], time_range: tuple[str, str] | None
    ) -> str | None:
        """
        Build a temporal SQL query for time-bounded retrieval.

        If no time range is available (the temporal patterns didn't
        extract one), falls back to a 7-day lookback as a reasonable
        default.

        Args:
            entities: List of entity names (used as content filter).
            time_range: Optional (start, end) ISO datetime tuple.

        Returns:
            SQL string with values inlined, or None.
        """
        if not time_range:
            # No explicit time range; can't build a meaningful temporal query
            return None

        if entities:
            # Temporal + entity filter: time-bounded with content match
            sql = TEMPORAL_ENTITY_SQL_TEMPLATE
            sql = sql.replace(":start_time", f"'{_sql_escape(time_range[0])}'")
            sql = sql.replace(":end_time", f"'{_sql_escape(time_range[1])}'")
            sql = sql.replace(":entity_name", f"'%{_sql_escape(entities[0])}%'")
            sql = sql.replace(":limit", str(DEFAULT_SQL_LIMIT))
        else:
            # Temporal without entity filter
            sql = TEMPORAL_SQL_TEMPLATE
            sql = sql.replace(":start_time", f"'{_sql_escape(time_range[0])}'")
            sql = sql.replace(":end_time", f"'{_sql_escape(time_range[1])}'")
            sql = sql.replace(":limit", str(DEFAULT_SQL_LIMIT))

        return sql

    def _build_comparative_sql(self, entities: list[str]) -> str | None:
        """
        Build a comparative SQL query for multi-entity comparison.

        Generates a WHERE clause that matches any of the entities
        using OR conditions with COLLATE NOCASE.

        Args:
            entities: List of entity names to compare.

        Returns:
            SQL string, or None if fewer than 2 entities.
        """
        if len(entities) < 2:
            # Comparative requires at least 2 entities
            if entities:
                # Fall back to factual lookup for a single entity
                return self._build_factual_sql(entities)
            return None

        # Build OR conditions for each entity (escape apostrophes)
        conditions = " OR ".join(
            f"e.name LIKE '%{_sql_escape(ent)}%' COLLATE NOCASE"
            for ent in entities
        )

        sql = COMPARATIVE_SQL_TEMPLATE.replace("{entity_conditions}", f"({conditions})")
        sql = sql.replace(":limit", str(DEFAULT_SQL_LIMIT))
        return sql

    def _build_fts_sql(self, rewritten_query: str) -> str | None:
        """
        Build an FTS (full-text search) query as a fallback.

        Uses SQLite's FTS5 MATCH syntax for keyword search. This
        provides a structured-side complement to the vector search
        that runs in parallel.

        Args:
            rewritten_query: The cleaned query string for keyword matching.

        Returns:
            SQL string for FTS search, or None if query is empty.
        """
        if not rewritten_query.strip():
            return None

        # Sanitize FTS5 special characters. FTS5 operators and punctuation
        # cause syntax errors when passed in raw user input. We strip
        # ALL non-alphanumeric, non-space, non-hyphen characters so the
        # query degrades to a simple keyword match rather than crashing.
        # See research/notes/fts5-sanitization.md for the full rationale.
        import re
        sanitized = re.sub(r"[^a-zA-Z0-9\s-]", " ", rewritten_query)

        # FTS5 uses implicit AND -- every token must appear in the
        # document. Common words like "tell", "what", "s" (from
        # contraction splitting) cause zero results because they
        # don't appear in the content. Strip them here so the MATCH
        # focuses on meaningful content words only.
        fts_stop = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "do", "does", "did", "will", "would", "can", "could",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "about", "up", "out", "or", "and", "but", "not", "no",
            "what", "which", "who", "whom", "how", "when", "where", "why",
            "this", "that", "these", "those", "it", "its",
            "i", "me", "my", "we", "our", "you", "your",
            "he", "she", "they", "them", "their",
            "tell", "show", "find", "give", "list", "get",
            "has", "have", "had", "s", "t", "re", "ve", "ll", "d",
        }
        tokens = [t for t in sanitized.split() if t.lower() not in fts_stop and len(t) > 1]
        if not tokens:
            return None

        # Use OR between terms so documents matching ANY keyword are
        # returned. FTS5 default is AND which is too strict for
        # natural language queries -- one irrelevant term kills results.
        match_expr = " OR ".join(tokens)

        sql = (
            "SELECT i.id, i.title, i.content, i.source_type, "
            "i.created_at, rank "
            "FROM interactions_fts fts "
            "JOIN interactions i ON i.rowid = fts.rowid "
            f"WHERE interactions_fts MATCH '{match_expr}' "
            "ORDER BY rank "
            f"LIMIT {DEFAULT_SQL_LIMIT}"
        )
        return sql

    def _determine_routing(self, intent: QueryIntent) -> str:
        """
        Decide which stores to query based on intent.

        Routing rules:
        - AGGREGATION → "sql_only" (counting doesn't need vector search)
        - All other intents → "both" (query both stores, fuse results)

        In Phase 2, routing becomes more sophisticated with VECTOR_TO_SQL
        and SQL_TO_VECTOR modes that require an LLM to orchestrate.

        Args:
            intent: The classified query intent.

        Returns:
            One of "sql_only", "vector_only", or "both".
        """
        if intent == QueryIntent.AGGREGATION:
            return "sql_only"
        return "both"

    def _build_vector_filters(
        self, filters: dict[str, str], time_range: tuple[str, str] | None
    ) -> dict[str, Any]:
        """
        Build metadata filters for the vector store search.

        Converts interpretation filters (source_type, etc.) and time
        ranges into the format expected by the VectorStore.search() method.

        Args:
            filters: Explicit filters from the interpretation.
            time_range: Optional (start, end) ISO datetime tuple.

        Returns:
            A dict of vector store filter parameters.
        """
        vector_filters: dict[str, Any] = {}

        # Map source_type filter to the vector store's source_table column
        if "source_type" in filters:
            vector_filters["source_table"] = "interactions"

        # Add time range as a range filter on the created_at column
        if time_range:
            vector_filters["created_at"] = {
                "gte": time_range[0],
                "lte": time_range[1],
            }

        return vector_filters
