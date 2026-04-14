# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Rule-Based Query Interpreter
=============================

This module implements the Phase 1 query interpreter. It takes a user's
natural language question and extracts structured information: which
entities are mentioned, what the user's intent is (factual, temporal,
aggregation, etc.), and a cleaned query string suitable for vector search.

The interpreter works in three steps:
    1. Regex intent classification -- uses the patterns from intent.py to
       determine the query type (aggregation, temporal, comparative, etc.)
    2. Entity extraction -- matches query noun phrases against entities
       stored in the SQL store using case-insensitive LIKE queries
    3. Refinement -- decides between FACTUAL and SEMANTIC intent based on
       whether entities were found, and builds the rewritten query

This is the "good enough for Phase 1" implementation. Phase 2 replaces
it with LLMQueryInterpreter, which understands domain-specific entity
references and handles ambiguous questions that regex can't parse.

Depends on:
    - re (regex for time range extraction)
    - ctxmtg.interfaces.query (QueryInterpreter ABC)
    - ctxmtg.interfaces.storage (SQLStore for entity lookup)
    - ctxmtg.models.query (QueryIntent enum)
    - ctxmtg.models.profile (DomainProfile)
    - ctxmtg.query.intent (pattern constants, classify_intent, helpers)

Used by:
    - ctxmtg.query.executor (calls interpret() before planning)
    - tests/test_query/test_interpreter.py
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from ctxmtg.interfaces.query import QueryInterpreter
from ctxmtg.interfaces.storage import SQLStore
from ctxmtg.models.profile import DomainProfile
from ctxmtg.models.query import QueryIntent
from ctxmtg.query.intent import (
    classify_intent,
    has_semantic_markers,
    remove_stop_words,
)

# ---------------------------------------------------------------
# Module logger -- structured output for query debugging.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.interpreter")

# ---------------------------------------------------------------
# Regex patterns for extracting temporal bounds from queries.
# These convert natural language time references to ISO datetime
# strings that can be used in SQL WHERE clauses.
# ---------------------------------------------------------------
TIME_RANGE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "last week" → 7 days ago to now
    (re.compile(r"\blast\s+week\b", re.IGNORECASE), "last_week"),
    # "yesterday" → 1 day ago to 1 day ago end-of-day
    (re.compile(r"\byesterday\b", re.IGNORECASE), "yesterday"),
    # "this week" → start of current week to now
    (re.compile(r"\bthis\s+week\b", re.IGNORECASE), "this_week"),
    # "this month" → start of current month to now
    (re.compile(r"\bthis\s+month\b", re.IGNORECASE), "this_month"),
    # "last month" → start of previous month to end of previous month
    (re.compile(r"\blast\s+month\b", re.IGNORECASE), "last_month"),
    # "today" → start of today to now
    (re.compile(r"\btoday\b", re.IGNORECASE), "today"),
    # "last N days" → N days ago to now
    (re.compile(r"\blast\s+(\d+)\s+days?\b", re.IGNORECASE), "last_n_days"),
]


def _compute_time_range(label: str, match: re.Match[str] | None = None) -> tuple[str, str] | None:
    """
    Convert a time label to an (ISO start, ISO end) tuple.

    Uses UTC timestamps throughout. The returned strings are suitable
    for direct use in SQL BETWEEN clauses or vector store time filters.

    Args:
        label: One of the predefined labels from TIME_RANGE_PATTERNS.
        match: The regex match object (needed for "last_n_days" to
               extract the number).

    Returns:
        A (start_iso, end_iso) tuple, or None if the label is unknown.
    """
    now = datetime.now(timezone.utc)

    if label == "last_week":
        start = now - timedelta(days=7)
        return (start.isoformat(), now.isoformat())

    if label == "yesterday":
        yesterday = now - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        return (start.isoformat(), end.isoformat())

    if label == "this_week":
        # Monday = 0 in weekday(), compute start of week
        days_since_monday = now.weekday()
        start = (now - timedelta(days=days_since_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return (start.isoformat(), now.isoformat())

    if label == "this_month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return (start.isoformat(), now.isoformat())

    if label == "last_month":
        # Go to the first of this month, then subtract one day to get last month
        first_of_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_day_prev = first_of_this - timedelta(days=1)
        first_of_prev = last_day_prev.replace(day=1)
        return (first_of_prev.isoformat(), last_day_prev.isoformat())

    if label == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return (start.isoformat(), now.isoformat())

    if label == "last_n_days" and match is not None:
        # Extract the number of days from the regex match
        n = int(match.group(1))
        start = now - timedelta(days=n)
        return (start.isoformat(), now.isoformat())

    return None


def _extract_time_range(query: str) -> tuple[str, str] | None:
    """
    Scan the query string for temporal references and return a time range.

    Checks each temporal pattern in order and returns the first match.
    If no temporal patterns are found, returns None.

    Args:
        query: The user's natural language question.

    Returns:
        A (start_iso, end_iso) tuple, or None if no time reference found.
    """
    for pattern, label in TIME_RANGE_PATTERNS:
        m = pattern.search(query)
        if m:
            return _compute_time_range(label, m)
    return None


def _extract_noun_phrases(query: str) -> list[str]:
    """
    Extract candidate noun phrases from the query for entity matching.

    Uses a simple heuristic: split on common delimiters and filter out
    very short tokens and stop words. This is a lightweight alternative
    to running spaCy's noun chunk extractor on every query.

    The extracted phrases are used to search the entities table for
    matching entity names (case-insensitive).

    Args:
        query: The user's natural language question.

    Returns:
        A list of candidate noun phrases (strings) to match against
        the entities table.
    """
    # Remove common question words and punctuation to isolate noun phrases.
    # This is deliberately simple -- Phase 2 uses spaCy / LLM for better
    # noun phrase extraction.
    cleaned = re.sub(r"[?!.,;:\"'()\[\]]", " ", query)

    # Split into words and filter out short words, stop words, and
    # common question words that aren't useful for entity matching
    words = cleaned.split()
    question_words = {
        "what",
        "who",
        "where",
        "when",
        "how",
        "why",
        "did",
        "does",
        "do",
        "is",
        "are",
        "was",
        "were",
        "many",
        "much",
        "tell",
        "show",
        "find",
        "give",
        "list",
        "get",
        "me",
        "about",
        "the",
        "a",
        "an",
    }

    # Keep words that are likely entity references (capitalised or long enough)
    candidates = []
    for word in words:
        lower = word.lower()
        if lower in question_words:
            continue
        if len(word) < 2:
            continue
        candidates.append(word)

    return candidates


def _extract_filters(query: str) -> dict[str, str]:
    """
    Extract explicit filters from the query string.

    Looks for structured filter expressions like "source:meeting" or
    "type:email" in the query. These are converted to key-value pairs
    that the planner uses to add WHERE clauses to the SQL query.

    Args:
        query: The user's natural language question.

    Returns:
        A dict of filter key-value pairs. Empty dict if no filters found.
    """
    filters: dict[str, str] = {}

    # Look for "source_type" references in the query. This is a simple
    # pattern match -- Phase 2 handles more complex filter expressions.
    source_patterns = [
        (re.compile(r"\bmeeting(?:s)?\b", re.IGNORECASE), "meeting"),
        (re.compile(r"\bemail(?:s)?\b", re.IGNORECASE), "email"),
        (re.compile(r"\bdocument(?:s)?\b", re.IGNORECASE), "doc"),
        (re.compile(r"\bslack\b", re.IGNORECASE), "slack"),
    ]

    for pattern, source_type in source_patterns:
        if pattern.search(query):
            filters["source_type"] = source_type
            break

    return filters


class RuleBasedQueryInterpreter(QueryInterpreter):
    """
    Phase 1 query interpreter using regex + entity name matching.

    This interpreter does NOT require an LLM. It uses:
    1. Compiled regex patterns (from intent.py) to classify intent
    2. SQL queries against the entities table to find entity references
    3. Simple noun phrase extraction to identify search terms
    4. Time range extraction for temporal queries

    The output is a structured interpretation dict consumed by the
    TemplateQueryPlanner to generate SQL and vector queries.

    Limitations (fixed in Phase 2):
    - Cannot understand domain-specific entity references
      (e.g., "the Smith motion" in legal context)
    - Cannot resolve ambiguous queries ("what about that thing from Tuesday?")
    - Cannot handle multi-hop questions ("who does Alice's manager report to?")

    Usage:
        interpreter = RuleBasedQueryInterpreter(sql_store=sqlite_store)
        result = await interpreter.interpret(
            query="What did Alice propose last week?",
            profile=general_profile,
        )
        # result["entities"] → ["Alice"]
        # result["intent"] → QueryIntent.TEMPORAL
        # result["time_range"] → ("2026-03-09T...", "2026-03-16T...")
    """

    def __init__(self, sql_store: SQLStore) -> None:
        """
        Initialise with a reference to the SQL store for entity lookups.

        The SQL store is queried during interpretation to find entity
        references in the user's question. Entity matching uses
        COLLATE NOCASE for case-insensitive comparison.

        Args:
            sql_store: An initialised SQLStore instance (usually SQLiteStore).
        """
        self._sql_store = sql_store

    async def interpret(self, query: str, profile: DomainProfile) -> dict[str, Any]:  # type: ignore[override]
        """
        Extract structured information from a user query.

        Runs the three-step interpretation pipeline:
        1. Classify intent via regex patterns
        2. Extract entity references via SQL store lookups
        3. Refine intent and build the interpretation dict

        Args:
            query: The user's natural language question.
            profile: The active domain profile (used for future
                     domain-specific entity resolution in Phase 2).

        Returns:
            A dict with keys:
                - entities: list[str] -- matched entity names
                - intent: QueryIntent -- classified intent
                - time_range: tuple[str, str] | None -- temporal bounds
                - filters: dict[str, str] -- explicit filters
                - rewritten_query: str -- cleaned query for vector search
        """
        # Step 1: Classify intent from regex patterns.
        # This gives us AGGREGATION, TEMPORAL, COMPARATIVE, or UNKNOWN.
        intent = classify_intent(query)

        # Step 2: Extract entity references by matching query noun phrases
        # against the entities table in the SQL store.
        noun_phrases = _extract_noun_phrases(query)
        matched_entities = await self._match_entities(noun_phrases)

        # Step 3: Refine UNKNOWN intent based on entity detection results.
        # If entities were found and no semantic markers → FACTUAL.
        # If semantic markers present or no entities → SEMANTIC.
        if intent == QueryIntent.UNKNOWN:
            if has_semantic_markers(query):
                intent = QueryIntent.SEMANTIC
            elif matched_entities:
                intent = QueryIntent.FACTUAL
            else:
                # Default to SEMANTIC for open-ended queries with no
                # specific entities or structured patterns
                intent = QueryIntent.SEMANTIC

        # Step 4: Extract time range for temporal queries.
        time_range = _extract_time_range(query)

        # Step 5: Extract explicit filters (source_type, etc.).
        filters = _extract_filters(query)

        # Step 6: Build the rewritten query for vector search.
        # Remove stop words and add entity context for better embeddings.
        rewritten = remove_stop_words(query)

        logger.info(
            "query_interpreted",
            query=query,
            intent=intent.value,
            entities=matched_entities,
            has_time_range=time_range is not None,
            filter_count=len(filters),
        )

        return {
            "entities": matched_entities,
            "intent": intent,
            "time_range": time_range,
            "filters": filters,
            "rewritten_query": rewritten,
        }

    async def _match_entities(self, noun_phrases: list[str]) -> list[str]:
        """
        Match noun phrases against entities in the SQL store.

        For each candidate noun phrase, queries the entities table
        using case-insensitive LIKE matching (COLLATE NOCASE). Returns
        the list of matched entity names (deduplicated).

        This is the core of entity extraction in Phase 1. Phase 2
        replaces this with LLM-based entity resolution that can
        handle aliases, abbreviations, and domain-specific references.

        Args:
            noun_phrases: Candidate strings from the user's query.

        Returns:
            Deduplicated list of entity names that were found in the store.
        """
        matched: list[str] = []
        seen: set[str] = set()

        for phrase in noun_phrases:
            try:
                # Query with COLLATE NOCASE via the get_entities method.
                # name_like wraps the phrase in wildcards internally.
                entities = await self._sql_store.get_entities(name_like=phrase, limit=5)
                for entity in entities:
                    # Deduplicate by lowercased name to avoid
                    # "Alice" and "alice" appearing as separate matches
                    key = entity.name.lower()
                    if key not in seen:
                        seen.add(key)
                        matched.append(entity.name)
            except Exception:
                # Graceful degradation: if entity lookup fails for one
                # phrase, continue with the rest. Log the error.
                logger.warning(
                    "entity_lookup_failed",
                    error_code="CTXMTG-QRY-001",
                    phrase=phrase,
                    exc_info=True,
                )

        return matched
