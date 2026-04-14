# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Intent Classification Patterns
==============================

This module defines the regex patterns and constants used by the
RuleBasedQueryInterpreter to classify the intent of a user's
natural language query. Each pattern set maps to a QueryIntent enum
value (AGGREGATION, TEMPORAL, COMPARATIVE, FACTUAL, SEMANTIC).

The patterns are compiled once at import time for performance -- regex
compilation is expensive, and these patterns are used on every query.

Why regex instead of a classifier?
    Phase 1 operates without an LLM. Regex intent classification is
    fast, deterministic, and good enough for structured questions like
    "how many meetings?" or "what happened last week?". Phase 2
    replaces this with an LLM-based interpreter that handles ambiguous
    and domain-specific queries.

Pattern design philosophy:
    - Each pattern list is ordered from most specific to least specific
    - Patterns use IGNORECASE so "How Many" and "how many" both match
    - Word boundary anchors (\\b) prevent false positives on partial words
    - The first matching pattern determines intent (short-circuit)

Depends on:
    - re (Python's regex library)
    - ctxmtg.models.query (QueryIntent enum)

Used by:
    - ctxmtg.query.interpreter (consumes these patterns for classification)
"""

from __future__ import annotations

import re

from ctxmtg.models.query import QueryIntent

# =====================================================================
# Aggregation patterns: questions about counts, totals, averages.
# These typically want SQL COUNT / SUM / AVG queries, not semantic search.
# Examples: "how many meetings?", "total action items", "average confidence"
# =====================================================================
AGGREGATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bhow\s+many\b", re.IGNORECASE),
    re.compile(r"\bcount\b", re.IGNORECASE),
    re.compile(r"\btotal\b", re.IGNORECASE),
    re.compile(r"\baverage\b", re.IGNORECASE),
    re.compile(r"\bsum\b", re.IGNORECASE),
    re.compile(r"\bnumber\s+of\b", re.IGNORECASE),
]

# =====================================================================
# Temporal patterns: questions about specific time periods.
# These add a time filter to the SQL/vector queries.
# Examples: "last week", "yesterday", "since March 1", "between X and Y"
# =====================================================================
TEMPORAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\blast\s+week\b", re.IGNORECASE),
    re.compile(r"\byesterday\b", re.IGNORECASE),
    re.compile(r"\bthis\s+month\b", re.IGNORECASE),
    re.compile(r"\blast\s+month\b", re.IGNORECASE),
    re.compile(r"\bthis\s+week\b", re.IGNORECASE),
    re.compile(r"\bsince\b", re.IGNORECASE),
    re.compile(r"\bbefore\b", re.IGNORECASE),
    re.compile(r"\bbetween\s+.+?\s+and\b", re.IGNORECASE),
    re.compile(r"\blast\s+\d+\s+days?\b", re.IGNORECASE),
    re.compile(r"\brecently\b", re.IGNORECASE),
    re.compile(r"\btoday\b", re.IGNORECASE),
]

# =====================================================================
# Comparative patterns: questions comparing entities or time periods.
# These need data from multiple entities for side-by-side comparison.
# Examples: "compare Alice and Bob", "differences between X and Y"
# =====================================================================
COMPARATIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bdifference(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bmore\s+than\b", re.IGNORECASE),
    re.compile(r"\bless\s+than\b", re.IGNORECASE),
]

# =====================================================================
# Semantic marker patterns: indicators that the query needs
# similarity-based vector search rather than exact lookups.
# These are soft signals that push toward SEMANTIC intent.
# Examples: "topics related to", "discussions about", "similar to"
# =====================================================================
SEMANTIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brelated\s+to\b", re.IGNORECASE),
    re.compile(r"\bsimilar\s+to\b", re.IGNORECASE),
    re.compile(r"\bdiscussions?\s+about\b", re.IGNORECASE),
    re.compile(r"\btopics?\s+(?:about|on|related)\b", re.IGNORECASE),
    re.compile(r"\blike\b", re.IGNORECASE),
]

# =====================================================================
# Stop words: common English words removed from queries before vector
# search to improve embedding quality. The vector model handles these
# implicitly, but removing them from the rewritten query improves
# keyword-based components (FTS, TF-IDF reranking).
# =====================================================================
STOP_WORDS: set[str] = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "can",
    "could",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "about",
    "up",
    "out",
    "if",
    "or",
    "and",
    "but",
    "not",
    "no",
    "so",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "she",
    "it",
    "they",
    "them",
    "their",
    "how",
    "when",
    "where",
    "why",
}


# =====================================================================
# Intent-to-pattern mapping: maps each QueryIntent to its pattern list.
# The classifier iterates this list in priority order. The first intent
# whose patterns match the query wins. Order matters:
#   1. AGGREGATION (most structured -- count/sum queries)
#   2. COMPARATIVE (explicit comparison keywords)
#   3. TEMPORAL (time-bounded queries)
#   4. SEMANTIC is the fallback when no specific patterns match
#   5. FACTUAL is assigned when entities are detected but no semantic markers
# =====================================================================
INTENT_PATTERNS: list[tuple[QueryIntent, list[re.Pattern[str]]]] = [
    (QueryIntent.AGGREGATION, AGGREGATION_PATTERNS),
    (QueryIntent.COMPARATIVE, COMPARATIVE_PATTERNS),
    (QueryIntent.TEMPORAL, TEMPORAL_PATTERNS),
]


def classify_intent(query: str) -> QueryIntent:
    """
    Classify the intent of a user query using regex pattern matching.

    Iterates through the pattern sets in priority order and returns
    the first matching intent. If no specific patterns match, returns
    UNKNOWN (the caller should determine FACTUAL vs SEMANTIC based on
    entity detection results).

    Args:
        query: The raw user question string.

    Returns:
        The classified QueryIntent. Returns UNKNOWN if no patterns
        match -- the interpreter refines this to FACTUAL or SEMANTIC
        based on whether entities were detected.
    """
    # Check each intent's patterns in priority order.
    # The first matching pattern determines the intent.
    for intent, patterns in INTENT_PATTERNS:
        for pattern in patterns:
            if pattern.search(query):
                return intent

    # No structured intent patterns matched. Return UNKNOWN so the
    # interpreter can decide between FACTUAL and SEMANTIC based on
    # entity detection results.
    return QueryIntent.UNKNOWN


def has_semantic_markers(query: str) -> bool:
    """
    Check if the query contains semantic search indicators.

    Returns True if any semantic patterns match the query. This helps
    the interpreter distinguish between FACTUAL ("what did Alice say?")
    and SEMANTIC ("topics related to security") when the primary
    classifier returns UNKNOWN.

    Args:
        query: The raw user question string.

    Returns:
        True if semantic markers are present, False otherwise.
    """
    return any(pattern.search(query) for pattern in SEMANTIC_PATTERNS)


def remove_stop_words(text: str) -> str:
    """
    Remove common stop words from text for cleaner vector search.

    Splits the text into words, removes stop words, and rejoins.
    This improves the quality of the rewritten query used for
    vector search and TF-IDF reranking.

    Args:
        text: The input text to clean.

    Returns:
        The text with stop words removed. If all words are stop words,
        returns the original text unchanged (to avoid empty queries).
    """
    words = text.split()
    filtered = [w for w in words if w.lower() not in STOP_WORDS]
    # If filtering removed everything, keep the original text
    # to avoid sending an empty query to the vector store
    return " ".join(filtered) if filtered else text
