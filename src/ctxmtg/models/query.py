# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Query Data Models
=================

This module defines the data models for the query subsystem: retrieval
modes, query intents, query plans, search results, and final query results.

These models describe how a user's question flows through the query pipeline:
1. The user asks a question (natural language string)
2. The intent classifier determines what kind of question it is (QueryIntent)
3. The planner creates a QueryPlan (SQL query + vector query + routing)
4. The executor runs both queries, producing SearchResult objects
5. The fuser combines results into a final QueryResult

In the system architecture, these models are used by the query/ package
(intent classifier, planner, executor, fuser, reranker) and returned
to the CLI or API layer for display.

Depends on:
    - pydantic (validation, serialization)
    - enum (retrieval modes, query intents)

Used by:
    - ctxmtg.query.intent (produces QueryIntent)
    - ctxmtg.query.planner (produces QueryPlan)
    - ctxmtg.query.executor (produces SearchResult)
    - ctxmtg.query.fusion (produces QueryResult)
    - ctxmtg.cli (displays QueryResult to user)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------
# RetrievalMode enum: the 4 user-selectable retrieval strategies.
# These control HOW the system queries the dual stores (SQL + vector).
# Phase 1 supports PARALLEL only; others are Phase 2+.
# ---------------------------------------------------------------
class RetrievalMode(str, Enum):
    """
    User-selectable retrieval modes (4 modes).

    Controls how the system queries the dual SQL + vector stores:
    - PARALLEL: Both stores queried independently, results fused with RRF
    - VECTOR_TO_SQL: Vector discovers semantically similar content, LLM formulates SQL
    - SQL_TO_VECTOR: SQL facts first, LLM targets vector search
    - BIDIRECTIONAL: Both informed paths in parallel, full synthesis

    Phase 1 implements PARALLEL only (no LLM needed).
    VECTOR_TO_SQL and SQL_TO_VECTOR require Tier 1+ (local LLM).
    BIDIRECTIONAL requires Tier 2+ (larger local LLM).
    """

    PARALLEL = "parallel"             # All tiers -- both stores independently, RRF fusion
    VECTOR_TO_SQL = "vector_to_sql"   # Tier 1+ -- vector discovers, LLM formulates SQL
    SQL_TO_VECTOR = "sql_to_vector"   # Tier 1+ -- SQL facts first, LLM targets vector search
    BIDIRECTIONAL = "bidirectional"   # Tier 2+ -- both informed paths, full synthesis


# ---------------------------------------------------------------
# QueryIntent enum: what the user is trying to find out.
# The intent classifier maps natural language questions to one
# of these categories, which determines the query strategy.
# ---------------------------------------------------------------
class QueryIntent(str, Enum):
    """
    Classified intent of a user query.

    The query interpreter analyzes the user's question and assigns
    one of these intents, which determines the execution strategy:
    - FACTUAL: precise lookups ("Who proposed OAuth2?")
    - SEMANTIC: similarity search ("discussions about scaling")
    - AGGREGATION: counting/summing ("how many meetings this week?")
    - TEMPORAL: time-filtered ("what happened last Tuesday?")
    - COMPARATIVE: comparing entities ("Alice vs Bob contributions")
    - UNKNOWN: fallback when intent can't be determined
    """

    FACTUAL = "factual"           # Precise fact lookups
    SEMANTIC = "semantic"         # Similarity-based search
    AGGREGATION = "aggregation"   # Counting, summing, averaging
    TEMPORAL = "temporal"         # Time-filtered queries
    COMPARATIVE = "comparative"   # Comparing entities or time periods
    UNKNOWN = "unknown"           # Fallback: intent unclear


# ---------------------------------------------------------------
# QueryPlan model: the execution blueprint for a query.
# Created by the planner from the user's interpreted question,
# this tells the executor exactly what SQL and vector queries to run.
# ---------------------------------------------------------------
class QueryPlan(BaseModel):
    """
    Plan for executing a query against the dual stores.

    The planner creates this plan from the user's question and its
    interpreted intent. It specifies the SQL query to run against SQLite,
    the vector query to run against LanceDB, any metadata filters for
    the vector search, and the routing decision (which stores to query).

    The executor takes this plan and runs both queries in parallel.
    """

    # The original question the user asked
    original_query: str

    # Classified intent of the query (determines strategy)
    intent: QueryIntent

    # SQL query to run against the SQL store (None if vector-only)
    sql_query: str | None = None

    # Text query for the vector store (None if SQL-only)
    vector_query: str | None = None

    # Metadata filters for the vector search (e.g., source_type, time range)
    vector_filters: dict[str, Any] = Field(default_factory=dict)

    # Which stores to query: "sql_only", "vector_only", or "both"
    routing: str = "both"


# ---------------------------------------------------------------
# SearchResult model: a single result from either SQL or vector store.
# The executor produces these, and the fuser combines them into a
# final ranked list.
# ---------------------------------------------------------------
class SearchResult(BaseModel):
    """
    A single result from either store.

    Each search result carries a score indicating relevance, metadata
    about which store it came from, and the actual content snippet.
    The fuser takes results from both stores and combines them using
    Reciprocal Rank Fusion (RRF) to produce a unified ranking.
    """

    # Unique identifier of the matching record
    id: str

    # Which store this result came from: "sql" or "vector"
    source_store: str

    # The matching content (full text or snippet)
    content: str

    # Relevance score (meaning varies by store -- similarity for vector, rank for SQL)
    score: float

    # Additional metadata (e.g., entity_type, source_type, created_at)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------
# QueryResult model: the final answer to a user's query.
# Contains fused results from both stores, along with stats
# and an optional LLM-synthesized answer (Tier 2+).
# ---------------------------------------------------------------
class QueryResult(BaseModel):
    """
    Complete result of a query, after fusion.

    This is the final output returned to the user. It contains the
    fused and ranked results from both SQL and vector stores, counts
    from each store for transparency, the retrieval mode used, latency
    information, and an optional LLM-synthesized answer (available
    only in Tier 2+ when a local LLM is loaded).
    """

    # The original question that was asked
    query: str

    # Which retrieval mode was used (PARALLEL, VECTOR_TO_SQL, etc.)
    mode: RetrievalMode

    # Fused and ranked results from both stores
    results: list[SearchResult]

    # Total number of results (before any top-k cutoff)
    total_results: int

    # How many results came from the SQL store
    sql_results_count: int

    # How many results came from the vector store
    vector_results_count: int

    # LLM-generated synthesized answer (Tier 2+ only, None for Tier 0-1)
    synthesis: str | None = None

    # If the executor fell back from the requested retrieval mode
    # (e.g., v2s → parallel because LLM was unavailable), this field
    # records the reason.  None means no fallback occurred.
    fallback_reason: str | None = None

    # How long the query took to execute (in milliseconds)
    latency_ms: float
