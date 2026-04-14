# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Query Interface ABCs
====================

This module defines the abstract base classes for the query subsystem:
interpreting user questions, planning query execution, fusing results
from multiple stores, and re-ranking for quality.

The query pipeline transforms a natural language question into
structured results through these stages:
1. QueryInterpreter: extracts entities, intent, and context from the query
2. QueryPlanner: creates an execution plan (SQL + vector queries)
3. ResultFuser: combines results from SQL and vector stores (RRF)
4. Reranker: re-ranks fused results for relevance

Phase 1 uses rule-based implementations (regex patterns, TF-IDF).
Phase 2 replaces the interpreter and fuser with LLM-powered versions
for domain-specific understanding and natural language synthesis.

Depends on:
    - abc (Python's Abstract Base Class machinery)
    - ctxmtg.models.query (QueryPlan, SearchResult)
    - ctxmtg.models.profile (DomainProfile -- controls query behavior)

Used by:
    - ctxmtg.query.intent (implements QueryInterpreter)
    - ctxmtg.query.planner (implements QueryPlanner)
    - ctxmtg.query.fusion (implements ResultFuser)
    - ctxmtg.query.reranker (implements Reranker)
    - ctxmtg.query.executor (orchestrates the full query pipeline)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# ---------------------------------------------------------------
# Import the data models used by the query interfaces.
# QueryPlan describes what queries to execute, SearchResult holds
# individual results, and DomainProfile controls behavior.
# ---------------------------------------------------------------
from ctxmtg.models.profile import DomainProfile
from ctxmtg.models.query import QueryPlan, SearchResult

# =====================================================================
# QueryInterpreter ABC -- Natural Language Query Understanding
# =====================================================================


class QueryInterpreter(ABC):
    """
    Interprets user queries to extract entities, intent, and context.

    Phase 1: RuleBasedInterpreter (regex patterns, entity name matching).
    Phase 2: LLMInterpreter (dedicated LLM role "Query Interpreter" --
    assigned its own model in the model assignment table).

    This is a key differentiator for domain-specific use cases. A legal
    professional asking "What did Judge Morrison rule on the Smith motion?"
    needs the interpreter to understand Judge Morrison is a person, Smith
    is a case reference, and motion is a legal instrument. Regex cannot
    do this; a domain-tuned LLM can.

    The interpreter produces a structured interpretation dict that the
    planner uses to generate SQL and vector queries.

    Usage:
        interpreter = RuleBasedQueryInterpreter(entity_store)
        interpretation = interpreter.interpret(
            query="What did Alice propose last week?",
            profile=general_profile,
        )
        # interpretation["entities"] → ["Alice"]
        # interpretation["intent"] → QueryIntent.FACTUAL
        # interpretation["time_range"] → ("2026-03-03", "2026-03-10")
    """

    @abstractmethod
    def interpret(self, query: str, profile: DomainProfile) -> dict[str, Any]:
        """
        Extract structured information from a user query.

        Analyzes the natural language query to identify:
        - Entity references (people, projects, tools mentioned)
        - Query intent (factual, semantic, aggregation, temporal, comparative)
        - Time range (if temporal bounds are mentioned)
        - Explicit filters (if the user specifies constraints)
        - A cleaned/rewritten query for vector search

        The interpretation is used by the QueryPlanner to generate
        SQL and vector queries tailored to what the user is asking.

        Args:
            query: The user's natural language question.
            profile: The active domain profile, which may influence
                     how entity references are resolved (e.g., legal
                     profiles understand case numbers).

        Returns:
            A dict with the following keys:
                - entities: list[str] -- entity names/references found
                - intent: QueryIntent -- classified intent
                - time_range: tuple[str, str] | None -- temporal bounds
                  as ISO datetime strings, or None if no time reference
                - filters: dict[str, str] -- any explicit filters the
                  user specified (e.g., source_type="meeting")
                - rewritten_query: str -- cleaned query optimized for
                  vector search (stopwords removed, entities expanded)
        """
        ...


# =====================================================================
# QueryPlanner ABC -- Query Execution Planning
# =====================================================================


class QueryPlanner(ABC):
    """
    Converts interpreted query to an execution plan.

    Takes the structured interpretation from the QueryInterpreter
    and produces a QueryPlan that specifies:
    - Which SQL query to run against the SQL store
    - Which vector query to run against the vector store
    - Any metadata filters for the vector search
    - Routing: which stores to query (sql_only, vector_only, both)

    Phase 1: TemplateQueryPlanner uses SQL templates for each intent
    type (factual → fact lookup, aggregation → COUNT query, etc.).
    Phase 2: LLMQueryPlanner uses the LLM to generate dynamic SQL
    for complex queries.

    Usage:
        planner = TemplateQueryPlanner()
        plan = planner.plan(
            query="What did Alice propose?",
            interpretation={"entities": ["Alice"], "intent": "factual", ...},
            profile=general_profile,
        )
        # plan.sql_query → "SELECT * FROM facts WHERE ..."
        # plan.vector_query → "Alice proposal"
    """

    @abstractmethod
    def plan(
        self, query: str, interpretation: dict[str, Any], profile: DomainProfile
    ) -> QueryPlan:
        """
        Produce an execution plan from a query and its interpretation.

        Generates the specific SQL and vector queries to execute based
        on the interpreted user question. The plan also specifies
        routing (which stores to query) and vector filters.

        Args:
            query: The original user question (for reference).
            interpretation: The structured interpretation dict from
                            the QueryInterpreter, containing entities,
                            intent, time_range, filters, and rewritten_query.
            profile: The active domain profile, which may influence
                     query generation (e.g., legal profiles generate
                     different SQL templates).

        Returns:
            A QueryPlan object specifying the SQL query, vector query,
            filters, and routing strategy to execute.
        """
        ...


# =====================================================================
# ResultFuser ABC -- Multi-Store Result Combination
# =====================================================================


class ResultFuser(ABC):
    """
    Combines results from multiple stores.

    After the executor runs queries against both SQL and vector stores,
    the fuser combines the two result sets into a single ranked list.

    Phase 1: RRFFuser uses Reciprocal Rank Fusion (k=60) -- a purely
    mathematical approach that doesn't require an LLM. It assigns
    each result a score based on its rank in each store's result list,
    then sorts by combined score.

    Phase 2: LLMFuser sends both result sets to the LLM, which
    synthesizes a natural language answer. This produces better
    results but requires a local LLM.

    Usage:
        fuser = RRFFuser()
        fused = fuser.fuse(
            sql_results=[...],
            vector_results=[...],
            k=60,
        )
    """

    @abstractmethod
    def fuse(
        self,
        sql_results: list[SearchResult],
        vector_results: list[SearchResult],
        k: int = 60,
    ) -> list[SearchResult]:
        """
        Fuse results from SQL and vector stores using RRF.

        Combines two result lists into a single ranked list using
        Reciprocal Rank Fusion. For each result, the fused score is:
            score = sum(1 / (k + rank_in_store)) for each store

        Results that appear in both stores get scores from both.
        Results that appear in only one store get a score from that
        store only.

        Args:
            sql_results: Results from the SQL store query, ordered
                         by relevance (most relevant first).
            vector_results: Results from the vector store search,
                            ordered by similarity (most similar first).
            k: The RRF constant (default 60). Higher k gives more
               equal weighting across ranks; lower k gives more
               weight to top-ranked results.

        Returns:
            A single list of SearchResult objects, sorted by fused
            score (highest first). Metadata from both stores is
            merged for results that appear in both.
        """
        ...


# =====================================================================
# Reranker ABC -- Result Quality Improvement
# =====================================================================


class Reranker(ABC):
    """
    Re-ranks fused results for quality improvement.

    After fusion, the reranker optionally re-scores results based on
    a more detailed relevance assessment. This adds a second pass of
    quality filtering on top of the initial ranking.

    Phase 1: TFIDFReranker uses TF-IDF cosine similarity between the
    query and each result's content. Lightweight and fast.
    Phase 4: CrossEncoderReranker uses a cross-encoder model for
    more accurate but slower re-ranking.

    Usage:
        reranker = TFIDFReranker()
        reranked = reranker.rerank(
            query="What did Alice propose?",
            results=fused_results,
            top_k=10,
        )
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Re-rank results by relevance to the query.

        Takes the fused results and re-scores them using a more
        detailed relevance model (TF-IDF, cross-encoder, etc.).
        Returns the top_k most relevant results.

        Args:
            query: The original user question.
            results: The fused results from the ResultFuser.
            top_k: Number of top results to return (default 10).
                   The reranker may receive more results than this
                   and returns only the top_k best.

        Returns:
            A list of at most top_k SearchResult objects, re-ranked
            by relevance. Scores are updated to reflect the reranker's
            assessment.
        """
        ...
