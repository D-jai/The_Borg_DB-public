# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Bidirectional Informed Retrieval (Mode 4)
==========================================

This module implements the most thorough retrieval mode: Bidirectional.
It runs BOTH the V→SQL and S→V paths, producing 4 result sets that
the LLM synthesizes into a comprehensive answer.

The Bidirectional retriever is designed for complex, multi-faceted
questions where completeness matters more than speed. It discovers
entities the user didn't mention (via vector), provides structured
context for known entities (via SQL), detects cross-store connections
in both directions, and flags contradictions between stores.

Cost: 4-5 LLM calls (most expensive mode).
Availability: Tier 2+ with 7B+ LLM recommended.

Pipeline:
    1. Run V→SQL path: vector discovery → LLM bridge → SQL facts
    2. Run S→V path: SQL briefing → LLM bridge → vector chunks
    3. Combine all 4 result sets (initial vector, vector-informed SQL,
       initial SQL briefing, SQL-informed vector)
    4. LLM synthesizes the combined results into a comprehensive answer

If the LLM is unavailable, returns None to signal the caller should
fall back to Parallel mode.

Depends on:
    - time (latency measurement)
    - structlog (structured logging)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider)
    - ctxmtg.llm.prompt_assembler (PromptAssembler)
    - ctxmtg.models.profile (DomainProfile)
    - ctxmtg.models.query (QueryResult, SearchResult, RetrievalMode)
    - ctxmtg.query.briefing (SQLBriefingBuilder)
    - ctxmtg.query.informed_retrieval (VectorToSQLRetriever, SQLToVectorRetriever)
    - ctxmtg.query.synthesizer (LLMSynthesizer)

Used by:
    - ctxmtg.query.executor (dispatches to this for BIDIRECTIONAL mode)
    - tests/test_query/test_bidirectional.py
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.llm.prompt_assembler import PromptAssembler
from ctxmtg.models.profile import DomainProfile
from ctxmtg.models.query import QueryResult, RetrievalMode, SearchResult
from ctxmtg.query.briefing import SQLBriefingBuilder
from ctxmtg.query.informed_retrieval import SQLToVectorRetriever, VectorToSQLRetriever
from ctxmtg.query.synthesizer import LLMSynthesizer

# ---------------------------------------------------------------
# Module logger -- logs bidirectional retrieval steps and timings.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.bidirectional")

# ---------------------------------------------------------------
# Default top-k for individual path results.
# ---------------------------------------------------------------
DEFAULT_TOP_K = 10


class BidirectionalRetriever:
    """
    Mode 4: Bidirectional informed retrieval.

    Runs BOTH V→SQL and S→V paths and synthesizes all result sets
    into a comprehensive answer. This is the most thorough mode,
    producing the richest results at the highest cost.

    Pipeline:
        1. Run VectorToSQL and SQLToVector retrievers (sequentially
           since both share the same stores and LLM)
        2. Combine all result sets from both paths
        3. Deduplicate overlapping results
        4. LLM synthesizes the combined results

    If the LLM is unavailable, returns None to signal the caller
    should fall back to Parallel mode.

    Usage:
        retriever = BidirectionalRetriever(
            sql_store=sqlite_store,
            vector_store=lancedb_store,
            llm=llm_provider,
            prompt_assembler=assembler,
            profile=general_profile,
            briefing_builder=SQLBriefingBuilder(),
        )
        result = await retriever.retrieve(
            query="What is the full picture on the OAuth2 migration?",
            interpretation={"entities": ["OAuth2"], ...},
        )
    """

    def __init__(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        llm: LLMProvider,
        prompt_assembler: PromptAssembler,
        profile: DomainProfile,
        briefing_builder: SQLBriefingBuilder,
        embedding_fn: Any | None = None,
    ) -> None:
        """
        Initialize the Bidirectional retriever.

        Creates the underlying V→SQL and S→V retrievers and a
        synthesizer for the final combined synthesis.

        Args:
            sql_store: The SQL store for structured queries.
            vector_store: The vector store for semantic search.
            llm: The LLM provider for bridge queries and synthesis.
            prompt_assembler: The 4-layer prompt assembler.
            profile: Active domain profile.
            briefing_builder: Builder for SQL profiling queries.
            embedding_fn: Optional callable that converts text to a
                          vector. Required for vector search.
        """
        self._llm = llm
        self._prompt_assembler = prompt_assembler
        self._profile = profile

        # Create the two sub-retrievers that handle each path.
        self._v2s_retriever = VectorToSQLRetriever(
            sql_store=sql_store,
            vector_store=vector_store,
            llm=llm,
            prompt_assembler=prompt_assembler,
            profile=profile,
            briefing_builder=briefing_builder,
            embedding_fn=embedding_fn,
        )

        self._s2v_retriever = SQLToVectorRetriever(
            sql_store=sql_store,
            vector_store=vector_store,
            llm=llm,
            prompt_assembler=prompt_assembler,
            profile=profile,
            briefing_builder=briefing_builder,
            embedding_fn=embedding_fn,
        )

        # Synthesizer for the final combined answer.
        self._synthesizer = LLMSynthesizer(
            llm=llm,
            prompt_assembler=prompt_assembler,
            profile=profile,
        )

    async def retrieve(
        self,
        query: str,
        interpretation: dict[str, Any],
        top_k: int = DEFAULT_TOP_K,
    ) -> QueryResult | None:
        """
        Run both V→SQL and S→V paths and synthesize all results.

        Steps:
            1. Run V→SQL path → produces vector + SQL results
            2. Run S→V path → produces SQL briefing + vector results
            3. Combine and deduplicate all result sets
            4. LLM synthesizes the combined results

        Both paths run sequentially (they share LLM and stores).
        Future optimization: if the LLM supports concurrent calls,
        these could be parallelized with asyncio.gather.

        Args:
            query: The user's natural language question.
            interpretation: Structured interpretation from the interpreter.
            top_k: Number of final results to return.

        Returns:
            A QueryResult with results from all paths and LLM synthesis,
            or None if the LLM is unavailable.
        """
        start_time = time.monotonic()

        # ---------------------------------------------------------------
        # Graceful degradation: if LLM unavailable, signal fallback.
        # ---------------------------------------------------------------
        if not self._llm.is_available():
            logger.info("bidirectional_llm_unavailable_fallback", query=query)
            return None

        try:
            # ---------------------------------------------------------------
            # Step 1: Run V→SQL path.
            # ---------------------------------------------------------------
            v2s_result = await self._v2s_retriever.retrieve(
                query=query,
                interpretation=interpretation,
                top_k=top_k,
            )

            v2s_results: list[SearchResult] = []
            v2s_sql_count = 0
            v2s_vector_count = 0
            if v2s_result is not None:
                v2s_results = v2s_result.results
                v2s_sql_count = v2s_result.sql_results_count
                v2s_vector_count = v2s_result.vector_results_count

            logger.info(
                "bidirectional_v2s_done",
                query=query,
                result_count=len(v2s_results),
            )

            # ---------------------------------------------------------------
            # Step 2: Run S→V path.
            # ---------------------------------------------------------------
            s2v_result = await self._s2v_retriever.retrieve(
                query=query,
                interpretation=interpretation,
                top_k=top_k,
            )

            s2v_results: list[SearchResult] = []
            s2v_sql_count = 0
            s2v_vector_count = 0
            if s2v_result is not None:
                s2v_results = s2v_result.results
                s2v_sql_count = s2v_result.sql_results_count
                s2v_vector_count = s2v_result.vector_results_count

            logger.info(
                "bidirectional_s2v_done",
                query=query,
                result_count=len(s2v_results),
            )

            # ---------------------------------------------------------------
            # Step 3: Combine and deduplicate all result sets.
            # Results from both paths are merged. Duplicate IDs are kept
            # only once (first occurrence wins, highest score preserved).
            # ---------------------------------------------------------------
            combined = self._merge_results(v2s_results, s2v_results)

            total_sql = v2s_sql_count + s2v_sql_count
            total_vector = v2s_vector_count + s2v_vector_count

            # ---------------------------------------------------------------
            # Step 4: LLM synthesizes the combined results.
            # ---------------------------------------------------------------
            synthesis = self._synthesizer.synthesize(
                query=query,
                results=combined,
                mode=RetrievalMode.BIDIRECTIONAL,
            )

            latency_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "bidirectional_retrieval_complete",
                query=query,
                v2s_count=len(v2s_results),
                s2v_count=len(s2v_results),
                combined_count=len(combined),
                has_synthesis=synthesis is not None,
                latency_ms=round(latency_ms, 2),
            )

            return QueryResult(
                query=query,
                mode=RetrievalMode.BIDIRECTIONAL,
                results=combined[:top_k],
                total_results=len(combined),
                sql_results_count=total_sql,
                vector_results_count=total_vector,
                synthesis=synthesis,
                latency_ms=round(latency_ms, 2),
            )

        except Exception as exc:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "bidirectional_retrieval_failed",
                error_code="CTXMTG-QRY-001",
                error=str(exc),
                latency_ms=round(latency_ms, 2),
            )
            return None

    @staticmethod
    def _merge_results(
        v2s_results: list[SearchResult],
        s2v_results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Merge results from both retrieval paths, deduplicating by ID.

        Results with the same ID appear in only one copy. The version
        with the higher score is kept. Results are sorted by score
        descending.

        Args:
            v2s_results: Results from the V→SQL path.
            s2v_results: Results from the S→V path.

        Returns:
            A merged, deduplicated, score-sorted list of SearchResult.
        """
        result_map: dict[str, SearchResult] = {}

        # Add V→SQL results first.
        for result in v2s_results:
            existing = result_map.get(result.id)
            if existing is None or result.score > existing.score:
                result_map[result.id] = result

        # Add S→V results, keeping higher-scored version.
        for result in s2v_results:
            existing = result_map.get(result.id)
            if existing is None or result.score > existing.score:
                result_map[result.id] = result

        # Sort by score descending.
        return sorted(
            result_map.values(),
            key=lambda r: r.score,
            reverse=True,
        )
