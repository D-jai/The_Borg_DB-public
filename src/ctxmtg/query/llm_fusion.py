# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
LLM-Powered Result Fusion
===========================

This module implements LLM-powered result fusion for Phase 2+. Instead
of the mathematical Reciprocal Rank Fusion (RRF) from Phase 1, the LLM
receives both result streams and reorders them by relevance based on
semantic understanding of the user's query.

The LLM reads:
    - SQL results (structured facts with entities and predicates)
    - Vector results (semantic chunks from full-text content)

And returns a relevance-ordered list of result IDs. The LLMFuser then
reorders the original SearchResult objects based on the LLM's ranking.

Graceful degradation: if the LLM is unavailable or returns an
unparseable response, the fuser falls back to the RRFFuser from
Phase 1. This ensures the system always produces a result, even when
the LLM is broken.

Depends on:
    - json (parsing LLM JSON responses)
    - structlog (structured logging)
    - ctxmtg.interfaces.llm (LLMProvider ABC)
    - ctxmtg.interfaces.query (ResultFuser ABC)
    - ctxmtg.llm.prompt_assembler (PromptAssembler)
    - ctxmtg.models.profile (DomainProfile)
    - ctxmtg.models.query (SearchResult)
    - ctxmtg.query.fusion (RRFFuser for fallback)

Used by:
    - ctxmtg.query.executor (calls fuse() after parallel execution)
    - tests/test_query/test_llm_fusion.py
"""

from __future__ import annotations

import json

import structlog

from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.interfaces.query import ResultFuser
from ctxmtg.llm.prompt_assembler import PromptAssembler
from ctxmtg.models.profile import DomainProfile
from ctxmtg.models.query import SearchResult
from ctxmtg.query.fusion import RRFFuser

# ---------------------------------------------------------------
# Module logger -- structured output for LLM fusion debugging.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.llm_fusion")


class LLMFuser(ResultFuser):
    """
    LLM-powered result fusion for Tier 2+.

    Instead of mathematical rank fusion (RRF), the LLM receives both
    result streams and reorders them by relevance. This produces better
    results for complex queries where the LLM can understand which
    facts are most relevant to the user's question.

    The fuser formats both result sets as numbered text, asks the LLM
    to rank them by relevance, and returns the reordered results with
    LLM-assigned relevance scores.

    Falls back to RRFFuser if:
    - The LLM is unavailable (is_available() returns False)
    - The LLM returns invalid JSON
    - The LLM call raises an exception

    Usage:
        fuser = LLMFuser(
            llm=llm_provider,
            prompt_assembler=assembler,
            profile=general_profile,
        )
        fused = fuser.fuse(sql_results, vector_results, k=60)
    """

    def __init__(
        self,
        llm: LLMProvider,
        prompt_assembler: PromptAssembler,
        profile: DomainProfile,
        fallback: ResultFuser | None = None,
    ) -> None:
        """
        Initialize the LLM-powered fuser.

        Args:
            llm: The LLM provider for generating relevance rankings.
            prompt_assembler: The 4-layer prompt assembler.
            profile: Default domain profile for prompt assembly.
            fallback: Optional fallback fuser. If None, creates an
                      RRFFuser as the default fallback.
        """
        self._llm = llm
        self._prompt_assembler = prompt_assembler
        self._profile = profile
        self._fallback = fallback if fallback is not None else RRFFuser()

    def fuse(
        self,
        sql_results: list[SearchResult],
        vector_results: list[SearchResult],
        k: int = 60,
    ) -> list[SearchResult]:
        """
        LLM fuses results by relevance. Falls back to RRF if LLM fails.

        Formats both result sets as text, sends them to the LLM for
        relevance ranking, and returns the reordered results. The LLM
        returns a JSON list of result IDs in relevance order.

        Args:
            sql_results: Results from the SQL store.
            vector_results: Results from the vector store.
            k: The RRF constant (used only by the fallback fuser).

        Returns:
            A single list of SearchResult objects, ordered by LLM
            relevance (or RRF score if LLM fails).
        """
        # ---------------------------------------------------------------
        # Handle empty inputs -- no need for LLM if nothing to fuse.
        # ---------------------------------------------------------------
        if not sql_results and not vector_results:
            return []

        # If only one stream has results, return them directly
        # (no fusion needed -- just assign scores).
        if not sql_results:
            return self._assign_scores(vector_results)
        if not vector_results:
            return self._assign_scores(sql_results)

        # ---------------------------------------------------------------
        # Check LLM availability. Fall back to RRF if unavailable.
        # ---------------------------------------------------------------
        if not self._llm.is_available():
            logger.info("llm_unavailable_using_rrf_fallback")
            return self._fallback.fuse(sql_results, vector_results, k=k)

        try:
            # ---------------------------------------------------------------
            # Step 1: Format both result sets as numbered text.
            # ---------------------------------------------------------------
            sql_text = self._format_results(sql_results, "SQL")
            vector_text = self._format_results(vector_results, "VEC")

            # ---------------------------------------------------------------
            # Step 2: Build the system prompt for retrieval/fusion.
            # We use the retrieval stage prompt for this task.
            # ---------------------------------------------------------------
            system_prompt = self._prompt_assembler.assemble(
                stage="retrieval",
                profile=self._profile,
            )

            # ---------------------------------------------------------------
            # Step 3: Build the user prompt asking the LLM to rank results.
            # ---------------------------------------------------------------
            user_prompt = (
                "You have results from two knowledge stores for a user query.\n\n"
                "SQL results (structured facts):\n"
                f"{sql_text}\n\n"
                "Vector results (semantic chunks):\n"
                f"{vector_text}\n\n"
                "Rank ALL results by relevance. Return a JSON list of result IDs "
                "in order of relevance (most relevant first).\n"
                'Output JSON: {"ranked_ids": ["id1", "id2", ...]}'
            )

            # ---------------------------------------------------------------
            # Step 4: Call the LLM with json_mode=True.
            # ---------------------------------------------------------------
            stage_params = self._prompt_assembler.get_stage_params(
                "retrieval", self._profile
            )

            response = self._llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=stage_params.temperature,
                max_tokens=stage_params.max_tokens,
                top_p=stage_params.top_p,
                json_mode=True,
            )

            # ---------------------------------------------------------------
            # Step 5: Parse the LLM response and reorder results.
            # ---------------------------------------------------------------
            ranked_results = self._parse_and_reorder(
                response, sql_results, vector_results
            )

            logger.info(
                "llm_fusion_completed",
                sql_count=len(sql_results),
                vector_count=len(vector_results),
                fused_count=len(ranked_results),
            )

            return ranked_results

        except Exception as exc:
            # ---------------------------------------------------------------
            # LLM fusion failed. Fall back to RRF.
            # ---------------------------------------------------------------
            logger.warning(
                "llm_fusion_failed_using_rrf_fallback",
                error_code="CTXMTG-QRY-001",
                error=str(exc),
                exc_info=True,
            )
            return self._fallback.fuse(sql_results, vector_results, k=k)

    def _parse_and_reorder(
        self,
        response: str,
        sql_results: list[SearchResult],
        vector_results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Parse the LLM's ranked ID list and reorder the results.

        The LLM returns a JSON object with a "ranked_ids" list. We
        look up each ID in the combined result sets and build a new
        list in the LLM's ranking order. Results not mentioned by the
        LLM are appended at the end.

        Args:
            response: The raw JSON string from the LLM.
            sql_results: Original SQL results.
            vector_results: Original vector results.

        Returns:
            Reordered list of SearchResult objects.

        Raises:
            ValueError: If the response is not valid JSON.
        """
        data = json.loads(response)

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        ranked_ids = data.get("ranked_ids", [])
        if not isinstance(ranked_ids, list):
            raise ValueError("ranked_ids must be a list")

        # ---------------------------------------------------------------
        # Build a lookup map from result ID to result + metadata.
        # Track which store each result came from for source_store.
        # ---------------------------------------------------------------
        result_map: dict[str, SearchResult] = {}
        for result in sql_results:
            if result.id not in result_map:
                result_map[result.id] = result
            else:
                # Result exists in both stores -- mark as "both"
                existing = result_map[result.id]
                result_map[result.id] = SearchResult(
                    id=result.id,
                    source_store="both",
                    content=max(existing.content, result.content, key=len),
                    score=existing.score,
                    metadata={**existing.metadata, **result.metadata},
                )

        for result in vector_results:
            if result.id not in result_map:
                result_map[result.id] = result
            else:
                # Result exists in both stores -- mark as "both"
                existing = result_map[result.id]
                result_map[result.id] = SearchResult(
                    id=result.id,
                    source_store="both",
                    content=max(existing.content, result.content, key=len),
                    score=existing.score,
                    metadata={**existing.metadata, **result.metadata},
                )

        # ---------------------------------------------------------------
        # Build the ranked output based on LLM ordering.
        # Assign scores based on rank position (1.0 for first, decreasing).
        # ---------------------------------------------------------------
        ranked: list[SearchResult] = []
        seen_ids: set[str] = set()
        total = len(result_map)

        for rank, result_id in enumerate(ranked_ids):
            result_id = str(result_id)
            if result_id in result_map and result_id not in seen_ids:
                seen_ids.add(result_id)
                original = result_map[result_id]
                # Score based on LLM rank: highest rank gets highest score
                score = (total - rank) / total if total > 0 else 0.0
                ranked.append(
                    SearchResult(
                        id=original.id,
                        source_store=original.source_store,
                        content=original.content,
                        score=score,
                        metadata=original.metadata,
                    )
                )

        # ---------------------------------------------------------------
        # Append any results not mentioned by the LLM at the end.
        # These get lower scores than any LLM-ranked result.
        # ---------------------------------------------------------------
        for result_id, result in result_map.items():
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                # Score below the lowest LLM-ranked score
                score = 0.0
                ranked.append(
                    SearchResult(
                        id=result.id,
                        source_store=result.source_store,
                        content=result.content,
                        score=score,
                        metadata=result.metadata,
                    )
                )

        return ranked

    @staticmethod
    def _format_results(results: list[SearchResult], prefix: str) -> str:
        """
        Format a list of SearchResult objects as numbered text for LLM.

        Each result is formatted as:
            [n] id=<id> | content: <content>

        Args:
            results: The results to format.
            prefix: Label for the results ("SQL" or "VEC").

        Returns:
            Formatted string with numbered results.
        """
        if not results:
            return "(none)"

        lines: list[str] = []
        for i, result in enumerate(results, start=1):
            content = result.content[:300] if len(result.content) > 300 else result.content
            lines.append(f"[{prefix}:{i}] id={result.id} | {content}")

        return "\n".join(lines)

    @staticmethod
    def _assign_scores(results: list[SearchResult]) -> list[SearchResult]:
        """
        Assign rank-based scores to a single stream of results.

        Used when only one store has results (no fusion needed).
        Scores decrease linearly from 1.0 for the first result.

        Args:
            results: The results to score.

        Returns:
            New SearchResult list with rank-based scores.
        """
        total = len(results)
        scored: list[SearchResult] = []
        for rank, result in enumerate(results):
            score = (total - rank) / total if total > 0 else 0.0
            scored.append(
                SearchResult(
                    id=result.id,
                    source_store=result.source_store,
                    content=result.content,
                    score=score,
                    metadata=result.metadata,
                )
            )
        return scored
