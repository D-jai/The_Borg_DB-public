# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
LLM Synthesizer
=================

This module implements natural language answer synthesis from query
results. The synthesizer takes fused results from both SQL and vector
stores and generates a human-readable answer with source citations.

The synthesis stage is the final step in the Tier 2+ query pipeline.
Instead of returning raw ranked results (Tier 0-1), the LLM reads
both result streams and produces a coherent natural language answer
with [SQL:n] and [VEC:n] citations referencing specific sources.

The synthesizer can optionally receive a SQL briefing (from the
SQLBriefingBuilder) which provides additional structured context
for more comprehensive answers.

Citation format:
    - [SQL:n] references the nth SQL result (1-indexed)
    - [VEC:n] references the nth vector result (1-indexed)

The LLM is instructed to note contradictions between sources and
flag missing information rather than fabricating answers.

Depends on:
    - structlog (structured logging)
    - ctxmtg.interfaces.llm (LLMProvider ABC)
    - ctxmtg.llm.prompt_assembler (PromptAssembler for 4-layer prompts)
    - ctxmtg.models.profile (DomainProfile)
    - ctxmtg.models.query (SearchResult)

Used by:
    - ctxmtg.query.executor (calls synthesize() after fusion, Tier 2+)
    - tests/test_query/test_synthesizer.py
"""

from __future__ import annotations

import structlog

from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.llm.prompt_assembler import PromptAssembler
from ctxmtg.models.profile import DomainProfile
from ctxmtg.models.query import RetrievalMode, SearchResult

# ---------------------------------------------------------------
# Module logger -- structured output for synthesis debugging.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.synthesizer")

# ---------------------------------------------------------------
# Maximum number of results to include in synthesis context.
# Limits the amount of text sent to the LLM to avoid context
# window overflow. Top results from each store are included.
# ---------------------------------------------------------------
MAX_SQL_RESULTS_FOR_SYNTHESIS = 10
MAX_VECTOR_RESULTS_FOR_SYNTHESIS = 10


class LLMSynthesizer:
    """
    Generates natural language answers from query results.

    Takes the fused results from both stores and produces a
    human-readable answer with source citations. Uses the synthesis
    prompt template for 4-layer prompt assembly.

    The synthesizer separates results by source store (SQL vs vector)
    and formats them as numbered lists for the LLM. The LLM then
    synthesizes a comprehensive answer using [SQL:n] and [VEC:n]
    citations to reference specific sources.

    If the LLM is unavailable or fails, returns None (the caller
    should display raw results instead).

    Usage:
        synthesizer = LLMSynthesizer(
            llm=llm_provider,
            prompt_assembler=assembler,
            profile=general_profile,
        )
        answer = synthesizer.synthesize(
            query="What did Alice propose?",
            results=fused_results,
            sql_briefing=briefing_text,
        )
        # answer → "Alice proposed migrating to OAuth2 [SQL:1]..."
    """

    def __init__(
        self,
        llm: LLMProvider,
        prompt_assembler: PromptAssembler,
        profile: DomainProfile,
    ) -> None:
        """
        Initialize the LLM synthesizer.

        Args:
            llm: The LLM provider for generating answers.
            prompt_assembler: The 4-layer prompt assembler.
            profile: Default domain profile for prompt assembly.
        """
        self._llm = llm
        self._prompt_assembler = prompt_assembler
        self._profile = profile

    def synthesize(
        self,
        query: str,
        results: list[SearchResult],
        sql_briefing: str | None = None,
        mode: RetrievalMode | None = None,
    ) -> str | None:
        """
        Generate a natural language answer from query results.

        Separates results by source store, formats them as numbered
        lists, and sends them to the LLM with the synthesis prompt.
        The LLM produces a coherent answer with citations.

        Args:
            query: The original user question.
            results: Fused results from both stores (sorted by
                     relevance). Each result has source_store="sql",
                     "vector", or "both".
            sql_briefing: Optional SQL briefing text from the
                          SQLBriefingBuilder. Provides additional
                          structured context for richer answers.
            mode: The retrieval mode used. When provided, adds a
                  mode-specific hint to the synthesis prompt so the
                  LLM structures its answer appropriately (e.g.,
                  V2S by entity/status, S2V by causality/context).

        Returns:
            A natural language answer string with [SQL:n] and [VEC:n]
            citations, or None if the LLM is unavailable or fails.
        """
        # ---------------------------------------------------------------
        # Check LLM availability before attempting synthesis.
        # ---------------------------------------------------------------
        if not self._llm.is_available():
            logger.info("llm_unavailable_no_synthesis")
            return None

        try:
            # ---------------------------------------------------------------
            # Step 1: Separate results by source store for citation tracking.
            # Results from "both" stores are included in the SQL stream
            # (they have structured fact data).
            # ---------------------------------------------------------------
            sql_results = [
                r for r in results
                if r.source_store in ("sql", "both")
            ][:MAX_SQL_RESULTS_FOR_SYNTHESIS]

            vector_results = [
                r for r in results
                if r.source_store in ("vector", "both")
            ][:MAX_VECTOR_RESULTS_FOR_SYNTHESIS]

            # ---------------------------------------------------------------
            # Step 2: Format results as numbered text for the LLM.
            # Each result becomes "[SQL:n] content" or "[VEC:n] content".
            # ---------------------------------------------------------------
            sql_text = self._format_results(sql_results, "SQL")
            vector_text = self._format_results(vector_results, "VEC")

            # ---------------------------------------------------------------
            # Step 3: Assemble the system prompt using the synthesis template.
            # The synthesis template has slots for {{sql_results}},
            # {{vector_results}}, {{query}}, and {{domain_description}}.
            # We fill these slots in the user prompt instead.
            # ---------------------------------------------------------------
            system_prompt = self._prompt_assembler.assemble(
                stage="synthesis",
                profile=self._profile,
            )

            # ---------------------------------------------------------------
            # Step 4: Build the user prompt with results and context.
            # ---------------------------------------------------------------
            user_prompt = self._build_user_prompt(
                query=query,
                sql_text=sql_text,
                vector_text=vector_text,
                sql_briefing=sql_briefing,
                mode=mode,
            )

            # ---------------------------------------------------------------
            # Step 5: Get stage parameters for synthesis generation.
            # Synthesis uses moderate temperature for natural language.
            # ---------------------------------------------------------------
            stage_params = self._prompt_assembler.get_stage_params(
                "synthesis", self._profile
            )

            # ---------------------------------------------------------------
            # Step 6: Generate the synthesized answer.
            # ---------------------------------------------------------------
            answer = self._llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=stage_params.temperature,
                max_tokens=stage_params.max_tokens,
                top_p=stage_params.top_p,
            )

            logger.info(
                "synthesis_completed",
                query=query,
                sql_result_count=len(sql_results),
                vector_result_count=len(vector_results),
                answer_length=len(answer),
                has_briefing=sql_briefing is not None,
            )

            return answer

        except Exception as exc:
            # ---------------------------------------------------------------
            # Synthesis failed. Log and return None. The caller should
            # display raw results instead of a synthesized answer.
            # ---------------------------------------------------------------
            logger.warning(
                "synthesis_failed",
                error_code="CTXMTG-QRY-001",
                error=str(exc),
                exc_info=True,
            )
            return None

    @staticmethod
    def _format_results(results: list[SearchResult], prefix: str) -> str:
        """
        Format a list of SearchResult objects as numbered text.

        Each result is formatted as:
            [PREFIX:1] content (score: 0.85)
            [PREFIX:2] content (score: 0.72)

        This format allows the LLM to reference specific results
        using [SQL:n] or [VEC:n] citations in its answer.

        Args:
            results: The results to format.
            prefix: The citation prefix ("SQL" or "VEC").

        Returns:
            A formatted string with numbered results, or "(none)"
            if the results list is empty.
        """
        if not results:
            return "(none)"

        lines: list[str] = []
        for i, result in enumerate(results, start=1):
            # Include score for LLM context about confidence
            score_str = f"{result.score:.2f}"
            # Truncate very long content to avoid context overflow
            content = result.content[:500] if len(result.content) > 500 else result.content
            lines.append(f"[{prefix}:{i}] {content} (score: {score_str})")

        return "\n".join(lines)

    # ---------------------------------------------------------------
    # Mode-specific synthesis hints.  These one-liners tell the LLM
    # how to structure its answer based on the retrieval mode used.
    # V2S discovered entities via vector, so the answer should be
    # organised by entity and status.  S2V started from structured
    # facts, so the answer should explain causality and context.
    # ---------------------------------------------------------------
    _MODE_HINTS: dict[RetrievalMode, str] = {
        RetrievalMode.VECTOR_TO_SQL: (
            "Structure the answer by entity and current status."
        ),
        RetrievalMode.SQL_TO_VECTOR: (
            "Explain causality and contextual reasons."
        ),
        RetrievalMode.BIDIRECTIONAL: (
            "Provide a comprehensive answer covering both entity "
            "status and contextual reasons."
        ),
    }

    @staticmethod
    def _build_user_prompt(
        query: str,
        sql_text: str,
        vector_text: str,
        sql_briefing: str | None = None,
        mode: RetrievalMode | None = None,
    ) -> str:
        """
        Build the user prompt for synthesis, including results and context.

        The user prompt contains:
        - The original question
        - Formatted SQL results
        - Formatted vector results
        - Optional SQL briefing for additional context
        - Optional mode-specific hint for answer structure

        Args:
            query: The original user question.
            sql_text: Formatted SQL results text.
            vector_text: Formatted vector results text.
            sql_briefing: Optional SQL briefing text.
            mode: Optional retrieval mode for mode-specific hints.

        Returns:
            The complete user prompt string.
        """
        parts: list[str] = [
            f"User question: {query}",
            "",
            "SQL results (structured facts):",
            sql_text,
            "",
            "Vector results (semantic chunks):",
            vector_text,
        ]

        # Append SQL briefing if available -- provides additional
        # structured context from the Pass 1 profiling queries.
        if sql_briefing:
            parts.extend([
                "",
                "Additional context from SQL briefing:",
                sql_briefing,
            ])

        parts.extend([
            "",
            "Synthesize a comprehensive answer with [SQL:n] and [VEC:n] citations.",
        ])

        # Append a mode-specific hint so the LLM knows how to
        # structure its answer based on the retrieval path used.
        if mode is not None:
            hint = LLMSynthesizer._MODE_HINTS.get(mode)
            if hint:
                parts.append(hint)

        return "\n".join(parts)
