# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
LLM-Powered Query Interpreter
===============================

This module implements the Phase 2 query interpreter that uses a local
LLM to understand domain-specific natural language queries. It replaces
the regex-based intent classification with LLM understanding, enabling
the system to parse complex domain-specific questions like:

    "What did Judge Morrison rule on the Smith motion?"

where the LLM understands that Judge Morrison is a person, Smith is a
case reference, and motion is a legal instrument -- something regex
patterns cannot do.

The interpreter sends the user's query to the LLM using the
query_planning prompt template. The LLM returns a JSON response with
entities, intent, time_range, filters, and a rewritten query optimized
for vector search.

Graceful degradation: if the LLM fails (unavailable, invalid JSON,
timeout), the interpreter falls back to the RuleBasedQueryInterpreter
from Phase 1. This ensures the system always returns a result, even
when the LLM is broken.

Depends on:
    - json (parsing LLM JSON responses)
    - structlog (structured logging)
    - ctxmtg.interfaces.llm (LLMProvider ABC)
    - ctxmtg.interfaces.query (QueryInterpreter ABC)
    - ctxmtg.llm.prompt_assembler (PromptAssembler for 4-layer prompts)
    - ctxmtg.models.profile (DomainProfile)
    - ctxmtg.models.query (QueryIntent enum)
    - ctxmtg.query.interpreter (RuleBasedQueryInterpreter for fallback)

Used by:
    - ctxmtg.query.executor (calls interpret() before planning)
    - tests/test_query/test_llm_interpreter.py
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.interfaces.query import QueryInterpreter
from ctxmtg.llm.prompt_assembler import PromptAssembler
from ctxmtg.models.profile import DomainProfile
from ctxmtg.models.query import QueryIntent

# ---------------------------------------------------------------
# Module logger -- structured output for LLM interpreter debugging.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.llm_interpreter")

# ---------------------------------------------------------------
# Valid intent strings that the LLM can return. Mapped to the
# QueryIntent enum. Any unrecognised string defaults to SEMANTIC.
# ---------------------------------------------------------------
_VALID_INTENTS: dict[str, QueryIntent] = {
    "factual": QueryIntent.FACTUAL,
    "semantic": QueryIntent.SEMANTIC,
    "aggregation": QueryIntent.AGGREGATION,
    "temporal": QueryIntent.TEMPORAL,
    "comparative": QueryIntent.COMPARATIVE,
}


class LLMQueryInterpreter(QueryInterpreter):
    """
    LLM-powered query interpretation for Phase 2+.

    Replaces regex-based intent classification with LLM understanding
    of domain-specific queries. This is a key differentiator: the LLM
    can parse "What did Judge Morrison rule on the Smith motion?" in
    legal contexts, identifying entities and intent that regex cannot.

    The interpreter assembles a query_planning prompt from the 4-layer
    prompt system, sends it to the LLM, and parses the JSON response
    into a standard interpretation dict.

    Falls back to the RuleBasedQueryInterpreter if:
    - The LLM is unavailable (is_available() returns False)
    - The LLM returns invalid JSON (parse error)
    - The LLM call raises an exception (timeout, OOM, etc.)

    Usage:
        interpreter = LLMQueryInterpreter(
            llm=llm_provider,
            prompt_assembler=assembler,
            profile=legal_profile,
        )
        result = await interpreter.interpret(
            query="What did Judge Morrison rule on the Smith motion?",
            profile=legal_profile,
        )
        # result["entities"] → ["Judge Morrison", "Smith"]
        # result["intent"] → QueryIntent.FACTUAL
    """

    def __init__(
        self,
        llm: LLMProvider,
        prompt_assembler: PromptAssembler,
        profile: DomainProfile,
        fallback: QueryInterpreter | None = None,
    ) -> None:
        """
        Initialize the LLM-powered query interpreter.

        Args:
            llm: The LLM provider for generating interpretations.
            prompt_assembler: The 4-layer prompt assembler.
            profile: Default domain profile for prompt assembly.
            fallback: Optional fallback interpreter. If None, LLM
                      failures return a default SEMANTIC interpretation.
        """
        self._llm = llm
        self._prompt_assembler = prompt_assembler
        self._default_profile = profile
        self._fallback = fallback

    async def interpret(self, query: str, profile: DomainProfile) -> dict[str, Any]:  # type: ignore[override]
        """
        LLM interprets the query. Falls back to regex if LLM fails.

        Sends the query to the LLM using the query_planning prompt
        template. The LLM returns a JSON object with entities, intent,
        time_range, filters, and a rewritten query.

        If the LLM is unavailable or returns invalid data, falls back
        to the RuleBasedQueryInterpreter (Phase 1 behavior).

        Args:
            query: The user's natural language question.
            profile: The active domain profile.

        Returns:
            A dict with keys: entities, intent, time_range, filters,
            rewritten_query. Same structure as RuleBasedQueryInterpreter.
        """
        # ---------------------------------------------------------------
        # Check LLM availability before attempting a call. If unavailable,
        # fall back immediately rather than wasting time on a failed call.
        # ---------------------------------------------------------------
        if not self._llm.is_available():
            logger.info(
                "llm_unavailable_falling_back",
                query=query,
            )
            return await self._do_fallback(query, profile)

        try:
            # ---------------------------------------------------------------
            # Step 1: Assemble the system prompt using the query_planning
            # template + domain profile (4-layer composition).
            # ---------------------------------------------------------------
            system_prompt = self._prompt_assembler.assemble(
                stage="query_planning",
                profile=profile,
            )

            # ---------------------------------------------------------------
            # Step 2: Build the user prompt with the actual query.
            # ---------------------------------------------------------------
            user_prompt = f"Interpret this query:\n\n{query}"

            # ---------------------------------------------------------------
            # Step 3: Get stage parameters for temperature, max_tokens, etc.
            # Query planning uses low temperature for precise output.
            # ---------------------------------------------------------------
            stage_params = self._prompt_assembler.get_stage_params(
                "query_planning", profile
            )

            # ---------------------------------------------------------------
            # Step 4: Call the LLM with json_mode=True for structured output.
            # ---------------------------------------------------------------
            response = self._llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=stage_params.temperature,
                max_tokens=stage_params.max_tokens,
                top_p=stage_params.top_p,
                json_mode=True,
            )

            # ---------------------------------------------------------------
            # Step 5: Parse the JSON response and validate required fields.
            # ---------------------------------------------------------------
            interpretation = self._parse_response(response, query)

            logger.info(
                "llm_query_interpreted",
                query=query,
                intent=interpretation["intent"].value,
                entity_count=len(interpretation["entities"]),
            )

            return interpretation

        except Exception as exc:
            # ---------------------------------------------------------------
            # LLM call failed. Log the error and fall back to Phase 1.
            # This handles: invalid JSON, LLM timeout, OOM, network errors.
            # ---------------------------------------------------------------
            logger.warning(
                "llm_interpretation_failed",
                error_code="CTXMTG-QRY-001",
                error=str(exc),
                exc_info=True,
            )
            return await self._do_fallback(query, profile)

    def _parse_response(self, response: str, original_query: str) -> dict[str, Any]:
        """
        Parse the LLM's JSON response into a standard interpretation dict.

        Validates the response structure and normalizes values to match
        the expected format from the QueryInterpreter interface.

        Args:
            response: The raw JSON string from the LLM.
            original_query: The original user query (for fallback values).

        Returns:
            A validated interpretation dict.

        Raises:
            ValueError: If the response is not valid JSON or is missing
                        required fields.
        """
        # ---------------------------------------------------------------
        # Parse JSON. The LLM should return a valid JSON object.
        # If not, we raise and the caller falls back to Phase 1.
        # ---------------------------------------------------------------
        data = json.loads(response)

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        # ---------------------------------------------------------------
        # Extract and validate each field. Use sensible defaults when
        # the LLM omits optional fields.
        # ---------------------------------------------------------------

        # entities: list of entity name strings
        entities = data.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        entities = [str(e) for e in entities if e]

        # intent: string → QueryIntent enum
        intent_str = str(data.get("intent", "semantic")).lower()
        intent = _VALID_INTENTS.get(intent_str, QueryIntent.SEMANTIC)

        # time_range: null or [start, end] → tuple or None
        time_range_raw = data.get("time_range")
        time_range = None
        if isinstance(time_range_raw, (list, tuple)) and len(time_range_raw) == 2:
            time_range = (str(time_range_raw[0]), str(time_range_raw[1]))

        # filters: optional dict of filter key-value pairs
        filters = data.get("filters", {})
        if not isinstance(filters, dict):
            filters = {}

        # rewritten_query: cleaned query for vector search
        rewritten_query = str(data.get("rewritten_query", original_query))

        return {
            "entities": entities,
            "intent": intent,
            "time_range": time_range,
            "filters": filters,
            "rewritten_query": rewritten_query,
        }

    async def _do_fallback(self, query: str, profile: DomainProfile) -> dict[str, Any]:
        """
        Fall back to the configured fallback interpreter or return defaults.

        If a fallback interpreter was provided (typically a
        RuleBasedQueryInterpreter), delegate to it. Otherwise, return
        a minimal SEMANTIC interpretation that allows the query to
        proceed with vector search.

        Args:
            query: The user's natural language question.
            profile: The active domain profile.

        Returns:
            An interpretation dict from the fallback interpreter, or
            a minimal default interpretation.
        """
        if self._fallback is not None:
            logger.info("using_fallback_interpreter", query=query)
            # The fallback may be async (RuleBasedQueryInterpreter) or sync.
            # We handle both cases.
            result = self._fallback.interpret(query, profile)
            # If the fallback returns a coroutine (async), await it
            if hasattr(result, "__await__"):
                result = await result
            return result

        # No fallback configured -- return a minimal interpretation
        # that defaults to semantic search on the original query.
        logger.info("no_fallback_using_defaults", query=query)
        return {
            "entities": [],
            "intent": QueryIntent.SEMANTIC,
            "time_range": None,
            "filters": {},
            "rewritten_query": query,
        }
