# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Informed Retrieval Modes (V→SQL, S→V)
=======================================

This module implements two informed retrieval modes where one store's
results are used by the LLM to formulate a targeted query for the
other store. This produces richer results than the Parallel mode
(Phase 1) because the second query is informed by what the first
store found.

Two classes:
    VectorToSQLRetriever (Mode 2):
        1. Vector search → LLM reads results → formulates SQL → SQL results
        2. LLM synthesizes both result sets into a QueryResult.
        Best for exploratory questions ("what changed?", "tell me about...")

    SQLToVectorRetriever (Mode 3):
        1. SQL briefing → LLM reads it → formulates targeted vector query
        2. Vector returns semantic chunks
        3. LLM synthesizes both result sets into a QueryResult.
        Best for depth queries ("why did we...?", "what led to...?")

Both retrievers gracefully degrade: if the LLM is unavailable, they
fall back to Parallel mode (Phase 1 behavior) by returning a simple
parallel query result.

Depends on:
    - json (parsing LLM JSON responses)
    - structlog (structured logging)
    - time (latency measurement)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider)
    - ctxmtg.llm.prompt_assembler (PromptAssembler)
    - ctxmtg.models.profile (DomainProfile)
    - ctxmtg.models.query (QueryResult, SearchResult, RetrievalMode)
    - ctxmtg.query.briefing (SQLBriefingBuilder)
    - ctxmtg.query.synthesizer (LLMSynthesizer)

Used by:
    - ctxmtg.query.executor (dispatches to these for mode selection)
    - ctxmtg.query.bidirectional (runs both in parallel)
    - tests/test_query/test_informed_retrieval.py
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.llm.prompt_assembler import PromptAssembler
from ctxmtg.models.profile import DomainProfile
from ctxmtg.models.query import QueryResult, RetrievalMode, SearchResult
from ctxmtg.query.briefing import SQLBriefingBuilder
from ctxmtg.query.synthesizer import LLMSynthesizer

# ---------------------------------------------------------------
# Module logger -- logs retrieval steps and timings.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.informed_retrieval")

# ---------------------------------------------------------------
# Default top-k for intermediate searches (before synthesis).
# More results give the LLM richer context for the bridge query.
# ---------------------------------------------------------------
DEFAULT_INTERMEDIATE_TOP_K = 10

# ---------------------------------------------------------------
# Maximum number of results to include in LLM bridge prompts.
# Limits context window usage while preserving key information.
# ---------------------------------------------------------------
MAX_RESULTS_FOR_BRIDGE = 8


# =====================================================================
# VectorToSQLRetriever -- Mode 2: Vector→SQL Informed Retrieval
# =====================================================================


class VectorToSQLRetriever:
    """
    Mode 2: Vector→SQL informed retrieval.

    Pipeline:
        1. Run vector search (semantic discovery)
        2. LLM reads vector results, discovers entities/terms/time ranges
        3. LLM formulates precise SQL using discovered context
        4. SQL returns structured facts
        5. LLM synthesizes both result sets into QueryResult

    Best for exploratory questions where the user doesn't know the
    exact entities or timeframes -- vector search discovers them, then
    SQL fills in the structured details.

    If the LLM is unavailable, returns None to signal the caller should
    fall back to Parallel mode.

    Usage:
        retriever = VectorToSQLRetriever(
            sql_store=sqlite_store,
            vector_store=lancedb_store,
            llm=llm_provider,
            prompt_assembler=assembler,
            profile=general_profile,
            briefing_builder=SQLBriefingBuilder(),
        )
        result = await retriever.retrieve(
            query="What changed in the project?",
            interpretation={"entities": [], "intent": "semantic", ...},
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
        Initialize the V→SQL retriever.

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
        self._sql_store = sql_store
        self._vector_store = vector_store
        self._llm = llm
        self._prompt_assembler = prompt_assembler
        self._profile = profile
        self._briefing_builder = briefing_builder
        self._embedding_fn = embedding_fn

        # Reuse the synthesizer for final answer generation.
        self._synthesizer = LLMSynthesizer(
            llm=llm,
            prompt_assembler=prompt_assembler,
            profile=profile,
        )

    async def retrieve(
        self,
        query: str,
        interpretation: dict[str, Any],
        top_k: int = DEFAULT_INTERMEDIATE_TOP_K,
    ) -> QueryResult | None:
        """
        Run the V→SQL informed retrieval pipeline.

        Steps:
            1. Vector search for semantic discovery
            2. LLM reads vector results, formulates SQL via bridge prompt
            3. Execute the LLM-designed SQL query
            4. LLM synthesizes both result sets

        Args:
            query: The user's natural language question.
            interpretation: Structured interpretation from the interpreter.
            top_k: Number of results for intermediate searches.

        Returns:
            A QueryResult with results from both stores and optional
            LLM synthesis, or None if the LLM is unavailable.
        """
        start_time = time.monotonic()

        # ---------------------------------------------------------------
        # Graceful degradation: if LLM unavailable, signal fallback.
        # ---------------------------------------------------------------
        if not self._llm.is_available():
            logger.info("v2s_llm_unavailable_fallback", query=query)
            return None

        try:
            # ---------------------------------------------------------------
            # Step 1: Run vector search (semantic discovery).
            # Use the rewritten query from interpretation for better matches.
            # ---------------------------------------------------------------
            vector_query = interpretation.get("rewritten_query", query)
            vector_results = await self._run_vector_search(
                vector_query, top_k
            )

            logger.info(
                "v2s_vector_search_done",
                query=query,
                vector_count=len(vector_results),
            )

            # ---------------------------------------------------------------
            # Step 2: LLM reads vector results, formulates SQL query.
            # The bridge prompt tells the LLM what the vector store found
            # and asks it to formulate a targeted SQL query.
            # ---------------------------------------------------------------
            bridge_response = self._formulate_sql_from_vectors(
                query, vector_results
            )

            # ---------------------------------------------------------------
            # Step 3: Execute the LLM-designed SQL query with guardrails.
            # ---------------------------------------------------------------
            sql_results = await self._execute_bridge_sql(bridge_response)

            logger.info(
                "v2s_sql_bridge_done",
                query=query,
                sql_count=len(sql_results),
            )

            # ---------------------------------------------------------------
            # Step 4: LLM synthesizes both result sets.
            # ---------------------------------------------------------------
            all_results = vector_results + sql_results
            synthesis = self._synthesizer.synthesize(
                query=query,
                results=all_results,
                mode=RetrievalMode.VECTOR_TO_SQL,
            )

            latency_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "v2s_retrieval_complete",
                query=query,
                vector_count=len(vector_results),
                sql_count=len(sql_results),
                has_synthesis=synthesis is not None,
                latency_ms=round(latency_ms, 2),
            )

            return QueryResult(
                query=query,
                mode=RetrievalMode.VECTOR_TO_SQL,
                results=all_results[:top_k],
                total_results=len(all_results),
                sql_results_count=len(sql_results),
                vector_results_count=len(vector_results),
                synthesis=synthesis,
                latency_ms=round(latency_ms, 2),
            )

        except Exception as exc:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "v2s_retrieval_failed",
                error_code="CTXMTG-QRY-001",
                error=str(exc),
                latency_ms=round(latency_ms, 2),
            )
            # Return None to signal caller to fall back to Parallel.
            return None

    async def _run_vector_search(
        self, query_text: str, top_k: int
    ) -> list[SearchResult]:
        """
        Execute a vector search against the vector store.

        Converts the query text to an embedding vector and searches
        for similar content. Returns empty list if the embedding
        function is unavailable.

        Args:
            query_text: The text query to embed and search.
            top_k: Number of results to retrieve.

        Returns:
            List of SearchResult objects from vector search.
        """
        if self._embedding_fn is None:
            return []

        try:
            # Get the embedding vector for the query text.
            import asyncio

            result = self._embedding_fn(query_text)
            if asyncio.iscoroutine(result):
                result = await result

            if result is None:
                return []

            query_vector = list(result)

            # Search the vector store.
            results = await self._vector_store.search(
                query_vector=query_vector,
                top_k=top_k,
            )
            return results

        except Exception as exc:
            logger.warning(
                "v2s_vector_search_failed",
                error_code="CTXMTG-QRY-001",
                error=str(exc),
            )
            return []

    def _formulate_sql_from_vectors(
        self,
        query: str,
        vector_results: list[SearchResult],
    ) -> dict[str, Any]:
        """
        LLM reads vector results and formulates a targeted SQL query.

        Sends the vector results to the LLM using the retrieval bridge
        prompt. The LLM discovers entities, terms, and time ranges from
        the vector results and formulates a precise SQL query.

        Args:
            query: The original user question.
            vector_results: Results from the vector search.

        Returns:
            A dict with keys: query (SQL string), filters (dict),
            reasoning (str). Returns default values on LLM failure.
        """
        # Format vector results as text for the LLM.
        results_text = self._format_results_for_bridge(
            vector_results, "vector"
        )

        # Assemble the system prompt using the retrieval template.
        system_prompt = self._prompt_assembler.assemble(
            stage="retrieval",
            profile=self._profile,
        )

        # Build the user prompt for the bridge query.
        user_prompt = (
            f"User question: {query}\n\n"
            f"Mode: vector_to_sql\n"
            f"Vector search results (semantic discovery):\n{results_text}\n\n"
            f"Based on these vector results, formulate a targeted SQL query "
            f"to find structured facts that complement the semantic findings.\n"
            f"Focus on entities, predicates, and time ranges discovered "
            f"in the vector results.\n\n"
            f"Available tables: facts, entities, interactions\n"
            f"Output JSON: {{\"query\": \"SELECT ...\", "
            f"\"filters\": {{}}, \"reasoning\": \"...\"}}"
        )

        # Get stage parameters for the retrieval bridge.
        stage_params = self._prompt_assembler.get_stage_params(
            "retrieval", self._profile
        )

        try:
            response = self._llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=stage_params.temperature,
                max_tokens=stage_params.max_tokens,
                top_p=stage_params.top_p,
                json_mode=True,
            )

            logger.info(
                "v2s_bridge_raw_response",
                response_length=len(response),
                response_preview=response[:300],
            )

            # Strip markdown code fences if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                # Remove opening fence (```json or ```)
                first_newline = cleaned.index("\n")
                cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)
            if isinstance(data, dict) and "query" in data:
                logger.info(
                    "v2s_bridge_sql_formulated",
                    sql_preview=str(data["query"])[:200],
                    reasoning=str(data.get("reasoning", ""))[:200],
                )
                return data

        except (json.JSONDecodeError, Exception) as exc:
            logger.warning(
                "v2s_bridge_formulation_failed",
                error_code="CTXMTG-QRY-001",
                error=str(exc),
                # Log raw response for debugging
                raw_response=response[:300] if 'response' in dir() else "no response",
            )

        # Default: return a generic query.
        return {
            "query": "SELECT e.name, f.predicate, f.object_literal "
                     "FROM entities e JOIN facts f "
                     "ON f.subject_entity_id = e.id LIMIT 10",
            "filters": {},
            "reasoning": "Fallback: LLM bridge formulation failed.",
        }

    async def _execute_bridge_sql(
        self, bridge_response: dict[str, Any]
    ) -> list[SearchResult]:
        """
        Execute the LLM-designed SQL query with safety guardrails.

        Uses the SQLBriefingBuilder's execute_llm_queries method
        which validates the SQL against safety rules (allowlisted
        tables, no DDL, mandatory LIMIT).

        Args:
            bridge_response: The LLM bridge response dict with "query" key.

        Returns:
            List of SearchResult objects from the SQL query.
        """
        sql_query = str(bridge_response.get("query", ""))
        if not sql_query:
            return []

        queries = [{"purpose": "v2s_bridge", "sql": sql_query}]
        results = await self._briefing_builder.execute_llm_queries(
            sql_store=self._sql_store,
            queries=queries,
            row_limit=20,
        )

        # Convert rows to SearchResult objects.
        search_results: list[SearchResult] = []
        for result_entry in results:
            if result_entry.get("error"):
                logger.warning(
                    "v2s_bridge_sql_error",
                    error_code="CTXMTG-QRY-001",
                    error=result_entry["error"],
                )
                continue

            for rank, row in enumerate(result_entry.get("rows", [])):
                content = self._row_to_content(row)
                result_id = str(row.get("id", f"v2s-sql-{rank}"))
                search_results.append(
                    SearchResult(
                        id=result_id,
                        source_store="sql",
                        content=content,
                        score=1.0 / (rank + 1),
                        metadata={
                            k: str(v)
                            for k, v in row.items()
                            if k != "content" and v is not None
                        },
                    )
                )

        return search_results

    @staticmethod
    def _format_results_for_bridge(
        results: list[SearchResult], store_name: str
    ) -> str:
        """
        Format results as numbered text for the LLM bridge prompt.

        Each result is formatted as a numbered entry with content
        preview and metadata. Limited to MAX_RESULTS_FOR_BRIDGE
        to stay within context budget.

        Args:
            results: The search results to format.
            store_name: Label for the results ("vector" or "sql").

        Returns:
            Formatted string with numbered results, or "(none)".
        """
        if not results:
            return "(none)"

        lines: list[str] = []
        for i, result in enumerate(results[:MAX_RESULTS_FOR_BRIDGE], start=1):
            content = (
                result.content[:400]
                if len(result.content) > 400
                else result.content
            )
            lines.append(
                f"[{store_name}:{i}] (score: {result.score:.2f}) {content}"
            )
        return "\n".join(lines)

    @staticmethod
    def _row_to_content(row: dict[str, Any]) -> str:
        """
        Build a content string from a SQL result row.

        Adapts to whatever columns are available in the row.

        Args:
            row: A dict of column_name → value from a SQL query.

        Returns:
            A formatted content string.
        """
        if row.get("content"):
            return str(row["content"])

        parts: list[str] = []
        if "name" in row:
            parts.append(str(row["name"]))
        if "entity_name" in row:
            parts.append(str(row["entity_name"]))
        if "predicate" in row:
            parts.append(str(row["predicate"]))
        if row.get("object_literal"):
            parts.append(str(row["object_literal"]))
        if row.get("title"):
            parts.append(str(row["title"]))

        return " | ".join(parts) if parts else str(row)


# =====================================================================
# SQLToVectorRetriever -- Mode 3: SQL→Vector Informed Retrieval
# =====================================================================


class SQLToVectorRetriever:
    """
    Mode 3: SQL→Vector informed retrieval.

    Pipeline:
        1. Run SQL briefing (Pass 1 statistical profile)
        2. LLM reads briefing, identifies gaps and important terms
        3. LLM formulates a targeted vector search query
        4. Vector returns semantic chunks
        5. LLM synthesizes both result sets into QueryResult

    Best for depth and context queries where structured facts come
    first, and vector search fills in the why, context, and nuance.

    If the LLM is unavailable, returns None to signal the caller
    should fall back to Parallel mode.

    Usage:
        retriever = SQLToVectorRetriever(
            sql_store=sqlite_store,
            vector_store=lancedb_store,
            llm=llm_provider,
            prompt_assembler=assembler,
            profile=general_profile,
            briefing_builder=SQLBriefingBuilder(),
        )
        result = await retriever.retrieve(
            query="Why did Alice propose OAuth2?",
            interpretation={"entities": ["Alice"], "intent": "factual", ...},
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
        Initialize the S→V retriever.

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
        self._sql_store = sql_store
        self._vector_store = vector_store
        self._llm = llm
        self._prompt_assembler = prompt_assembler
        self._profile = profile
        self._briefing_builder = briefing_builder
        self._embedding_fn = embedding_fn

        # Reuse the synthesizer for final answer generation.
        self._synthesizer = LLMSynthesizer(
            llm=llm,
            prompt_assembler=prompt_assembler,
            profile=profile,
        )

    async def retrieve(
        self,
        query: str,
        interpretation: dict[str, Any],
        top_k: int = DEFAULT_INTERMEDIATE_TOP_K,
    ) -> QueryResult | None:
        """
        Run the S→V informed retrieval pipeline.

        Steps:
            1. Run SQL briefing (Pass 1 statistical profile)
            2. LLM reads briefing, formulates targeted vector query
            3. Execute the targeted vector search
            4. LLM synthesizes both result sets

        Args:
            query: The user's natural language question.
            interpretation: Structured interpretation from the interpreter.
            top_k: Number of results for intermediate searches.

        Returns:
            A QueryResult with results from both stores and optional
            LLM synthesis, or None if the LLM is unavailable.
        """
        start_time = time.monotonic()

        # ---------------------------------------------------------------
        # Graceful degradation: if LLM unavailable, signal fallback.
        # ---------------------------------------------------------------
        if not self._llm.is_available():
            logger.info("s2v_llm_unavailable_fallback", query=query)
            return None

        try:
            # ---------------------------------------------------------------
            # Step 1: Run SQL briefing (Pass 1 statistical profile).
            # Extracts entity names from the interpretation for focusing.
            # ---------------------------------------------------------------
            query_terms = interpretation.get("entities", [])
            sql_briefing = await self._briefing_builder.build_briefing(
                sql_store=self._sql_store,
                query_terms=query_terms,
            )

            logger.info(
                "s2v_sql_briefing_done",
                query=query,
                briefing_length=len(sql_briefing),
            )

            # ---------------------------------------------------------------
            # Step 2: LLM reads briefing, formulates targeted vector query.
            # ---------------------------------------------------------------
            bridge_response = self._formulate_vector_from_sql(
                query, sql_briefing
            )

            # ---------------------------------------------------------------
            # Step 3: Execute the targeted vector search.
            # ---------------------------------------------------------------
            targeted_query = str(bridge_response.get("query", query))
            vector_results = await self._run_vector_search(
                targeted_query, top_k
            )

            logger.info(
                "s2v_vector_bridge_done",
                query=query,
                targeted_query=targeted_query,
                vector_count=len(vector_results),
            )

            # ---------------------------------------------------------------
            # Build SQL results from briefing data for the QueryResult.
            # The briefing is text, so we wrap it as a synthetic result.
            # ---------------------------------------------------------------
            sql_results = [
                SearchResult(
                    id="s2v-briefing-0",
                    source_store="sql",
                    content=sql_briefing,
                    score=1.0,
                    metadata={"type": "briefing"},
                )
            ]

            # ---------------------------------------------------------------
            # Step 4: LLM synthesizes both result sets.
            # ---------------------------------------------------------------
            all_results = sql_results + vector_results
            synthesis = self._synthesizer.synthesize(
                query=query,
                results=all_results,
                sql_briefing=sql_briefing,
                mode=RetrievalMode.SQL_TO_VECTOR,
            )

            latency_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "s2v_retrieval_complete",
                query=query,
                vector_count=len(vector_results),
                has_synthesis=synthesis is not None,
                latency_ms=round(latency_ms, 2),
            )

            return QueryResult(
                query=query,
                mode=RetrievalMode.SQL_TO_VECTOR,
                results=all_results[:top_k],
                total_results=len(all_results),
                sql_results_count=len(sql_results),
                vector_results_count=len(vector_results),
                synthesis=synthesis,
                latency_ms=round(latency_ms, 2),
            )

        except Exception as exc:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "s2v_retrieval_failed",
                error_code="CTXMTG-QRY-001",
                error=str(exc),
                latency_ms=round(latency_ms, 2),
            )
            return None

    async def _run_vector_search(
        self, query_text: str, top_k: int
    ) -> list[SearchResult]:
        """
        Execute a vector search against the vector store.

        Converts the query text to an embedding vector and searches
        for similar content. Returns empty list if the embedding
        function is unavailable.

        Args:
            query_text: The text query to embed and search.
            top_k: Number of results to retrieve.

        Returns:
            List of SearchResult objects from vector search.
        """
        if self._embedding_fn is None:
            return []

        try:
            import asyncio

            result = self._embedding_fn(query_text)
            if asyncio.iscoroutine(result):
                result = await result

            if result is None:
                return []

            query_vector = list(result)

            results = await self._vector_store.search(
                query_vector=query_vector,
                top_k=top_k,
            )
            return results

        except Exception as exc:
            logger.warning(
                "s2v_vector_search_failed",
                error_code="CTXMTG-QRY-001",
                error=str(exc),
            )
            return []

    def _formulate_vector_from_sql(
        self,
        query: str,
        sql_briefing: str,
    ) -> dict[str, Any]:
        """
        LLM reads SQL briefing and formulates a targeted vector query.

        Sends the SQL briefing to the LLM using the retrieval bridge
        prompt. The LLM identifies what contextual gaps exist and
        formulates a targeted vector search query to fill them.

        Args:
            query: The original user question.
            sql_briefing: The SQL briefing text from Pass 1.

        Returns:
            A dict with keys: query (str), filters (dict),
            reasoning (str). Returns defaults on LLM failure.
        """
        # Assemble the system prompt using the retrieval template.
        system_prompt = self._prompt_assembler.assemble(
            stage="retrieval",
            profile=self._profile,
        )

        # Build the user prompt for the bridge query.
        user_prompt = (
            f"User question: {query}\n\n"
            f"Mode: sql_to_vector\n"
            f"SQL briefing (structured facts and statistics):\n"
            f"{sql_briefing}\n\n"
            f"Based on this SQL briefing, formulate a targeted vector "
            f"search query to find semantic content that fills gaps in "
            f"the structured data. Focus on the 'why', context, and "
            f"nuance that SQL facts don't capture.\n\n"
            f"Output JSON: {{\"query\": \"...\", "
            f"\"filters\": {{}}, \"reasoning\": \"...\"}}"
        )

        # Get stage parameters for the retrieval bridge.
        stage_params = self._prompt_assembler.get_stage_params(
            "retrieval", self._profile
        )

        try:
            response = self._llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=stage_params.temperature,
                max_tokens=stage_params.max_tokens,
                top_p=stage_params.top_p,
                json_mode=True,
            )

            logger.info(
                "s2v_bridge_raw_response",
                response_length=len(response),
                response_preview=response[:300],
            )

            # Strip markdown code fences if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                first_newline = cleaned.index("\n")
                cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)
            if isinstance(data, dict) and "query" in data:
                logger.info(
                    "s2v_bridge_query_formulated",
                    query_preview=str(data["query"])[:200],
                    reasoning=str(data.get("reasoning", ""))[:200],
                )
                return data

        except (json.JSONDecodeError, Exception) as exc:
            logger.warning(
                "s2v_bridge_formulation_failed",
                error_code="CTXMTG-QRY-001",
                error=str(exc),
                raw_response=response[:300] if 'response' in dir() else "no response",
            )

        # Default: use the original query for vector search.
        return {
            "query": query,
            "filters": {},
            "reasoning": "Fallback: LLM bridge formulation failed.",
        }
