# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Query Executor
==============

This module orchestrates the full query pipeline: interpretation,
planning, parallel execution against both stores, fusion, and reranking.
It is the main entry point for the query subsystem.

The executor coordinates the following steps:
    1. Interpret the user's query (via QueryInterpreter)
    2. Plan the execution (via QueryPlanner)
    3. Execute SQL and vector queries in parallel (asyncio.gather)
    4. Convert raw results to SearchResult objects
    5. Hydrate top-N vector results with full content from SQL
    6. Fuse results from both stores (via ResultFuser)
    7. Rerank fused results (via Reranker)
    8. Build the final QueryResult

Graceful degradation: if one store fails (SQL error, vector index
not built), the executor logs a warning and returns results from
the healthy store only. This ensures the system is always useful
even when partially broken.

Depends on:
    - asyncio (parallel execution via gather)
    - time (latency measurement)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- optional, for informed retrieval)
    - ctxmtg.interfaces.query (QueryInterpreter, QueryPlanner, ResultFuser, Reranker)
    - ctxmtg.models.query (QueryPlan, SearchResult, QueryResult, RetrievalMode)
    - ctxmtg.models.profile (DomainProfile)
    - ctxmtg.exceptions (QueryError)
    - ctxmtg.query.informed_retrieval (VectorToSQLRetriever, SQLToVectorRetriever)
    - ctxmtg.query.bidirectional (BidirectionalRetriever)
    - ctxmtg.query.briefing (SQLBriefingBuilder)
    - ctxmtg.llm.prompt_assembler (PromptAssembler)

Used by:
    - ctxmtg.query.server (handles query requests)
    - ctxmtg.cli (CLI query command)
    - tests/test_query/test_executor.py
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from ctxmtg.exceptions import QueryError
from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.interfaces.query import QueryInterpreter, QueryPlanner, Reranker, ResultFuser
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.llm.prompt_assembler import PromptAssembler
from ctxmtg.models.profile import DomainProfile
from ctxmtg.models.query import QueryPlan, QueryResult, RetrievalMode, SearchResult
from ctxmtg.query.bidirectional import BidirectionalRetriever
from ctxmtg.query.briefing import SQLBriefingBuilder
from ctxmtg.query.informed_retrieval import SQLToVectorRetriever, VectorToSQLRetriever
from ctxmtg.query.quality_logger import QueryQualityLogger

# ---------------------------------------------------------------
# Module logger -- logs query execution steps and timings.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.executor")

# ---------------------------------------------------------------
# Constants: default values for query execution.
# ---------------------------------------------------------------

# Number of top vector results to hydrate with full content from SQL.
# Hydration fetches the full interaction text (not just the 200-char
# preview) for the most relevant results. Limited to 5 for Tier 0-1
# to keep response size manageable; configurable for higher tiers
# where an LLM will read and synthesize the full content.
DEFAULT_HYDRATION_COUNT = 5

# Default number of results to return after fusion + reranking.
DEFAULT_TOP_K = 10


class QueryExecutor:
    """
    Orchestrates the full query pipeline from question to ranked results.

    This is the main entry point for the query subsystem. It coordinates
    all the components: interpreter, planner, stores, fuser, and reranker.

    The executor runs SQL and vector queries in parallel using
    asyncio.gather, which significantly reduces latency compared to
    sequential execution (typical improvement: 40-60%).

    Graceful degradation ensures the system works even when one store
    is unavailable:
    - If the vector store fails → returns SQL results only
    - If the SQL store fails → returns vector results only
    - If both fail → raises QueryError

    Usage:
        executor = QueryExecutor(
            sql_store=sqlite_store,
            vector_store=lancedb_store,
            interpreter=RuleBasedQueryInterpreter(sqlite_store),
            planner=TemplateQueryPlanner(),
            fuser=RRFFuser(),
            reranker=TFIDFReranker(),
        )
        result = await executor.execute(
            query="What did Alice propose?",
            profile=general_profile,
        )
        # result.results → [SearchResult(...), ...]
    """

    def __init__(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        interpreter: QueryInterpreter,
        planner: QueryPlanner,
        fuser: ResultFuser,
        reranker: Reranker,
        embedding_fn: Any | None = None,
        llm: LLMProvider | None = None,
        prompt_assembler: PromptAssembler | None = None,
        profile: DomainProfile | None = None,
        quality_logger: QueryQualityLogger | None = None,
    ) -> None:
        """
        Initialize the executor with all pipeline components.

        All components are injected via constructor for testability.
        In production, these are wired up by the application factory.

        Args:
            sql_store: The SQL store for structured queries.
            vector_store: The vector store for semantic search.
            interpreter: Interprets natural language queries.
            planner: Generates SQL/vector query plans.
            fuser: Combines results from both stores (RRF).
            reranker: Re-ranks fused results for quality.
            embedding_fn: Optional callable that converts text to a
                          vector. Required for vector search. If None,
                          vector search is skipped.
            llm: Optional LLM provider for informed retrieval modes.
                 Required for VECTOR_TO_SQL, SQL_TO_VECTOR, and
                 BIDIRECTIONAL modes. If None, those modes fall back
                 to PARALLEL.
            prompt_assembler: Optional prompt assembler for LLM modes.
                              Required when llm is provided.
            profile: Optional default domain profile for LLM-powered
                     retrieval modes. If None, uses the profile passed
                     to execute().
            quality_logger: Optional logger that records query metrics
                            to query_quality_log for the self-learning
                            feedback loop. If None, logging is skipped.
        """
        self._sql_store = sql_store
        self._vector_store = vector_store
        self._interpreter = interpreter
        self._planner = planner
        self._fuser = fuser
        self._reranker = reranker
        self._embedding_fn = embedding_fn
        self._llm = llm
        self._prompt_assembler = prompt_assembler
        self._default_profile = profile
        self._quality_logger = quality_logger

        # Warn at init time if the vector store is provided but
        # no embedding function is given. This means the executor
        # will silently skip vector search on every query.
        if vector_store is not None and embedding_fn is None:
            logger.warning(
                "executor_no_embedding_fn",
                error_code="CTXMTG-QRY-001",
                detail="Vector store provided but no embedding_fn. "
                       "Vector search will be skipped for all queries.",
            )
        # Track last query ID + time for refinement detection.
        # When a user re-phrases within 60s, the previous query
        # is flagged as a refinement (implicit negative feedback).
        self._last_query_id: str | None = None
        self._last_query_time: float = 0.0

    async def execute(
        self,
        query: str,
        profile: DomainProfile,
        mode: RetrievalMode = RetrievalMode.PARALLEL,
        top_k: int = DEFAULT_TOP_K,
    ) -> QueryResult:
        """
        Execute a full query pipeline and return ranked results.

        Steps:
        1. Interpret the query (extract entities, intent, time range)
        2. Plan the execution (generate SQL + vector queries)
        3. Execute SQL and vector queries in parallel
        4. Hydrate top vector results with full content from SQL
        5. Fuse results from both stores using RRF
        6. Rerank fused results for quality
        7. Return the final QueryResult

        Args:
            query: The user's natural language question.
            profile: The active domain profile.
            mode: The retrieval mode (PARALLEL for Phase 1).
            top_k: Number of top results to return.

        Returns:
            A QueryResult containing fused, reranked results from
            both stores, along with execution metadata.

        Raises:
            QueryError: If both stores fail or the query is invalid.
        """
        start_time = time.monotonic()

        try:
            # Step 1: Interpret the query.
            # The ABC's interpret() is sync, but our RuleBasedQueryInterpreter
            # overrides it as async (needs SQL store access). We await here.
            interpretation = await self._interpreter.interpret(query, profile)  # type: ignore[misc]

            # ---------------------------------------------------------------
            # Mode dispatch: for informed retrieval modes, delegate to the
            # appropriate retriever. If the retriever returns None (LLM
            # unavailable), fall back to PARALLEL mode.
            # ---------------------------------------------------------------
            # Track the originally requested mode so we can record
            # a fallback_reason if we end up using PARALLEL instead.
            requested_mode = mode
            fallback_reason: str | None = None

            if mode != RetrievalMode.PARALLEL:
                informed_result = await self._dispatch_informed(
                    mode=mode,
                    query=query,
                    interpretation=interpretation,
                    profile=profile,
                    top_k=top_k,
                )
                if informed_result is not None:
                    await self._log_query_quality(informed_result)
                    return informed_result

                # Fall back to PARALLEL mode if informed retrieval
                # returned None (LLM unavailable or error).
                fallback_reason = f"{requested_mode.value}_unavailable"
                logger.info(
                    "informed_retrieval_fallback_to_parallel",
                    original_mode=mode.value,
                    fallback_reason=fallback_reason,
                    query=query,
                )
                mode = RetrievalMode.PARALLEL

            # ---------------------------------------------------------------
            # PARALLEL mode: the Phase 1 path (always available).
            # ---------------------------------------------------------------

            # Step 2: Plan the execution
            plan = self._planner.plan(query, interpretation, profile)

            # Step 3: Execute SQL and vector queries in parallel
            sql_results, vector_results = await self._execute_parallel(plan)

            # Step 4: Hydrate top vector results with full content
            vector_results = await self._hydrate_results(vector_results, DEFAULT_HYDRATION_COUNT)

            # Step 5: Fuse results from both stores
            fused = self._fuser.fuse(sql_results, vector_results)

            # Step 6: Rerank fused results
            reranked = self._reranker.rerank(query, fused, top_k=top_k)

            # Calculate latency
            latency_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "query_executed",
                query=query,
                intent=plan.intent.value,
                sql_count=len(sql_results),
                vector_count=len(vector_results),
                fused_count=len(fused),
                final_count=len(reranked),
                latency_ms=round(latency_ms, 2),
            )

            result = QueryResult(
                query=query,
                mode=mode,
                results=reranked,
                total_results=len(fused),
                sql_results_count=len(sql_results),
                vector_results_count=len(vector_results),
                synthesis=None,  # LLM synthesis is Tier 2+ only
                fallback_reason=fallback_reason,
                latency_ms=round(latency_ms, 2),
            )

            # Log query metrics for the self-learning feedback loop.
            # Fire-and-forget: logging failures must never break queries.
            await self._log_query_quality(result)

            return result

        except QueryError:
            raise
        except Exception as exc:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "query_failed",
                error_code="CTXMTG-QRY-008",
                query=query,
                error=str(exc),
                latency_ms=round(latency_ms, 2),
            )
            raise QueryError(
                f"Query execution failed: {exc}",
                error_code="CTXMTG-QRY-008",
            ) from exc

    async def _dispatch_informed(
        self,
        mode: RetrievalMode,
        query: str,
        interpretation: dict[str, Any],
        profile: DomainProfile,
        top_k: int,
    ) -> QueryResult | None:
        """
        Dispatch to the appropriate informed retrieval mode.

        Creates the retriever for the requested mode and runs it.
        Returns None if the LLM is unavailable or the retriever
        is not configured (missing LLM or prompt_assembler).

        Args:
            mode: The informed retrieval mode to use.
            query: The user's natural language question.
            interpretation: Structured interpretation from the interpreter.
            profile: The active domain profile.
            top_k: Number of results to return.

        Returns:
            A QueryResult from the informed retriever, or None if
            the mode is not available (LLM missing, error, etc.).
        """
        # ---------------------------------------------------------------
        # Check that LLM infrastructure is available. Without an LLM
        # and prompt assembler, informed retrieval cannot function.
        # ---------------------------------------------------------------
        if self._llm is None or self._prompt_assembler is None:
            logger.info(
                "informed_retrieval_no_llm",
                mode=mode.value,
                query=query,
            )
            return None

        # Use the profile from the execute() call, or the default.
        active_profile = profile or self._default_profile
        if active_profile is None:
            logger.warning("informed_retrieval_no_profile", mode=mode.value,
                error_code="CTXMTG-QRY-001",
            )
            return None

        briefing_builder = SQLBriefingBuilder()

        if mode == RetrievalMode.VECTOR_TO_SQL:
            retriever = VectorToSQLRetriever(
                sql_store=self._sql_store,
                vector_store=self._vector_store,
                llm=self._llm,
                prompt_assembler=self._prompt_assembler,
                profile=active_profile,
                briefing_builder=briefing_builder,
                embedding_fn=self._embedding_fn,
            )
            return await retriever.retrieve(query, interpretation, top_k)

        elif mode == RetrievalMode.SQL_TO_VECTOR:
            retriever = SQLToVectorRetriever(
                sql_store=self._sql_store,
                vector_store=self._vector_store,
                llm=self._llm,
                prompt_assembler=self._prompt_assembler,
                profile=active_profile,
                briefing_builder=briefing_builder,
                embedding_fn=self._embedding_fn,
            )
            return await retriever.retrieve(query, interpretation, top_k)

        elif mode == RetrievalMode.BIDIRECTIONAL:
            retriever = BidirectionalRetriever(
                sql_store=self._sql_store,
                vector_store=self._vector_store,
                llm=self._llm,
                prompt_assembler=self._prompt_assembler,
                profile=active_profile,
                briefing_builder=briefing_builder,
                embedding_fn=self._embedding_fn,
            )
            return await retriever.retrieve(query, interpretation, top_k)

        # Unknown mode -- fall back to PARALLEL.
        logger.warning("unknown_retrieval_mode", mode=mode.value,
            error_code="CTXMTG-QRY-001",
        )
        return None

    # =================================================================
    # Quality logging for the self-learning feedback loop
    # =================================================================

    _REFINEMENT_WINDOW_SECONDS = 60.0

    async def _log_query_quality(self, result: QueryResult) -> None:
        """
        Log query metrics to query_quality_log and detect refinements.

        If the user issues a new query within 60 seconds of the
        previous one, the previous query is flagged as a refinement
        (implicit negative feedback consumed by FeedbackLoopStage).

        Errors are caught and logged -- never propagated. Logging
        must not break the query pipeline.
        """
        if self._quality_logger is None:
            return

        try:
            now = time.monotonic()

            # Detect refinement: if a query arrives within 60s of the
            # previous one, mark the previous query as refined.
            if (
                self._last_query_id is not None
                and (now - self._last_query_time) < self._REFINEMENT_WINDOW_SECONDS
            ):
                await self._quality_logger.mark_refinement(self._last_query_id)

            # Log this query's metrics
            result_ids = [r.id for r in result.results]
            query_id = await self._quality_logger.log_query(
                query_text=result.query,
                mode=result.mode.value,
                result_ids=result_ids,
                sql_result_count=result.sql_results_count,
                vector_result_count=result.vector_results_count,
                latency_ms=result.latency_ms,
            )

            # Track for refinement detection on the next query
            self._last_query_id = query_id
            self._last_query_time = now

        except Exception as exc:
            logger.warning(
                "quality_logging_failed",
                error_code="CTXMTG-QRY-007",
                error=str(exc),
            )

    async def _execute_parallel(
        self, plan: QueryPlan
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        """
        Execute SQL and vector queries in parallel using asyncio.gather.

        Both queries run concurrently. If one fails, the other's results
        are still returned (graceful degradation). If both fail, raises
        QueryError.

        Args:
            plan: The QueryPlan specifying which queries to execute.

        Returns:
            A tuple of (sql_results, vector_results). Either list may
            be empty if that store was not queried or failed.
        """
        # Determine which stores to query based on routing
        run_sql = plan.routing in ("both", "sql_only")
        run_vector = plan.routing in ("both", "vector_only")

        # Create tasks for parallel execution
        tasks: dict[str, Any] = {}

        if run_sql and plan.sql_query:
            tasks["sql"] = self._execute_sql(plan.sql_query)

        if run_vector and plan.vector_query and self._embedding_fn is not None:
            tasks["vector"] = self._execute_vector(plan.vector_query, plan.vector_filters)
        elif run_vector and plan.vector_query and self._embedding_fn is None:
            logger.warning(
                "vector_search_skipped",
                error_code="CTXMTG-QRY-001",
                reason="no_embedding_fn",
                routing=plan.routing,
            )

        # If no tasks to run, return empty results
        if not tasks:
            return ([], [])

        # Run tasks in parallel with graceful error handling.
        # Track both results AND whether each store raised an exception.
        # Empty results (0 rows) is a valid outcome, not a failure.
        sql_results: list[SearchResult] = []
        vector_results: list[SearchResult] = []
        sql_failed = False
        vector_failed = False

        if len(tasks) == 1:
            # Only one store to query -- run it directly
            key = next(iter(tasks))
            try:
                result = await tasks[key]
                if key == "sql":
                    sql_results = result
                else:
                    vector_results = result
            except Exception as exc:
                _code = "CTXMTG-QRY-003" if key == "sql" else "CTXMTG-QRY-004"
                logger.warning(
                    f"{key}_store_failed",
                    error_code=_code,
                    error=str(exc),
                )
                if key == "sql":
                    sql_failed = True
                else:
                    vector_failed = True
        else:
            # Both stores -- run in parallel
            results = await asyncio.gather(
                tasks["sql"],
                tasks["vector"],
                return_exceptions=True,
            )

            # Process SQL results (gather returns BaseException | result)
            if isinstance(results[0], BaseException):
                logger.warning(
                    "sql_store_failed",
                    error_code="CTXMTG-QRY-003",
                    error=str(results[0]),
                )
                sql_failed = True
            else:
                sql_results = list(results[0])

            # Process vector results
            if isinstance(results[1], BaseException):
                logger.warning(
                    "vector_store_failed",
                    error_code="CTXMTG-QRY-004",
                    error=str(results[1]),
                )
                vector_failed = True
            else:
                vector_results = list(results[1])

        # Only raise if every queried store threw an exception.
        # Empty results (0 rows found) is valid -- not an error.
        all_queried_failed = (
            ("sql" not in tasks or sql_failed)
            and ("vector" not in tasks or vector_failed)
            and (sql_failed or vector_failed)
        )
        if all_queried_failed:
            logger.error(
                "total_query_failed",
                error_code="CTXMTG-QRY-008",
            )
            raise QueryError(
                "Both SQL and vector stores failed to return results.",
                error_code="CTXMTG-QRY-008",
            )

        return (sql_results, vector_results)

    async def _execute_sql(self, sql_query: str) -> list[SearchResult]:
        """
        Execute a SQL query against the SQL store and convert to SearchResults.

        Runs the query through SQLStore.execute_sql() and converts each
        result row to a SearchResult object. The score is set based on
        row position (higher rank = higher score).

        Args:
            sql_query: The SQL query string to execute.

        Returns:
            A list of SearchResult objects from the SQL query.
        """
        try:
            rows = await self._sql_store.execute_sql(sql_query)
        except Exception as exc:
            logger.warning(
                "sql_execution_failed",
                error_code="CTXMTG-QRY-003",
                error=str(exc),
            )
            raise

        results: list[SearchResult] = []
        for rank, row in enumerate(rows):
            # Use position-based scoring: first result gets highest score.
            # Score = 1.0 / (rank + 1) gives a natural rank decay.
            score = 1.0 / (rank + 1)

            # Extract the result ID from the row. Different query templates
            # produce rows with different column names.
            result_id = str(row.get("id", f"sql-row-{rank}"))

            # Build content from available columns
            content = self._row_to_content(row)

            # Build metadata from the row (exclude content fields)
            metadata = {
                k: str(v) for k, v in row.items() if k not in ("content",) and v is not None
            }

            results.append(
                SearchResult(
                    id=result_id,
                    source_store="sql",
                    content=content,
                    score=score,
                    metadata=metadata,
                )
            )

        return results

    async def _execute_vector(
        self,
        query_text: str,
        filters: dict[str, Any],
    ) -> list[SearchResult]:
        """
        Execute a vector search against the vector store.

        Converts the query text to a vector using the embedding function,
        then searches the vector store for similar vectors.

        Args:
            query_text: The text query to embed and search for.
            filters: Metadata filters to narrow the search.

        Returns:
            A list of SearchResult objects from vector search.
        """
        if self._embedding_fn is None:
            return []

        # Convert query text to embedding vector
        query_vector = await self._get_embedding(query_text)

        if query_vector is None:
            return []

        # Search the vector store
        results = await self._vector_store.search(
            query_vector=query_vector,
            top_k=DEFAULT_TOP_K * 2,  # Fetch more for fusion
            filters=filters if filters else None,
        )

        return results

    async def _get_embedding(self, text: str) -> list[float] | None:
        """
        Convert text to an embedding vector using the configured function.

        Handles both sync and async embedding functions. Returns None
        if the embedding fails (graceful degradation).

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats, or None on failure.
        """
        try:
            embed_fn = self._embedding_fn
            if embed_fn is None:
                return None
            result = embed_fn(text)
            # Handle both sync and async embedding functions
            if asyncio.iscoroutine(result):
                result = await result
            return list(result) if result is not None else None
        except Exception as exc:
            logger.warning(
                "query_embedding_failed",
                error_code="CTXMTG-QRY-004",
                error=str(exc),
            )
            return None

    async def _hydrate_results(
        self,
        vector_results: list[SearchResult],
        count: int,
    ) -> list[SearchResult]:
        """
        Hydrate top-N vector results with full content from SQL.

        Vector search results only contain a 200-char content_preview.
        This method fetches the full interaction content from the SQL
        store for the top N results, replacing the preview with the
        complete text.

        Only the top-N results are hydrated to limit SQL queries.
        Lower-ranked results keep their preview content.

        Args:
            vector_results: The vector search results to hydrate.
            count: Number of top results to hydrate (default 5).

        Returns:
            The same results list, with top-N contents replaced by
            full interaction text from SQL.
        """
        if not vector_results:
            return vector_results

        # Hydrate only the top-N results
        to_hydrate = vector_results[:count]
        rest = vector_results[count:]

        hydrated: list[SearchResult] = []
        for result in to_hydrate:
            # Get the source_id from metadata to look up in SQL
            source_id = result.metadata.get("source_id", "")
            if source_id:
                try:
                    interaction = await self._sql_store.get_interaction(source_id)
                    if interaction:
                        # Replace preview content with full content
                        hydrated.append(
                            SearchResult(
                                id=result.id,
                                source_store=result.source_store,
                                content=interaction.content,
                                score=result.score,
                                metadata={
                                    **result.metadata,
                                    "title": interaction.title or "",
                                    "source_type": interaction.source_type.value,
                                    "hydrated": "true",
                                },
                            )
                        )
                        continue
                except Exception:
                    # If hydration fails, keep the original preview
                    logger.warning(
                        "hydration_failed",
                        error_code="CTXMTG-QRY-003",
                        source_id=source_id,
                    )

            # Keep original result if hydration was not possible
            hydrated.append(result)

        return hydrated + rest

    @staticmethod
    def _row_to_content(row: dict[str, Any]) -> str:
        """
        Build a content string from a SQL result row.

        Constructs a human-readable content string from the available
        columns in the row. Different query templates produce different
        columns, so this method adapts to whatever is available.

        Args:
            row: A dict of column_name → value from a SQL query.

        Returns:
            A formatted content string.
        """
        # If the row has a "content" column, use it directly
        if row.get("content"):
            return str(row["content"])

        # Build content from fact columns (factual/comparative queries)
        parts: list[str] = []

        if "entity_name" in row:
            parts.append(str(row["entity_name"]))
        if "predicate" in row:
            parts.append(str(row["predicate"]))
        if row.get("object_literal"):
            parts.append(str(row["object_literal"]))
        if row.get("title"):
            parts.append(str(row["title"]))

        # For aggregation results, show the count
        if "total_count" in row:
            parts.append(f"Count: {row['total_count']}")

        return " | ".join(parts) if parts else str(row)
