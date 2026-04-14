# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
RRF Result Fusion
=================

This module implements Reciprocal Rank Fusion (RRF) for combining results
from the SQL and vector stores into a single ranked list.

RRF is a well-studied rank aggregation technique that works without
requiring the scores from different rankers to be on the same scale.
This is critical for our dual-store architecture because:
    - SQL store scores are position-based (1/rank)
    - Vector store scores are cosine similarity (0.0 to 1.0)
    - These scales are incomparable; RRF uses only ranks

The RRF formula for each result:
    fused_score = sum( 1 / (k + rank_in_store) ) across all stores

Where k is a smoothing constant (default 60). Higher k gives more
equal weighting across all ranks; lower k amplifies top-ranked results.

The k=60 value comes from the original RRF paper (Cormack, Clarke, Butt
2009) and is the standard default in most IR systems.

Phase 2 adds LLMFuser: instead of mathematical fusion, the LLM reads
both result streams and synthesizes a natural language answer. RRF
remains as the Tier 0-1 fallback when no LLM is available.

Depends on:
    - ctxmtg.interfaces.query (ResultFuser ABC)
    - ctxmtg.models.query (SearchResult)

Used by:
    - ctxmtg.query.executor (calls fuse() after parallel execution)
    - tests/test_query/test_fusion.py
"""

from __future__ import annotations

import structlog

from ctxmtg.interfaces.query import ResultFuser
from ctxmtg.models.query import SearchResult

# ---------------------------------------------------------------
# Module logger -- logs fusion statistics.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.fusion")


class RRFFuser(ResultFuser):
    """
    Reciprocal Rank Fusion (RRF) with configurable k parameter.

    Combines results from SQL and vector stores into a single ranked
    list using the RRF formula. Results that appear in both stores
    get scores from both, producing a natural boost for results that
    are relevant in both structured and semantic dimensions.

    The fuser handles three cases:
    1. Result in SQL only → score from SQL rank only
    2. Result in vector only → score from vector rank only
    3. Result in both stores → merged scores from both ranks + merged metadata

    Usage:
        fuser = RRFFuser()
        fused = fuser.fuse(sql_results, vector_results, k=60)
        # fused[0] has the highest combined rank across both stores
    """

    def fuse(
        self,
        sql_results: list[SearchResult],
        vector_results: list[SearchResult],
        k: int = 60,
    ) -> list[SearchResult]:
        """
        Fuse results from SQL and vector stores using RRF.

        For each unique result (identified by ID), computes:
            score = sum(1 / (k + rank)) across all stores it appears in

        Results appearing in both stores get a higher combined score
        than results appearing in only one. Metadata from both stores
        is merged for dual-appearing results.

        Args:
            sql_results: Results from the SQL store, ordered by relevance.
            vector_results: Results from the vector store, ordered by
                            similarity.
            k: The RRF smoothing constant (default 60). From the
               original paper by Cormack, Clarke, and Butt (2009).

        Returns:
            A single list of SearchResult objects, sorted by fused
            score (highest first). Each result's score field contains
            the RRF score.
        """
        # Build a map of result_id → accumulated RRF data.
        # We track the RRF score, the best content, and merged metadata.
        fused_map: dict[str, _FusedEntry] = {}

        # Process SQL results: assign RRF scores based on rank (0-indexed)
        for rank, result in enumerate(sql_results):
            rrf_score = 1.0 / (k + rank + 1)
            entry = fused_map.get(result.id)

            if entry is None:
                # First time seeing this result -- create a new entry
                fused_map[result.id] = _FusedEntry(
                    result_id=result.id,
                    rrf_score=rrf_score,
                    content=result.content,
                    metadata=dict(result.metadata),
                    source_stores=["sql"],
                )
            else:
                # Result already seen in another store -- merge
                entry.rrf_score += rrf_score
                entry.source_stores.append("sql")
                entry.metadata.update(result.metadata)
                # Prefer longer content (SQL usually has full content)
                if len(result.content) > len(entry.content):
                    entry.content = result.content

        # Process vector results: assign RRF scores based on rank (0-indexed)
        for rank, result in enumerate(vector_results):
            rrf_score = 1.0 / (k + rank + 1)
            entry = fused_map.get(result.id)

            if entry is None:
                # First time seeing this result
                fused_map[result.id] = _FusedEntry(
                    result_id=result.id,
                    rrf_score=rrf_score,
                    content=result.content,
                    metadata=dict(result.metadata),
                    source_stores=["vector"],
                )
            else:
                # Result already seen in SQL store -- merge
                entry.rrf_score += rrf_score
                entry.source_stores.append("vector")
                # Merge metadata from vector store
                for mk, mv in result.metadata.items():
                    if mk not in entry.metadata:
                        entry.metadata[mk] = mv
                # Prefer longer content
                if len(result.content) > len(entry.content):
                    entry.content = result.content

        # Sort by RRF score (highest first) and convert to SearchResult
        sorted_entries = sorted(
            fused_map.values(),
            key=lambda e: e.rrf_score,
            reverse=True,
        )

        # Convert _FusedEntry objects back to SearchResult objects
        fused_results: list[SearchResult] = []
        for entry in sorted_entries:
            # The source_store field indicates where the result came from.
            # "both" if it appeared in both stores, otherwise the single store.
            source_store = "both" if len(entry.source_stores) > 1 else entry.source_stores[0]

            fused_results.append(
                SearchResult(
                    id=entry.result_id,
                    source_store=source_store,
                    content=entry.content,
                    score=entry.rrf_score,
                    metadata=entry.metadata,
                )
            )

        # Log fusion statistics for debugging and farming feedback
        both_count = sum(1 for e in sorted_entries if len(e.source_stores) > 1)
        logger.info(
            "rrf_fusion_completed",
            sql_count=len(sql_results),
            vector_count=len(vector_results),
            fused_count=len(fused_results),
            both_stores_count=both_count,
            k=k,
        )

        return fused_results


class _FusedEntry:
    """
    Internal data structure for accumulating RRF fusion data.

    Tracks the running RRF score, best content, merged metadata,
    and which stores contributed to this result. Used only within
    the RRFFuser during the fusion process.
    """

    __slots__ = ("content", "metadata", "result_id", "rrf_score", "source_stores")

    def __init__(
        self,
        result_id: str,
        rrf_score: float,
        content: str,
        metadata: dict,
        source_stores: list[str],
    ) -> None:
        self.result_id = result_id
        self.rrf_score = rrf_score
        self.content = content
        self.metadata = metadata
        self.source_stores = source_stores
