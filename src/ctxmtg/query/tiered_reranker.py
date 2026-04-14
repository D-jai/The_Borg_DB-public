# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Tiered Reranker
===============

This module implements a tiered reranking strategy that composes
TF-IDF and cross-encoder rerankers for a two-stage relevance pipeline.

Why tiered reranking?
    - Cross-encoders are accurate but slow (O(n) forward passes)
    - TF-IDF is fast but less accurate for semantic relevance
    - By running TF-IDF first to narrow candidates, then cross-encoder
      on the smaller set, we get the best of both worlds
    - Graceful degradation: if the cross-encoder is unavailable, the
      tiered reranker returns TF-IDF results directly

Architecture:
    Stage 1 (TF-IDF): Narrow from full result set → tfidf_top_k (e.g., 50)
    Stage 2 (Cross-encoder): Re-score the tfidf_top_k candidates → top_k

    This mirrors the "retrieval → reranking" pattern used in production
    search systems (e.g., Elasticsearch → BERT reranker).

Fallback behavior:
    If cross_encoder is None (model not available), the tiered reranker
    degrades to TF-IDF-only reranking. This ensures the system always
    returns results, even without the ONNX model.

Depends on:
    - ctxmtg.interfaces.query (Reranker ABC)
    - ctxmtg.models.query (SearchResult)
    - ctxmtg.query.reranker (TFIDFReranker -- Stage 1)
    - ctxmtg.query.cross_encoder_reranker (CrossEncoderReranker -- Stage 2)

Used by:
    - ctxmtg.query.reranker_factory (factory creates this)
    - ctxmtg.query.executor (calls rerank() after fusion)
    - tests/test_query/test_tiered_reranker.py
"""

from __future__ import annotations

import structlog

from ctxmtg.interfaces.query import Reranker
from ctxmtg.models.query import SearchResult
from ctxmtg.query.cross_encoder_reranker import CrossEncoderReranker
from ctxmtg.query.reranker import TFIDFReranker

# ---------------------------------------------------------------
# Module logger -- logs tiered reranking decisions and statistics.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.tiered_reranker")


class TieredReranker(Reranker):
    """
    Two-stage reranker: TF-IDF narrowing + cross-encoder scoring.

    Composes a TFIDFReranker (Stage 1) and a CrossEncoderReranker
    (Stage 2) into a single reranking pipeline. Stage 1 quickly
    narrows the candidate set, then Stage 2 applies the more
    accurate (but slower) cross-encoder to the survivors.

    If the cross-encoder is not available (passed as None), the
    tiered reranker degrades gracefully to TF-IDF-only mode. This
    ensures the query pipeline always returns results.

    Usage:
        # Full tiered reranking (cross-encoder available)
        tiered = TieredReranker(
            tfidf_reranker=TFIDFReranker(),
            cross_encoder=CrossEncoderReranker(model_path="/path/to/model.onnx"),
            tfidf_top_k=50,
        )

        # Fallback mode (cross-encoder unavailable)
        tiered = TieredReranker(
            tfidf_reranker=TFIDFReranker(),
            cross_encoder=None,  # Falls back to TF-IDF only
        )

        reranked = tiered.rerank("What did Alice propose?", results, top_k=10)
    """

    def __init__(
        self,
        tfidf_reranker: TFIDFReranker,
        cross_encoder: CrossEncoderReranker | None,
        tfidf_top_k: int = 50,
    ) -> None:
        """
        Initialize the tiered reranker with both stages.

        Args:
            tfidf_reranker: The TF-IDF reranker for Stage 1 narrowing.
            cross_encoder: The cross-encoder reranker for Stage 2 scoring.
                           If None, the tiered reranker uses TF-IDF only.
            tfidf_top_k: How many results to keep after Stage 1 (default 50).
                         This is the candidate set size passed to the
                         cross-encoder. Larger values are more accurate but
                         slower (each candidate needs a forward pass).
        """
        self._tfidf = tfidf_reranker
        self._cross_encoder = cross_encoder
        self._tfidf_top_k = tfidf_top_k

        # Log the configuration for debugging
        logger.info(
            "tiered_reranker_initialized",
            has_cross_encoder=cross_encoder is not None,
            tfidf_top_k=tfidf_top_k,
        )

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Re-rank results using the two-stage TF-IDF + cross-encoder pipeline.

        Stage 1: TF-IDF narrows results to tfidf_top_k candidates.
        Stage 2: Cross-encoder re-scores the candidates for final ranking.

        If the cross-encoder is not available (None), returns the TF-IDF
        results directly, limited to top_k.

        Args:
            query: The original user question.
            results: The fused results from the ResultFuser.
            top_k: Number of top results to return (default 10).

        Returns:
            A list of at most top_k SearchResult objects, re-ranked
            by the highest-quality available method.
        """
        # Handle empty input gracefully
        if not results:
            return []

        # Stage 1: TF-IDF narrowing to tfidf_top_k candidates
        # Use tfidf_top_k as the Stage 1 cutoff, not top_k
        tfidf_results = self._tfidf.rerank(
            query=query,
            results=results,
            top_k=self._tfidf_top_k,
        )

        logger.debug(
            "tiered_stage1_completed",
            input_count=len(results),
            tfidf_output_count=len(tfidf_results),
        )

        # Stage 2: Cross-encoder scoring (if available)
        if self._cross_encoder is not None:
            # Run the cross-encoder on the TF-IDF narrowed set
            final_results = self._cross_encoder.rerank(
                query=query,
                results=tfidf_results,
                top_k=top_k,
            )

            logger.info(
                "tiered_reranking_completed",
                mode="cross_encoder",
                input_count=len(results),
                stage1_count=len(tfidf_results),
                output_count=len(final_results),
            )

            return final_results

        # Fallback: cross-encoder not available, return TF-IDF results
        # Trim to top_k since Stage 1 may have returned more
        fallback_results = tfidf_results[:top_k]

        logger.info(
            "tiered_reranking_completed",
            mode="tfidf_fallback",
            input_count=len(results),
            output_count=len(fallback_results),
            detail="Cross-encoder unavailable; using TF-IDF results only",
        )

        return fallback_results
