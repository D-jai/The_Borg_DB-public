# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Reranker Factory
================

This module provides a factory function for creating reranker instances
based on configuration. It encapsulates the logic for choosing between
TF-IDF, cross-encoder, and tiered rerankers, including graceful fallback
when the cross-encoder model is not available.

Why a factory?
    - Centralizes reranker creation logic in one place
    - Handles graceful degradation (cross-encoder → TF-IDF fallback)
    - Config-driven: the reranker type can be changed without code changes
    - Simplifies testing: callers don't need to know about model paths

Configuration keys (under `query.` in config):
    - reranker: "tfidf" | "cross_encoder" | "tiered" (default: "tfidf")
    - cross_encoder_model_path: path to the ONNX model file (default: None)
    - cross_encoder_model: model name for reference (default: "ms-marco-MiniLM-L6-v2")
    - tiered_tfidf_top_k: Stage 1 candidate set size (default: 50)

Fallback behavior:
    If "cross_encoder" or "tiered" is requested but the model cannot be
    loaded (file not found, import error, etc.), the factory logs a warning
    and returns a TFIDFReranker instead. The system always has a working
    reranker -- it just may not be the most accurate one.

Depends on:
    - ctxmtg.interfaces.query (Reranker ABC)
    - ctxmtg.query.reranker (TFIDFReranker)
    - ctxmtg.query.cross_encoder_reranker (CrossEncoderReranker, RerankerModelError)
    - ctxmtg.query.tiered_reranker (TieredReranker)

Used by:
    - ctxmtg.query.executor (gets reranker from factory)
    - tests/test_query/test_reranker_factory.py
"""

from __future__ import annotations

import structlog

from ctxmtg.interfaces.query import Reranker
from ctxmtg.query.cross_encoder_reranker import (
    CrossEncoderReranker,
    RerankerModelError,
)
from ctxmtg.query.reranker import TFIDFReranker
from ctxmtg.query.tiered_reranker import TieredReranker

# ---------------------------------------------------------------
# Module logger -- logs factory decisions and fallback events.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.reranker_factory")


def create_reranker(config: dict | None = None) -> Reranker:
    """
    Create a reranker instance based on configuration.

    Reads the reranker type from config and instantiates the appropriate
    reranker. If the requested type requires a cross-encoder model that
    is not available, falls back to TF-IDF with a warning log.

    Config structure expected (nested dict from YAML):
        {
            "query": {
                "reranker": "tfidf",          # or "cross_encoder" or "tiered"
                "cross_encoder_model_path": null,  # path to ONNX model
                "tiered_tfidf_top_k": 50,     # Stage 1 candidate count
            }
        }

    If config is None or empty, defaults to TF-IDF reranker.

    Args:
        config: The full configuration dict (typically loaded from YAML).
                If None, uses defaults (TF-IDF reranker).

    Returns:
        A Reranker instance (TFIDFReranker, CrossEncoderReranker,
        or TieredReranker), ready to use in the query pipeline.
    """
    # Extract query-specific config, defaulting to empty dict
    query_config = (config or {}).get("query", {})

    # Determine which reranker type to create (default: tfidf)
    reranker_type = query_config.get("reranker", "tfidf")

    # Extract cross-encoder configuration
    model_path = query_config.get("cross_encoder_model_path")

    # Extract tiered reranker configuration
    tfidf_top_k = query_config.get("tiered_tfidf_top_k", 50)

    logger.info(
        "creating_reranker",
        reranker_type=reranker_type,
        model_path=model_path,
    )

    # --- TF-IDF: the default, always-available reranker ---
    if reranker_type == "tfidf":
        logger.info("reranker_created", reranker_type="tfidf")
        return TFIDFReranker()

    # --- Cross-encoder: higher accuracy, requires ONNX model ---
    if reranker_type == "cross_encoder":
        try:
            reranker = CrossEncoderReranker(model_path=model_path)
            logger.info(
                "reranker_created",
                reranker_type="cross_encoder",
                model_path=model_path,
            )
            return reranker
        except RerankerModelError as exc:
            # Cross-encoder unavailable -- fall back to TF-IDF
            logger.warning(
                "cross_encoder_fallback_to_tfidf",
                error_code="CTXMTG-QRY-001",
                error=str(exc),
                detail="Cross-encoder model unavailable; falling back to TF-IDF",
            )
            return TFIDFReranker()

    # --- Tiered: TF-IDF + cross-encoder two-stage pipeline ---
    if reranker_type == "tiered":
        # Create the TF-IDF reranker for Stage 1
        tfidf_reranker = TFIDFReranker()

        # Attempt to create the cross-encoder for Stage 2
        cross_encoder: CrossEncoderReranker | None = None
        try:
            cross_encoder = CrossEncoderReranker(model_path=model_path)
        except RerankerModelError as exc:
            # Cross-encoder unavailable -- tiered degrades to TF-IDF-only
            logger.warning(
                "tiered_cross_encoder_unavailable",
                error_code="CTXMTG-QRY-001",
                error=str(exc),
                detail="Cross-encoder unavailable; tiered reranker will use TF-IDF only",
            )

        reranker = TieredReranker(
            tfidf_reranker=tfidf_reranker,
            cross_encoder=cross_encoder,
            tfidf_top_k=tfidf_top_k,
        )

        logger.info(
            "reranker_created",
            reranker_type="tiered",
            has_cross_encoder=cross_encoder is not None,
            tfidf_top_k=tfidf_top_k,
        )
        return reranker

    # --- Unknown reranker type: warn and fall back to TF-IDF ---
    logger.warning(
        "unknown_reranker_type",
        error_code="CTXMTG-QRY-001",
        reranker_type=reranker_type,
        detail=f"Unknown reranker type '{reranker_type}'; falling back to TF-IDF",
    )
    return TFIDFReranker()
