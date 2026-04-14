# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Cross-Encoder Reranker
======================

This module implements a cross-encoder reranker that scores (query, document)
pairs using an ONNX cross-encoder model for more accurate relevance ranking
than keyword-based approaches like TF-IDF.

Why a cross-encoder?
    - Cross-encoders jointly encode query + document, capturing semantic
      interactions that bi-encoders and TF-IDF cannot
    - Much more accurate than TF-IDF for nuanced relevance judgment
    - ONNX format enables CPU-only inference on edge devices (Pi, laptop)
    - The tradeoff is speed: cross-encoders score each pair individually,
      so they are slower than TF-IDF for large result sets

Architecture:
    The CrossEncoderReranker is designed to sit after an initial coarse
    ranking (TF-IDF or RRF fusion) that narrows results to a manageable
    set (e.g., top 50). The cross-encoder then re-scores these candidates
    for high-precision final ranking.

Graceful degradation:
    If the ONNX model file is not found or fails to load, the reranker
    raises RerankerModelError. The TieredReranker and reranker_factory
    handle this by falling back to TF-IDF-only reranking.

Depends on:
    - onnxruntime (ONNX model inference -- may not be available)
    - numpy (array operations for model input/output)
    - ctxmtg.interfaces.query (Reranker ABC)
    - ctxmtg.models.query (SearchResult)

Used by:
    - ctxmtg.query.tiered_reranker (TieredReranker composes this)
    - ctxmtg.query.reranker_factory (factory creates this)
    - tests/test_query/test_cross_encoder.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog

from ctxmtg.interfaces.query import Reranker
from ctxmtg.models.query import SearchResult

# ---------------------------------------------------------------
# Module logger -- logs model loading and scoring events.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.cross_encoder_reranker")


# ---------------------------------------------------------------
# Custom exception for cross-encoder model loading failures.
# This is separate from the generic EmbeddingError because the
# cross-encoder is optional -- callers should catch this and
# fall back to TF-IDF rather than treating it as a fatal error.
# ---------------------------------------------------------------
class RerankerModelError(Exception):
    """
    Raised when the cross-encoder ONNX model cannot be loaded.

    This error signals that the cross-encoder is not available,
    and the caller should fall back to TF-IDF reranking. Common
    causes include:
    - Model file not found at the specified path
    - Model file is corrupted or incompatible
    - ONNX runtime cannot load the model (e.g., missing operators)
    """


# ---------------------------------------------------------------
# Simple tokenizer for cross-encoder input preparation.
# In a production setup this would use the model's own tokenizer
# (e.g., Hugging Face tokenizers). Here we use a word-level
# tokenizer that produces token IDs from a basic vocabulary,
# compatible with the ONNX model's expected input format.
# ---------------------------------------------------------------
def _simple_tokenize(
    text_a: str,
    text_b: str,
    max_length: int = 512,
) -> dict[str, Any]:
    """
    Tokenize a (query, document) pair for cross-encoder input.

    Produces a simplified token representation compatible with
    transformer-style models: [CLS] text_a [SEP] text_b [SEP].
    Token IDs are simple word hashes (suitable for testing and
    fallback; a real deployment uses the model's tokenizer).

    The output dict contains:
    - input_ids: list of integer token IDs
    - attention_mask: list of 1s (all tokens attended to)
    - token_type_ids: 0 for text_a tokens, 1 for text_b tokens

    Args:
        text_a: The query text (first segment).
        text_b: The document text (second segment).
        max_length: Maximum total sequence length (default 512).

    Returns:
        Dict with input_ids, attention_mask, and token_type_ids.
    """
    # Tokenize both texts into lowercase word tokens
    tokens_a = re.findall(r"\b[a-z0-9]+\b", text_a.lower())
    tokens_b = re.findall(r"\b[a-z0-9]+\b", text_b.lower())

    # Reserve 3 slots for special tokens: [CLS], [SEP], [SEP]
    available = max_length - 3

    # Split available space between text_a and text_b (text_b gets more)
    max_a = min(len(tokens_a), available // 3)
    max_b = min(len(tokens_b), available - max_a)

    # Truncate to fit within max_length
    tokens_a = tokens_a[:max_a]
    tokens_b = tokens_b[:max_b]

    # Build input_ids using hash-based token IDs
    # CLS=101, SEP=102 (matching BERT convention)
    cls_id = 101
    sep_id = 102

    # Hash each token to a positive integer (simulate vocabulary lookup)
    ids_a = [hash(t) % 30000 + 1000 for t in tokens_a]
    ids_b = [hash(t) % 30000 + 1000 for t in tokens_b]

    # Assemble: [CLS] tokens_a [SEP] tokens_b [SEP]
    input_ids = [cls_id] + ids_a + [sep_id] + ids_b + [sep_id]

    # Attention mask: 1 for all real tokens
    attention_mask = [1] * len(input_ids)

    # Token type IDs: 0 for text_a segment, 1 for text_b segment
    token_type_ids = [0] * (1 + len(ids_a) + 1) + [1] * (len(ids_b) + 1)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder reranker using an ONNX model for relevance scoring.

    Scores each (query, document) pair by jointly encoding them through
    a cross-encoder transformer model. This captures fine-grained
    semantic interactions between query and document that keyword-based
    methods (TF-IDF) and bi-encoder methods (embedding similarity) miss.

    The model is loaded from an ONNX file at construction time. If the
    model file is not found, a RerankerModelError is raised so the
    caller can fall back to TF-IDF.

    Architecture:
        1. For each result, tokenize (query, result.content) as a pair
        2. Run the tokenized pair through the ONNX cross-encoder
        3. Extract the relevance score from the model output
        4. Sort results by score and return top_k

    Usage:
        try:
            reranker = CrossEncoderReranker(model_path="/path/to/model.onnx")
        except RerankerModelError:
            # Model not available -- use TF-IDF fallback
            reranker = TFIDFReranker()

        reranked = reranker.rerank("What did Alice propose?", results, top_k=10)
    """

    def __init__(
        self,
        model_path: str | None = None,
        max_length: int = 512,
    ) -> None:
        """
        Initialize the cross-encoder reranker with an ONNX model.

        Attempts to load the ONNX model from the specified path. If
        the path is None or the file does not exist, raises a
        RerankerModelError so the caller can fall back gracefully.

        Args:
            model_path: Path to the ONNX cross-encoder model file.
                        If None, raises RerankerModelError immediately.
            max_length: Maximum token sequence length for the model
                        input (default 512). Longer inputs are truncated.

        Raises:
            RerankerModelError: If the model cannot be loaded (file
                                missing, import error, or runtime error).
        """
        self._max_length = max_length
        self._session = None  # ONNX InferenceSession (set below)
        self._model_path = model_path

        # If no model path provided, signal that the model is unavailable
        if model_path is None:
            logger.warning(
                "cross_encoder_no_model_path",
                error_code="CTXMTG-QRY-001",
                detail="No model path provided; cross-encoder is unavailable",
            )
            raise RerankerModelError(
                "No model path provided. Cross-encoder reranker requires "
                "an ONNX model file. Use TF-IDF reranker as fallback."
            )

        # Verify the model file exists before attempting to load
        model_file = Path(model_path)
        if not model_file.exists():
            logger.warning(
                "cross_encoder_model_not_found",
                error_code="CTXMTG-QRY-001",
                detail="Model file does not exist at the specified path",
            )
            raise RerankerModelError(
                f"Cross-encoder model not found at '{model_path}'. "
                f"Download the model with scripts/download_models.py or "
                f"specify a valid path."
            )

        # Attempt to load the ONNX model using onnxruntime
        try:
            import onnxruntime as ort  # noqa: F811

            # Create an inference session with CPU provider (edge-friendly)
            self._session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            logger.info(
                "cross_encoder_model_loaded",
                model_path=model_path,
                max_length=max_length,
            )
        except ImportError:
            # onnxruntime not installed -- cannot use cross-encoder
            logger.warning(
                "cross_encoder_onnxruntime_missing",
                error_code="CTXMTG-QRY-001",
                detail="onnxruntime is not installed; cross-encoder unavailable",
            )
            raise RerankerModelError(
                "onnxruntime is not installed. Install it with: "
                "pip install onnxruntime"
            )
        except Exception as exc:
            # Any other error during model loading (corrupt file, etc.)
            logger.warning(
                "cross_encoder_model_load_failed",
                error_code="CTXMTG-QRY-001",
                error=str(exc),
            )
            raise RerankerModelError(
                f"Failed to load cross-encoder model from '{model_path}': {exc}"
            ) from exc

    def _score_pair(self, query: str, document: str) -> float:
        """
        Score a single (query, document) pair using the cross-encoder.

        Tokenizes the pair, runs it through the ONNX model, and extracts
        the relevance score from the model output. The score is a float
        where higher values indicate more relevance.

        For models with a two-class output (relevant/not-relevant), we
        use the logit for the "relevant" class. For single-output models,
        we use the output directly.

        Args:
            query: The user's query text.
            document: The document content to score against the query.

        Returns:
            A float relevance score (higher = more relevant).
        """
        import numpy as np

        # Tokenize the (query, document) pair
        tokens = _simple_tokenize(query, document, max_length=self._max_length)

        # Prepare numpy arrays for the ONNX model
        # Shape: (batch_size=1, sequence_length)
        input_ids = np.array([tokens["input_ids"]], dtype=np.int64)
        attention_mask = np.array([tokens["attention_mask"]], dtype=np.int64)
        token_type_ids = np.array([tokens["token_type_ids"]], dtype=np.int64)

        # Run inference through the ONNX model
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        # Only include inputs that the model expects (some models
        # don't use token_type_ids)
        model_inputs = {inp.name for inp in self._session.get_inputs()}
        filtered_inputs = {k: v for k, v in inputs.items() if k in model_inputs}

        outputs = self._session.run(None, filtered_inputs)

        # Extract the relevance score from model output
        # Output shape is typically (1, num_classes) or (1,)
        logits = outputs[0]

        if logits.shape[-1] == 2:
            # Two-class model: use the "relevant" class logit (index 1)
            score = float(logits[0][1])
        else:
            # Single-output model: use the output directly
            score = float(logits[0][0])

        return score

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Re-rank results using cross-encoder relevance scoring.

        Scores each (query, result.content) pair through the ONNX
        cross-encoder model, then sorts by score (highest first) and
        returns the top_k most relevant results.

        This is the most accurate reranking method available, but also
        the slowest since each pair requires a full forward pass through
        the model. Best used after TF-IDF narrows the candidate set.

        Args:
            query: The original user question.
            results: The candidate results to re-rank (typically
                     pre-filtered by TF-IDF to a manageable set).
            top_k: Number of top results to return (default 10).

        Returns:
            A list of at most top_k SearchResult objects, re-ranked
            by cross-encoder relevance score (highest first).
        """
        # Handle empty input gracefully
        if not results:
            return []

        # Score each result against the query
        scored: list[tuple[float, int]] = []
        for idx, result in enumerate(results):
            score = self._score_pair(query, result.content)
            scored.append((score, idx))

        # Sort by score descending and take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored[:top_k]

        # Build the output list with updated scores
        reranked: list[SearchResult] = []
        for score, idx in top_scored:
            original = results[idx]
            reranked.append(
                SearchResult(
                    id=original.id,
                    source_store=original.source_store,
                    content=original.content,
                    score=score,
                    metadata=original.metadata,
                )
            )

        # Log reranking statistics
        logger.info(
            "cross_encoder_reranking_completed",
            input_count=len(results),
            output_count=len(reranked),
            top_score=round(reranked[0].score, 4) if reranked else 0.0,
        )

        return reranked
