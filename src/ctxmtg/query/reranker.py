# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
TF-IDF Reranker
===============

This module implements a lightweight reranker using TF-IDF cosine
similarity between the user's query and each search result's content.
This provides a second pass of relevance scoring on top of RRF fusion.

Why TF-IDF?
    - No external dependencies needed (manual implementation)
    - Fast enough for edge devices (Pi, laptop)
    - Captures keyword relevance that vector embeddings might miss
    - Good complement to semantic similarity (vectors capture meaning,
      TF-IDF captures keyword overlap)

The implementation is manual (no scikit-learn) to keep the dependency
footprint small for edge deployment. The TF-IDF math is:
    TF(term, doc) = count(term in doc) / len(doc)
    IDF(term) = log(N / df(term)) where N = number of docs, df = docs containing term
    TF-IDF(term, doc) = TF * IDF
    cosine_sim = dot(tfidf_query, tfidf_doc) / (norm_query * norm_doc)

Phase 4 replaces this with a cross-encoder model for higher accuracy
at the cost of more computation.

Depends on:
    - math (log, sqrt for TF-IDF computation)
    - ctxmtg.interfaces.query (Reranker ABC)
    - ctxmtg.models.query (SearchResult)

Used by:
    - ctxmtg.query.executor (calls rerank() after fusion)
    - tests/test_query/test_fusion.py (reranker tests in fusion test file)
"""

from __future__ import annotations

import math
import re
from collections import Counter

import structlog

from ctxmtg.interfaces.query import Reranker
from ctxmtg.models.query import SearchResult

# ---------------------------------------------------------------
# Module logger -- logs reranking statistics.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.reranker")


def _tokenize(text: str) -> list[str]:
    """
    Simple tokenizer: lowercase, split on non-alphanumeric chars.

    Produces a list of lowercase tokens for TF-IDF computation.
    This is intentionally simple -- a production system would use
    a proper tokenizer with stemming and stop word removal, but
    for Phase 1 this is sufficient.

    Args:
        text: The input text to tokenize.

    Returns:
        A list of lowercase tokens.
    """
    # Convert to lowercase and split on word boundaries
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def _compute_tf(tokens: list[str]) -> dict[str, float]:
    """
    Compute Term Frequency (TF) for a list of tokens.

    TF(term) = count(term) / total_tokens

    Args:
        tokens: List of tokens from a document.

    Returns:
        Dict mapping each term to its TF value.
    """
    if not tokens:
        return {}

    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def _compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """
    Compute Inverse Document Frequency (IDF) across all documents.

    IDF(term) = log(N / df(term))
    where N = number of documents, df = number of documents containing term.

    Uses log(N / df) + 1 to avoid zero IDF for terms in all documents.

    Args:
        documents: List of tokenized documents (each is a list of tokens).

    Returns:
        Dict mapping each term to its IDF value.
    """
    n = len(documents)
    if n == 0:
        return {}

    # Count how many documents contain each term
    df: Counter[str] = Counter()
    for doc_tokens in documents:
        # Use set to count each term only once per document
        unique_terms = set(doc_tokens)
        for term in unique_terms:
            df[term] += 1

    # Compute IDF with smoothing (add 1 to avoid log(0))
    idf: dict[str, float] = {}
    for term, doc_freq in df.items():
        idf[term] = math.log(n / doc_freq) + 1.0

    return idf


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """
    Compute cosine similarity between two sparse TF-IDF vectors.

    cosine_sim = dot(A, B) / (||A|| * ||B||)

    Both vectors are represented as dicts mapping terms to weights.
    Terms not present in a vector have weight 0 (sparse representation).

    Args:
        vec_a: First TF-IDF vector (query).
        vec_b: Second TF-IDF vector (document).

    Returns:
        Cosine similarity in [0, 1]. Returns 0 if either vector is zero.
    """
    # Compute dot product (only terms present in both vectors contribute)
    dot_product = sum(vec_a[term] * vec_b.get(term, 0.0) for term in vec_a)

    # Compute norms
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

    # Avoid division by zero
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class TFIDFReranker(Reranker):
    """
    Lightweight reranker using TF-IDF cosine similarity.

    Recomputes relevance scores for fused results by measuring the
    keyword overlap between the query and each result's content. Results
    are sorted by the TF-IDF similarity score (highest first).

    The reranker's score replaces the RRF fusion score in the output.
    This is intentional: TF-IDF provides a more interpretable relevance
    signal than the mathematical RRF score.

    Implementation is fully manual (no scikit-learn) to keep the
    dependency footprint small for edge deployment.

    Usage:
        reranker = TFIDFReranker()
        reranked = reranker.rerank(
            query="What did Alice propose?",
            results=fused_results,
            top_k=10,
        )
        # reranked[0] has the highest TF-IDF similarity to the query
    """

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Re-rank results by TF-IDF cosine similarity to the query.

        Steps:
        1. Tokenize the query and all result contents
        2. Compute IDF across all documents (results + query)
        3. Compute TF-IDF vectors for query and each result
        4. Calculate cosine similarity between query and each result
        5. Sort by similarity and return top_k

        Args:
            query: The original user question.
            results: The fused results from the ResultFuser.
            top_k: Number of top results to return.

        Returns:
            A list of at most top_k SearchResult objects, sorted by
            TF-IDF similarity (highest first). Scores are updated to
            reflect the reranker's assessment.
        """
        if not results:
            return []

        # Step 1: Tokenize query and all documents
        query_tokens = _tokenize(query)
        doc_token_lists: list[list[str]] = [_tokenize(r.content) for r in results]

        # Step 2: Compute IDF across all documents (include query as a document)
        all_docs = [query_tokens, *doc_token_lists]
        idf = _compute_idf(all_docs)

        # Step 3: Compute TF-IDF vector for the query
        query_tf = _compute_tf(query_tokens)
        query_tfidf = {term: tf_val * idf.get(term, 1.0) for term, tf_val in query_tf.items()}

        # Step 4: Compute TF-IDF similarity for each result
        scored: list[tuple[float, int]] = []
        for idx, doc_tokens in enumerate(doc_token_lists):
            doc_tf = _compute_tf(doc_tokens)
            doc_tfidf = {term: tf_val * idf.get(term, 1.0) for term, tf_val in doc_tf.items()}
            similarity = _cosine_similarity(query_tfidf, doc_tfidf)
            scored.append((similarity, idx))

        # Step 5: Sort by similarity (highest first) and take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored[:top_k]

        # Build the output list with updated scores
        reranked: list[SearchResult] = []
        for similarity, idx in top_scored:
            original = results[idx]
            reranked.append(
                SearchResult(
                    id=original.id,
                    source_store=original.source_store,
                    content=original.content,
                    score=similarity,
                    metadata=original.metadata,
                )
            )

        logger.info(
            "tfidf_reranking_completed",
            input_count=len(results),
            output_count=len(reranked),
            top_score=round(reranked[0].score, 4) if reranked else 0.0,
        )

        return reranked
