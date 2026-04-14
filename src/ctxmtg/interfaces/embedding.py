# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Embedding Interface ABC
=======================

This module defines the abstract base class for embedding providers --
the components that convert text into numerical vectors (embeddings)
for semantic search.

Embeddings are dense vector representations of text that capture
semantic meaning. Texts with similar meaning produce vectors that
are close together in the embedding space. This enables "fuzzy"
search: finding content related to a query even when the exact
words don't match.

Phase 1 uses ONNX Runtime with the all-MiniLM-L6-v2 model (384
dimensions, fast on CPU). Phase 2+ may use larger models for
better quality, or GPU-accelerated models for speed.

Depends on:
    - abc (Python's Abstract Base Class machinery)

Used by:
    - ctxmtg.embedding.onnx_embedder (implements EmbeddingProvider)
    - ctxmtg.ingestion.worker (embeds text chunks after extraction)
    - ctxmtg.query.executor (embeds query text for vector search)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# =====================================================================
# EmbeddingProvider ABC -- Text-to-Vector Interface
# =====================================================================


class EmbeddingProvider(ABC):
    """
    Embedding model provider.

    Converts text into numerical vectors (embeddings) for semantic
    search. The embedding model maps text strings to points in a
    high-dimensional space where semantically similar texts are close
    together.

    Implementations handle model loading, tokenization, and inference.
    The provider interface is synchronous because embedding computation
    is CPU-bound (no I/O to await), and the ingestion pipeline that
    calls it runs synchronously.

    Key characteristics reported by the provider:
    - dimensions: the size of each embedding vector (e.g., 384)
    - model_name: which model is loaded (e.g., "all-MiniLM-L6-v2")
    - model_version: the specific version string of the model

    Usage:
        embedder = ONNXEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        vectors = embedder.embed(["Hello world", "Goodbye world"])
        # vectors[0] and vectors[1] are each 384-dimensional float lists
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts. Returns list of vectors.

        Converts multiple text strings into their embedding vectors
        in a single batch call. Batch processing is significantly
        faster than embedding one text at a time because it amortizes
        model overhead and can use hardware parallelism.

        Args:
            texts: List of text strings to embed. Each text is
                   tokenized and encoded independently. Empty
                   list returns empty list.

        Returns:
            A list of embedding vectors, one per input text.
            Each vector is a list of floats with length equal
            to get_dimensions(). The order matches the input order.
        """
        ...

    @abstractmethod
    def embed_single(self, text: str) -> list[float]:
        """
        Embed a single text. Returns one vector.

        Convenience method for embedding a single text string.
        Equivalent to embed([text])[0] but may be optimized for
        single-item inference in some implementations.

        Args:
            text: A single text string to embed.

        Returns:
            The embedding vector as a list of floats with length
            equal to get_dimensions().
        """
        ...

    @abstractmethod
    def get_dimensions(self) -> int:
        """
        Return the dimensionality of embeddings this model produces.

        Reports how many dimensions (floats) each embedding vector
        has. This is a fixed property of the loaded model:
        - all-MiniLM-L6-v2: 384 dimensions
        - all-mpnet-base-v2: 768 dimensions

        Used by the vector store to configure its schema and by
        the embedding metadata to record which model produced each
        embedding.

        Returns:
            The number of dimensions in each embedding vector (e.g., 384).
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the model name (e.g., 'all-MiniLM-L6-v2').

        Reports the name of the currently loaded embedding model.
        This is stored in EmbeddingMetadata records so the system
        knows which model produced each embedding and can re-embed
        content when models change.

        Returns:
            The model name as a string (e.g., "all-MiniLM-L6-v2").
        """
        ...

    @abstractmethod
    def get_model_version(self) -> str:
        """
        Return the model version string.

        Reports the version of the loaded model. Combined with the
        model name, this uniquely identifies the model and enables
        incremental re-embedding when the model version changes.

        Returns:
            The model version as a string (e.g., "1.0.0" or "2024.01").
        """
        ...
