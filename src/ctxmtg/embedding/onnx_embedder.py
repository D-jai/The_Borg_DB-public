# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
ONNX Embedding Provider
========================

This module implements the EmbeddingProvider interface using ONNX
Runtime for model inference and HuggingFace tokenizers for text
tokenisation. It converts text strings into dense embedding vectors
that capture semantic meaning -- texts with similar meaning produce
vectors that are close together in the embedding space.

The default model is all-MiniLM-L6-v2, which:
    - Produces 384-dimensional vectors
    - Runs efficiently on CPU (including Raspberry Pi)
    - Has good quality for general-purpose semantic search
    - Is ~80 MB in ONNX format (manageable for edge devices)

Why ONNX Runtime instead of PyTorch / sentence-transformers?
    - ONNX Runtime has a much smaller footprint (~50 MB vs ~2 GB)
    - It supports CPU, CUDA, CoreML, and DirectML execution providers
    - It can load quantised (INT8) models for even faster inference
    - No GPU driver requirements for CPU-only deployment
(See research/round-1/03-embedding-and-vectorization.md for analysis.)

Model lifecycle:
    1. On first use, the model is downloaded from HuggingFace Hub
       (or loaded from a local cache).
    2. The ONNX session is created once and reused for all batches
       (session creation is expensive, inference is cheap).
    3. The tokeniser is loaded from the same model directory.

Depends on:
    - onnxruntime (ONNX model inference engine)
    - transformers (HuggingFace tokeniser loading)
    - numpy (array operations for batching and normalisation)
    - ctxmtg.interfaces.embedding (EmbeddingProvider ABC)
    - ctxmtg.exceptions (EmbeddingError for error reporting)

Used by:
    - ctxmtg.ingestion.worker (embeds text chunks after extraction)
    - ctxmtg.query.executor (embeds query text for vector search)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog

from ctxmtg.exceptions import EmbeddingError
from ctxmtg.interfaces.embedding import EmbeddingProvider

# ---------------------------------------------------------------
# Module-level logger -- logs model metadata, not content.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.embedding.onnx_embedder")

# ---------------------------------------------------------------
# Default model configuration.
# all-MiniLM-L6-v2 is the best balance of quality vs. speed for
# edge deployment (384 dims, ~80 MB ONNX, fast on CPU).
# ---------------------------------------------------------------
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_VERSION = "1.0.0"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_SEQ_LENGTH = 256


def _detect_execution_providers() -> list[str]:
    """
    Detect which ONNX Runtime execution providers are available.

    ONNX Runtime supports multiple backends. We try the fastest
    first and fall back to CPU:
        1. CUDAExecutionProvider -- NVIDIA GPU (fastest, if available)
        2. CoreMLExecutionProvider -- Apple Silicon (macOS/iOS)
        3. CPUExecutionProvider -- always available (slowest)

    Returns:
        Ordered list of available provider names (best first).
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        logger.error(
            "onnxruntime_not_installed",
            error_code="CTXMTG-EMB-002",
            error=str(exc),
        )
        raise EmbeddingError(
            "onnxruntime is not installed. Run: pip install onnxruntime",
            error_code="CTXMTG-EMB-002",
        ) from exc

    available = ort.get_available_providers()
    # Build a preference-ordered list of providers to use.
    # ONNX Runtime tries them in order and uses the first one
    # that actually works for the loaded model.
    preferred_order = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    # Keep only providers that are actually available on this system
    providers = [p for p in preferred_order if p in available]

    # Always ensure CPU is present as the fallback
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    logger.info("execution_providers_detected", providers=providers)
    return providers


class ONNXEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using ONNX Runtime for inference.

    Loads a sentence-transformer model in ONNX format, tokenises
    input text with the model's HuggingFace tokeniser, and runs
    batched inference through ONNX Runtime. The result is a list
    of embedding vectors (one per input text) normalised to unit
    length for cosine similarity search.

    The provider detects the best available execution provider
    (CUDA > CoreML > CPU) and uses it automatically. For INT8
    quantised models, simply point model_name_or_path to the
    quantised model directory.

    Usage:
        provider = ONNXEmbeddingProvider(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectors = provider.embed(["Hello world", "Semantic search rocks"])
        # vectors is a list of 2 lists, each with 384 floats

    Args:
        model_name_or_path: HuggingFace model ID (downloaded on first use)
                            or local path to an ONNX model directory.
        batch_size: Number of texts to process in each inference batch.
                    Default 32 is the sweet spot per Round 2 research.
        max_seq_length: Maximum token sequence length. Texts exceeding
                        this are truncated. Default 256 matches the
                        default chunk_size from EmbeddingConfig.
        model_version: Version string for metadata tracking. Stored in
                       EmbeddingMetadata so the system knows which model
                       produced each embedding.
        execution_providers: Override the auto-detected providers. Pass
                             None (default) to auto-detect.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        model_version: str = DEFAULT_MODEL_VERSION,
        execution_providers: list[str] | None = None,
    ) -> None:
        """
        Configure the embedding provider (lazy initialization).

        The ONNX session and tokeniser are NOT loaded here. They are
        loaded lazily on the first call to embed() or embed_single().
        This keeps construction fast and allows the provider to be
        created even if the model isn't downloaded yet.
        """
        self._model_name_or_path = model_name_or_path
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._model_version = model_version
        self._execution_providers = execution_providers

        # These are set by _ensure_loaded() on first use
        self._session: Any = None  # onnxruntime.InferenceSession
        self._tokenizer: Any = None  # transformers.AutoTokenizer
        self._dimensions: int | None = None  # vector dimensionality
        self._model_display_name: str = self._extract_model_name(model_name_or_path)

    # =================================================================
    # EmbeddingProvider interface implementation
    # =================================================================

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts. Returns list of vectors.

        Processes texts in batches of self._batch_size for efficient
        ONNX inference. Each text is tokenised, padded/truncated to
        max_seq_length, and run through the model. The output vectors
        are L2-normalised for cosine similarity.

        Args:
            texts: List of text strings to embed. Empty list returns
                   empty list.

        Returns:
            A list of embedding vectors (list of floats), one per
            input text. Each vector has get_dimensions() elements.

        Raises:
            EmbeddingError: If model loading or inference fails.
        """
        if not texts:
            return []

        # Ensure model and tokeniser are loaded
        self._ensure_loaded()

        all_embeddings: list[list[float]] = []

        # Process in batches to avoid OOM on large inputs.
        # Batch size 32 is the sweet spot per Round 2 research.
        for i in range(0, len(texts), self._batch_size):
            batch_texts = texts[i : i + self._batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

        logger.debug(
            "batch_embedded",
            text_count=len(texts),
            batch_count=(len(texts) + self._batch_size - 1) // self._batch_size,
        )

        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """
        Embed a single text. Returns one vector.

        Convenience wrapper around embed() for single-text use cases
        like embedding a user query for vector search.

        Args:
            text: A single text string to embed.

        Returns:
            The embedding vector as a list of floats.

        Raises:
            EmbeddingError: If model loading or inference fails.
        """
        results = self.embed([text])
        return results[0]

    def get_dimensions(self) -> int:
        """
        Return the dimensionality of embeddings this model produces.

        For all-MiniLM-L6-v2 this is 384. The value is determined
        by running a dummy inference on first load.

        Returns:
            Number of dimensions per embedding vector.

        Raises:
            EmbeddingError: If the model hasn't been loaded yet and
                            loading fails.
        """
        self._ensure_loaded()
        assert self._dimensions is not None
        return self._dimensions

    def get_model_name(self) -> str:
        """
        Return the model name (e.g., 'all-MiniLM-L6-v2').

        This is the short display name extracted from the full model
        path or HuggingFace ID. Stored in EmbeddingMetadata records.

        Returns:
            The model name as a string.
        """
        return self._model_display_name

    def get_model_version(self) -> str:
        """
        Return the model version string.

        Returns:
            The model version configured at construction time.
        """
        return self._model_version

    # =================================================================
    # Lazy initialisation
    # =================================================================

    def _ensure_loaded(self) -> None:
        """
        Load the ONNX model and tokeniser if not already loaded.

        This is called automatically on the first embed() call.
        It downloads the model from HuggingFace (if not cached),
        creates the ONNX Runtime session, and loads the tokeniser.

        Raises:
            EmbeddingError: If the model cannot be loaded.
        """
        if self._session is not None:
            return  # Already loaded

        try:
            self._load_model()
        except EmbeddingError:
            raise
        except Exception as exc:
            logger.error(
                "embedding_model_load_failed",
                error_code="CTXMTG-EMB-002",
                model=self._model_name_or_path,
                error=str(exc),
            )
            raise EmbeddingError(
                f"Failed to load embedding model '{self._model_name_or_path}': {exc}",
                error_code="CTXMTG-EMB-002",
            ) from exc

    def _load_model(self) -> None:
        """
        Actually load the ONNX session and tokeniser.

        Steps:
        1. Resolve the model path (download from HuggingFace if needed).
        2. Load the tokeniser from the model directory.
        3. Create the ONNX Runtime inference session.
        4. Run a dummy inference to determine the output dimensionality.
        """
        import onnxruntime as ort
        from transformers import AutoTokenizer

        # Determine execution providers
        if self._execution_providers is None:
            providers = _detect_execution_providers()
        else:
            providers = self._execution_providers

        # Resolve the ONNX model file path
        onnx_path = self._resolve_onnx_path(self._model_name_or_path)

        logger.info(
            "loading_onnx_model",
            model=self._model_name_or_path,
            onnx_path=str(onnx_path),
            providers=providers,
        )

        # Load the tokeniser from the same model directory or HF hub.
        # The tokeniser handles text → token IDs conversion, including
        # padding, truncation, and special token insertion.
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)

        # Create the ONNX Runtime inference session.
        # The session loads the model graph and optimises it for the
        # selected execution provider. This is expensive (~1-2 sec)
        # but only happens once.
        session_options = ort.SessionOptions()
        # Enable graph optimisation for faster inference
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=providers,
        )

        # Determine the output dimensionality by running a single dummy
        # text through the model. This is more reliable than reading
        # the ONNX graph metadata (which may not include dimension info).
        dummy_embedding = self._embed_batch(["test"])
        self._dimensions = len(dummy_embedding[0])

        logger.info(
            "onnx_model_loaded",
            model=self._model_display_name,
            dimensions=self._dimensions,
            provider_used=self._session.get_providers()[0],
        )

    def _resolve_onnx_path(self, model_name_or_path: str) -> Path:
        """
        Resolve the path to the ONNX model file.

        If model_name_or_path is a local directory containing an ONNX
        file, use it directly. Otherwise, download from HuggingFace
        Hub and look for the ONNX file in the cached directory.

        For sentence-transformers models, the ONNX file is typically
        named 'model.onnx' or 'onnx/model.onnx'.

        Args:
            model_name_or_path: HuggingFace model ID or local path.

        Returns:
            Path to the .onnx file.

        Raises:
            EmbeddingError: If no ONNX file can be found.
        """
        local_path = Path(model_name_or_path)

        # Case 1: Direct path to an ONNX file
        if local_path.is_file() and local_path.suffix == ".onnx":
            return local_path

        # Case 2: Local directory -- look for model.onnx inside
        if local_path.is_dir():
            onnx_file = self._find_onnx_in_dir(local_path)
            if onnx_file is not None:
                return onnx_file

        # Case 3: HuggingFace model ID -- download and find ONNX
        return self._download_onnx_model(model_name_or_path)

    def _find_onnx_in_dir(self, directory: Path) -> Path | None:
        """
        Search a directory for an ONNX model file.

        Looks for common ONNX file names in the model directory.
        Prefers INT8 quantised models (smaller, faster) over
        full-precision models.

        Args:
            directory: Path to search for .onnx files.

        Returns:
            Path to the best ONNX file found, or None.
        """
        # Preference order: INT8 quantised > full precision
        candidates = [
            "model_quantized.onnx",  # INT8 quantised (preferred)
            "model_optimized.onnx",  # Optimised full-precision
            "model.onnx",  # Standard full-precision
            "onnx/model.onnx",  # Some models nest in onnx/ subdir
        ]

        for candidate in candidates:
            onnx_path = directory / candidate
            if onnx_path.is_file():
                return onnx_path

        # Fallback: find any .onnx file in the directory tree
        onnx_files = list(directory.rglob("*.onnx"))
        if onnx_files:
            return onnx_files[0]

        return None

    def _download_onnx_model(self, model_id: str) -> Path:
        """
        Download an ONNX model from HuggingFace Hub.

        Uses the huggingface_hub library (installed with transformers)
        to download the model files to a local cache directory. The
        cached files are reused on subsequent calls.

        Args:
            model_id: HuggingFace model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2").

        Returns:
            Path to the downloaded .onnx file.

        Raises:
            EmbeddingError: If the download fails or no ONNX file is found.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            logger.error(
                "huggingface_hub_not_installed",
                error_code="CTXMTG-EMB-006",
                error=str(exc),
            )
            raise EmbeddingError(
                "huggingface_hub is not installed. Run: pip install transformers huggingface_hub",
                error_code="CTXMTG-EMB-006",
            ) from exc

        try:
            # Download the entire model snapshot (tokeniser + ONNX)
            # to the default HF cache (~/.cache/huggingface/hub/).
            # Only .onnx files and tokeniser files are needed.
            cache_dir = snapshot_download(
                repo_id=model_id,
                allow_patterns=[
                    "*.onnx",
                    "*.json",
                    "*.txt",
                    "tokenizer*",
                    "vocab*",
                    "special_tokens_map*",
                    "config*",
                ],
            )
        except Exception as exc:
            logger.error(
                "model_download_failed",
                error_code="CTXMTG-EMB-006",
                model_id=model_id,
                error=str(exc),
            )
            raise EmbeddingError(
                f"Failed to download model '{model_id}' from HuggingFace: {exc}",
                error_code="CTXMTG-EMB-006",
            ) from exc

        # Find the ONNX file in the downloaded cache
        cache_path = Path(cache_dir)
        onnx_file = self._find_onnx_in_dir(cache_path)

        if onnx_file is None:
            logger.error(
                "onnx_file_not_found",
                error_code="CTXMTG-EMB-001",
                model_id=model_id,
                cache_dir=str(cache_dir),
            )
            raise EmbeddingError(
                f"No ONNX model file found in '{model_id}'. "
                f"This model may not have an ONNX export. "
                f"Try using a sentence-transformers model that includes "
                f"ONNX files, or export manually with optimum.",
                error_code="CTXMTG-EMB-001",
            )

        return onnx_file

    # =================================================================
    # Batched inference
    # =================================================================

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Run ONNX inference on a batch of texts.

        Steps:
        1. Tokenise all texts (pad to same length, truncate if needed).
        2. Convert token IDs to numpy arrays.
        3. Run the ONNX model.
        4. Pool the token-level embeddings to sentence-level (mean pooling).
        5. L2-normalise the vectors for cosine similarity.

        Args:
            texts: List of text strings (already batched by the caller).

        Returns:
            List of embedding vectors (list of floats), one per text.

        Raises:
            EmbeddingError: If inference fails.
        """
        try:
            # Tokenise the batch: converts texts to token IDs with
            # padding and truncation applied uniformly.
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self._max_seq_length,
                return_tensors="np",  # Return numpy arrays directly
            )

            # Build the ONNX input dict. Most sentence-transformer
            # models expect input_ids, attention_mask, and optionally
            # token_type_ids.
            onnx_inputs: dict[str, Any] = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }

            # Some models also expect token_type_ids (e.g., for BERT-
            # style models with segment embeddings). Include it if
            # the tokeniser produced it AND the model expects it.
            model_input_names = [inp.name for inp in self._session.get_inputs()]
            if "token_type_ids" in encoded and "token_type_ids" in model_input_names:
                onnx_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

            # Run inference through the ONNX model.
            # The output is typically a tuple: (token_embeddings, ...).
            # We take the first output (token-level embeddings) and
            # apply mean pooling to get sentence-level embeddings.
            outputs = self._session.run(None, onnx_inputs)

            # outputs[0] shape: (batch_size, seq_length, hidden_dim)
            token_embeddings = outputs[0]

            # Mean pooling: average the token embeddings, weighted by
            # the attention mask (so padding tokens don't contribute).
            attention_mask = encoded["attention_mask"]
            sentence_embeddings = self._mean_pooling(token_embeddings, attention_mask)

            # L2-normalise so that cosine similarity = dot product.
            # This makes similarity search faster and more numerically
            # stable.
            normalised = self._l2_normalize(sentence_embeddings)

            # Convert from numpy to plain Python lists
            result: list[list[float]] = normalised.tolist()
            return result

        except Exception as exc:
            logger.error(
                "onnx_inference_failed",
                error_code="CTXMTG-EMB-003",
                batch_size=len(texts),
                error=str(exc),
            )
            raise EmbeddingError(
                f"ONNX inference failed on batch of {len(texts)} texts: {exc}",
                error_code="CTXMTG-EMB-003",
            ) from exc

    @staticmethod
    def _mean_pooling(
        token_embeddings: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Pool token-level embeddings into sentence-level embeddings.

        Computes the mean of all non-padding token embeddings. Padding
        tokens (where attention_mask == 0) are excluded from the mean
        so they don't dilute the sentence representation.

        This is the standard pooling strategy for sentence-transformers
        models. CLS pooling is another option but mean pooling usually
        gives better results for similarity search.

        Args:
            token_embeddings: Shape (batch_size, seq_length, hidden_dim).
            attention_mask:   Shape (batch_size, seq_length), 1 for real
                              tokens, 0 for padding.

        Returns:
            Sentence embeddings, shape (batch_size, hidden_dim).
        """
        # Expand attention_mask to match embedding dimensions:
        # (batch, seq_len) → (batch, seq_len, 1) for broadcasting
        mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)

        # Zero out padding token embeddings, then sum
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)

        # Count the number of real tokens per sentence (avoid div by 0)
        sum_mask = np.sum(mask_expanded, axis=1)
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)

        # Mean = sum / count
        result: np.ndarray = sum_embeddings / sum_mask
        return result

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """
        L2-normalise each vector to unit length.

        After normalisation, cosine similarity between any two vectors
        equals their dot product: cos(a, b) = a · b / (|a| * |b|),
        and since |a| = |b| = 1 after normalisation, cos(a, b) = a · b.
        This makes similarity computation cheaper.

        Args:
            vectors: Shape (batch_size, hidden_dim).

        Returns:
            Normalised vectors, same shape. Each row has L2 norm = 1.
        """
        # Compute L2 norm for each vector (row)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        # Avoid division by zero for all-zero vectors
        norms = np.clip(norms, a_min=1e-12, a_max=None)

        result: np.ndarray = vectors / norms
        return result

    @staticmethod
    def _extract_model_name(model_name_or_path: str) -> str:
        """
        Extract a short display name from a model path or HF ID.

        Examples:
            "sentence-transformers/all-MiniLM-L6-v2" → "all-MiniLM-L6-v2"
            "/home/user/models/my-model" → "my-model"
            "all-MiniLM-L6-v2" → "all-MiniLM-L6-v2"

        Args:
            model_name_or_path: Full model identifier or path.

        Returns:
            Short model name suitable for display and metadata.
        """
        # Split on "/" and take the last part
        parts = model_name_or_path.rstrip("/").split("/")
        return parts[-1] if parts else model_name_or_path
