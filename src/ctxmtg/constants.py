# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Global Constants
================

This module defines project-wide constants used across all ctxmtg
modules. Centralizing constants here avoids magic numbers scattered
throughout the codebase and makes it easy to tune system behavior
from one place.

Constants are grouped by subsystem:
    - Storage: database paths, batch sizes, limits
    - Extraction: NER confidence thresholds, model names
    - Embedding: model identifiers, vector dimensions, chunk sizes
    - Query: fusion parameters, result limits
    - Farming: scheduling intervals, minimum data thresholds
    - General: application metadata, default paths

Depends on: nothing (leaf module)
Used by: virtually every other module in ctxmtg
"""

# ---------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------

# The application name, used in CLI output, log prefixes, and
# configuration file paths (~/.ctxmtg/).
APP_NAME = "ctxmtg"

# Default directory for user data (databases, vectors, profiles).
# On first run, the system creates this directory if it doesn't exist.
DEFAULT_DATA_DIR = "~/.ctxmtg"

# ---------------------------------------------------------------
# Storage constants
# ---------------------------------------------------------------

# Default file path for the SQLite knowledge database.
# This stores all structured data: interactions, entities, facts.
DEFAULT_DB_PATH = "~/.ctxmtg/knowledge.db"

# Default directory for the LanceDB vector store.
# This stores embedding vectors for semantic search.
DEFAULT_VECTOR_PATH = "~/.ctxmtg/vectors"

# Maximum number of records to insert in a single database
# transaction. Larger batches are faster but use more memory.
# 500 is a safe default for Raspberry Pi (512MB RAM).
STORAGE_BATCH_SIZE = 500

# ---------------------------------------------------------------
# Extraction constants
# ---------------------------------------------------------------

# Minimum confidence score for an extracted entity to be stored.
# Entities below this threshold are discarded as noise.
# 0.7 was chosen based on spaCy benchmarks in our evaluation suite
# (see tests/evaluation/test_ner_quality.py for the analysis).
NER_CONFIDENCE_THRESHOLD = 0.7

# Default spaCy model for Named Entity Recognition.
# "en_core_web_sm" is the smallest English model (~12MB).
# Larger models (en_core_web_md, en_core_web_lg) give better
# accuracy but require more RAM -- see research/round-1/03.
DEFAULT_SPACY_MODEL = "en_core_web_sm"

# ---------------------------------------------------------------
# Embedding constants
# ---------------------------------------------------------------

# Default ONNX embedding model name.
# all-MiniLM-L6-v2 produces 384-dimensional vectors and runs
# well on CPU (including Raspberry Pi). It's the best balance
# of quality vs. speed for edge deployment.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Dimensionality of the default embedding model's output vectors.
# Must match the model -- if you change the model, update this.
DEFAULT_EMBEDDING_DIM = 384

# Maximum number of tokens per text chunk before embedding.
# Longer chunks lose semantic specificity; shorter chunks lose
# context. 256 tokens ≈ 200 words, a good balance.
DEFAULT_CHUNK_SIZE = 256

# Number of tokens to overlap between consecutive chunks.
# Overlap prevents information loss at chunk boundaries.
# 32 tokens ≈ 25 words of context carried between chunks.
DEFAULT_CHUNK_OVERLAP = 32

# ---------------------------------------------------------------
# Query constants
# ---------------------------------------------------------------

# Reciprocal Rank Fusion (RRF) smoothing parameter.
# Controls how much weight high-ranked results get vs lower-ranked.
# k=60 is the standard value from the original RRF paper
# (Cormack et al., 2009). See research/round-1/04 for analysis.
RRF_K = 60

# Default number of results returned from a query.
# Users can override this per query, but this is the default.
DEFAULT_TOP_K = 10

# Maximum number of results the system will consider during
# fusion, before trimming to top_k. A wider candidate pool
# gives better fusion quality but costs more compute.
MAX_CANDIDATE_POOL = 100

# ---------------------------------------------------------------
# Farming constants
# ---------------------------------------------------------------

# Minimum number of stored interactions before farming runs.
# Running farming on too few interactions produces noise.
# 50 interactions ≈ 2-3 weeks of regular meeting usage.
MIN_INTERACTIONS_FOR_FARMING = 50

# Default interval between automatic farming cycles (in seconds).
# 3600 = 1 hour. Farming runs during idle periods only.
DEFAULT_FARMING_INTERVAL = 3600

# ---------------------------------------------------------------
# LLM constants (Tier 2+)
# ---------------------------------------------------------------

# Default context window size for local LLM inference.
# 4096 tokens is safe for most 7B models on 8GB RAM.
DEFAULT_LLM_CONTEXT_SIZE = 4096

# ---------------------------------------------------------------
# Domain profile constants
# ---------------------------------------------------------------

# Default domain profile name, loaded when no profile is specified.
DEFAULT_PROFILE_NAME = "general"

# Default directory where domain profile YAML files are stored.
DEFAULT_PROFILE_DIR = "~/.ctxmtg/profiles"
