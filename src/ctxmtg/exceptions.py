# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Custom Exception Hierarchy
==========================

This module defines the exception hierarchy for the ctxmtg package.
All custom exceptions inherit from a single base class (CtxMtgError)
so that callers can catch "any ctxmtg error" with one except clause,
or catch specific error types for fine-grained handling.

Every exception carries an optional error_code of the form
CTXMTG-<MODULE>-<NNN>. The code is included in the exception string
and must also be passed to structlog so operators can look it up in
docs/error_codes_guidance.md for cause and remediation steps.

The hierarchy mirrors the system architecture:
    CtxMtgError (base)
    ├── StorageError      -- Database read/write failures       (CTXMTG-STG-*)
    ├── ExtractionError   -- NER, fact extraction, summarization (CTXMTG-EXT-*)
    ├── EmbeddingError    -- ONNX model loading or inference     (CTXMTG-EMB-*)
    ├── QueryError        -- Query planning, execution, fusion   (CTXMTG-QRY-*)
    ├── FarmingError      -- Farming pipeline stage failures     (CTXMTG-FRM-*)
    ├── ProfileError      -- Domain profile loading/validation   (CTXMTG-PRF-*)
    ├── ConfigError       -- Configuration parsing/validation    (CTXMTG-CFG-*)
    ├── IngestionError    -- Ingestion worker failures           (CTXMTG-ING-*)
    ├── SyncError         -- Multi-device sync failures          (CTXMTG-SYN-*)
    ├── CLIError          -- CLI argument/startup failures       (CTXMTG-CLI-*)
    ├── HealthError       -- Health monitor/metrics failures     (CTXMTG-HLT-*)
    └── IntakeError       -- Input handling/normalization        (CTXMTG-ING-*)

Usage example:
    from ctxmtg.exceptions import StorageError

    try:
        results = await store.query_facts(sql)
    except StorageError as e:
        logger.error("storage_failure", error_code=e.error_code, error=str(e))
    except CtxMtgError as e:
        logger.error("unexpected_ctxmtg_error", error_code=e.error_code, error=str(e))

Depends on: nothing (leaf module)
Used by: every module that raises or catches errors
"""


# ---------------------------------------------------------------
# Base exception: all ctxmtg exceptions inherit from this.
# The error_code attribute is what operators use to look up
# docs/error_codes_guidance.md for cause and remediation.
# ---------------------------------------------------------------
class CtxMtgError(Exception):
    """
    Base exception for all ctxmtg errors.

    Every raise site must pass error_code so the code appears in
    the exception string and in structured log fields.
    Callers access it via exc.error_code for programmatic branching.

    Args:
        message: Human-readable description of the failure.
        error_code: Structured code of the form CTXMTG-<MODULE>-<NNN>.
                    See docs/error_codes_guidance.md for the full list.
    """

    def __init__(self, message: str, error_code: str | None = None) -> None:
        # Store the code so callers can branch on it (e.g., retry logic).
        self.error_code = error_code
        # Prefix the message so the code appears in tracebacks and any
        # log system that only captures the exception string.
        formatted = f"[{error_code}] {message}" if error_code else message
        super().__init__(formatted)


# ---------------------------------------------------------------
# Storage layer errors (SQLite, LanceDB) — CTXMTG-STG-001..015
# ---------------------------------------------------------------
class StorageError(CtxMtgError):
    """
    Database read/write failure (SQLite or LanceDB).
    Error codes: CTXMTG-STG-001 through CTXMTG-STG-015.

    Raised when:
    - A database connection cannot be established         (STG-001)
    - A SQL batch write fails (disk full, lock timeout)   (STG-002, STG-015)
    - A SQL read/query fails                              (STG-003)
    - Schema migration fails                              (STG-004)
    - Transaction lock timeout                            (STG-005)
    - LanceDB init/insert/search/delete fails             (STG-006..009)
    - Embedding metadata or insight write fails           (STG-010, STG-013)
    """


# ---------------------------------------------------------------
# Extraction pipeline errors — CTXMTG-EXT-001..010
# ---------------------------------------------------------------
class ExtractionError(CtxMtgError):
    """
    NER, fact extraction, or summarization failure.
    Error codes: CTXMTG-EXT-001 through CTXMTG-EXT-010.

    Raised when:
    - The spaCy model cannot be loaded                    (EXT-001)
    - NER inference fails on input text                   (EXT-002)
    - A regex pattern fails to compile                    (EXT-003)
    - Dependency parse fails                              (EXT-004)
    - Summarization fails (empty/too-short text)          (EXT-005)
    - Entity deduplication logic errors                   (EXT-006)
    - Chunking fails (empty text or invalid chunk_size)   (EXT-007)
    - A mandatory pipeline stage fails                    (EXT-008)
    """


# ---------------------------------------------------------------
# Embedding errors — CTXMTG-EMB-001..008
# ---------------------------------------------------------------
class EmbeddingError(CtxMtgError):
    """
    ONNX model loading or inference failure.
    Error codes: CTXMTG-EMB-001 through CTXMTG-EMB-008.

    Raised when:
    - The ONNX model file is missing                      (EMB-001)
    - ONNX session creation fails                         (EMB-002)
    - Inference fails (shape mismatch, OOM)               (EMB-003)
    - Text exceeds model max sequence length              (EMB-004)
    - Batch embedding partially fails                     (EMB-005)
    - Model download fails                                (EMB-006)
    - Dimension mismatch vs. stored vectors               (EMB-007)
    - Requested execution provider unavailable            (EMB-008)
    """


# ---------------------------------------------------------------
# Query system errors — CTXMTG-QRY-001..009
# ---------------------------------------------------------------
class QueryError(CtxMtgError):
    """
    Query planning, execution, or fusion failure.
    Error codes: CTXMTG-QRY-001 through CTXMTG-QRY-009.

    Raised when:
    - Intent classification produces no confident match   (QRY-001)
    - No SQL template matches the detected intent         (QRY-002)
    - SQL query execution fails                           (QRY-003)
    - Vector query fails                                  (QRY-004)
    - RRF fusion receives incompatible input              (QRY-005)
    - TF-IDF reranking fails                              (QRY-006)
    - Query log write fails                               (QRY-007)
    - Both stores fail (total query failure)              (QRY-008)
    - Invalid query parameters                            (QRY-009)
    """


# ---------------------------------------------------------------
# Farming pipeline errors — CTXMTG-FRM-001..010
# ---------------------------------------------------------------
class FarmingError(CtxMtgError):
    """
    Farming pipeline stage failure.
    Error codes: CTXMTG-FRM-001 through CTXMTG-FRM-010.

    Raised when:
    - A stage raises an unhandled exception               (FRM-001)
    - Entity analytics computation fails                  (FRM-002)
    - Trend detection has insufficient data               (FRM-003)
    - Clustering convergence fails                        (FRM-004)
    - Topic modeling fails                                (FRM-005)
    - Graph/PageRank computation fails                    (FRM-006)
    - Insight storage fails                               (FRM-007)
    - Scheduler idle-detection fails                      (FRM-008)
    - Feedback loop read fails                            (FRM-009)
    - Checkpoint write fails                              (FRM-010)
    """


# ---------------------------------------------------------------
# Domain profile errors — CTXMTG-PRF-001..006
# ---------------------------------------------------------------
class ProfileError(CtxMtgError):
    """
    Domain profile loading or validation failure.
    Error codes: CTXMTG-PRF-001 through CTXMTG-PRF-006.

    Raised when:
    - Profile YAML file is missing                        (PRF-001)
    - YAML syntax is malformed                            (PRF-002)
    - Pydantic validation of profile fields fails         (PRF-003)
    - Profile version is incompatible                     (PRF-004)
    - An entity type in the profile is unknown            (PRF-005)
    - Profile switch fails                                (PRF-006)
    """


# ---------------------------------------------------------------
# Configuration errors — CTXMTG-CFG-001..005
# ---------------------------------------------------------------
class ConfigError(CtxMtgError):
    """
    Configuration parsing or validation failure.
    Error codes: CTXMTG-CFG-001 through CTXMTG-CFG-005.

    Raised when:
    - Config YAML file is missing                         (CFG-001)
    - Config YAML has invalid syntax                      (CFG-002)
    - Pydantic BaseSettings validation fails              (CFG-003)
    - An env var override has an invalid value            (CFG-004)
    - A required setting has no value and no default      (CFG-005)
    """


# ---------------------------------------------------------------
# Ingestion worker errors — CTXMTG-ING-001..005
# ---------------------------------------------------------------
class IngestionError(CtxMtgError):
    """
    Ingestion worker failure (file I/O, pipeline, or storage step).
    Error codes: CTXMTG-ING-001 through CTXMTG-ING-005.

    Raised when:
    - Worker cannot initialize (stores/pipeline not ready) (ING-001)
    - Input file is missing or unreadable                  (ING-002)
    - Input text is too short to process                   (ING-003)
    - A pipeline stage fails during ingestion              (ING-004)
    - A duplicate interaction is detected (warning level)  (ING-005)
    """


# ---------------------------------------------------------------
# Synchronization errors — CTXMTG-SYN-001..005
# ---------------------------------------------------------------
class SyncError(CtxMtgError):
    """
    Multi-device synchronization failure.
    Error codes: CTXMTG-SYN-001 through CTXMTG-SYN-005.

    Raised when:
    - Remote sync endpoint is unreachable                 (SYN-001)
    - CRDT conflict cannot be resolved                    (SYN-002)
    - Remote rejects the push                             (SYN-003)
    - Pull returns invalid data                           (SYN-004)
    - Vector delta replay fails                           (SYN-005)
    """


# ---------------------------------------------------------------
# CLI errors — CTXMTG-CLI-001..004
# ---------------------------------------------------------------
class CLIError(CtxMtgError):
    """
    CLI argument validation or server startup failure.
    Error codes: CTXMTG-CLI-001 through CTXMTG-CLI-004.

    Raised when:
    - An invalid or missing CLI argument is passed        (CLI-001)
    - Server port is in use or permission denied          (CLI-002)
    - Query server connection fails                       (CLI-003)
    - Graceful shutdown times out                         (CLI-004)
    """


# ---------------------------------------------------------------
# Health monitor errors — CTXMTG-HLT-001..003
# ---------------------------------------------------------------
class HealthError(CtxMtgError):
    """
    Health monitor or metrics write failure.
    Error codes: CTXMTG-HLT-001 through CTXMTG-HLT-003.

    Raised when:
    - Health monitor cannot initialize                    (HLT-001)
    - Metrics JSONL write fails                           (HLT-002)
    - OS resource query (psutil) fails                    (HLT-003)
    """


# ---------------------------------------------------------------
# Intake errors (input handling and format normalization)
# Maps to CTXMTG-ING-* range for file/format failures.
# ---------------------------------------------------------------
class IntakeError(CtxMtgError):
    """
    Input handling and format normalization failure.
    Shares the CTXMTG-ING-* error code range with IngestionError.

    Raised when:
    - An input file format is unsupported or cannot be parsed (ING-002)
    - Content normalization fails (encoding, corrupted data)
    - Input validation rejects malformed or empty content     (ING-003)
    """
