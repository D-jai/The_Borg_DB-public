# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Interfaces Package
==================

This package contains abstract base classes (ABCs) that define the
contracts between ctxmtg components. Every major subsystem (storage,
extraction, embedding, query, farming, LLM, sync, intake) has an
abstract interface here that concrete implementations must fulfill.

Why interfaces?
    By programming against abstractions rather than concrete classes,
    we can swap implementations without changing the code that uses them.
    For example, SQLite can be replaced with PostgreSQL by implementing
    the same SQLStore interface. This is critical for supporting
    multiple hardware tiers (Pi through server).

All ABCs use Python's abc module (Abstract Base Class). Attempting to
instantiate any ABC directly raises TypeError -- you must create a
concrete subclass that implements every abstract method.

Submodules:
    - storage.py     : SQLStore, VectorStore ABCs
    - extraction.py  : NERProvider, FactExtractor, Summarizer,
                       ExtractionPipeline ABCs
    - embedding.py   : EmbeddingProvider ABC
    - query.py       : QueryInterpreter, QueryPlanner, ResultFuser,
                       Reranker ABCs
    - farming.py     : FarmingStage ABC
    - llm.py         : LLMProvider ABC
    - sync.py        : SyncProvider ABC
    - intake.py      : IntakeGateway ABC

All public ABCs are re-exported here for convenient access:
    from ctxmtg.interfaces import SQLStore, VectorStore
    from ctxmtg.interfaces import ExtractionPipeline
"""

# ---------------------------------------------------------------
# Re-export all public ABCs from submodules so callers can do:
#     from ctxmtg.interfaces import SQLStore, VectorStore
# instead of:
#     from ctxmtg.interfaces.storage import SQLStore, VectorStore
# ---------------------------------------------------------------

# --- Storage interfaces (SQL and vector stores) ---
# --- Embedding interface (text-to-vector conversion) ---
from ctxmtg.interfaces.embedding import EmbeddingProvider

# --- Extraction interfaces (NER, facts, summary, pipeline) ---
from ctxmtg.interfaces.extraction import (
    ExtractionPipeline,
    FactExtractor,
    NERProvider,
    Summarizer,
)

# --- Farming interface (meta-intelligence pattern mining) ---
from ctxmtg.interfaces.farming import FarmingStage

# --- Intake interface (Traffic Cop classification and transformation) ---
from ctxmtg.interfaces.intake import IntakeGateway

# --- LLM interface (local language model generation) ---
from ctxmtg.interfaces.llm import LLMProvider

# --- Query interfaces (interpretation, planning, fusion, reranking) ---
from ctxmtg.interfaces.query import (
    QueryInterpreter,
    QueryPlanner,
    Reranker,
    ResultFuser,
)
from ctxmtg.interfaces.storage import SQLStore, VectorStore

# --- Sync interface (multi-device synchronization) ---
from ctxmtg.interfaces.sync import SyncProvider

# ---------------------------------------------------------------
# __all__ defines the public API of this package.
# Only symbols listed here are exported by `from ctxmtg.interfaces import *`.
# ---------------------------------------------------------------
__all__ = [
    "EmbeddingProvider",
    "ExtractionPipeline",
    "FactExtractor",
    "FarmingStage",
    "IntakeGateway",
    "LLMProvider",
    "NERProvider",
    "QueryInterpreter",
    "QueryPlanner",
    "Reranker",
    "ResultFuser",
    "SQLStore",
    "Summarizer",
    "SyncProvider",
    "VectorStore",
]
