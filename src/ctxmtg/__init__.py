# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
ctxmtg -- Local-First Multi-Agent Knowledge System
====================================================

This is the top-level package for ctxmtg, a system that extracts
structured intelligence from human interactions (meetings, emails,
conversations), stores it in a dual SQL + vector architecture, and
answers hybrid queries that combine precise lookups with semantic
search.

The system is designed to run locally on any hardware tier -- from a
Raspberry Pi (Tier 0) to a full server with PostgreSQL (Tier 4). All
data stays on the user's device unless they explicitly enable sync.

Architecture overview:
    - extraction/   : NER, fact extraction, summarization
    - embedding/    : ONNX-based text embedding and chunking
    - storage/      : SQLite + LanceDB dual-store
    - query/        : Intent classification, SQL/vector planning, fusion
    - farming/      : Meta-intelligence mining (patterns, trends, clusters)
    - llm/          : Optional local LLM integration (Tier 2+)
    - sync/         : Multi-device CRDT synchronization (Phase 4)
    - profile/      : Domain profiles (legal, medical, engineering, etc.)
    - ingestion/    : Orchestrates extraction + embedding for new content
    - intake/       : Input handling and format normalization
    - config/       : Configuration management (env vars, YAML, defaults)
    - health/       : Health monitoring and metrics
    - interfaces/   : Abstract base classes defining contracts
    - models/       : Pydantic data models shared across modules

For full documentation, see the project README and research/ directory.
"""

# ---------------------------------------------------------------
# Package version: follows semantic versioning (major.minor.patch).
# This is the single source of truth for the version number.
# It is referenced by pyproject.toml and CLI --version output.
# ---------------------------------------------------------------
__version__ = "0.1.0"
