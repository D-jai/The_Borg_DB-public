# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Data Models Package
===================

This package contains Pydantic data models shared across all ctxmtg
modules. These models define the shape of data as it flows through
the system -- from raw interaction text through extraction, storage,
querying, and farming.

Using Pydantic models gives us:
    - Automatic validation (type checking at runtime)
    - Serialization to/from JSON and dictionaries
    - Clear documentation of data structures
    - IDE autocompletion and type checking support

Submodules:
    - interaction.py : Interaction, Entity, Fact, EmbeddingMetadata,
                       ExtractionResult, SourceType, EntityType, IntakeAction
    - query.py       : QueryPlan, QueryResult, SearchResult,
                       RetrievalMode, QueryIntent
    - profile.py     : DomainProfile, StageConfig, StageParams, NERConfig,
                       FarmingConfig, EmbeddingConfig, IntakeRule, IntakeConfig
    - farming.py     : FarmingInsight

All public models are re-exported here for convenient access:
    from ctxmtg.models import Interaction, Entity, Fact
    from ctxmtg.models import QueryPlan, QueryResult
    from ctxmtg.models import DomainProfile
    from ctxmtg.models import FarmingInsight
"""

# ---------------------------------------------------------------
# Re-export all public models from submodules so callers can do:
#     from ctxmtg.models import Interaction, Entity, Fact
# instead of:
#     from ctxmtg.models.interaction import Interaction, Entity, Fact
# ---------------------------------------------------------------

# --- Interaction models (core data types) ---
# --- Farming models (meta-intelligence data types) ---
from ctxmtg.models.farming import FarmingInsight
from ctxmtg.models.interaction import (
    EmbeddingMetadata,
    Entity,
    EntityType,
    ExtractionResult,
    Fact,
    IntakeAction,
    Interaction,
    SourceType,
)

# --- Profile models (domain configuration data types) ---
from ctxmtg.models.profile import (
    DomainProfile,
    EmbeddingConfig,
    FarmingConfig,
    IntakeConfig,
    IntakeRule,
    NERConfig,
    StageConfig,
    StageParams,
)

# --- Query models (query pipeline data types) ---
from ctxmtg.models.query import (
    QueryIntent,
    QueryPlan,
    QueryResult,
    RetrievalMode,
    SearchResult,
)

# ---------------------------------------------------------------
# __all__ defines the public API of this package.
# Only symbols listed here are exported by `from ctxmtg.models import *`.
# ---------------------------------------------------------------
__all__ = [
    "DomainProfile",
    "EmbeddingConfig",
    "EmbeddingMetadata",
    "Entity",
    "EntityType",
    "ExtractionResult",
    "Fact",
    "FarmingConfig",
    "FarmingInsight",
    "IntakeAction",
    "IntakeConfig",
    "IntakeRule",
    "Interaction",
    "NERConfig",
    "QueryIntent",
    "QueryPlan",
    "QueryResult",
    "RetrievalMode",
    "SearchResult",
    "SourceType",
    "StageConfig",
    "StageParams",
]
