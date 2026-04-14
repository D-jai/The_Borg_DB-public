# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Storage Package
===============

This package implements the dual-store architecture: SQLite for
structured data (entities, facts, metadata) and LanceDB for vector
embeddings (semantic search). Together, these two stores enable
hybrid queries that combine precise SQL lookups with semantic
similarity search.

This is the persistence layer -- everything that the system "knows"
is stored here. The extraction pipeline writes data in, the query
system reads data out, and the farming pipeline reads accumulated
data for pattern mining.

Why dual stores instead of one?
    SQL is great for precise queries ("meetings this week") but
    can't do semantic similarity. Vectors are great for semantic
    search ("discussions about scaling") but can't do aggregation.
    Combining both gives the best of both worlds. See
    research/round-1/02-dual-store-architecture.md for the full
    analysis.

Submodules:
    - sqlite.py       : SQLite implementation of SQLStore
    - lancedb_store.py: LanceDB implementation of VectorStore
    - schema.py       : DDL definitions and migration logic
    - id_gen.py       : UUIDv5 deterministic ID generation
"""

from ctxmtg.storage.id_gen import (
    CTXMTG_NAMESPACE,
    generate_embedding_id,
    generate_entity_id,
    generate_fact_id,
    generate_insight_id,
    generate_interaction_id,
)
from ctxmtg.storage.schema import (
    SCHEMA_VERSION,
    apply_pragmas,
    apply_schema,
    get_schema_version,
    migrate,
)
from ctxmtg.storage.sqlite import SQLiteStore

__all__ = [
    "CTXMTG_NAMESPACE",
    "SCHEMA_VERSION",
    "SQLiteStore",
    "apply_pragmas",
    "apply_schema",
    "generate_embedding_id",
    "generate_entity_id",
    "generate_fact_id",
    "generate_insight_id",
    "generate_interaction_id",
    "get_schema_version",
    "migrate",
]
