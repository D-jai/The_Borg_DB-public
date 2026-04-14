# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Storage Interface ABCs
======================

This module defines the abstract base classes for the dual-store
architecture: SQLStore for structured (SQL) data and VectorStore
for semantic (embedding) data. Together, these two stores form the
backbone of ctxmtg's knowledge storage system.

SQLStore handles precise lookups and structured queries -- "how many
meetings this week?" or "what facts mention Alice?". VectorStore
handles semantic similarity search -- "discussions about scaling
concerns" or "topics related to security".

Every method in these ABCs is abstract: concrete implementations
(SQLiteStore, LanceDBStore, PostgreSQLStore, etc.) MUST implement
every method. The rest of the system codes against these interfaces,
never against concrete implementations. This is what allows swapping
SQLite for PostgreSQL without touching the query engine.

Depends on:
    - abc (Python's Abstract Base Class machinery)
    - ctxmtg.models.interaction (Interaction, Entity, Fact, EmbeddingMetadata)
    - ctxmtg.models.query (SearchResult)
    - ctxmtg.models.farming (FarmingInsight)

Used by:
    - ctxmtg.storage.sqlite (implements SQLStore)
    - ctxmtg.storage.lancedb_store (implements VectorStore)
    - ctxmtg.ingestion.worker (writes to both stores)
    - ctxmtg.query.executor (reads from both stores)
    - ctxmtg.farming.pipeline (reads from both stores)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# ---------------------------------------------------------------
# Import the data models that flow through the storage interfaces.
# These are the shared "lingua franca" of the system -- every
# component reads and writes these same structures.
# ---------------------------------------------------------------
from ctxmtg.models.farming import FarmingInsight
from ctxmtg.models.interaction import (
    EmbeddingMetadata,
    Entity,
    Fact,
    Interaction,
)
from ctxmtg.models.query import SearchResult

# =====================================================================
# SQLStore ABC -- Structured Data Store Interface
# =====================================================================


class SQLStore(ABC):
    """
    Abstract interface for the structured (SQL) data store.

    Implementations: SQLiteStore (Phase 1), PostgreSQLStore (Phase 4).
    This interface defines every operation the rest of the system
    can perform on the SQL store. All methods are async because
    database I/O should not block the event loop.

    The SQL store manages:
    - Interactions (meetings, emails, documents)
    - Entities (people, orgs, projects, topics extracted from interactions)
    - Facts (subject-predicate-object triples linking entities)
    - Embedding metadata (links SQL records to their vector store entries)
    - Farming insights (patterns discovered by the farming pipeline)

    Usage:
        store = SQLiteStore(db_path="/path/to/knowledge.db")
        await store.initialize()   # Create tables if needed
        await store.store_interaction(interaction)
        results = await store.get_entities(entity_type="person")
        await store.close()
    """

    # -----------------------------------------------------------------
    # Lifecycle methods: setup and teardown for the database connection.
    # Every implementation must handle creating tables on first run
    # and closing connections cleanly on shutdown.
    # -----------------------------------------------------------------

    @abstractmethod
    async def initialize(self) -> None:
        """
        Create tables and indexes if they don't exist.

        Called once at startup. Implementations should:
        1. Open/verify the database connection
        2. Run DDL to create tables if they're missing
        3. Create indexes for common query patterns
        4. Set database pragmas (e.g., WAL mode for SQLite)

        This method is idempotent -- calling it multiple times
        on an existing database should be safe (CREATE IF NOT EXISTS).
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """
        Close the database connection cleanly.

        Called on shutdown. Implementations should:
        1. Flush any pending writes
        2. Close the database connection
        3. Release any held resources (locks, file handles)
        """
        ...

    # -----------------------------------------------------------------
    # Interaction CRUD: create, read, list interactions.
    # Interactions are the fundamental data unit -- every piece of
    # content the user ingests becomes an Interaction record.
    # -----------------------------------------------------------------

    @abstractmethod
    async def store_interaction(self, interaction: Interaction) -> str:
        """
        Store an interaction. Returns the interaction ID.

        Persists the interaction to the database. If an interaction
        with the same ID already exists, implementations may choose
        to update it (upsert) or skip it (idempotent insert).

        Args:
            interaction: The Interaction object to store.

        Returns:
            The interaction's ID (same as interaction.id).
        """
        ...

    @abstractmethod
    async def get_interaction(self, interaction_id: str) -> Interaction | None:
        """
        Retrieve an interaction by ID. Returns None if not found.

        Looks up a single interaction by its unique identifier.
        Returns the full Interaction object including metadata,
        participants, and content.

        Args:
            interaction_id: The unique ID of the interaction to retrieve.

        Returns:
            The Interaction object, or None if no interaction exists
            with that ID.
        """
        ...

    @abstractmethod
    async def list_interactions(
        self,
        source_type: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Interaction]:
        """
        List interactions with optional filters.

        Retrieves a paginated list of interactions, optionally filtered
        by source type and/or time range. Results are ordered by
        created_at descending (most recent first).

        Args:
            source_type: Filter by source type (e.g., "meeting", "email").
                         None means all source types.
            since: ISO datetime string -- only interactions created after
                   this time. None means no lower time bound.
            until: ISO datetime string -- only interactions created before
                   this time. None means no upper time bound.
            limit: Maximum number of results to return (default 100).
            offset: Number of results to skip (for pagination).

        Returns:
            A list of Interaction objects matching the filters.
        """
        ...

    # -----------------------------------------------------------------
    # Entity CRUD: batch-store and query entities.
    # Entities are people, orgs, projects, topics, etc. extracted
    # from interactions by the NER pipeline.
    # -----------------------------------------------------------------

    @abstractmethod
    async def store_entities(self, entities: list[Entity]) -> int:
        """
        Batch-store entities. Returns count of inserted (non-duplicate) entities.

        Takes a list of Entity objects and inserts them into the entities
        table. Duplicate entities (same ID) are silently skipped.
        All inserts happen in a single transaction for performance.

        Args:
            entities: List of Entity objects to store.

        Returns:
            Number of entities actually inserted (excludes duplicates).
        """
        ...

    @abstractmethod
    async def get_entities(
        self,
        interaction_id: str | None = None,
        entity_type: str | None = None,
        name_like: str | None = None,
        limit: int = 100,
    ) -> list[Entity]:
        """
        Query entities with optional filters.

        Retrieves entities matching the given criteria. All name matching
        is case-insensitive (COLLATE NOCASE). Filters are ANDed together.

        Args:
            interaction_id: Filter by source interaction. None means all.
            entity_type: Filter by entity type (e.g., "person"). None means all.
            name_like: Partial name match (case-insensitive). None means all.
            limit: Maximum number of results to return (default 100).

        Returns:
            A list of Entity objects matching the filters.
        """
        ...

    # -----------------------------------------------------------------
    # Fact CRUD: batch-store and query subject-predicate-object triples.
    # Facts are the structured knowledge atoms extracted from text.
    # -----------------------------------------------------------------

    @abstractmethod
    async def store_facts(self, facts: list[Fact]) -> int:
        """
        Batch-store facts. Returns count of inserted facts.

        Takes a list of Fact objects (subject-predicate-object triples)
        and inserts them into the facts table. All inserts happen in
        a single transaction for performance.

        Args:
            facts: List of Fact objects to store.

        Returns:
            Number of facts actually inserted.
        """
        ...

    @abstractmethod
    async def get_facts(
        self,
        interaction_id: str | None = None,
        subject_entity_id: str | None = None,
        predicate: str | None = None,
        limit: int = 100,
    ) -> list[Fact]:
        """
        Query facts with optional filters.

        Retrieves facts matching the given criteria. Filters are
        ANDed together -- all specified conditions must match.

        Args:
            interaction_id: Filter by source interaction. None means all.
            subject_entity_id: Filter by subject entity. None means all.
            predicate: Filter by predicate string. None means all.
            limit: Maximum number of results to return (default 100).

        Returns:
            A list of Fact objects matching the filters.
        """
        ...

    # -----------------------------------------------------------------
    # Embedding metadata: links SQL records to their vector store entries.
    # This is the bridge between the structured and semantic halves
    # of the dual-store architecture.
    # -----------------------------------------------------------------

    @abstractmethod
    async def store_embedding_metadata(self, metadata: list[EmbeddingMetadata]) -> int:
        """
        Store embedding metadata records.

        Records which SQL records have been embedded and in which
        vector store entries. This enables re-embedding when models
        change and tracing vector results back to source records.

        Args:
            metadata: List of EmbeddingMetadata objects to store.

        Returns:
            Number of metadata records actually inserted.
        """
        ...

    # -----------------------------------------------------------------
    # Farming insights: patterns discovered by the farming pipeline.
    # The farming pipeline reads from storage, discovers patterns,
    # and writes insights back here for future queries.
    # -----------------------------------------------------------------

    @abstractmethod
    async def store_insight(self, insight: FarmingInsight) -> str:
        """
        Store a farming insight. Returns the insight ID.

        Persists a pattern or insight discovered by the farming pipeline
        (co-occurrences, trends, clusters, anomalies) back to the
        SQL store so it can be queried later.

        Args:
            insight: The FarmingInsight object to store.

        Returns:
            The insight's ID (same as insight.id).
        """
        ...

    @abstractmethod
    async def get_insights(
        self,
        insight_type: str | None = None,
        limit: int = 50,
    ) -> list[FarmingInsight]:
        """
        Query stored insights.

        Retrieves farming insights, optionally filtered by type.
        Results are ordered by created_at descending (most recent first).

        Args:
            insight_type: Filter by insight type (e.g., "trend", "cluster").
                          None means all insight types.
            limit: Maximum number of results to return (default 50).

        Returns:
            A list of FarmingInsight objects matching the filter.
        """
        ...

    # -----------------------------------------------------------------
    # Raw SQL execution: for the query engine to run dynamic queries.
    # The query planner generates SQL from user questions, and the
    # executor runs it through this method.
    # -----------------------------------------------------------------

    @abstractmethod
    async def execute_sql(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a raw SQL query and return results as dicts.

        Provides direct SQL access for the query engine, which
        generates SQL queries from user questions. Results are
        returned as a list of dictionaries (column_name → value).

        WARNING: Implementations MUST use parameterized queries
        to prevent SQL injection. The 'params' dict maps placeholder
        names to values.

        Args:
            sql: The SQL query string (with named placeholders).
            params: Optional dict of parameter values for the query.
                    None means no parameters.

        Returns:
            A list of dicts, one per result row. Each dict maps
            column names to their values.
        """
        ...

    # -----------------------------------------------------------------
    # Full-text search: keyword-based search across interaction content.
    # Uses the database's FTS engine (FTS5 for SQLite) for fast
    # keyword matching. This complements semantic vector search.
    # -----------------------------------------------------------------

    @abstractmethod
    async def search_fts(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Full-text search across interactions. Returns matching rows.

        Runs a keyword search against the FTS index on interaction
        content. This is the structured complement to semantic vector
        search -- it finds exact keyword matches rather than semantic
        similarity.

        Args:
            query: The search query string (FTS syntax).
            limit: Maximum number of results to return (default 10).

        Returns:
            A list of dicts representing matching rows, including
            the interaction ID, content snippet, and relevance rank.
        """
        ...


# =====================================================================
# VectorStore ABC -- Semantic Data Store Interface
# =====================================================================


class VectorStore(ABC):
    """
    Abstract interface for the vector (embedding) store.

    Implementations: LanceDBStore (Phase 1), FAISSStore (Phase 4).
    This interface defines operations for storing and searching
    vector embeddings -- the semantic representations of text.

    The vector store enables semantic search: finding content that
    is conceptually similar to a query, even if the exact words
    differ. For example, searching for "authentication concerns"
    can find a discussion about "OAuth2 security issues".

    All methods are async because vector operations may involve
    disk I/O (reading index files) or network I/O (future: remote
    vector stores).

    Usage:
        store = LanceDBStore(path="/path/to/vectors")
        await store.initialize()
        await store.insert(ids, vectors, metadata)
        results = await store.search(query_vector, top_k=10)
        await store.close()
    """

    # -----------------------------------------------------------------
    # Lifecycle methods: create the vector collection on first run,
    # and close it cleanly on shutdown.
    # -----------------------------------------------------------------

    @abstractmethod
    async def initialize(self) -> None:
        """
        Create the vector collection/table if it doesn't exist.

        Called once at startup. Implementations should:
        1. Open/verify the vector store connection
        2. Create the collection/table if it doesn't exist
        3. Verify the embedding dimensionality matches the expected value

        This method is idempotent -- calling it on an existing
        collection should be safe.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """
        Close the store cleanly.

        Called on shutdown. Implementations should:
        1. Flush any pending writes to disk
        2. Release any held resources (memory-mapped files, locks)
        """
        ...

    # -----------------------------------------------------------------
    # Insert: add new vectors (embeddings) with metadata.
    # Each vector represents a chunk of text that has been run
    # through an embedding model (e.g., all-MiniLM-L6-v2).
    # -----------------------------------------------------------------

    @abstractmethod
    async def insert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> int:
        """
        Insert vectors with metadata. Returns count inserted.

        Adds a batch of embedding vectors to the store. Each vector
        has an ID (for later retrieval/deletion) and metadata (for
        filtering and attribution).

        Args:
            ids: Unique identifiers for each vector. Must be same
                 length as vectors and metadata.
            vectors: The embedding vectors (list of float lists).
                     All vectors must have the same dimensionality.
            metadata: Metadata dicts for each vector. Typically includes
                      source_table, source_id, chunk_index, and
                      a content_preview for debugging.

        Returns:
            Number of vectors actually inserted.
        """
        ...

    # -----------------------------------------------------------------
    # Search: find vectors most similar to a query vector.
    # This is the core semantic search operation -- given an embedded
    # query, find the top-k most similar stored vectors.
    # -----------------------------------------------------------------

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Semantic search. Returns top-k results ranked by similarity.

        Finds the vectors most similar to the query vector using
        cosine similarity (or the store's configured distance metric).
        Results can be filtered by metadata fields.

        Args:
            query_vector: The embedding vector of the search query.
                          Must have the same dimensionality as stored vectors.
            top_k: Number of results to return (default 10).
            filters: Optional metadata filters to narrow the search.
                     For example, {"source_table": "interactions"}
                     searches only interaction embeddings.

        Returns:
            A list of SearchResult objects, ranked by similarity
            score (highest first). Each result includes the matched
            vector's ID, content preview, score, and metadata.
        """
        ...

    # -----------------------------------------------------------------
    # Delete: remove vectors by ID.
    # Used when interactions are re-ingested (old embeddings deleted,
    # new ones inserted) or when data is purged.
    # -----------------------------------------------------------------

    @abstractmethod
    async def delete(self, ids: list[str]) -> int:
        """
        Delete vectors by ID. Returns count deleted.

        Removes vectors from the store. Used for re-embedding
        (delete old vectors, insert new ones) and data purging.

        Args:
            ids: List of vector IDs to delete.

        Returns:
            Number of vectors actually deleted (IDs that existed).
        """
        ...

    # -----------------------------------------------------------------
    # Batch retrieval: fetch specific vectors by ID.
    # Used by farming stages (clustering) to process specific entity
    # vectors without iterating the entire store.
    # -----------------------------------------------------------------

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[tuple[str, list[float]]]:
        """
        Retrieve vectors by their IDs.

        Returns a list of (id, vector) tuples for each ID found.
        IDs that don't exist in the store are silently skipped --
        the returned list may be shorter than the input.

        Used by farming clustering to fetch vectors for specific
        entities without needing to iterate the entire store.

        Args:
            ids: List of vector IDs to retrieve.

        Returns:
            A list of (id, vector) tuples. The vector is a list
            of floats with the same dimensionality as stored vectors.
        """
        ...

    # -----------------------------------------------------------------
    # Storage optimization: compact and clean up the vector store.
    # Used by the Defragmenter maintenance agent to remove tombstones,
    # merge small data files, and rebuild degraded indexes.
    # -----------------------------------------------------------------

    @abstractmethod
    async def compact(self) -> dict[str, Any]:
        """
        Run storage optimization (compaction, tombstone cleanup).

        Performs maintenance operations on the underlying vector store:
        - Remove tombstones from deleted vectors
        - Merge small data files into larger ones
        - Update index statistics

        Returns a metrics dict with information about what was done,
        e.g., {"tombstones_removed": 5, "files_merged": 3, "duration_ms": 150}.
        The exact keys depend on the implementation.

        Returns:
            A dict of optimization metrics.
        """
        ...

    # -----------------------------------------------------------------
    # Count: return the total number of vectors in the store.
    # Used by the health monitor and for diagnostics.
    # -----------------------------------------------------------------

    @abstractmethod
    async def count(self) -> int:
        """
        Return total number of vectors stored.

        Returns the count of all vectors in the store. Used by
        the health monitor to report system status and by tests
        to verify insertion/deletion operations.

        Returns:
            Total count of vectors currently in the store.
        """
        ...
