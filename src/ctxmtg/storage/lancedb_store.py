# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
LanceDB Vector Store
====================

This module provides the LanceDB implementation of the VectorStore
abstract interface. It handles all semantic (embedding) data: storing
vectors alongside metadata, and performing cosine similarity searches
to find content that is conceptually similar to a query.

In the system architecture, this is the "right side" of the dual-store
design. The "left side" is the SQL store (SQLite). Together they answer
hybrid queries -- vectors handle semantic search ("discussions about
scaling concerns") while SQL handles precise lookups ("how many meetings
this week?").

LanceDB is an embedded vector database (no server needed) that stores
data on disk in the Lance columnar format. It is lightweight enough to
run on a Raspberry Pi yet fast enough for 100K+ vectors.

Key design decisions (references to research):
    - Disk-backed storage for production, temp directory for tests
      (research/round-1/02-dual-store-architecture.md)
    - Cosine similarity as the default distance metric
      (research/round-1/04-hybrid-query-orchestration.md)
    - Table auto-creation on first insert to simplify startup
      (countermeasures 4.2)
    - Native async API via lancedb.connect_async
      (matches the async VectorStore ABC contract)

Depends on:
    - lancedb (embedded vector database)
    - pyarrow (columnar data interchange, used by LanceDB internally)
    - structlog (structured JSON logging)
    - ctxmtg.interfaces.storage (the abstract contract this implements)
    - ctxmtg.models.query (SearchResult data model)
    - ctxmtg.exceptions (StorageError for error handling)

Used by:
    - ctxmtg.ingestion.worker (writes embedding vectors here)
    - ctxmtg.query.executor (searches vectors during queries)
    - ctxmtg.farming.pipeline (reads vectors for clustering/analysis)
"""

from __future__ import annotations

from typing import Any

import structlog

# ---------------------------------------------------------------
# Import the abstract interface this class implements, the data
# model it returns, and the exception type it raises. These are
# the only ctxmtg dependencies -- everything else is third-party.
# ---------------------------------------------------------------
from ctxmtg.exceptions import StorageError
from ctxmtg.interfaces.storage import VectorStore
from ctxmtg.models.query import SearchResult

# ---------------------------------------------------------------
# Module-level logger. Every log message includes the module name
# so operators can filter logs by component.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.storage.lancedb_store")

# ---------------------------------------------------------------
# Constants: table name and default settings. The table name is
# fixed because the system expects exactly one embeddings table.
# ---------------------------------------------------------------

# The name of the LanceDB table that stores all embeddings.
# Every vector in the system lives in this single table, with
# metadata columns (source_table, source_id) that distinguish
# where each vector came from.
EMBEDDINGS_TABLE_NAME = "embeddings"

# Default distance metric for similarity search. Cosine similarity
# measures the angle between two vectors, making it robust to
# differences in vector magnitude. This is the standard choice for
# text embeddings (see research/round-1/04-hybrid-query-orchestration.md).
DEFAULT_DISTANCE_METRIC = "cosine"


class LanceDBStore(VectorStore):
    """
    LanceDB-based storage for vector (embedding) data.

    This is the vector half of the dual-store architecture. It stores
    embedding vectors alongside metadata, and supports fast cosine
    similarity search to find semantically similar content.

    LanceDB is an embedded database (like SQLite but for vectors) --
    no server process needed, data lives on disk, and it runs on any
    hardware from a Raspberry Pi to a beefy desktop.

    Why LanceDB and not FAISS or ChromaDB?
    LanceDB provides disk-backed storage with lazy loading, so it
    doesn't need to load all vectors into RAM. This is critical for
    the Pi tier (1-4GB RAM). FAISS is faster for pure in-memory
    search but requires loading the entire index into RAM.
    (See research/round-1/02-dual-store-architecture.md)

    Schema (all columns in the "embeddings" table):
        - id: str           -- Unique identifier for this embedding
        - vector: list[f32] -- The embedding vector (fixed-size float32 list)
        - source_table: str -- Which SQL table the source record lives in
                               (e.g., "interactions", "entities")
        - source_id: str    -- The ID of the source record in SQL
        - chunk_index: int  -- Which chunk of the source this vector represents
                               (0 for single-chunk records)
        - content_preview: str -- First 200 chars of the original text
                                  (for debugging, not for search)
        - created_at: str   -- ISO datetime string of when this was embedded

    Usage:
        store = LanceDBStore(db_path="/path/to/vectors")
        await store.initialize()
        await store.insert(ids, vectors, metadata_list)
        results = await store.search(query_vector, top_k=5)
        await store.close()
    """

    def __init__(self, db_path: str) -> None:
        """
        Initialize the LanceDB store with a path to the database directory.

        The directory will be created if it doesn't exist. LanceDB stores
        data as Lance files inside this directory -- one subdirectory per
        table.

        Args:
            db_path: Filesystem path where LanceDB will store its data.
                     For production, use a persistent path like
                     "~/.ctxmtg/vectors". For tests, use a temp directory.
        """
        # Store the path for later connection. We don't connect here
        # because the connection is async (done in initialize()).
        self._db_path = db_path

        # These will be set during initialize(). We track them as
        # instance attributes so other methods can use them.
        self._db: Any = None      # The async LanceDB connection
        self._table: Any = None   # The embeddings table (or None if not created yet)

        # Track whether initialize() has been called. Methods that
        # need the database will check this and raise if not.
        self._initialized = False

    async def initialize(self) -> None:
        """
        Open the LanceDB connection and load the embeddings table if it exists.

        This method is idempotent -- calling it multiple times is safe.
        If the embeddings table already exists (from a previous run),
        it is opened. If not, the table will be created lazily on the
        first insert.

        We don't create the table here because LanceDB requires at
        least one row of data to infer the schema (including the vector
        dimensionality). Since we don't know the embedding dimensions
        until the first insert, we defer table creation.
        """
        # Import lancedb here to keep it as a lazy dependency.
        # This way, import errors are caught at initialization time
        # rather than at module import time.
        import lancedb

        try:
            # Open an async connection to the LanceDB database directory.
            # connect_async returns a native async connection that doesn't
            # block the event loop during I/O operations.
            self._db = await lancedb.connect_async(self._db_path)

            # Check if the embeddings table already exists from a previous
            # session. If so, open it so we can search/insert immediately.
            table_list_result = await self._db.list_tables()
            existing_tables = table_list_result.tables

            if EMBEDDINGS_TABLE_NAME in existing_tables:
                # Table exists -- open it for reading and writing.
                self._table = await self._db.open_table(EMBEDDINGS_TABLE_NAME)
                logger.info(
                    "lancedb_table_opened",
                    table=EMBEDDINGS_TABLE_NAME,
                    db_path=self._db_path,
                )
            else:
                # Table doesn't exist yet. It will be created on the
                # first call to insert(). This is fine -- search() and
                # count() handle the no-table case gracefully.
                self._table = None
                logger.info(
                    "lancedb_initialized_no_table",
                    table=EMBEDDINGS_TABLE_NAME,
                    db_path=self._db_path,
                )

            self._initialized = True

        except Exception as exc:
            logger.error(
                "lancedb_init_failed",
                error_code="CTXMTG-STG-006",
                db_path=self._db_path,
                error=str(exc),
            )
            raise StorageError(
                f"Failed to initialize LanceDB at {self._db_path}: {exc}",
                error_code="CTXMTG-STG-006",
            ) from exc

    async def close(self) -> None:
        """
        Close the LanceDB connection and release resources.

        After calling close(), the store cannot be used until
        initialize() is called again.
        """
        # LanceDB's async connection doesn't require explicit close
        # in all cases, but we reset our state to prevent accidental
        # use after close.
        self._table = None
        self._db = None
        self._initialized = False

        logger.info("lancedb_closed", db_path=self._db_path)

    async def insert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> int:
        """
        Insert embedding vectors with metadata into the store.

        Each vector represents a chunk of text that has been run through
        an embedding model (e.g., all-MiniLM-L6-v2). The metadata carries
        provenance information linking the vector back to its SQL source.

        If the embeddings table doesn't exist yet (first insert after a
        fresh install), this method creates it automatically. The schema
        is inferred from the first batch of data.

        Args:
            ids: Unique identifiers for each vector. Length must match
                 vectors and metadata.
            vectors: Embedding vectors as lists of floats. All vectors
                     must have the same dimensionality.
            metadata: Metadata dicts for each vector. Expected keys:
                      source_table, source_id, chunk_index,
                      content_preview, created_at.

        Returns:
            Number of vectors actually inserted.

        Raises:
            StorageError: If the insert fails (disk full, schema mismatch, etc.)
        """
        # Guard: ensure initialize() was called first.
        self._ensure_initialized()

        # Guard: all three lists must have the same length.
        if not (len(ids) == len(vectors) == len(metadata)):
            logger.error(
                "lancedb_insert_length_mismatch",
                error_code="CTXMTG-STG-007",
                ids=len(ids),
                vectors=len(vectors),
                metadata=len(metadata),
            )
            raise StorageError(
                f"Insert length mismatch: ids={len(ids)}, "
                f"vectors={len(vectors)}, metadata={len(metadata)}",
                error_code="CTXMTG-STG-007",
            )

        # Guard: nothing to insert.
        if len(ids) == 0:
            return 0

        try:
            # Build the row dicts that LanceDB expects. Each row has
            # all the schema columns: id, vector, and metadata fields.
            rows = self._build_rows(ids, vectors, metadata)

            if self._table is None:
                # First insert ever -- create the table with this data.
                # LanceDB infers the schema (including vector dimensions)
                # from the data itself.
                self._table = await self._db.create_table(
                    EMBEDDINGS_TABLE_NAME,
                    data=rows,
                    mode="overwrite",
                )
                logger.info(
                    "lancedb_table_created",
                    table=EMBEDDINGS_TABLE_NAME,
                    initial_rows=len(rows),
                    vector_dim=len(vectors[0]),
                )
            else:
                # Table already exists -- append the new rows.
                await self._table.add(rows)
                logger.info(
                    "lancedb_vectors_inserted",
                    count=len(rows),
                )

            return len(rows)

        except StorageError:
            raise
        except Exception as exc:
            logger.error(
                "lancedb_insert_failed",
                error_code="CTXMTG-STG-007",
                vector_count=len(ids),
                error=str(exc),
            )
            raise StorageError(
                f"Failed to insert vectors: {exc}",
                error_code="CTXMTG-STG-007",
            ) from exc

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Find the top-k most similar vectors using cosine similarity.

        This is the core semantic search operation. Given a query vector
        (typically the embedding of a user's question), find the stored
        vectors that are most similar. Results are returned as
        SearchResult objects ranked by similarity (highest first).

        Cosine similarity measures the angle between two vectors:
        - 1.0 = identical direction (maximum similarity)
        - 0.0 = orthogonal (no similarity)
        - -1.0 = opposite direction (maximum dissimilarity)

        LanceDB returns a "distance" where 0 = identical and 1 = orthogonal.
        We convert this to a similarity score: similarity = 1.0 - distance.

        Args:
            query_vector: The embedding vector of the search query. Must
                          have the same dimensionality as stored vectors.
            top_k: Number of results to return (default 10).
            filters: Optional metadata filters to narrow the search. Keys
                     map to column names, values are exact-match filters.
                     Example: {"source_table": "interactions"} searches
                     only interaction embeddings.

        Returns:
            A list of SearchResult objects, sorted by similarity score
            (highest first). Empty list if the table doesn't exist yet
            or no results match the filters.

        Raises:
            StorageError: If the search fails (dimension mismatch, etc.)
        """
        # Guard: ensure initialize() was called.
        self._ensure_initialized()

        # If the table hasn't been created yet (no inserts have happened),
        # return an empty list. This is not an error -- it's normal during
        # initial startup before any data is ingested.
        if self._table is None:
            return []

        try:
            # Build the search query with cosine distance metric.
            # vector_search() is the async-native search method.
            search_query = (
                self._table.vector_search(query_vector)
                .distance_type(DEFAULT_DISTANCE_METRIC)
                .limit(top_k)
            )

            # Apply metadata filters if provided. LanceDB uses SQL-like
            # WHERE clause syntax for filtering.
            if filters:
                where_clause = self._build_where_clause(filters)
                if where_clause:
                    search_query = search_query.where(where_clause)

            # Execute the search and get results as a PyArrow table.
            # to_arrow() avoids the pandas dependency.
            arrow_results = await search_query.to_arrow()

            # Convert PyArrow results to our SearchResult data model.
            results = self._arrow_to_search_results(arrow_results)

            logger.info(
                "lancedb_search_completed",
                top_k=top_k,
                results_found=len(results),
                has_filters=filters is not None,
            )

            return results

        except StorageError:
            raise
        except Exception as exc:
            logger.error(
                "lancedb_search_failed",
                error_code="CTXMTG-STG-008",
                top_k=top_k,
                error=str(exc),
            )
            raise StorageError(
                f"Vector search failed: {exc}",
                error_code="CTXMTG-STG-008",
            ) from exc

    async def delete(self, ids: list[str]) -> int:
        """
        Delete vectors by their IDs.

        Used when interactions are re-ingested (old embeddings deleted,
        new ones inserted) or when data is purged.

        Args:
            ids: List of vector IDs to delete. IDs that don't exist
                 are silently ignored.

        Returns:
            Number of vectors actually deleted. Note: LanceDB's delete
            doesn't return a count of affected rows, so we estimate by
            counting before and after.

        Raises:
            StorageError: If the delete operation fails.
        """
        # Guard: ensure initialize() was called.
        self._ensure_initialized()

        # If the table doesn't exist, there's nothing to delete.
        if self._table is None:
            return 0

        # If the ids list is empty, nothing to do.
        if not ids:
            return 0

        try:
            # Count rows before deletion so we can report how many
            # were actually removed. LanceDB's delete() doesn't
            # return affected row counts.
            count_before: int = await self._table.count_rows()

            # Build a SQL-style IN clause for the delete predicate.
            # LanceDB accepts SQL WHERE clause syntax for deletes.
            # We need to escape single quotes in IDs to prevent
            # injection-like issues.
            escaped_ids = [id_val.replace("'", "''") for id_val in ids]
            id_list = ", ".join(f"'{id_val}'" for id_val in escaped_ids)
            delete_predicate = f"id IN ({id_list})"

            await self._table.delete(delete_predicate)

            # Count rows after deletion to determine how many were removed.
            count_after: int = await self._table.count_rows()
            deleted_count = count_before - count_after

            logger.info(
                "lancedb_vectors_deleted",
                requested=len(ids),
                actual_deleted=deleted_count,
            )

            return deleted_count

        except StorageError:
            raise
        except Exception as exc:
            logger.error(
                "lancedb_delete_failed",
                error_code="CTXMTG-STG-009",
                id_count=len(ids),
                error=str(exc),
            )
            raise StorageError(
                f"Failed to delete vectors: {exc}",
                error_code="CTXMTG-STG-009",
            ) from exc

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
        # Guard: ensure initialize() was called.
        self._ensure_initialized()

        # If the table doesn't exist or ids list is empty, nothing to fetch.
        if self._table is None or not ids:
            return []

        try:
            # Build a SQL-style IN clause to filter by the requested IDs.
            # Escape single quotes to prevent injection-like issues
            # (same approach used in delete()).
            escaped_ids = [id_val.replace("'", "''") for id_val in ids]
            id_list = ", ".join(f"'{id_val}'" for id_val in escaped_ids)
            where_clause = f"id IN ({id_list})"

            # Query the table with the ID filter. We only need the id
            # and vector columns -- skip metadata to reduce I/O.
            # LanceDB async tables support query().where().select().
            arrow_table = await (
                self._table
                .query()
                .where(where_clause)
                .select(["id", "vector"])
                .to_arrow()
            )

            # Convert the PyArrow table to a list of (id, vector) tuples.
            data = arrow_table.to_pydict()
            num_rows = arrow_table.num_rows

            results: list[tuple[str, list[float]]] = []
            for i in range(num_rows):
                vec_id = data["id"][i]
                # LanceDB stores vectors as fixed-size lists. Convert
                # to a plain Python list of floats for compatibility.
                vector = [float(v) for v in data["vector"][i]]
                results.append((vec_id, vector))

            logger.info(
                "lancedb_get_by_ids",
                requested=len(ids),
                found=len(results),
            )

            return results

        except StorageError:
            raise
        except Exception as exc:
            logger.error(
                "lancedb_get_by_ids_failed",
                error_code="CTXMTG-STG-008",
                id_count=len(ids),
                error=str(exc),
            )
            raise StorageError(
                f"Failed to get vectors by IDs: {exc}",
                error_code="CTXMTG-STG-008",
            ) from exc

    async def compact(self) -> dict[str, Any]:
        """
        Run storage optimization (compaction, tombstone cleanup).

        Performs maintenance operations on the LanceDB table:
        - Removes tombstones from previously deleted vectors
        - Merges small Lance data files into larger ones
        - Cleans up old table versions

        LanceDB uses a copy-on-write strategy for deletes and updates,
        so deleted rows leave behind tombstones that waste disk space
        and slow down scans. Compaction removes these tombstones and
        consolidates small files.

        Returns:
            A dict of optimization metrics describing what was done.

        Raises:
            StorageError: If compaction fails.
        """
        # Guard: ensure initialize() was called.
        self._ensure_initialized()

        # If the table doesn't exist, there's nothing to compact.
        if self._table is None:
            return {"status": "skipped", "reason": "no_table"}

        import time

        metrics: dict[str, Any] = {}
        start_ms = time.monotonic()

        try:
            # Step 1: Compact data files -- merges small Lance fragments
            # and removes tombstones from deleted rows.
            try:
                compact_result = await self._table.optimize.compact_files()
                # compact_files() returns a CompactionMetrics object.
                # Extract useful fields if available.
                metrics["files_removed"] = getattr(
                    compact_result, "files_removed", 0
                )
                metrics["files_added"] = getattr(
                    compact_result, "files_added", 0
                )
                metrics["fragments_removed"] = getattr(
                    compact_result, "fragments_removed", 0
                )
                metrics["fragments_added"] = getattr(
                    compact_result, "fragments_added", 0
                )
            except (AttributeError, NotImplementedError):
                # compact_files() may not be available in all LanceDB
                # versions. This is best-effort.
                metrics["compact_files"] = "not_supported"

            # Step 2: Clean up old table versions. LanceDB keeps
            # previous versions for time-travel queries. Pruning
            # old versions reclaims disk space.
            try:
                await self._table.optimize.cleanup_old_versions()
                metrics["old_versions_cleaned"] = True
            except (AttributeError, NotImplementedError):
                # cleanup_old_versions() may not be available in all
                # LanceDB versions. This is best-effort.
                metrics["old_versions_cleaned"] = "not_supported"

            elapsed_ms = int((time.monotonic() - start_ms) * 1000)
            metrics["duration_ms"] = elapsed_ms
            metrics["status"] = "completed"

            logger.info("lancedb_compaction_completed", **metrics)

            return metrics

        except StorageError:
            raise
        except Exception as exc:
            logger.error(
                "lancedb_compact_failed",
                error_code="CTXMTG-STG-006",
                error=str(exc),
            )
            raise StorageError(
                f"Compaction failed: {exc}",
                error_code="CTXMTG-STG-006",
            ) from exc

    async def count(self) -> int:
        """
        Return the total number of vectors stored.

        Used by the health monitor to report system status and by
        tests to verify insertion/deletion operations.

        Returns:
            Total count of vectors. Returns 0 if the table hasn't
            been created yet.

        Raises:
            StorageError: If counting fails.
        """
        # Guard: ensure initialize() was called.
        self._ensure_initialized()

        # If the table doesn't exist yet, count is 0.
        if self._table is None:
            return 0

        try:
            total: int = await self._table.count_rows()
            return total

        except Exception as exc:
            logger.error(
                "lancedb_count_failed",
                error_code="CTXMTG-STG-006",
                error=str(exc),
            )
            raise StorageError(
                f"Failed to count vectors: {exc}",
                error_code="CTXMTG-STG-006",
            ) from exc

    # =================================================================
    # Private helper methods
    # =================================================================

    def _ensure_initialized(self) -> None:
        """
        Check that initialize() has been called before any operation.

        Raises StorageError with a helpful message if the store hasn't
        been initialized. This prevents confusing errors from trying
        to use a None database connection.
        """
        if not self._initialized:
            logger.error(
                "lancedb_not_initialized",
                error_code="CTXMTG-STG-006",
                db_path=self._db_path,
            )
            raise StorageError(
                "LanceDBStore not initialized. Call await store.initialize() first.",
                error_code="CTXMTG-STG-006",
            )

    def _build_rows(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Build row dicts for LanceDB insertion from separate id/vector/metadata lists.

        Each row dict has the full schema: id, vector, and all metadata
        columns. Missing metadata fields get sensible defaults.

        Args:
            ids: Vector identifiers.
            vectors: Embedding vectors.
            metadata: Metadata dicts (may have varying keys).

        Returns:
            A list of dicts ready for LanceDB insertion.
        """
        rows = []
        for vec_id, vector, meta in zip(ids, vectors, metadata, strict=True):
            # Build the row with all schema columns. Metadata fields
            # that are missing get empty-string or zero defaults.
            row = {
                "id": vec_id,
                "vector": vector,
                "source_table": meta.get("source_table", ""),
                "source_id": meta.get("source_id", ""),
                "chunk_index": int(meta.get("chunk_index", 0)),
                # Truncate content_preview to 200 chars for storage efficiency.
                # This is for debugging only -- the full content lives in SQL.
                "content_preview": str(meta.get("content_preview", ""))[:200],
                "created_at": meta.get("created_at", ""),
            }
            rows.append(row)

        return rows

    def _build_where_clause(self, filters: dict[str, Any]) -> str:
        """
        Convert a filters dict to a LanceDB SQL-style WHERE clause.

        Supports simple equality filters on metadata columns. Multiple
        filters are ANDed together. Values are escaped to prevent
        injection-like issues.

        Args:
            filters: Dict mapping column names to expected values.
                     Example: {"source_table": "interactions"} becomes
                     "source_table = 'interactions'".

        Returns:
            A SQL-style WHERE clause string, or empty string if no
            valid filters were provided.
        """
        # Whitelist of columns that can be filtered. This prevents
        # users from filtering on the vector column (which doesn't
        # make sense) or injecting arbitrary SQL.
        filterable_columns = {
            "source_table",
            "source_id",
            "chunk_index",
            "created_at",
        }

        clauses = []
        for col, value in filters.items():
            # Skip columns that aren't in our whitelist.
            if col not in filterable_columns:
                logger.warning(
                    "lancedb_filter_skipped",
                    error_code="CTXMTG-STG-002",
                    column=col,
                    reason="not a filterable column",
                )
                continue

            if isinstance(value, str):
                escaped_value = value.replace("'", "''")
                clauses.append(f"{col} = '{escaped_value}'")
            elif isinstance(value, int):
                clauses.append(f"{col} = {value}")
            elif isinstance(value, dict):
                range_parts = self._build_range_clause(col, value)
                if range_parts:
                    clauses.extend(range_parts)
            else:
                logger.warning(
                    "lancedb_filter_skipped",
                    error_code="CTXMTG-STG-002",
                    column=col,
                    reason=f"unsupported value type: {type(value).__name__}",
                )

        # Join all clauses with AND.
        return " AND ".join(clauses)

    def _build_range_clause(self, col: str, range_dict: dict[str, Any]) -> list[str]:
        """
        Build range filter clauses from a dict with comparison operators.

        Supports "gte" (>=), "lte" (<=), "gt" (>), "lt" (<) keys.
        Used for temporal filtering (e.g., "created_at >= '2026-01-01'").

        Args:
            col: The column name to filter on.
            range_dict: Dict with comparison operator keys and values.

        Returns:
            A list of SQL clause strings.
        """
        # Map our operator names to SQL operators.
        operator_map = {
            "gte": ">=",
            "lte": "<=",
            "gt": ">",
            "lt": "<",
        }

        clauses = []
        for op_key, op_value in range_dict.items():
            sql_op = operator_map.get(op_key)
            if sql_op is None:
                continue

            if isinstance(op_value, str):
                escaped = op_value.replace("'", "''")
                clauses.append(f"{col} {sql_op} '{escaped}'")
            elif isinstance(op_value, (int, float)):
                clauses.append(f"{col} {sql_op} {op_value}")

        return clauses

    def _arrow_to_search_results(self, arrow_table: Any) -> list[SearchResult]:
        """
        Convert a PyArrow table of search results to SearchResult objects.

        LanceDB returns results as a PyArrow table with a special
        '_distance' column. We convert the distance to a similarity
        score (1.0 - distance for cosine) and package everything into
        our SearchResult data model.

        Args:
            arrow_table: PyArrow table from LanceDB search. Expected
                         columns: id, vector, source_table, source_id,
                         chunk_index, content_preview, created_at, _distance.

        Returns:
            A list of SearchResult objects sorted by similarity (highest first).
        """
        results = []

        # Convert the PyArrow table to a dict-of-lists for easier access.
        # This is more efficient than row-by-row iteration for PyArrow.
        data = arrow_table.to_pydict()

        # Get the number of results.
        num_rows = arrow_table.num_rows

        for i in range(num_rows):
            # Convert LanceDB's cosine distance to a similarity score.
            # Cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite.
            # Similarity = 1.0 - distance gives: 1.0 = identical, 0.0 = orthogonal.
            distance = data["_distance"][i]
            similarity_score = 1.0 - distance

            # Build the metadata dict from the non-vector columns.
            # We exclude 'id', 'vector', and '_distance' since they
            # are represented in other SearchResult fields.
            metadata = {
                "source_table": data.get("source_table", [""])[i]
                if "source_table" in data
                else "",
                "source_id": data.get("source_id", [""])[i]
                if "source_id" in data
                else "",
                "chunk_index": data.get("chunk_index", [0])[i]
                if "chunk_index" in data
                else 0,
                "created_at": data.get("created_at", [""])[i]
                if "created_at" in data
                else "",
            }

            # Use content_preview as the SearchResult content. This is
            # a truncated preview (200 chars max). The query executor
            # can hydrate this with full content from SQL if needed.
            content = (
                data.get("content_preview", [""])[i]
                if "content_preview" in data
                else ""
            )

            result = SearchResult(
                id=data["id"][i],
                source_store="vector",
                content=content,
                score=similarity_score,
                metadata=metadata,
            )
            results.append(result)

        return results
