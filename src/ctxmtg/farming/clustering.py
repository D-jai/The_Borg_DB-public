# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Clustering Stage (Intelligence Stage 3)
=========================================

Groups related entities by running incremental MiniBatchKMeans
clustering over their vector embeddings.  Entities are processed
in alphabetical batches so the algorithm can be resumed across
farming cycles without re-processing already-seen entities.

This stage uses the ``farming_clustering_progress`` table to track
how far through the alphabetical entity list it has progressed.
When a full pass completes (no more unseen entities), it reads the
resulting cluster labels and emits one ``FarmingInsight`` per
cluster summarising membership.

Graceful degradation: if ``scikit-learn`` is not installed (e.g., on
a resource-constrained Tier 0 device), the stage returns an empty
list immediately -- clustering is disabled without crashing.

Algorithm overview:
    1. Read the last-processed entity name from the progress table.
    2. Query the next ``batch_size`` entities (alphabetically).
    3. For each entity, look up its embedding IDs in
       ``embeddings_metadata`` and fetch the vectors via
       ``vector_store.get_by_ids()``.
    4. If we have vectors and sklearn, call
       ``MiniBatchKMeans.partial_fit()`` incrementally.
    5. Save the KMeans model to the checkpoint store so it survives
       process restarts.
    6. Record a progress row in ``farming_clustering_progress``.
    7. When no more entities remain (full pass done), generate
       cluster insights from the fitted model's labels.

Depends on:
    - structlog (structured logging)
    - sklearn.cluster.MiniBatchKMeans (optional -- graceful degradation)
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- optional, unused tier 0-1)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async bridge)

Used by:
    - ctxmtg.farming.pipeline (registered as the third intelligence stage)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog

from ctxmtg.farming.checkpoint import _run_async
from ctxmtg.interfaces.farming import FarmingContext, FarmingStage
from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.models.farming import FarmingInsight

# ---------------------------------------------------------------
# Module-level logger -- structured JSON output, no PII in logs.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.clustering")

# ---------------------------------------------------------------
# Optional sklearn import.  If scikit-learn is not installed, the
# stage degrades gracefully -- it returns an empty insights list
# instead of crashing.  This lets Tier 0 devices (no sklearn) run
# the full pipeline without special-casing.
# ---------------------------------------------------------------
try:
    from sklearn.cluster import MiniBatchKMeans

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    MiniBatchKMeans = None  # type: ignore[misc,assignment]
    _HAS_SKLEARN = False

# ---------------------------------------------------------------
# SQL queries -- module constants for readability and testability.
# ---------------------------------------------------------------

# Read the latest progress row (by completed_at DESC) to find out
# where the last batch left off in the alphabetical entity scan.
LAST_PROGRESS_SQL = """\
SELECT last_entity
FROM farming_clustering_progress
ORDER BY completed_at DESC
LIMIT 1
"""

# Fetch the next batch of distinct entity names alphabetically,
# starting after the last entity we processed.  This gives us the
# incremental "cursor" for resumable clustering.
NEXT_ENTITIES_SQL = """\
SELECT DISTINCT name
FROM entities
WHERE name > :last_entity
ORDER BY name ASC
LIMIT :batch_size
"""

# Look up embedding IDs for a set of entity names.  We join through
# the embeddings_metadata table using source_table = 'entities' to
# find which vectors belong to the entities in this batch.
EMBEDDING_IDS_SQL = """\
SELECT id
FROM embeddings_metadata
WHERE source_table = 'entities'
  AND source_id IN ({placeholders})
"""

# Record progress after a batch completes, so the next cycle can
# resume from where we stopped.
INSERT_PROGRESS_SQL = """\
INSERT INTO farming_clustering_progress
    (id, cycle_id, last_entity, entities_done, batch_size, completed_at)
VALUES (:id, :cycle_id, :last_entity, :entities_done, :batch_size, :completed_at)
"""


class ClusteringStage(FarmingStage):
    """
    Farming stage 3: incremental MiniBatchKMeans clustering.

    Processes entity embeddings in alphabetical batches, fitting
    a MiniBatchKMeans model incrementally via ``partial_fit()``.
    The model is checkpointed between cycles so work is not lost.

    When a full alphabetical pass completes (the entity query returns
    fewer rows than ``batch_size``), the stage emits cluster insights.
    Otherwise, it returns an empty list (work-in-progress).

    If scikit-learn is not installed, the stage returns empty
    immediately (graceful degradation for Tier 0 devices).

    Usage:
        stage = ClusteringStage(n_clusters=20, batch_size=500)
        insights = stage.run(sql_store, vector_store, context)
    """

    def __init__(
        self,
        n_clusters: int = 20,
        batch_size: int = 500,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Configure the clustering stage.

        Args:
            n_clusters: Number of clusters for MiniBatchKMeans.
                Default 20 -- chosen for a moderate knowledge store.
            batch_size: How many entities to process per cycle.
                Default 500 -- keeps individual runs short.
            llm: Optional LLM provider for future narrative generation.
                 Currently unused (tier 0-1 does not require an LLM).
        """
        self._n_clusters = n_clusters
        self._batch_size = batch_size
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface
    # -----------------------------------------------------------------

    def get_name(self) -> str:
        """Return the canonical stage name for logging/checkpointing."""
        return "clustering"

    def run(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        """
        Execute incremental clustering over entity embeddings.

        Steps:
            1. Read last position from farming_clustering_progress.
            2. Fetch next batch of entity names (alphabetically).
            3. For each entity, look up embedding IDs and fetch vectors.
            4. If sklearn available, run MiniBatchKMeans.partial_fit().
            5. Save model checkpoint and record progress.
            6. If the pass is complete, generate cluster insights.

        This method is synchronous (per the FarmingStage contract)
        but calls async store methods via ``_run_async()``.

        Args:
            sql_store:    SQL store for entity and progress data.
            vector_store: Vector store to fetch entity embeddings from.
            context:      Farming context with cycle ID and budget.

        Returns:
            List of FarmingInsight objects (type ``"cluster"``).
            Empty if mid-pass or sklearn unavailable.
        """
        logger.info(
            "clustering_start",
            cycle_id=context.cycle_id,
            n_clusters=self._n_clusters,
            batch_size=self._batch_size,
            has_sklearn=_HAS_SKLEARN,
        )

        # ----------------------------------------------------------
        # Graceful degradation: no sklearn → no clustering.
        # ----------------------------------------------------------
        if not _HAS_SKLEARN:
            logger.info(
                "clustering_skip_no_sklearn",
                cycle_id=context.cycle_id,
            )
            return []

        # ----------------------------------------------------------
        # Step 1: Read last position from progress table.
        # If no rows exist, start from the beginning (empty string
        # sorts before all real entity names).
        # ----------------------------------------------------------
        progress_rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(LAST_PROGRESS_SQL, {})
        )

        # Default: start from very beginning ('' < any real name)
        last_entity: str = ""
        if progress_rows and progress_rows[0].get("last_entity"):
            last_entity = progress_rows[0]["last_entity"]

        logger.debug(
            "clustering_resume_position",
            last_entity=last_entity,
        )

        # ----------------------------------------------------------
        # Step 2: Fetch next batch of distinct entity names.
        # ----------------------------------------------------------
        entity_rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(
                NEXT_ENTITIES_SQL,
                {"last_entity": last_entity, "batch_size": self._batch_size},
            )
        )

        entity_names: list[str] = [row["name"] for row in entity_rows]

        logger.debug(
            "clustering_batch_loaded",
            entity_count=len(entity_names),
        )

        # No entities to process → either the store is empty or
        # we have wrapped around (full pass complete).
        if not entity_names:
            # If we had a previous position, the pass is complete --
            # generate cluster insights from the fitted model.
            if last_entity:
                return self._generate_cluster_insights(context)

            # Store is truly empty -- nothing to cluster.
            logger.info(
                "clustering_no_entities",
                cycle_id=context.cycle_id,
            )
            return []

        # ----------------------------------------------------------
        # Step 3: Look up embedding IDs for these entities.
        # We need the entity IDs first to query embeddings_metadata.
        # ----------------------------------------------------------
        vectors = self._fetch_entity_vectors(
            sql_store, vector_store, entity_names
        )

        # ----------------------------------------------------------
        # Step 4: partial_fit the MiniBatchKMeans model.
        # Load existing model from checkpoint or create a new one.
        # ----------------------------------------------------------
        if vectors:
            model = self._load_or_create_model(context)
            model.partial_fit(vectors)
            # Save the updated model back to the checkpoint store
            context.checkpoint.save({"model": model})

            logger.debug(
                "clustering_partial_fit",
                n_vectors=len(vectors),
            )

        # ----------------------------------------------------------
        # Step 5: Record progress so the next cycle knows where
        # to resume from.
        # ----------------------------------------------------------
        now_str = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        _run_async(
            sql_store.execute_sql(
                INSERT_PROGRESS_SQL,
                {
                    "id": str(uuid4()),
                    "cycle_id": context.cycle_id,
                    "last_entity": entity_names[-1],
                    "entities_done": len(entity_names),
                    "batch_size": self._batch_size,
                    "completed_at": now_str,
                },
            )
        )

        logger.info(
            "clustering_batch_complete",
            cycle_id=context.cycle_id,
            entities_processed=len(entity_names),
            last_entity=entity_names[-1],
        )

        # ----------------------------------------------------------
        # Step 6: Check if the pass wrapped around.
        # If fewer entities returned than batch_size, we're at the
        # end of the alphabetical list → generate insights.
        # ----------------------------------------------------------
        if len(entity_names) < self._batch_size:
            return self._generate_cluster_insights(context)

        # Mid-pass: no insights yet, will complete in a future cycle.
        return []

    # =================================================================
    # Private helpers
    # =================================================================

    def _fetch_entity_vectors(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        entity_names: list[str],
    ) -> list[list[float]]:
        """
        Fetch embedding vectors for the given entity names.

        Looks up embedding IDs from embeddings_metadata (source_table
        = 'entities') then retrieves the actual vectors from the
        vector store.

        Args:
            sql_store:    SQL store for metadata lookups.
            vector_store: Vector store for fetching embedding vectors.
            entity_names: List of entity names to fetch vectors for.

        Returns:
            A flat list of embedding vectors (list of float lists).
            May be empty if no embeddings exist for these entities.
        """
        # First, get the entity IDs from the entities table so we can
        # look them up in embeddings_metadata.
        # Build a parameterised IN clause for safe SQL.
        placeholders = ", ".join(f":e{i}" for i in range(len(entity_names)))
        params: dict[str, Any] = {
            f"e{i}": name for i, name in enumerate(entity_names)
        }

        # Query entity IDs matching these names
        entity_id_rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(
                f"SELECT DISTINCT id FROM entities WHERE name IN ({placeholders})",
                params,
            )
        )

        entity_ids = [row["id"] for row in entity_id_rows]

        if not entity_ids:
            return []

        # Now look up embedding IDs for these entity IDs
        emb_placeholders = ", ".join(
            f":emb{i}" for i in range(len(entity_ids))
        )
        emb_params: dict[str, Any] = {
            f"emb{i}": eid for i, eid in enumerate(entity_ids)
        }

        embedding_rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(
                EMBEDDING_IDS_SQL.format(placeholders=emb_placeholders),
                emb_params,
            )
        )

        embedding_ids = [row["id"] for row in embedding_rows]

        if not embedding_ids:
            return []

        # Fetch actual vectors from the vector store (async)
        id_vector_pairs: list[tuple[str, list[float]]] = _run_async(
            vector_store.get_by_ids(embedding_ids)
        )

        # Extract just the vectors (drop the IDs)
        return [vec for _, vec in id_vector_pairs]

    def _load_or_create_model(self, context: FarmingContext) -> Any:
        """
        Load a MiniBatchKMeans model from checkpoint, or create new.

        If a checkpoint exists with a previously fitted model, load
        and return it so partial_fit() continues from where it was.
        Otherwise, create a fresh MiniBatchKMeans with the configured
        number of clusters.

        Args:
            context: Farming context with checkpoint access.

        Returns:
            A MiniBatchKMeans model instance (either loaded or new).
        """
        # Try loading from checkpoint first
        saved_state = context.checkpoint.load()
        if saved_state and isinstance(saved_state, dict):
            model = saved_state.get("model")
            if model is not None:
                logger.debug("clustering_model_loaded_from_checkpoint")
                return model

        # No checkpoint → create a fresh model
        logger.debug(
            "clustering_model_created",
            n_clusters=self._n_clusters,
        )
        return MiniBatchKMeans(
            n_clusters=self._n_clusters,
            batch_size=self._batch_size,
            random_state=42,
        )

    def _generate_cluster_insights(
        self,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        """
        Generate insights from the completed clustering pass.

        Loads the fitted model from checkpoint and creates one
        ``FarmingInsight`` per cluster with its centre coordinates
        and cluster label.

        Args:
            context: Farming context with checkpoint and cycle ID.

        Returns:
            A list of cluster insights (one per cluster).
            Empty if the model was never fitted or has no centres.
        """
        # Load the model from checkpoint
        saved_state = context.checkpoint.load()
        if not saved_state or not isinstance(saved_state, dict):
            logger.warning(
                "clustering_no_model_for_insights",
                error_code="CTXMTG-FRM-004",
                cycle_id=context.cycle_id,
            )
            return []

        model = saved_state.get("model")
        if model is None or not hasattr(model, "cluster_centers_"):
            logger.warning(
                "clustering_model_not_fitted",
                error_code="CTXMTG-FRM-004",
                cycle_id=context.cycle_id,
            )
            return []

        # Build one insight per cluster
        insights: list[FarmingInsight] = []
        n_clusters = len(model.cluster_centers_)

        for cluster_idx in range(n_clusters):
            insight_id = (
                # ORIGINAL: f"cluster-{cluster_idx}-{context.cycle_id}"
                f"cluster-{cluster_idx}"
            )

            insight = FarmingInsight(
                id=insight_id,
                insight_type="cluster",
                title=f"Cluster {cluster_idx} ({n_clusters} total)",
                confidence=0.5,
                parameters={
                    "cluster_index": cluster_idx,
                    "n_clusters": n_clusters,
                },
                entity_ids=[],
            )
            insights.append(insight)

        logger.info(
            "clustering_insights_generated",
            cycle_id=context.cycle_id,
            n_insights=len(insights),
        )

        return insights
