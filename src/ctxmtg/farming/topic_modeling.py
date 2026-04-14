# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Topic Modeling Stage (Intelligence Stage 4)
=============================================

Labels clusters discovered by the ClusteringStage using TF-IDF
keyword extraction.  For each cluster, gathers interaction content
associated with the cluster's entities, extracts the top keywords
via TfidfVectorizer, and emits a ``"topic"`` insight naming the
cluster's dominant theme.

This stage depends on the clustering stage having completed at least
one full pass -- it reads from the ``farming_clustering_progress``
table to verify that clustering data exists.  If no clustering data
is found, it returns empty immediately.

Graceful degradation: if ``scikit-learn`` is not installed (no
TfidfVectorizer available), the stage returns an empty list instead
of crashing.  This mirrors the clustering stage's behaviour on
Tier 0 devices.

Algorithm overview:
    1. Check the ``farming_clustering_progress`` table for evidence
       of a completed clustering pass.
    2. Read cluster assignments (progress rows map entity batches to
       cluster labels implicitly via their ordering).
    3. For each unique cluster, gather interaction content by joining
       entities → interactions through shared ``interaction_id``.
    4. Run ``TfidfVectorizer`` on the concatenated content per cluster
       to extract the top ``top_k_keywords`` terms.
    5. Emit one ``FarmingInsight`` per cluster with insight_type
       ``"topic"`` and a title composed of the top 3 keywords.

Depends on:
    - structlog (structured logging)
    - sklearn.feature_extraction.text.TfidfVectorizer (optional)
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- optional, unused tier 0-1)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async bridge)

Used by:
    - ctxmtg.farming.pipeline (registered as the fourth intelligence stage)
"""

from __future__ import annotations

from typing import Any

import structlog

from ctxmtg.farming.checkpoint import _run_async
from ctxmtg.interfaces.farming import FarmingContext, FarmingStage
from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.models.farming import FarmingInsight

# ---------------------------------------------------------------
# Module-level logger -- structured JSON output, no PII in logs.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.topic_modeling")

# ---------------------------------------------------------------
# Optional sklearn import.  If scikit-learn is not installed, the
# stage degrades gracefully -- returns an empty insights list.
# TfidfVectorizer is the only sklearn component used here.
# ---------------------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore[misc,assignment]
    _HAS_SKLEARN = False

# ---------------------------------------------------------------
# SQL queries -- kept as module constants for readability.
# ---------------------------------------------------------------

# Check whether clustering has completed at least once by counting
# distinct last_entity values.  If > 0, clustering has run.
CLUSTERING_DONE_SQL = """\
SELECT COUNT(DISTINCT last_entity) as cluster_count
FROM farming_clustering_progress
"""

# Read the distinct cluster assignments from the progress table.
# Each row's last_entity is the "end marker" for that batch.
# We use the set of all last_entity values as proxy cluster labels.
CLUSTER_ASSIGNMENTS_SQL = """\
SELECT DISTINCT last_entity
FROM farming_clustering_progress
ORDER BY last_entity ASC
"""

# For a range of entities (alphabetical window), get the interaction
# content.  We join entities → interactions to collect the raw text
# associated with each entity batch.
CLUSTER_CONTENT_SQL = """\
SELECT DISTINCT i.content
FROM entities e
JOIN interactions i ON e.interaction_id = i.id
WHERE e.name > :start_entity AND e.name <= :end_entity
LIMIT 200
"""

# Fallback query: when start_entity is '' (first batch), we need
# entities from the very beginning through end_entity.
CLUSTER_CONTENT_FIRST_SQL = """\
SELECT DISTINCT i.content
FROM entities e
JOIN interactions i ON e.interaction_id = i.id
WHERE e.name <= :end_entity
LIMIT 200
"""


class TopicModelingStage(FarmingStage):
    """
    Farming stage 4: TF-IDF keyword extraction for cluster labelling.

    Reads clustering progress to identify entity batches, gathers
    interaction content for each batch, and uses TfidfVectorizer
    to extract dominant keywords.  Emits ``"topic"`` insights with
    titles formed from the top 3 keywords.

    If scikit-learn is not installed or clustering has not yet run,
    the stage returns empty gracefully.

    Usage:
        stage = TopicModelingStage(max_features=500, top_k_keywords=5)
        insights = stage.run(sql_store, vector_store, context)
    """

    def __init__(
        self,
        max_features: int = 500,
        top_k_keywords: int = 5,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Configure the topic modeling stage.

        Args:
            max_features: Maximum number of TF-IDF features (vocabulary
                size).  Default 500 -- keeps computation lightweight.
            top_k_keywords: Number of top keywords to extract per
                cluster.  Default 5 -- enough for a descriptive label.
            llm: Optional LLM provider for future narrative generation.
                 Currently unused (tier 0-1 does not require an LLM).
        """
        self._max_features = max_features
        self._top_k_keywords = top_k_keywords
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface
    # -----------------------------------------------------------------

    def get_name(self) -> str:
        """Return the canonical stage name for logging/checkpointing."""
        return "topic_modeling"

    def run(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        """
        Execute topic modeling over clustered entity groups.

        Steps:
            1. Check if clustering has completed at least once.
            2. Read cluster assignments (entity batch endpoints).
            3. For each cluster batch, gather interaction content.
            4. Use TfidfVectorizer to extract top keywords.
            5. Emit one "topic" insight per cluster.

        This method is synchronous (per the FarmingStage contract)
        but calls async store methods via ``_run_async()``.

        Args:
            sql_store:    SQL store for entity/interaction data.
            vector_store: Vector store (unused by this stage).
            context:      Farming context with cycle ID and budget.

        Returns:
            List of FarmingInsight objects (type ``"topic"``).
            Empty if clustering hasn't run or sklearn unavailable.
        """
        logger.info(
            "topic_modeling_start",
            cycle_id=context.cycle_id,
            max_features=self._max_features,
            top_k_keywords=self._top_k_keywords,
            has_sklearn=_HAS_SKLEARN,
        )

        # ----------------------------------------------------------
        # Graceful degradation: no sklearn → no topic modeling.
        # ----------------------------------------------------------
        if not _HAS_SKLEARN:
            logger.info(
                "topic_modeling_skip_no_sklearn",
                cycle_id=context.cycle_id,
            )
            return []

        # ----------------------------------------------------------
        # Step 1: Check if clustering has completed at least once.
        # We look for any rows in farming_clustering_progress.
        # ----------------------------------------------------------
        count_rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(CLUSTERING_DONE_SQL, {})
        )

        # Extract the cluster count from the result
        cluster_count = 0
        if count_rows:
            cluster_count = count_rows[0].get("cluster_count", 0)

        if cluster_count == 0:
            logger.info(
                "topic_modeling_skip_no_clusters",
                cycle_id=context.cycle_id,
            )
            return []

        logger.debug(
            "topic_modeling_clusters_found",
            cluster_count=cluster_count,
        )

        # ----------------------------------------------------------
        # Step 2: Read cluster assignments (batch end-markers).
        # Each distinct last_entity marks the end of a batch.
        # ----------------------------------------------------------
        assignment_rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(CLUSTER_ASSIGNMENTS_SQL, {})
        )

        # List of batch end-markers (alphabetical order)
        batch_endpoints: list[str] = [
            row["last_entity"] for row in assignment_rows
        ]

        if not batch_endpoints:
            return []

        # ----------------------------------------------------------
        # Steps 3-5: For each cluster batch, gather content and
        # extract keywords via TF-IDF.
        # ----------------------------------------------------------
        insights: list[FarmingInsight] = []

        # Track the start entity for each batch window.  The first
        # batch starts at '' (the beginning of the alphabet).
        prev_entity: str = ""

        for cluster_idx, end_entity in enumerate(batch_endpoints):
            # Gather interaction content for this entity window
            content_docs = self._gather_cluster_content(
                sql_store, prev_entity, end_entity
            )

            # Extract keywords if we have content
            keywords: list[str] = []
            if content_docs:
                keywords = self._extract_keywords(content_docs)

            # Build the insight title from top 3 keywords
            if keywords:
                title_keywords = keywords[:3]
                title = ", ".join(title_keywords)
            else:
                # Fallback title when no keywords extracted
                title = f"Cluster {cluster_idx}"

            # Deterministic ID for deduplication across cycles
            insight_id = (
                # ORIGINAL: f"topic-{cluster_idx}-{context.cycle_id}"
                f"topic-{cluster_idx}"
            )

            # Confidence: higher when we have more keywords
            confidence = min(len(keywords) / self._top_k_keywords, 1.0)
            confidence = max(confidence, 0.1)

            insight = FarmingInsight(
                id=insight_id,
                insight_type="topic",
                title=title,
                confidence=confidence,
                parameters={
                    "cluster_index": cluster_idx,
                    "keywords": keywords,
                    "doc_count": len(content_docs),
                },
                entity_ids=[],
            )
            insights.append(insight)

            # Move the start marker for the next batch window
            prev_entity = end_entity

        logger.info(
            "topic_modeling_complete",
            cycle_id=context.cycle_id,
            insights_produced=len(insights),
        )

        return insights

    # =================================================================
    # Private helpers
    # =================================================================

    def _gather_cluster_content(
        self,
        sql_store: SQLStore,
        start_entity: str,
        end_entity: str,
    ) -> list[str]:
        """
        Gather interaction content for entities in the given window.

        Queries interactions joined through entities for the
        alphabetical range (start_entity, end_entity].  Returns a
        list of content strings (one per interaction).

        Args:
            sql_store:    SQL store for entity/interaction data.
            start_entity: Exclusive lower bound of the entity name range.
            end_entity:   Inclusive upper bound of the entity name range.

        Returns:
            List of interaction content strings in this cluster.
            May be empty if no interactions found.
        """
        # Use the first-batch query when start_entity is empty
        # (beginning of alphabet → no lower bound).
        if not start_entity:
            rows: list[dict[str, Any]] = _run_async(
                sql_store.execute_sql(
                    CLUSTER_CONTENT_FIRST_SQL,
                    {"end_entity": end_entity},
                )
            )
        else:
            rows = _run_async(
                sql_store.execute_sql(
                    CLUSTER_CONTENT_SQL,
                    {
                        "start_entity": start_entity,
                        "end_entity": end_entity,
                    },
                )
            )

        # Extract content strings, filtering out None/empty values
        return [
            row["content"]
            for row in rows
            if row.get("content")
        ]

    def _extract_keywords(
        self,
        documents: list[str],
    ) -> list[str]:
        """
        Extract top keywords from documents using TF-IDF.

        Fits a TfidfVectorizer on the provided documents and returns
        the feature names with the highest aggregate TF-IDF scores.

        Args:
            documents: List of text documents to analyse.

        Returns:
            List of top keywords (up to ``top_k_keywords``), sorted
            by descending TF-IDF score.  Empty if TF-IDF fails or
            no features extracted.
        """
        if not documents or TfidfVectorizer is None:
            return []

        try:
            # Fit the vectorizer on the cluster's documents.
            # stop_words='english' removes common words; max_features
            # caps the vocabulary size for efficiency.
            vectorizer = TfidfVectorizer(
                max_features=self._max_features,
                stop_words="english",
            )
            tfidf_matrix = vectorizer.fit_transform(documents)

            # Get feature names (vocabulary terms)
            feature_names = vectorizer.get_feature_names_out()

            # Sum TF-IDF scores across all documents for each term.
            # Higher aggregate score → more important to this cluster.
            scores = tfidf_matrix.sum(axis=0).A1  # type: ignore[union-attr]

            # Sort by score descending and take top_k
            top_indices = scores.argsort()[::-1][: self._top_k_keywords]
            keywords = [str(feature_names[i]) for i in top_indices]

            return keywords

        except Exception as exc:
            # Graceful: if TF-IDF fails for any reason (empty vocab,
            # encoding issues, etc.), log and return empty.
            logger.warning(
                "topic_modeling_tfidf_failed",
                error_code="CTXMTG-FRM-005",
                error=str(exc),
            )
            return []
