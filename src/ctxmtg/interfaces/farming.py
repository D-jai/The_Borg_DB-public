# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Farming Interface ABC
=====================

This module defines the abstract base class for farming stages --
the components that mine accumulated knowledge data to discover
higher-order patterns (co-occurrences, trends, clusters, anomalies).

It also provides FarmingContext (the operational context passed to
every stage), CheckpointStore (a protocol for saving/loading partial
progress), and the FarmingStage ABC itself. Together, these form
the contract between the pipeline orchestrator and the 15 stages
(7 intelligence + 7 maintenance + 1 calibrator).

Farming is the system's long-term intelligence layer. While extraction
processes individual interactions (short-term memory), farming looks
across many interactions to find patterns that no single interaction
reveals (long-term memory).

Each farming stage focuses on one type of pattern analysis:
- Entity analytics (co-occurrence, frequency)
- Trend detection (temporal patterns)
- Clustering (grouping related interactions)
- Topic modeling (discovering themes)
- Graph analysis (relationship networks)
- Insight generation (meta-insights from other stages)
- Maintenance agents (consolidation, pruning, archival, defragmentation)
- Calibrator (adjusts pipeline weights from feedback signals)

Farming runs during idle time and writes its discoveries back to
the SQL store as FarmingInsight records. Stages receive a
FarmingContext with their time budget, checkpoint access, and
stage-specific config from the domain profile.

Depends on:
    - abc (Python's Abstract Base Class machinery)
    - dataclasses (FarmingContext)
    - typing (Protocol for CheckpointStore, Any for config values)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore -- data sources)
    - ctxmtg.models.farming (FarmingInsight -- output model)

Used by:
    - ctxmtg.farming.entity_analytics (implements FarmingStage)
    - ctxmtg.farming.trend_detection (implements FarmingStage)
    - ctxmtg.farming.clustering (implements FarmingStage)
    - ctxmtg.farming.pipeline (orchestrates multiple FarmingStage instances)
    - ctxmtg.farming.consolidator (maintenance: merges duplicate insights)
    - ctxmtg.farming.pruner (maintenance: expires stale insights)
    - ctxmtg.farming.calibrator (adjusts pipeline from feedback)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

# ---------------------------------------------------------------
# Import the storage interfaces and output model.
# FarmingStage reads from both stores and produces FarmingInsight objects.
# We import the ABCs here because farming stages need to accept
# any implementation of the stores (SQLite, PostgreSQL, etc.).
# ---------------------------------------------------------------
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.models.farming import FarmingInsight


# =====================================================================
# CheckpointStore Protocol -- Save/Load Partial Stage Progress
# =====================================================================


class CheckpointStore(Protocol):
    """Protocol for saving/loading farming stage checkpoint state.

    Implemented by SQLiteCheckpointStore which persists state as
    a BLOB in the farming_checkpoints table. Stages use this to
    save partial progress (e.g., a half-fitted KMeans model) so
    work isn't lost if the farming window ends or the process crashes.
    """

    def save(self, state: Any) -> None:
        """Serialize and persist the stage's current state."""
        ...

    def load(self) -> Any | None:
        """Load previously saved state. Returns None if no checkpoint."""
        ...


# =====================================================================
# FarmingContext -- Operational Context for Each Stage Execution
# =====================================================================


@dataclass
class FarmingContext:
    """Operational context passed to every farming stage.

    The pipeline orchestrator builds one FarmingContext per stage
    execution, providing the stage with its budget, checkpoint
    access, and stage-specific configuration from the domain profile.

    Attributes:
        cycle_id: The farming cycle number (monotonically increasing).
        budget_seconds: Maximum wall-clock time this stage may use.
            When exceeded, the stage should save checkpoint and return
            partial results. The orchestrator enforces this as a hard
            timeout as a safety net.
        checkpoint: Interface for saving/loading stage state across
            farming cycles. Used by long-running stages (e.g., clustering)
            to avoid re-processing on resume.
        config: Stage-specific configuration dict read from the domain
            profile's farming section. Each stage reads its own knobs
            (e.g., clustering reads n_clusters, batch_size).
    """

    cycle_id: int
    budget_seconds: float
    checkpoint: CheckpointStore
    config: dict[str, Any] = field(default_factory=dict)


# =====================================================================
# FarmingStage ABC -- Single Stage of the Farming Pipeline
# =====================================================================


class FarmingStage(ABC):
    """
    One stage of the farming pipeline.

    Each stage performs a specific type of pattern analysis over the
    accumulated knowledge data. Stages are independent and can run
    in any order, though the pipeline typically runs them sequentially
    (entity analytics → trends → clusters → topics → graph → insights).

    A stage reads from both the SQL and vector stores, analyzes the
    data, and returns a list of discovered insights. The pipeline
    orchestrator stores these insights back in the SQL store.

    Stages must be idempotent: running the same stage twice on the
    same data should produce the same (or very similar) insights.
    The pipeline handles deduplication of insights.

    The FarmingContext provides each stage with its time budget,
    checkpoint access (for saving partial progress), and any
    stage-specific configuration from the domain profile. Stages
    should check their budget and checkpoint periodically during
    long-running computations.

    Usage:
        context = FarmingContext(
            cycle_id=42,
            budget_seconds=30.0,
            checkpoint=sqlite_checkpoint_store,
            config={"min_co_occurrence": 3},
        )
        stage = EntityAnalyticsStage()
        insights = stage.run(sql_store, vector_store, context)
        for insight in insights:
            await sql_store.store_insight(insight)
    """

    @abstractmethod
    def run(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        """
        Execute this farming stage. Returns discovered insights.

        Reads data from both stores, performs pattern analysis, and
        returns any patterns or insights discovered. The caller is
        responsible for persisting the returned insights.

        This method is synchronous because farming computation is
        CPU-bound. If it needs to read from async stores, it should
        use asyncio.run() or equivalent internally.

        Stages should respect context.budget_seconds and save a
        checkpoint via context.checkpoint.save() before returning
        partial results when the budget is exhausted.

        Args:
            sql_store: The SQL store to read structured data from
                       (entities, facts, interactions, existing insights).
            vector_store: The vector store to read embeddings from
                          (for clustering, similarity analysis, etc.).
            context: Operational context with cycle ID, time budget,
                     checkpoint access, and stage-specific config.

        Returns:
            A list of FarmingInsight objects representing the patterns
            discovered during this analysis pass. May be empty if no
            new patterns are found.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the stage name (for logging and checkpointing).

        Each farming stage has a unique name used for:
        - Logging which stage is currently running
        - Checkpointing progress (resume after interruption)
        - Metrics reporting (how long each stage takes)

        Returns:
            A descriptive string name for this stage
            (e.g., "entity_analytics", "trend_detection", "clustering").
        """
        ...
