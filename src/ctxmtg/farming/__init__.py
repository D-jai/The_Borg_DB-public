# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Farming Package
===============

This package implements "meta-intelligence farming" -- the process of
mining accumulated knowledge data to discover higher-order patterns,
trends, and insights that weren't explicitly stated in any single
interaction.

The farming pipeline runs in 17 stages organized into three groups:

Intelligence stages (7 stages -- discover new patterns):
    1. Entity analytics: frequency analysis, co-occurrence matrices
    2. Trend detection: temporal pattern detection over time windows
    3. Clustering: K-Means / HDBSCAN grouping of related entities
    4. Topic modeling: LDA / BERTopic for theme discovery
    5. Graph analysis: relationship graph construction + PageRank
    6. Insight generation: meta-insight storage and indexing
    7. Causal mining: causal relationship discovery across interactions

Self-learning (1 stage):
    8.  Feedback loop: mines query_quality_log for gap signals

Maintenance stages (9 stages -- keep the knowledge store healthy):
    9.  Consolidator: merges duplicate / near-duplicate insights
    10. Pruner: expires stale insights that are no longer relevant
    11. Completionist: fills gaps in entity profiles and fact coverage
    12. Linker: cross-references entities across different interactions
    13. Verifier: validates insight consistency and confidence scores
    14. Calibrator: adjusts pipeline weights from query-quality feedback
    15. Distiller: condenses entities into compact summaries with scores
    16. Archivist: moves old low-value data to cold storage
    17. Defragmenter: compacts and re-indexes storage for performance

Farming runs during idle periods (when the user isn't actively
querying) and feeds its results back into the query system to
improve future answer quality. See research/round-1/05-meta-
intelligence-farming.md for the complete design.

Submodules:
    - pipeline.py          : 16-stage farming pipeline orchestrator
    - entity_analytics.py  : Stage 1: frequency, co-occurrence
    - trend_detection.py   : Stage 2: temporal trends
    - clustering.py        : Stage 3: K-Means / HDBSCAN
    - topic_modeling.py    : Stage 4: LDA / BERTopic
    - graph_analysis.py    : Stage 5: relationship graph + PageRank
    - insight_generator.py : Stage 6: meta-insight storage
    - causal_miner.py      : Stage 7: causal relationship discovery
    - feedback_loop.py     : Stage 8: self-learning from query quality signals
    - consolidator.py      : Stage 9: duplicate insight merging
    - pruner.py            : Stage 10: stale insight expiration
    - completionist.py     : Stage 11: entity/fact gap filling
    - linker.py            : Stage 12: cross-interaction entity linking
    - verifier.py          : Stage 13: insight consistency validation
    - calibrator.py        : Stage 14: feedback-driven weight tuning
    - distiller.py         : Stage 15: entity summarisation + relevance scoring
    - archivist.py         : Stage 16: cold storage archival
    - defragmenter.py      : Stage 17: storage compaction + re-indexing
    - checkpoint.py        : Checkpoint persistence for stage progress
    - scheduler.py         : Idle-time scheduling
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ctxmtg.farming.archivist import ArchivistStage
from ctxmtg.farming.calibrator import CalibratorStage
from ctxmtg.farming.causal_miner import CausalMinerStage
from ctxmtg.farming.clustering import ClusteringStage
from ctxmtg.farming.completionist import CompletionistStage
from ctxmtg.farming.consolidator import ConsolidatorStage
from ctxmtg.farming.defragmenter import DefragmenterStage
from ctxmtg.farming.distiller import DistillerStage
from ctxmtg.farming.entity_analytics import EntityAnalyticsStage
from ctxmtg.farming.feedback_loop import FeedbackLoopStage
from ctxmtg.farming.graph_analysis import GraphAnalysisStage
from ctxmtg.farming.insight_generator import InsightGeneratorStage
from ctxmtg.farming.linker import LinkerStage
from ctxmtg.farming.pipeline import FarmingPipeline
from ctxmtg.farming.pruner import PrunerStage
from ctxmtg.farming.rationalizer import RationalizerStage
from ctxmtg.farming.scheduler import FarmingScheduler
from ctxmtg.farming.topic_modeling import TopicModelingStage
from ctxmtg.farming.trend_detection import TrendDetectionStage
from ctxmtg.farming.verifier import VerifierStage

if TYPE_CHECKING:
    from ctxmtg.interfaces.farming import FarmingStage
    from ctxmtg.interfaces.llm import LLMProvider


def create_default_stages(llm: LLMProvider | None = None) -> list[FarmingStage]:
    """
    Build the default ordered list of all 17 farming stages.

    Order: intelligence (7) -> self-learning (1) -> maintenance (9).
    Defragmenter runs last because it compacts storage after all
    other stages have finished writing.

    Args:
        llm: Optional LLM provider passed to stages that benefit
             from LLM enhancement. All stages degrade gracefully
             when llm is None.

    Returns:
        Ordered list of FarmingStage instances with default config.
    """
    # Derive archive.db path from the runtime data root (see
    # ctxmtg.paths). For multi-instance setups (e.g. Local_Emails)
    # CTXMTG_DB_PATH still wins so the archive lives next to the
    # knowledge.db on a custom path.
    db_path = os.environ.get("CTXMTG_DB_PATH", "")
    if db_path:
        archive_path = os.path.join(os.path.dirname(db_path), "archive.db")
    else:
        from ctxmtg import paths as _paths
        archive_path = str(_paths.get_archive_db_path())

    return [
        # Intelligence stages (1-7): discover new patterns
        EntityAnalyticsStage(llm=llm),
        TrendDetectionStage(),
        ClusteringStage(),
        TopicModelingStage(),
        GraphAnalysisStage(),
        InsightGeneratorStage(llm=llm),
        CausalMinerStage(),
        # Self-learning (8): mine query quality signals
        FeedbackLoopStage(llm=llm),
        # Maintenance stages (9-18): keep the knowledge store healthy
        RationalizerStage(),             # marks garbage entities (confidence → 0.1)
        ConsolidatorStage(),
        PrunerStage(llm=llm),
        CompletionistStage(),
        LinkerStage(),
        VerifierStage(),
        CalibratorStage(llm=llm),
        DistillerStage(llm=llm),
        ArchivistStage(archive_db_path=archive_path),  # archives garbage + cold entities
        DefragmenterStage(),
    ]


__all__ = [
    "ArchivistStage",
    "CalibratorStage",
    "CausalMinerStage",
    "ClusteringStage",
    "CompletionistStage",
    "ConsolidatorStage",
    "DefragmenterStage",
    "DistillerStage",
    "EntityAnalyticsStage",
    "FarmingPipeline",
    "FarmingScheduler",
    "FeedbackLoopStage",
    "GraphAnalysisStage",
    "InsightGeneratorStage",
    "LinkerStage",
    "PrunerStage",
    "RationalizerStage",
    "TopicModelingStage",
    "TrendDetectionStage",
    "VerifierStage",
    "create_default_stages",
]
