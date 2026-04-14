# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Hive Farming Stages
====================

Three intelligence-mining stages that run over the hive's merged
collective data.  These discover patterns that no single local can see.

Stage 1: Cross-Stream Scoring + Coverage Gaps
Stage 2: Latent Relationship Discovery (2-hop co-entity graph)
Stage 3: Insight Correlation (independent discoveries from different locals)
"""

from ctxmtg.sync.stages.cross_stream import CrossStreamStage
from ctxmtg.sync.stages.insight_correlation import InsightCorrelationStage
from ctxmtg.sync.stages.latent_discovery import LatentDiscoveryStage

__all__ = [
    "CrossStreamStage",
    "LatentDiscoveryStage",
    "InsightCorrelationStage",
]
