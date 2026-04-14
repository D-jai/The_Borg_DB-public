# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Farming Data Models
===================

This module defines the data models for the meta-intelligence farming
pipeline. Farming is the process of mining accumulated knowledge data
(entities, facts, interactions) to discover higher-order patterns:
co-occurrence relationships, temporal trends, topic clusters, and
anomalies.

The farming pipeline runs during idle time and produces FarmingInsight
objects that are stored back in the SQL database. These insights are
then available for queries ("what are the trending topics this month?")
and for improving future extractions (feedback loop).

In the system architecture, farming sits "above" the storage layer.
It reads from both SQL and vector stores, discovers patterns, and
writes insights back to SQL. It's the system's long-term memory
and pattern recognition capability.

Depends on:
    - pydantic (validation, serialization, field constraints)
    - datetime (timestamps for insight creation and expiration)

Used by:
    - ctxmtg.farming.pipeline (produces FarmingInsight objects)
    - ctxmtg.farming.entity_analytics (co-occurrence insights)
    - ctxmtg.farming.trend_detection (temporal trend insights)
    - ctxmtg.farming.clustering (cluster insights)
    - ctxmtg.storage.sqlite (stores and retrieves FarmingInsight)
    - ctxmtg.query.executor (queries can return farming insights)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------
# FarmingInsight model: a pattern or relationship discovered by
# the farming pipeline. Each insight has a type (cluster, trend,
# anomaly, relationship), evidence (IDs of supporting records),
# and a confidence score.
# ---------------------------------------------------------------
class FarmingInsight(BaseModel):
    """
    A pattern or insight discovered by the farming pipeline.

    Farming insights represent higher-order knowledge that emerges from
    analyzing accumulated data over time. They are not directly extracted
    from any single interaction -- instead, they are discovered by mining
    patterns across many interactions.

    Types of insights:
    - "cluster": a group of related interactions sharing a theme
    - "trend": an entity or topic whose frequency is changing over time
    - "anomaly": something unusual (sudden spike, unexpected co-occurrence)
    - "relationship": a strong connection between two entities

    Insights have an optional expiration (expires_at) because patterns
    can become stale. The farming pipeline refreshes insights on each
    cycle and may supersede or remove expired ones.

    Usage:
        insight = FarmingInsight(
            id="insight-001",
            insight_type="trend",
            title="OAuth2 mentions increasing",
            description="OAuth2 appeared in 80% of meetings this week, up from 20%",
            evidence=["interaction-001", "interaction-005", "interaction-012"],
            confidence=0.85,
            parameters={"slope": 0.6, "window_days": 7},
        )
    """

    # Unique identifier for this insight
    id: str

    # What kind of pattern this represents: "cluster", "trend", "anomaly", "relationship"
    insight_type: str

    # Human-readable title summarizing the insight
    title: str

    # Optional longer description with details about the pattern
    description: str | None = None

    # IDs of interactions, entities, or facts that support this insight
    evidence: list[str] = Field(default_factory=list)

    # IDs of entities this insight relates to. Enables efficient lookup
    # by the ContextEnricher when building enriched entity context.
    # For example, a 'relationship' insight about Alice and Bob would
    # list both entity IDs here.
    entity_ids: list[str] = Field(default_factory=list)

    # How confident the farming pipeline is in this insight (0.0 to 1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    # Additional numerical/structural parameters about the pattern
    # (e.g., {"slope": 0.6, "cluster_size": 15, "purity": 0.82})
    parameters: dict[str, Any] = Field(default_factory=dict)

    # When this insight was discovered
    created_at: datetime | None = None

    # When this insight should be considered stale (None = never expires)
    expires_at: datetime | None = None
