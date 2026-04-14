# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Graph Analysis Stage (Intelligence Stage 5)
==============================================

Builds a co-occurrence graph from entity data and applies the
PageRank algorithm to identify the most "central" entities --
those that are well-connected hubs in the knowledge graph.
For example, if "Alice" co-occurs with many other entities
across multiple interactions, she'll receive a high PageRank
score, suggesting she is a central figure.

This is the fifth of seven intelligence stages in the farming
pipeline.  It uses a pure-Python iterative power-method PageRank
(no external graph libraries like NetworkX) to keep the edge
deployment footprint minimal.

Algorithm overview:
    1. SEED: Query co-occurrence pairs from the entities table --
       pairs of entities that appear in the same interaction at
       least twice.  This ensures we only build graph edges for
       relationships with meaningful evidence.
    2. Build a bidirectional adjacency dict from the co-occurrence
       pairs.  Each edge carries a weight (shared interaction count).
    3. Compute PageRank using the iterative power method with a
       configurable damping factor (default 0.85) and iteration
       limit (default 50).  Convergence is detected when the L1
       norm of the rank-change vector drops below 1e-6.
    4. Emit ``FarmingInsight`` objects of type ``"relationship"``
       for the top 10 entities by PageRank score, annotated with
       their rank value and connection count.

Depends on:
    - structlog (structured logging)
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- optional, unused tier 0-1)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async bridge)

Used by:
    - ctxmtg.farming.pipeline (registered as the fifth intelligence stage)
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
logger = structlog.get_logger("ctxmtg.farming.graph_analysis")

# ---------------------------------------------------------------
# SQL query: co-occurrence pairs with at least 2 shared
# interactions.  The a.id < b.id condition deduplicates pairs
# (so we only get (Alice, Bob), not also (Bob, Alice)).
# Results are ordered by weight DESC and capped at 200 rows
# to keep the graph manageable on edge devices.
# ---------------------------------------------------------------
# ORIGINAL (disabled 2026-04-07): Hardcoded LIMIT 200 with no OFFSET.
# CO_OCCURRENCE_GRAPH_SQL = "... LIMIT 200"
CO_OCCURRENCE_GRAPH_SQL = """\
SELECT a.name as entity_a, b.name as entity_b,
       COUNT(DISTINCT a.interaction_id) as weight
FROM entities a
JOIN entities b
    ON a.interaction_id = b.interaction_id AND a.id < b.id
GROUP BY a.name, b.name
HAVING weight >= 2
ORDER BY weight DESC
LIMIT 200 OFFSET :offset
"""


class GraphAnalysisStage(FarmingStage):
    """
    Farming stage 5: relationship graph construction and PageRank.

    Builds a co-occurrence graph from entity data and uses the
    iterative power-method PageRank algorithm to find the most
    central entities in the knowledge base.  Central entities
    are those that co-occur with many other entities across
    multiple interactions, suggesting they are important hubs.

    The ``damping`` factor (default 0.85) controls how much rank
    is distributed via edges vs. the uniform "teleport" baseline.
    Higher damping means edges matter more; lower damping spreads
    rank more evenly.  The ``iterations`` parameter caps the
    number of power-method steps (default 50); convergence is
    usually achieved in 10-20 iterations.

    The optional ``llm`` parameter is accepted for API consistency
    with higher-tier stages but is not used in this stage.

    Usage:
        stage = GraphAnalysisStage(damping=0.85, iterations=50)
        insights = stage.run(sql_store, vector_store, context)
    """

    def __init__(
        self,
        damping: float = 0.85,
        iterations: int = 50,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Configure the graph analysis stage.

        Args:
            damping: PageRank damping factor (probability of following
                an edge vs. teleporting to a random node).  Default 0.85
                per the original PageRank paper.
            iterations: Maximum number of power-method iterations.
                Default 50 -- convergence usually happens in 10-20.
            llm: Optional LLM provider for future narrative generation.
                 Currently unused (tier 0-1 does not require an LLM).
        """
        self._damping = damping
        self._iterations = iterations
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface
    # -----------------------------------------------------------------

    def get_name(self) -> str:
        """Return the canonical stage name for logging/checkpointing."""
        return "graph_analysis"

    def run(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        """
        Execute graph analysis: build co-occurrence graph + PageRank.

        Steps:
            1. Query co-occurrence pairs from the entities table.
            2. Build a bidirectional weighted adjacency dictionary.
            3. Run iterative PageRank (power method, pure Python).
            4. Emit relationship insights for top-10 entities by rank.

        This method is synchronous (per the FarmingStage contract)
        but calls async store methods via ``_run_async()``.

        Args:
            sql_store:    SQL store to read entity data from.
            vector_store: Vector store (unused by this stage).
            context:      Farming context with cycle ID and budget.

        Returns:
            List of FarmingInsight objects (type ``"relationship"``).
            Empty list if no qualifying co-occurrence pairs exist.
        """
        logger.info(
            "graph_analysis_start",
            cycle_id=context.cycle_id,
            damping=self._damping,
            iterations=self._iterations,
        )

        # ----------------------------------------------------------
        # Step 1: Query co-occurrence pairs from the entities table.
        # Each row gives (entity_a, entity_b, weight) where weight
        # is the number of shared interactions.
        # ----------------------------------------------------------
        from ctxmtg.farming.progress import get_offset_with_wrap, update_offset

        total_pair_rows = _run_async(
            sql_store.execute_sql(
                "SELECT COUNT(*) as cnt FROM ("
                "SELECT a.name, b.name FROM entities a "
                "JOIN entities b ON a.interaction_id = b.interaction_id AND a.id < b.id "
                "GROUP BY a.name, b.name HAVING COUNT(DISTINCT a.interaction_id) >= 2"
                ")", {}
            )
        )
        total_pairs = total_pair_rows[0]["cnt"] if total_pair_rows else 0

        graph_offset = _run_async(
            get_offset_with_wrap(sql_store, "graph_analysis", total_pairs, 200)
        )

        rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(CO_OCCURRENCE_GRAPH_SQL, {"offset": graph_offset})
        )

        logger.debug(
            "co_occurrence_pairs_loaded",
            pair_count=len(rows),
            offset=graph_offset,
        )

        # Early exit: no co-occurrence data means no graph to analyse.
        if not rows:
            logger.info(
                "graph_analysis_complete",
                cycle_id=context.cycle_id,
                insights_produced=0,
                reason="no_co_occurrence_data",
            )
            return []

        # ----------------------------------------------------------
        # Step 2: Build bidirectional adjacency dictionary.
        # adj[a][b] = weight means entity_a and entity_b share
        # `weight` interactions.  We make it symmetric so PageRank
        # treats the graph as undirected.
        # ----------------------------------------------------------
        adj: dict[str, dict[str, int]] = {}

        for row in rows:
            entity_a: str = row["entity_a"]
            entity_b: str = row["entity_b"]
            weight: int = row["weight"]

            # Forward edge: a -> b
            if entity_a not in adj:
                adj[entity_a] = {}
            adj[entity_a][entity_b] = weight

            # Backward edge: b -> a (undirected graph)
            if entity_b not in adj:
                adj[entity_b] = {}
            adj[entity_b][entity_a] = weight

        # Collect the full set of nodes in the graph.
        nodes: list[str] = list(adj.keys())
        n: int = len(nodes)

        logger.debug(
            "graph_built",
            node_count=n,
            edge_count=len(rows),
        )

        # ----------------------------------------------------------
        # Step 3: Compute PageRank using the iterative power method.
        #
        # PageRank formula (per iteration):
        #   PR(node) = (1 - d) / N + d * Σ [ PR(src) * w(src→node) / Σw(src) ]
        #
        # Where:
        #   d     = damping factor (default 0.85)
        #   N     = total number of nodes
        #   src   = each node that has an edge to `node`
        #   w     = edge weight
        #   Σw(src) = total outgoing weight from `src`
        #
        # Convergence: stop when L1 norm of rank changes < 1e-6.
        # ----------------------------------------------------------
        rank: dict[str, float] = {node: 1.0 / n for node in nodes}

        # Pre-compute total outgoing weight for each node (used as
        # the denominator in the rank distribution formula).
        out_weight: dict[str, float] = {
            node: float(sum(adj[node].values()))
            for node in nodes
        }

        for iteration in range(self._iterations):
            new_rank: dict[str, float] = {}

            for node in nodes:
                # Sum incoming rank contributions from all neighbours.
                # Each source distributes its rank proportionally to
                # the edge weight, normalised by its total out-weight.
                incoming = sum(
                    rank[src] * adj[src][node] / out_weight[src]
                    for src in adj
                    if node in adj[src]
                )

                # PageRank formula: teleport + damped incoming rank
                new_rank[node] = (1.0 - self._damping) / n + self._damping * incoming

            # Check convergence: L1 norm of rank change vector.
            delta = sum(abs(new_rank[nd] - rank[nd]) for nd in nodes)

            rank = new_rank

            # Early termination if ranks have stabilised.
            if delta < 1e-6:
                logger.debug(
                    "pagerank_converged",
                    iterations_used=iteration + 1,
                )
                break

        # ----------------------------------------------------------
        # Step 4: Emit FarmingInsight for top-10 entities by PageRank.
        # Sort entities by rank descending, take top 10.
        # ----------------------------------------------------------
        sorted_entities = sorted(
            rank.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]

        insights: list[FarmingInsight] = []

        for entity_name, pagerank_score in sorted_entities:
            # Number of direct connections (edges) for this entity
            connection_count = len(adj.get(entity_name, {}))

            # Confidence: proportional to PageRank (higher rank =
            # more evidence of centrality).  Capped at 1.0.
            confidence = min(pagerank_score * n, 1.0)
            # Floor at 0.1 to avoid near-zero confidence.
            confidence = max(confidence, 0.1)

            # Deterministic ID for deduplication across cycles.
            insight_id = (
                # ORIGINAL: f"graph-{entity_name}-{context.cycle_id}"
                f"graph-{entity_name}"
            )

            insight = FarmingInsight(
                id=insight_id,
                insight_type="relationship",
                title=(
                    f"{entity_name} is central "
                    f"(PageRank: {pagerank_score:.3f})"
                ),
                confidence=confidence,
                parameters={
                    "pagerank": round(pagerank_score, 6),
                    "connections": connection_count,
                },
                # No canonical entity IDs at this stage
                entity_ids=[],
            )
            insights.append(insight)

        # Advance offset for next cycle
        _run_async(update_offset(sql_store, "graph_analysis", graph_offset + 200, len(rows)))

        logger.info(
            "graph_analysis_complete",
            cycle_id=context.cycle_id,
            insights_produced=len(insights),
            offset=graph_offset,
            top_entity=(
                sorted_entities[0][0] if sorted_entities else None
            ),
        )

        return insights
