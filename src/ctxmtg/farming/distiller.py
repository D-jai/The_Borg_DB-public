# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Distiller Stage (Intelligence Stage: Entity Summarisation)
============================================================

Condenses raw entity data into compact, query-friendly summaries
stored in the ``distiller_summaries`` table.  For each unique entity
name in the knowledge base, the distiller gathers:

    - mention count and interaction spread (how many interactions
      reference this entity)
    - top predicates (the most common relationships this entity
      participates in, from the facts table)
    - co-entities (other entities that appear in the same interactions)
    - source instances (which ctxmtg instances contributed data for
      this entity)

It then computes a relevance score that balances frequency, spread,
and recency, and builds a short plain-text summary capped at
``max_summary_chars``.

Entities that fall below ``relevance_threshold`` are pruned from
the distiller_summaries table to keep it lean and focused on the
entities that matter most.

Algorithm overview:
    1. QUERY: Aggregate entity names with mention counts and last-seen
       timestamps from the entities table.
    2. ENRICH: For each entity, gather top predicates, co-entities,
       and source instances via targeted SQL queries.
    3. SCORE: Compute relevance = log(1 + mentions) * log(1 + interactions)
       / (1 + days_since_last / 30).
    4. SUMMARISE: Build a natural-language summary string capped at
       max_summary_chars.
    5. UPSERT: Write (INSERT OR REPLACE) into distiller_summaries.
    6. PRUNE: DELETE entities below the relevance threshold.
    7. REPORT: Return one FarmingInsight of type 'meta' summarising
       the distillation cycle.

Depends on:
    - math (log for relevance scoring)
    - json (serialise lists for TEXT columns)
    - datetime (parse timestamps, compute recency)
    - structlog (structured logging)
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- accepted for API consistency)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async bridge)

Used by:
    - ctxmtg.farming.pipeline (registered as a farming stage)
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
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
logger = structlog.get_logger("ctxmtg.farming.distiller")

# ---------------------------------------------------------------
# SQL queries -- kept as module constants for readability.
# ---------------------------------------------------------------

# Aggregate entity names with counts and last-seen timestamps.
# Groups by name (case-insensitive) to unify e.g. "alice" and "Alice".
ENTITY_AGGREGATE_SQL = """\
SELECT name, entity_type,
       COUNT(*) as mention_count,
       MAX(created_at) as last_seen
FROM entities
GROUP BY name COLLATE NOCASE
"""

# Top predicates for a specific entity by frequency.
# Joins facts → entities on subject_entity_id to find which
# predicates this entity appears in most often.
TOP_PREDICATES_SQL = """\
SELECT predicate, COUNT(*) as cnt
FROM facts f
JOIN entities e ON f.subject_entity_id = e.id
WHERE e.name = :name COLLATE NOCASE
  AND predicate NOT IN ('co_occurs_with')
GROUP BY predicate
ORDER BY cnt DESC
LIMIT :max_predicates
"""

# Co-entities: other entities that appear in the same interactions.
# Uses a self-join on interaction_id to find entities that share
# interactions with the target entity.
CO_ENTITIES_SQL = """\
SELECT DISTINCT e2.name
FROM entities e1
JOIN entities e2 ON e1.interaction_id = e2.interaction_id
WHERE e1.name = :name COLLATE NOCASE
  AND e2.name != e1.name COLLATE NOCASE
LIMIT :max_co_entities
"""

# Source instances: which ctxmtg instances contributed data for
# this entity (e.g., "laptop", "desktop", "pi").
SOURCE_INSTANCES_SQL = """\
SELECT DISTINCT source_instance
FROM entities
WHERE name = :name COLLATE NOCASE
"""

# Count distinct interactions for an entity (for relevance scoring).
INTERACTION_COUNT_SQL = """\
SELECT COUNT(DISTINCT interaction_id) as interaction_count
FROM entities
WHERE name = :name COLLATE NOCASE
"""

# UPSERT: write or overwrite the distiller summary for an entity.
# Uses INSERT OR REPLACE because entity_name is the PRIMARY KEY.
UPSERT_SUMMARY_SQL = """\
INSERT OR REPLACE INTO distiller_summaries
    (entity_name, entity_type, summary, source_instances,
     top_co_entities, top_predicates, relevance_score,
     updated_at, cycle_id)
VALUES
    (:entity_name, :entity_type, :summary, :source_instances,
     :top_co_entities, :top_predicates, :relevance_score,
     :updated_at, :cycle_id)
"""

# PRUNE: remove entities below the relevance threshold.
# Keeps the distiller_summaries table lean and focused.
PRUNE_SQL = """\
DELETE FROM distiller_summaries
WHERE relevance_score < :threshold
"""


class DistillerStage(FarmingStage):
    """
    Farming stage: entity summarisation and relevance scoring.

    Condenses raw entity data from the ``entities`` and ``facts``
    tables into compact summaries stored in ``distiller_summaries``.
    Each summary includes co-entity relationships, top predicates,
    source instance provenance, and a relevance score.

    Entities that fall below ``relevance_threshold`` are pruned from
    the summaries table to keep it focused on the entities that
    matter most to the user.

    The optional ``llm`` parameter is accepted for API consistency
    with higher-tier stages but is not used in this stage's current
    implementation.  Future iterations may use it for richer
    narrative summaries.

    Usage:
        stage = DistillerStage(max_summary_chars=500)
        insights = stage.run(sql_store, vector_store, context)
    """

    def __init__(
        self,
        max_summary_chars: int = 500,
        max_co_entities: int = 5,
        max_predicates: int = 5,
        relevance_threshold: float = 0.1,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Configure the distiller stage.

        Args:
            max_summary_chars: Maximum length of the plain-text summary
                per entity.  Longer summaries are truncated with '...'.
            max_co_entities:   Maximum number of co-entities to include
                in each summary.  Default 5.
            max_predicates:    Maximum number of top predicates to record
                per entity.  Default 5.
            relevance_threshold: Minimum relevance score an entity must
                have to be kept in distiller_summaries.  Entities scoring
                below this are pruned.  Default 0.1.
            llm: Optional LLM provider for future narrative generation.
                 Currently unused.
        """
        self._max_summary_chars = max_summary_chars
        self._max_co_entities = max_co_entities
        self._max_predicates = max_predicates
        self._relevance_threshold = relevance_threshold
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface
    # -----------------------------------------------------------------

    def get_name(self) -> str:
        """Return the canonical stage name for logging/checkpointing."""
        return "distiller"

    def run(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        """
        Execute the distiller: summarise, score, upsert, prune.

        Steps:
            1. Aggregate entity names with mention counts.
            2. For each entity, gather predicates, co-entities, sources.
            3. Compute relevance score.
            4. Build summary text capped at max_summary_chars.
            5. UPSERT into distiller_summaries.
            6. PRUNE entities below relevance_threshold.
            7. Return a meta insight summarising the cycle.

        This method is synchronous (per the FarmingStage contract)
        but calls async store methods via ``_run_async()``.

        Args:
            sql_store:    SQL store to read entity/fact data from.
            vector_store: Vector store (unused by this stage).
            context:      Farming context with cycle ID and budget.

        Returns:
            List containing one FarmingInsight of type 'meta'
            summarising the distillation results.  Empty list if
            no entities exist to distill.
        """
        logger.info(
            "distiller_start",
            cycle_id=context.cycle_id,
            max_summary_chars=self._max_summary_chars,
            relevance_threshold=self._relevance_threshold,
        )

        # ----------------------------------------------------------
        # Step 1: Aggregate entity names with counts and timestamps.
        # ----------------------------------------------------------
        entity_rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(ENTITY_AGGREGATE_SQL, {})
        )

        # Nothing to do if the entities table is empty.
        if not entity_rows:
            logger.info(
                "distiller_noop",
                cycle_id=context.cycle_id,
                reason="no_entities",
            )
            return []

        logger.debug(
            "distiller_entities_loaded",
            entity_count=len(entity_rows),
        )

        # ----------------------------------------------------------
        # Step 2-5: Process each entity: enrich, score, summarise,
        # and upsert into distiller_summaries.
        # ----------------------------------------------------------
        now_iso = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )

        # Track stats for the meta insight returned at the end.
        upserted_count = 0
        pruned_count = 0

        for row in entity_rows:
            entity_name: str = row["name"]
            entity_type: str = row["entity_type"]
            mention_count: int = row["mention_count"]
            last_seen: str = row.get("last_seen") or now_iso

            # -- 2a: Top predicates for this entity ----------------
            predicate_rows: list[dict[str, Any]] = _run_async(
                sql_store.execute_sql(
                    TOP_PREDICATES_SQL,
                    {"name": entity_name, "max_predicates": self._max_predicates},
                )
            )
            top_predicates = [r["predicate"] for r in predicate_rows]

            # -- 2b: Co-entities from shared interactions ----------
            co_entity_rows: list[dict[str, Any]] = _run_async(
                sql_store.execute_sql(
                    CO_ENTITIES_SQL,
                    {"name": entity_name, "max_co_entities": self._max_co_entities},
                )
            )
            co_entities = [r["name"] for r in co_entity_rows]

            # -- 2c: Source instances (which devices contributed) ---
            source_rows: list[dict[str, Any]] = _run_async(
                sql_store.execute_sql(
                    SOURCE_INSTANCES_SQL,
                    {"name": entity_name},
                )
            )
            source_instances = [r["source_instance"] for r in source_rows]

            # -- 2d: Interaction count (for relevance scoring) -----
            ic_rows: list[dict[str, Any]] = _run_async(
                sql_store.execute_sql(
                    INTERACTION_COUNT_SQL,
                    {"name": entity_name},
                )
            )
            interaction_count = ic_rows[0]["interaction_count"] if ic_rows else 0

            # -- 3: Compute relevance score ------------------------
            # relevance = log(1 + mentions) * log(1 + interactions)
            #             / (1 + days_since_last / 30)
            # This rewards entities that are both frequent and spread
            # across many interactions, with a recency decay factor.
            days_since_last = _days_since(last_seen)
            relevance_score = (
                math.log(1 + mention_count)
                * math.log(1 + interaction_count)
                / (1 + days_since_last / 30)
            )

            # -- 4: Build summary text -----------------------------
            summary = _build_summary(
                entity_name=entity_name,
                entity_type=entity_type,
                mention_count=mention_count,
                interaction_count=interaction_count,
                top_predicates=top_predicates,
                co_entities=co_entities,
                max_chars=self._max_summary_chars,
            )

            # -- 5: UPSERT into distiller_summaries ----------------
            _run_async(
                sql_store.execute_sql(
                    UPSERT_SUMMARY_SQL,
                    {
                        "entity_name": entity_name,
                        "entity_type": entity_type,
                        "summary": summary,
                        "source_instances": json.dumps(source_instances),
                        "top_co_entities": json.dumps(co_entities),
                        "top_predicates": json.dumps(top_predicates),
                        "relevance_score": round(relevance_score, 6),
                        "updated_at": now_iso,
                        "cycle_id": context.cycle_id,
                    },
                )
            )
            upserted_count += 1

        # ----------------------------------------------------------
        # Step 6: PRUNE entities below relevance_threshold.
        # ----------------------------------------------------------
        _run_async(
            sql_store.execute_sql(
                PRUNE_SQL,
                {"threshold": self._relevance_threshold},
            )
        )

        # Commit the writes explicitly.  The execute_sql method does
        # NOT auto-commit, so the UPSERT and PRUNE statements leave
        # an implicit transaction open.  If we don't commit here, the
        # pipeline's store_insight() call will fail with "cannot start
        # a transaction within a transaction".
        _run_async(_commit_db(sql_store))

        # Count how many were pruned (entities upserted minus those
        # remaining in the table).
        remaining_rows: list[dict[str, Any]] = _run_async(
            sql_store.execute_sql(
                "SELECT COUNT(*) as cnt FROM distiller_summaries", {}
            )
        )
        remaining = remaining_rows[0]["cnt"] if remaining_rows else 0
        pruned_count = upserted_count - remaining

        logger.info(
            "distiller_complete",
            cycle_id=context.cycle_id,
            upserted=upserted_count,
            pruned=pruned_count,
            remaining=remaining,
        )

        # ----------------------------------------------------------
        # Step 7: Return a meta insight summarising the cycle.
        # ----------------------------------------------------------
        insight = FarmingInsight(
            id=f"distiller-cycle-{context.cycle_id}",
            insight_type="meta",
            title=f"Distiller cycle {context.cycle_id}: "
                  f"{upserted_count} entities processed, "
                  f"{pruned_count} pruned",
            description=(
                f"Distilled {upserted_count} entities into summaries. "
                f"{pruned_count} fell below relevance threshold "
                f"({self._relevance_threshold}) and were pruned. "
                f"{remaining} summaries remain."
            ),
            confidence=1.0,
            parameters={
                "upserted": upserted_count,
                "pruned": pruned_count,
                "remaining": remaining,
                "relevance_threshold": self._relevance_threshold,
            },
        )

        return [insight]


# =====================================================================
# Private helpers
# =====================================================================


def _days_since(iso_timestamp: str) -> float:
    """
    Compute the number of days between ``iso_timestamp`` and now.

    Handles various ISO-8601 formats gracefully.  Returns 0.0 if the
    timestamp cannot be parsed (defensive -- avoids crashing the
    distiller on bad data).
    """
    try:
        # Try parsing with microseconds (the format used by SQLite's
        # strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        return max(delta.total_seconds() / 86400.0, 0.0)
    except (ValueError, AttributeError):
        # Unparseable timestamp -- treat as "just now" to avoid
        # penalising the entity with a huge recency penalty.
        return 0.0


def _build_summary(
    entity_name: str,
    entity_type: str,
    mention_count: int,
    interaction_count: int,
    top_predicates: list[str],
    co_entities: list[str],
    max_chars: int,
) -> str:
    """
    Build a plain-text summary for an entity, capped at max_chars.

    The summary includes the entity's type, mention frequency,
    interaction spread, top predicates, and co-entities.  If the
    full text exceeds max_chars it is truncated with '...'.

    Args:
        entity_name:      The entity's canonical name.
        entity_type:      The entity's type (person, org, etc.).
        mention_count:    Total mentions across all interactions.
        interaction_count: Number of distinct interactions.
        top_predicates:   Most common predicates (e.g., "proposed").
        co_entities:      Entities that co-occur with this one.
        max_chars:        Maximum summary length.

    Returns:
        A plain-text summary string, guaranteed ≤ max_chars.
    """
    # Start with the entity identity and frequency stats.
    parts = [
        f"{entity_name} ({entity_type}): "
        f"{mention_count} mentions across {interaction_count} interactions."
    ]

    # Add top predicates if available.
    if top_predicates:
        pred_str = ", ".join(top_predicates)
        parts.append(f"Key actions: {pred_str}.")

    # Add co-entities if available.
    if co_entities:
        co_str = ", ".join(co_entities)
        parts.append(f"Often appears with: {co_str}.")

    # Join all parts with spaces.
    full_summary = " ".join(parts)

    # Truncate if necessary, leaving room for '...'
    if len(full_summary) > max_chars:
        full_summary = full_summary[: max_chars - 3] + "..."

    return full_summary


async def _commit_db(sql_store: SQLStore) -> None:
    """
    Explicitly commit the underlying database connection.

    The SQLStore.execute_sql() method does NOT auto-commit write
    statements.  Farming stages that write data (like the distiller)
    must commit before returning, otherwise the pipeline's subsequent
    store_insight() call will fail with a transaction conflict.
    """
    db = sql_store._ensure_db()
    await db.commit()
