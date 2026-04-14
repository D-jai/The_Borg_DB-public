# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Context Enricher (Hive Side)
=============================

This module builds rich entity context from the full hive dataset.
For each entity, it queries ALL related facts, co-occurring entities,
temporal patterns, and predicate distributions across all instances.
The result is an enriched context summary suitable for re-embedding
with richer vectors than any single instance could produce.

Why is the hive context richer than instance context?
    The hive aggregates data from ALL instances (meetings, emails,
    Slack, CRM, etc.). An entity like "Alice Chen" might appear in
    meetings (proposed OAuth2), emails (follow-up to Bob), and Slack
    (mentioned in #engineering). The hive sees ALL of this; a single
    instance sees only its own slice. The enriched context captures
    the full picture, producing embeddings that surface entities with
    higher relevance for cross-domain queries.

Two summary modes:
    - LLM mode:      If an LLMProvider is available, the enricher sends
                      the raw data to the LLM for a natural-language summary.
    - Template mode:  If no LLM is available, a template-based summary is
                      generated (f"Entity '{name}' has {n} facts across
                      {m} instances. Top predicates: {p}.").

See research/notes/hive-sync-design.md § "Hive Enrichment Worker"
for the full design rationale.

Depends on:
    - ctxmtg.interfaces.storage (SQLStore for hive SQL queries)
    - ctxmtg.interfaces.llm (LLMProvider for optional LLM summaries)
    - ctxmtg.exceptions (SyncError for error reporting)

Used by:
    - ctxmtg.sync.hive_enrichment (HiveEnrichmentWorker enriches entities)
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from ctxmtg.exceptions import SyncError
from ctxmtg.interfaces.llm import LLMProvider

# ---------------------------------------------------------------
# Module-level logger -- structured JSON output, no PII in logs.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.sync.context_enricher")


class ContextEnricher:
    """
    Builds rich entity context from the full hive dataset.

    For each entity, queries ALL related facts (entity is subject or
    object), co-occurring entities (in same interactions), predicate
    distribution (GROUP BY predicate, COUNT), temporal patterns
    (earliest/latest/frequency), and source instance distribution.
    Produces an enriched context dict suitable for re-embedding.

    Usage:
        enricher = ContextEnricher(hive_store=hive_sql_store, llm=mock_llm)
        result = await enricher.enrich_entity("ent-001")
        # result = {"summary": "...", "co_entities": [...], ...}

        batch = await enricher.enrich_batch(["ent-001", "ent-002"])
    """

    def __init__(
        self,
        hive_store: Any,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Prepare the context enricher.

        Args:
            hive_store: The hive SQLStore instance (provides execute_sql
                        for querying the hive database).
            llm:        Optional LLMProvider for generating natural-language
                        summaries. If None, template-based summaries are used.
        """
        self._hive_store = hive_store
        self._llm = llm

    # =================================================================
    # Public API: enrich one entity
    # =================================================================

    async def enrich_entity(self, entity_id: str) -> dict[str, Any]:
        """
        Build enriched context for one entity.

        Queries the hive for all related facts, co-occurring entities,
        predicate distributions, temporal patterns, and source instance
        distributions.  If an LLM is available, generates a natural-
        language summary; otherwise uses a template-based summary.

        Args:
            entity_id: The unique ID of the entity to enrich.

        Returns:
            Dict with keys:
                - summary (str): Natural-language or template summary.
                - co_entities (list[str]): Names of co-occurring entities.
                - predicate_distribution (dict[str, int]): Predicate → count.
                - temporal_pattern (str): Description of temporal activity.
                - source_instances (list[str]): Instances that contributed.
                - fact_count (int): Total related facts.

        Raises:
            SyncError: If enrichment fails due to a database error.
        """
        try:
            # -------------------------------------------------------
            # Step 1: Fetch the entity's own record from the hive.
            # -------------------------------------------------------
            entity_info = await self._get_entity_info(entity_id)
            if entity_info is None:
                logger.warning(
                    "enrich_entity_not_found",
                    error_code="CTXMTG-SYN-004",
                    entity_id=entity_id,
                )
                return self._empty_enrichment(entity_id)

            entity_name = entity_info.get("name", "Unknown")
            entity_type = entity_info.get("entity_type", "other")

            # -------------------------------------------------------
            # Step 2: Query all related facts (entity as subject or object).
            # -------------------------------------------------------
            related_facts = await self._get_related_facts(entity_id)
            fact_count = len(related_facts)

            # -------------------------------------------------------
            # Step 3: Find co-occurring entities (entities in the same
            # interactions as this entity).
            # -------------------------------------------------------
            co_entities = await self._get_co_entities(entity_id)

            # -------------------------------------------------------
            # Step 4: Compute predicate distribution (GROUP BY predicate).
            # -------------------------------------------------------
            predicate_distribution = await self._get_predicate_distribution(entity_id)

            # -------------------------------------------------------
            # Step 5: Compute temporal pattern (earliest, latest, frequency).
            # -------------------------------------------------------
            temporal_pattern = await self._get_temporal_pattern(entity_id)

            # -------------------------------------------------------
            # Step 6: Get source instance distribution.
            # -------------------------------------------------------
            source_instances = await self._get_source_instances(entity_id)

            # -------------------------------------------------------
            # Step 7: Build the summary (LLM or template).
            # -------------------------------------------------------
            raw_data = {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "fact_count": fact_count,
                "co_entities": co_entities,
                "predicate_distribution": predicate_distribution,
                "temporal_pattern": temporal_pattern,
                "source_instances": source_instances,
                "related_facts": related_facts,
            }

            summary = await self._generate_summary(raw_data)

            return {
                "summary": summary,
                "co_entities": co_entities,
                "predicate_distribution": predicate_distribution,
                "temporal_pattern": temporal_pattern,
                "source_instances": source_instances,
                "fact_count": fact_count,
            }

        except SyncError:
            raise
        except Exception as exc:
            logger.error(
                "context_enrichment_failed",
                error_code="CTXMTG-SYN-004",
                entity_id=entity_id,
                error=str(exc),
            )
            raise SyncError(
                f"Failed to enrich entity {entity_id}: {exc}",
                error_code="CTXMTG-SYN-004",
            ) from exc

    # =================================================================
    # Public API: batch enrichment
    # =================================================================

    async def enrich_batch(
        self,
        entity_ids: list[str],
        batch_size: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Enrich a batch of entities.

        Processes entity_ids in chunks of batch_size to avoid
        overwhelming the database or LLM with too many concurrent
        requests.  Returns results in the same order as entity_ids.

        Args:
            entity_ids:  List of entity IDs to enrich.
            batch_size:  Maximum entities to process per batch.
                         Default 50 balances throughput and resource use.

        Returns:
            List of enrichment dicts (same order as entity_ids).
        """
        results: list[dict[str, Any]] = []

        for i in range(0, len(entity_ids), batch_size):
            batch = entity_ids[i : i + batch_size]
            for eid in batch:
                result = await self.enrich_entity(eid)
                results.append(result)

            logger.debug(
                "enrich_batch_progress",
                processed=min(i + batch_size, len(entity_ids)),
                total=len(entity_ids),
            )

        return results

    # =================================================================
    # Private helpers: database queries against the hive
    # =================================================================

    async def _get_entity_info(self, entity_id: str) -> dict[str, Any] | None:
        """
        Fetch basic entity information from the hive.

        Returns a dict with 'name', 'entity_type', 'interaction_id',
        etc.  Returns None if the entity is not found.
        """
        rows = await self._hive_store.execute_sql(
            "SELECT id, name, entity_type, interaction_id, source_instance, "
            "created_at FROM entities WHERE id = :entity_id",
            {"entity_id": entity_id},
        )
        if not rows:
            return None
        return rows[0]

    async def _get_related_facts(self, entity_id: str) -> list[dict[str, Any]]:
        """
        Fetch all facts where this entity is subject or object.

        This is the core query: it collects every fact triple that
        mentions the entity, across ALL instances and interactions.
        """
        rows = await self._hive_store.execute_sql(
            "SELECT f.id, f.interaction_id, f.predicate, "
            "f.subject_entity_id, f.object_entity_id, f.object_literal, "
            "f.confidence, f.source_instance, f.created_at "
            "FROM facts f "
            "WHERE f.subject_entity_id = :eid OR f.object_entity_id = :eid",
            {"eid": entity_id},
        )
        return rows

    async def _get_co_entities(self, entity_id: str) -> list[str]:
        """
        Find entities that co-occur with this entity.

        Co-occurring entities are those that appear in the same
        interactions as the target entity. Returns a deduplicated
        list of entity names (excluding the target entity itself).
        """
        # First, get the interaction IDs where this entity appears
        rows = await self._hive_store.execute_sql(
            "SELECT DISTINCT interaction_id FROM entities "
            "WHERE id = :eid",
            {"eid": entity_id},
        )
        if not rows:
            return []

        interaction_ids = [r["interaction_id"] for r in rows]

        # Then find other entities in those same interactions
        # (Use a parameterised IN clause)
        co_names: list[str] = []
        for iid in interaction_ids:
            co_rows = await self._hive_store.execute_sql(
                "SELECT DISTINCT name FROM entities "
                "WHERE interaction_id = :iid AND id != :eid",
                {"iid": iid, "eid": entity_id},
            )
            for r in co_rows:
                name = r["name"]
                if name not in co_names:
                    co_names.append(name)

        return co_names

    async def _get_predicate_distribution(
        self, entity_id: str
    ) -> dict[str, int]:
        """
        Compute the predicate distribution for an entity.

        Returns a dict mapping predicate strings to their occurrence
        counts (e.g., {"proposed": 3, "leads": 1, "decided": 2}).
        Counts facts where the entity is the subject.
        """
        rows = await self._hive_store.execute_sql(
            "SELECT predicate, COUNT(*) as cnt "
            "FROM facts "
            "WHERE subject_entity_id = :eid "
            "GROUP BY predicate "
            "ORDER BY cnt DESC",
            {"eid": entity_id},
        )
        return {r["predicate"]: r["cnt"] for r in rows}

    async def _get_temporal_pattern(self, entity_id: str) -> str:
        """
        Compute the temporal pattern for an entity.

        Returns a human-readable string describing when the entity
        is most active: earliest appearance, latest appearance, and
        total occurrence count.
        """
        rows = await self._hive_store.execute_sql(
            "SELECT MIN(f.created_at) as earliest, "
            "MAX(f.created_at) as latest, "
            "COUNT(*) as total "
            "FROM facts f "
            "WHERE f.subject_entity_id = :eid OR f.object_entity_id = :eid",
            {"eid": entity_id},
        )
        if not rows or rows[0]["total"] == 0:
            return "No temporal data available."

        row = rows[0]
        earliest = row["earliest"] or "unknown"
        latest = row["latest"] or "unknown"
        total = row["total"]

        return f"Active from {earliest} to {latest}, {total} occurrences."

    async def _get_source_instances(self, entity_id: str) -> list[str]:
        """
        Get the list of source instances that contributed data
        about this entity.

        Returns a deduplicated list of instance names (e.g.,
        ["laptop", "phone", "desktop"]).
        """
        rows = await self._hive_store.execute_sql(
            "SELECT DISTINCT source_instance FROM entities "
            "WHERE id = :eid",
            {"eid": entity_id},
        )
        # Also check facts for source instances
        fact_rows = await self._hive_store.execute_sql(
            "SELECT DISTINCT source_instance FROM facts "
            "WHERE subject_entity_id = :eid OR object_entity_id = :eid",
            {"eid": entity_id},
        )

        instances: list[str] = []
        for r in rows:
            inst = r["source_instance"]
            if inst and inst not in instances:
                instances.append(inst)
        for r in fact_rows:
            inst = r["source_instance"]
            if inst and inst not in instances:
                instances.append(inst)

        return instances

    # =================================================================
    # Summary generation
    # =================================================================

    async def _generate_summary(self, raw_data: dict[str, Any]) -> str:
        """
        Generate a summary from enrichment data.

        If an LLM is available and reports is_available(), sends the
        raw data to the LLM for a natural-language summary.  Otherwise
        falls back to a template-based summary.

        Args:
            raw_data: Dict with entity_name, entity_type, fact_count,
                      co_entities, predicate_distribution, temporal_pattern,
                      source_instances.

        Returns:
            A summary string.
        """
        if self._llm is not None and self._llm.is_available():
            return self._llm_summary(raw_data)
        return self._template_summary(raw_data)

    def _llm_summary(self, raw_data: dict[str, Any]) -> str:
        """
        Generate a natural-language summary using the LLM.

        Formats the raw enrichment data as a prompt and asks the LLM
        to produce a concise natural-language summary suitable for
        embedding.
        """
        assert self._llm is not None

        # Build a structured prompt with the enrichment data
        prompt_data = {
            "entity_name": raw_data["entity_name"],
            "entity_type": raw_data["entity_type"],
            "fact_count": raw_data["fact_count"],
            "co_entities": raw_data["co_entities"],
            "predicate_distribution": raw_data["predicate_distribution"],
            "temporal_pattern": raw_data["temporal_pattern"],
            "source_instances": raw_data["source_instances"],
        }

        prompt = (
            f"Summarize the following entity context for embedding.\n"
            f"Entity: {prompt_data['entity_name']} ({prompt_data['entity_type']})\n"
            f"Facts: {prompt_data['fact_count']} related facts\n"
            f"Co-occurring entities: {', '.join(prompt_data['co_entities']) or 'none'}\n"
            f"Predicates: {json.dumps(prompt_data['predicate_distribution'])}\n"
            f"Temporal: {prompt_data['temporal_pattern']}\n"
            f"Sources: {', '.join(prompt_data['source_instances']) or 'local'}\n\n"
            f"Write a concise, information-dense summary paragraph."
        )

        system_prompt = (
            "You are a knowledge summarization agent. Produce a concise "
            "paragraph summarizing an entity's context from a knowledge "
            "graph. Include key relationships, activities, and temporal "
            "patterns. The summary will be used for embedding, so pack "
            "in as much semantic information as possible."
        )

        try:
            result = self._llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=256,
            )
            logger.debug(
                "llm_summary_generated",
                entity=raw_data["entity_name"],
                summary_length=len(result),
            )
            return result
        except Exception as exc:
            # Graceful degradation: if LLM fails, fall back to template
            logger.warning(
                "llm_summary_failed",
                error_code="CTXMTG-SYN-004",
                error=str(exc),
            )
            return self._template_summary(raw_data)

    def _template_summary(self, raw_data: dict[str, Any]) -> str:
        """
        Generate a template-based summary (no LLM required).

        Produces a descriptive string that captures the key enrichment
        data in a format suitable for embedding.
        """
        name = raw_data["entity_name"]
        fact_count = raw_data["fact_count"]
        source_instances = raw_data["source_instances"]
        predicate_distribution = raw_data["predicate_distribution"]

        # Build the top predicates string
        # Sort by count descending, take top 5
        sorted_preds = sorted(
            predicate_distribution.items(), key=lambda x: x[1], reverse=True
        )
        top_preds = ", ".join(f"{p}" for p, _ in sorted_preds[:5])
        if not top_preds:
            top_preds = "none"

        num_instances = len(source_instances)

        return (
            f"Entity '{name}' has {fact_count} facts across "
            f"{num_instances} instance{'s' if num_instances != 1 else ''}. "
            f"Top predicates: {top_preds}."
        )

    # =================================================================
    # Empty enrichment result
    # =================================================================

    @staticmethod
    def _empty_enrichment(entity_id: str) -> dict[str, Any]:
        """
        Return an empty enrichment dict for entities not found in the hive.

        This ensures the caller always gets a consistent structure,
        even for missing entities.
        """
        return {
            "summary": f"Entity {entity_id} not found in hive.",
            "co_entities": [],
            "predicate_distribution": {},
            "temporal_pattern": "No temporal data available.",
            "source_instances": [],
            "fact_count": 0,
        }
