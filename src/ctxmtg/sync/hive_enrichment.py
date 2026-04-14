# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Hive Enrichment Worker (Hive Side)
===================================

This module implements the hive-side enrichment pipeline.  After the
instance-side sync worker pushes raw records to the hive, this worker
processes them: enriching entities with cross-instance context,
re-embedding with richer vectors, and updating the hive records.

The enrichment loop:
    1. Find entities WHERE enriched_at IS NULL (never enriched) or
       WHERE enriched_at < the entity's most recent fact created_at
       (stale enrichment -- new data has arrived since last enrichment).
    2. For each entity, build enriched context using ContextEnricher.
    3. Re-embed the enriched context summary (not just the entity name).
    4. Update the hive entity record with the enriched_context text,
       enriched_at timestamp, and new embedding vector.

Why re-embed?
    The hive's vectors encode cross-instance context.  "Alice Chen,
    tech lead, engineering. Proposed OAuth2 (from meetings). Sent
    follow-up emails (from email instance). Mentioned in Slack (from
    chat instance)."  This is far richer than any single instance's
    vector for Alice.  A query to the hive for "who is leading
    authentication changes?" will surface Alice with much higher
    relevance.

See research/notes/hive-sync-design.md § "Why Hive Vectors Are Better"
for the design rationale.

Depends on:
    - ctxmtg.interfaces.storage (SQLStore, VectorStore for hive stores)
    - ctxmtg.interfaces.embedding (EmbeddingProvider for re-embedding)
    - ctxmtg.sync.context_enricher (ContextEnricher builds the context)
    - ctxmtg.exceptions (SyncError for error reporting)

Used by:
    - ctxmtg.cli (``ctxmtg hive enrich`` command, future)
    - Future: scheduled background worker on the hive process
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import structlog

from ctxmtg.exceptions import SyncError
from ctxmtg.interfaces.embedding import EmbeddingProvider
from ctxmtg.sync.context_enricher import ContextEnricher

# ---------------------------------------------------------------
# Module-level logger -- structured JSON output, no PII in logs.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.sync.hive_enrichment")

# ---------------------------------------------------------------
# Batch sizes for enrichment processing.  These limit memory usage
# and prevent the enrichment worker from monopolizing the database
# connection for too long.
# ---------------------------------------------------------------
ENRICHMENT_BATCH_SIZE = 50


class HiveEnrichmentWorker:
    """
    Hive-side enrichment worker.  Runs after receiving synced records.

    The worker finds entities that need enrichment (never enriched or
    stale), enriches them with cross-instance context via ContextEnricher,
    re-embeds the enriched context using the configured EmbeddingProvider,
    and updates the hive's SQL and vector stores.

    The hive does the heavy compute so instances don't have to.  This
    is the heart of the "pull model" described in hive-sync-design.md.

    Usage:
        worker = HiveEnrichmentWorker(
            hive_store=hive_sql,
            hive_vector_store=hive_vec,
            enricher=context_enricher,
            embedding_provider=embedding_provider,
        )
        counts = await worker.process_pending()
        # counts = {"enriched": 12, "re_embedded": 12}
    """

    def __init__(
        self,
        hive_store: Any,
        hive_vector_store: Any,
        enricher: ContextEnricher,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        """
        Prepare the enrichment worker.

        Args:
            hive_store:          The hive SQLStore instance (execute_sql,
                                 plus direct DB access for UPDATE).
            hive_vector_store:   The hive VectorStore instance (insert,
                                 delete for re-embedding).
            enricher:            ContextEnricher that builds enriched
                                 context from the hive's full dataset.
            embedding_provider:  EmbeddingProvider for re-embedding
                                 enriched context summaries.
        """
        self._hive_store = hive_store
        self._hive_vector_store = hive_vector_store
        self._enricher = enricher
        self._embedding_provider = embedding_provider

    # =================================================================
    # Main processing loop
    # =================================================================

    async def process_pending(self) -> dict[str, int]:
        """
        Process all entities not yet enriched (or stale).

        Finds entities WHERE enriched_at IS NULL, enriches them in
        batches using ContextEnricher, re-embeds the enriched context,
        and updates the hive records.

        Returns:
            Dict with counts: {"enriched": n, "re_embedded": n}

        Raises:
            SyncError: If the enrichment process fails.
        """
        try:
            # -------------------------------------------------------
            # Step 1: Find entities that need enrichment.
            # Criteria: enriched_at IS NULL (never enriched)
            #           OR enriched_at < latest related fact created_at
            #           (new data arrived since last enrichment).
            # -------------------------------------------------------
            pending_ids = await self._find_pending_entities()

            if not pending_ids:
                logger.info("hive_enrichment_noop", reason="no_pending_entities")
                return {"enriched": 0, "re_embedded": 0}

            logger.info(
                "hive_enrichment_start",
                pending_count=len(pending_ids),
            )

            # -------------------------------------------------------
            # Step 2: Enrich in batches using ContextEnricher.
            # -------------------------------------------------------
            enriched_count = 0
            re_embedded_count = 0

            for i in range(0, len(pending_ids), ENRICHMENT_BATCH_SIZE):
                batch_ids = pending_ids[i : i + ENRICHMENT_BATCH_SIZE]

                # Enrich the batch
                enrichments = await self._enricher.enrich_batch(
                    batch_ids, batch_size=ENRICHMENT_BATCH_SIZE
                )

                # -------------------------------------------------------
                # Step 3: Re-embed and update each enriched entity.
                # -------------------------------------------------------
                for entity_id, enrichment in zip(batch_ids, enrichments):
                    summary = enrichment.get("summary", "")
                    if not summary:
                        continue

                    # Re-embed the enriched context summary
                    vector = self._embedding_provider.embed_single(summary)

                    # Store the new vector in the hive vector store.
                    # Use a deterministic vector ID based on the entity ID
                    # so re-embedding replaces the old vector.
                    vector_id = f"hive-entity-{entity_id}"

                    # Delete old vector (if any) before inserting new one
                    await self._hive_vector_store.delete([vector_id])

                    # Insert the new enriched vector with metadata
                    await self._hive_vector_store.insert(
                        ids=[vector_id],
                        vectors=[vector],
                        metadata=[{
                            "source_table": "entities",
                            "source_id": entity_id,
                            "content_preview": summary[:200],
                            "enriched": True,
                        }],
                    )
                    re_embedded_count += 1

                    # Update the hive entity record with enriched context
                    await self._update_entity_enrichment(
                        entity_id=entity_id,
                        enriched_context=summary,
                        source_instances=enrichment.get("source_instances", []),
                    )
                    enriched_count += 1

                logger.debug(
                    "hive_enrichment_batch_done",
                    batch_start=i,
                    batch_size=len(batch_ids),
                    enriched_so_far=enriched_count,
                )

            logger.info(
                "hive_enrichment_complete",
                enriched=enriched_count,
                re_embedded=re_embedded_count,
            )

            return {"enriched": enriched_count, "re_embedded": re_embedded_count}

        except SyncError:
            raise
        except Exception as exc:
            logger.error(
                "hive_enrichment_failed",
                error_code="CTXMTG-SYN-004",
                error=str(exc),
            )
            raise SyncError(
                f"Hive enrichment failed: {exc}",
                error_code="CTXMTG-SYN-004",
            ) from exc

    # =================================================================
    # Private helpers
    # =================================================================

    async def _find_pending_entities(self) -> list[str]:
        """
        Find entity IDs that need enrichment.

        Returns IDs of entities where:
            - enriched_at IS NULL (never enriched), OR
            - enriched_at < the entity's hive_synced_at (new data
              arrived since last enrichment).

        Results are ordered by created_at ASC so older entities
        are enriched first.
        """
        rows = await self._hive_store.execute_sql(
            "SELECT id FROM entities "
            "WHERE enriched_at IS NULL "
            "   OR enriched_at < hive_synced_at "
            "ORDER BY created_at ASC",
            {},
        )
        return [r["id"] for r in rows]

    async def _update_entity_enrichment(
        self,
        entity_id: str,
        enriched_context: str,
        source_instances: list[str],
    ) -> None:
        """
        Update a hive entity record with enriched context.

        Sets enriched_context (the summary text), enriched_at
        (current timestamp), and source_instances (JSON array of
        contributing instances).

        Uses execute_sql for the UPDATE to stay within the SQLStore
        interface.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        instances_json = json.dumps(source_instances)

        await self._hive_store.execute_sql(
            "UPDATE entities "
            "SET enriched_context = :ctx, "
            "    enriched_at = :ts, "
            "    source_instances = :inst "
            "WHERE id = :eid",
            {
                "ctx": enriched_context,
                "ts": now_iso,
                "inst": instances_json,
                "eid": entity_id,
            },
        )
