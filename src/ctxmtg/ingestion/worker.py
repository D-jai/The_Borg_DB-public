# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Ingestion Worker
================

This module orchestrates the full ingestion pipeline from raw file
to stored knowledge. It coordinates all the moving parts:

    1. Load file via the appropriate format loader → Interaction
    2. Run the Traffic Cop → (IntakeAction, modified Interaction)
    3. If ACCEPT: run extraction (or bypass for calendar/contact)
    4. Embed text chunks → vectors
    5. Store entities + facts in SQLite
    6. Store embeddings in LanceDB
    7. Log statistics

The worker is synchronous (CPU-bound extraction + embedding). For
batch ingestion, items are processed sequentially to keep memory
usage manageable on edge devices (Raspberry Pi, laptop).

Calendar (.ics) and contact (.vcf) files bypass the NLP extraction
pipeline entirely -- their structured data is converted directly to
entities and facts by their respective loaders. There's no point
running NER on data that's already perfectly structured.

Depends on:
    - ctxmtg.ingestion.loaders (FileLoaderRegistry, format-specific loaders)
    - ctxmtg.intake.rules (RuleBasedIntakeGateway -- Traffic Cop)
    - ctxmtg.extraction.pipeline (BasicExtractionPipeline)
    - ctxmtg.embedding.onnx_embedder (ONNXEmbeddingProvider)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.embedding (EmbeddingProvider)
    - ctxmtg.models.interaction (Interaction, IntakeAction, SourceType)
    - ctxmtg.models.profile (DomainProfile)

Used by:
    - ctxmtg.cli (CLI ingest command)
    - tests/test_integration/test_end_to_end.py
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import structlog

from ctxmtg.exceptions import IntakeError
from ctxmtg.ingestion.loaders import FileLoaderRegistry
from ctxmtg.ingestion.loaders.calendar_loader import CalendarLoadResult
from ctxmtg.ingestion.loaders.contact_loader import ContactLoadResult
from ctxmtg.ingestion.loaders.text_loader import load_text_string
from ctxmtg.interfaces.embedding import EmbeddingProvider
from ctxmtg.interfaces.extraction import ExtractionPipeline
from ctxmtg.interfaces.intake import IntakeGateway
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.models.interaction import (
    EmbeddingMetadata,
    IntakeAction,
    Interaction,
    SourceType,
)
from ctxmtg.storage.id_gen import generate_embedding_id

# ---------------------------------------------------------------
# Module-level logger -- logs pipeline statistics, not content.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.ingestion.worker")


class IngestionWorker:
    """
    Coordinates the full ingestion pipeline from file to stored knowledge.

    The worker ties together all pipeline components:
    - File loaders (format detection + parsing)
    - Traffic Cop (classification + content transformation)
    - Extraction pipeline (NER + facts + summary + chunking)
    - Embedding provider (text → vectors)
    - Dual stores (SQLite for structured data, LanceDB for vectors)

    The worker handles the special case of calendar and contact files,
    which bypass NLP extraction because their data is already structured.

    Usage:
        worker = IngestionWorker(
            sql_store=sqlite_store,
            vector_store=lancedb_store,
            extraction_pipeline=pipeline,
            embedding_provider=embedder,
            intake_gateway=traffic_cop,
        )

        # Ingest a file
        stats = worker.ingest_file(Path("meeting.txt"))

        # Ingest raw text
        stats = worker.ingest_text("Alice proposed OAuth2 migration.")

        # Batch ingest a directory
        all_stats = worker.ingest_directory(Path("./documents/"))
    """

    def __init__(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        extraction_pipeline: ExtractionPipeline | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        intake_gateway: IntakeGateway | None = None,
    ) -> None:
        """
        Initialize the ingestion worker with all pipeline components.

        All components are injected for testability. In production,
        these are wired up by the application factory in the CLI.

        Args:
            sql_store: The SQL store for structured data.
            vector_store: The vector store for embeddings.
            extraction_pipeline: The NER + fact extraction pipeline.
                                 None disables extraction.
            embedding_provider: The text-to-vector provider.
                                None disables embedding.
            intake_gateway: The Traffic Cop. None means accept all.
        """
        self._sql_store = sql_store
        self._vector_store = vector_store
        self._extraction = extraction_pipeline
        self._embedder = embedding_provider
        self._gateway = intake_gateway
        self._loader_registry = FileLoaderRegistry()

        # Accumulated statistics
        self._total_stats: dict[str, int] = {
            "files_processed": 0,
            "interactions_ingested": 0,
            "entities_stored": 0,
            "facts_stored": 0,
            "embeddings_stored": 0,
            "accepted": 0,
            "deferred": 0,
            "rejected": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        """Return a copy of the accumulated ingestion statistics."""
        return dict(self._total_stats)

    def ingest_file(self, file_path: Path) -> dict[str, Any]:
        """
        Ingest a single file through the full pipeline.

        Auto-detects the file format from the extension and dispatches
        to the appropriate loader. Returns per-file statistics.

        Args:
            file_path: Path to the file to ingest.

        Returns:
            A dict of statistics for this ingestion (entities_stored, etc.).

        Raises:
            IntakeError: If the file format is not supported or cannot be loaded.
        """
        start = time.monotonic()
        file_stats: dict[str, Any] = {"file": str(file_path), "status": "unknown"}

        # Determine file extension and get the appropriate loader
        ext = file_path.suffix.lower()

        if not self._loader_registry.can_load(ext):
            logger.error(
                "unsupported_file_format",
                error_code="CTXMTG-ING-002",
                extension=ext,
                file_path=str(file_path),
            )
            raise IntakeError(
                f"Unsupported file format: {ext}. "
                f"Supported: {', '.join(self._loader_registry.supported_extensions())}",
                error_code="CTXMTG-ING-002",
            )

        loader = self._loader_registry.get_loader(ext)

        # Load the file using the appropriate loader
        load_result = loader(file_path)

        # Handle different loader return types
        if ext == ".ics":
            # Calendar loader returns CalendarLoadResult list
            file_stats = self._process_calendar_results(load_result)
        elif ext == ".vcf":
            # Contact loader returns ContactLoadResult list
            file_stats = self._process_contact_results(load_result)
        elif ext == ".json":
            # JSON loader returns a list of Interactions
            file_stats = self._process_interactions(load_result)
        else:
            # Text/EML loaders return a single Interaction
            file_stats = self._process_interactions([load_result])

        elapsed = time.monotonic() - start
        file_stats["duration_ms"] = round(elapsed * 1000, 2)
        file_stats["file"] = str(file_path)

        self._total_stats["files_processed"] += 1

        logger.info(
            "file_ingested",
            file_path=str(file_path),
            duration_ms=file_stats["duration_ms"],
            entities=file_stats.get("entities_stored", 0),
            facts=file_stats.get("facts_stored", 0),
        )

        return file_stats

    def ingest_text(self, text: str, title: str | None = None) -> dict[str, Any]:
        """
        Ingest a raw text string through the pipeline.

        Used when the user passes text directly via CLI.

        Args:
            text: The raw text to ingest.
            title: Optional title for the interaction.

        Returns:
            A dict of ingestion statistics.
        """
        interaction = load_text_string(text, title=title)
        return self._process_interactions([interaction])

    def ingest_directory(self, dir_path: Path) -> list[dict[str, Any]]:
        """
        Batch ingest all supported files in a directory.

        Scans the directory for files with supported extensions and
        processes each one. Errors on individual files are logged
        but don't stop the batch.

        Args:
            dir_path: Path to the directory to scan.

        Returns:
            A list of per-file statistics dicts.
        """
        if not dir_path.is_dir():
            logger.error(
                "ingest_dir_not_found",
                error_code="CTXMTG-ING-002",
                dir_path=str(dir_path),
            )
            raise IntakeError(
                f"Not a directory: {dir_path}",
                error_code="CTXMTG-ING-002",
            )

        all_stats: list[dict[str, Any]] = []
        supported_exts = set(self._loader_registry.supported_extensions())

        # Collect all supported files, sorted for deterministic order
        files = sorted(
            f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in supported_exts
        )

        logger.info(
            "batch_ingest_started",
            directory=str(dir_path),
            file_count=len(files),
        )

        for file_path in files:
            try:
                stats = self.ingest_file(file_path)
                all_stats.append(stats)
            except Exception as exc:
                logger.error(
                    "file_ingest_failed",
                    error_code="CTXMTG-ING-004",
                    file_path=str(file_path),
                    error=str(exc),
                )
                all_stats.append({"file": str(file_path), "status": "error", "error": str(exc)})

        return all_stats

    def _process_interactions(self, interactions: list[Interaction]) -> dict[str, Any]:
        """
        Process a list of Interaction objects through the full pipeline.

        For each interaction:
        1. Run Traffic Cop (classify + transform)
        2. If ACCEPT: run extraction → entities + facts
        3. Embed chunks → vectors
        4. Store in both stores

        Args:
            interactions: The Interaction objects to process.

        Returns:
            Aggregated statistics for all processed interactions.
        """
        stats: dict[str, Any] = {
            "interactions_total": len(interactions),
            "accepted": 0,
            "deferred": 0,
            "rejected": 0,
            "entities_stored": 0,
            "facts_stored": 0,
            "embeddings_stored": 0,
            "status": "ok",
        }

        for interaction in interactions:
            # Step 1: Traffic Cop classification + transformation
            action, modified = self._run_traffic_cop(interaction)

            if action == IntakeAction.ACCEPT:
                stats["accepted"] += 1
                self._total_stats["accepted"] += 1

                # Step 2: Store the interaction in SQL
                asyncio.get_event_loop().run_until_complete(
                    self._sql_store.store_interaction(modified)
                ) if _has_event_loop() else _run_async(
                    self._sql_store.store_interaction(modified)
                )

                # Step 3: Run extraction pipeline
                entities, facts, chunks = self._run_extraction(modified)

                # Step 4: Store entities and facts
                if entities:
                    entity_count = _run_async(self._sql_store.store_entities(entities))
                    stats["entities_stored"] += entity_count
                    self._total_stats["entities_stored"] += entity_count

                if facts:
                    fact_count = _run_async(self._sql_store.store_facts(facts))
                    stats["facts_stored"] += fact_count
                    self._total_stats["facts_stored"] += fact_count

                # Step 5: Embed chunks and store vectors
                if chunks and self._embedder:
                    emb_count = self._embed_and_store(modified.id, chunks)
                    stats["embeddings_stored"] += emb_count
                    self._total_stats["embeddings_stored"] += emb_count

                self._total_stats["interactions_ingested"] += 1

            elif action == IntakeAction.DEFER:
                stats["deferred"] += 1
                self._total_stats["deferred"] += 1
                logger.info("interaction_deferred", interaction_id=interaction.id)

            elif action == IntakeAction.REJECT:
                stats["rejected"] += 1
                self._total_stats["rejected"] += 1
                logger.info("interaction_rejected", interaction_id=interaction.id)

        return stats

    def _process_calendar_results(self, results: list[CalendarLoadResult]) -> dict[str, Any]:
        """
        Process calendar load results (bypass NLP extraction).

        Calendar events have pre-built entities and facts from the
        structured .ics data. We store them directly without running
        the extraction pipeline.

        Args:
            results: CalendarLoadResult objects from the ICS loader.

        Returns:
            Aggregated statistics.
        """
        stats: dict[str, Any] = {
            "interactions_total": len(results),
            "accepted": 0,
            "entities_stored": 0,
            "facts_stored": 0,
            "embeddings_stored": 0,
            "status": "ok",
        }

        for result in results:
            # Traffic cop on the interaction
            action, modified = self._run_traffic_cop(result.interaction)

            if action == IntakeAction.ACCEPT:
                stats["accepted"] += 1
                self._total_stats["accepted"] += 1

                # Store the interaction
                _run_async(self._sql_store.store_interaction(modified))

                # Store pre-built entities directly (bypass NLP)
                if result.entities:
                    count = _run_async(self._sql_store.store_entities(result.entities))
                    stats["entities_stored"] += count
                    self._total_stats["entities_stored"] += count

                # Store pre-built facts directly (bypass NLP)
                if result.facts:
                    count = _run_async(self._sql_store.store_facts(result.facts))
                    stats["facts_stored"] += count
                    self._total_stats["facts_stored"] += count

                # Embed the interaction content
                if self._embedder:
                    chunks = [modified.content]
                    emb_count = self._embed_and_store(modified.id, chunks)
                    stats["embeddings_stored"] += emb_count
                    self._total_stats["embeddings_stored"] += emb_count

                self._total_stats["interactions_ingested"] += 1

        return stats

    def _process_contact_results(self, results: list[ContactLoadResult]) -> dict[str, Any]:
        """
        Process contact load results (bypass NLP extraction).

        Contacts have pre-built entities from the structured .vcf data.
        We store them directly without running the extraction pipeline.

        Args:
            results: ContactLoadResult objects from the VCF loader.

        Returns:
            Aggregated statistics.
        """
        stats: dict[str, Any] = {
            "interactions_total": len(results),
            "accepted": 0,
            "entities_stored": 0,
            "facts_stored": 0,
            "embeddings_stored": 0,
            "status": "ok",
        }

        for result in results:
            # Traffic cop on the interaction
            action, modified = self._run_traffic_cop(result.interaction)

            if action == IntakeAction.ACCEPT:
                stats["accepted"] += 1
                self._total_stats["accepted"] += 1

                # Store the interaction
                _run_async(self._sql_store.store_interaction(modified))

                # Store pre-built entities directly (bypass NLP)
                if result.entities:
                    count = _run_async(self._sql_store.store_entities(result.entities))
                    stats["entities_stored"] += count
                    self._total_stats["entities_stored"] += count

                self._total_stats["interactions_ingested"] += 1

        return stats

    def _run_traffic_cop(self, interaction: Interaction) -> tuple[IntakeAction, Interaction]:
        """
        Run the Traffic Cop on an interaction.

        If no gateway is configured, accepts everything unchanged.

        Args:
            interaction: The interaction to classify.

        Returns:
            The (action, modified_interaction) tuple.
        """
        if self._gateway is None:
            return (IntakeAction.ACCEPT, interaction)

        return self._gateway.process(interaction)

    def _run_extraction(self, interaction: Interaction) -> tuple[list, list, list[str]]:
        """
        Run the extraction pipeline on an interaction.

        Calendar and contact interactions are skipped (they bypass NLP).

        Args:
            interaction: The interaction to extract from.

        Returns:
            A tuple of (entities, facts, chunks).
        """
        # Calendar and contact sources bypass NLP extraction
        if interaction.source_type in (SourceType.CALENDAR, SourceType.CONTACT):
            return ([], [], [interaction.content])

        if self._extraction is None:
            # No extraction pipeline configured -- just return content as a chunk
            return ([], [], [interaction.content])

        try:
            result = self._extraction.process(interaction)
            return (result.entities, result.facts, result.chunks or [interaction.content])
        except Exception as exc:
            logger.error(
                "extraction_failed",
                error_code="CTXMTG-ING-004",
                interaction_id=interaction.id,
                error=str(exc),
            )
            # Graceful degradation: store the interaction content without extraction
            return ([], [], [interaction.content])

    def _embed_and_store(self, interaction_id: str, chunks: list[str]) -> int:
        """
        Embed text chunks and store vectors in the vector store.

        Each chunk gets its own embedding and metadata record linking
        back to the source interaction in the SQL store.

        Args:
            interaction_id: The ID of the source interaction.
            chunks: The text chunks to embed.

        Returns:
            The number of embeddings stored.
        """
        if not self._embedder or not chunks:
            return 0

        try:
            # Embed all chunks in one batch
            vectors = self._embedder.embed(chunks)

            # Build IDs and metadata for the vector store
            ids: list[str] = []
            metadata_list: list[dict[str, Any]] = []

            for idx, chunk in enumerate(chunks):
                emb_id = generate_embedding_id(interaction_id, idx)
                ids.append(emb_id)
                metadata_list.append({
                    "source_table": "interactions",
                    "source_id": interaction_id,
                    "chunk_index": idx,
                    "content_preview": chunk[:200],
                })

            # Store in the vector store
            count = _run_async(
                self._vector_store.insert(ids, vectors, metadata_list)
            )

            # Store embedding metadata in SQL
            emb_metadata = [
                EmbeddingMetadata(
                    id=ids[i],
                    source_table="interactions",
                    source_id=interaction_id,
                    chunk_index=i,
                    model_name=self._embedder.get_model_name(),
                    model_version=self._embedder.get_model_version(),
                    dimensions=self._embedder.get_dimensions(),
                )
                for i in range(len(ids))
            ]
            _run_async(self._sql_store.store_embedding_metadata(emb_metadata))

            return count

        except Exception as exc:
            logger.error(
                "embedding_failed",
                error_code="CTXMTG-ING-004",
                interaction_id=interaction_id,
                chunk_count=len(chunks),
                error=str(exc),
            )
            return 0


def _has_event_loop() -> bool:
    """Check if there's a running event loop."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _run_async(coro: Any) -> Any:
    """
    Run an async coroutine from synchronous code.

    Uses asyncio.run() to execute the coroutine. This is safe because
    the ingestion worker runs synchronously (no event loop conflict).

    Args:
        coro: The coroutine to execute.

    Returns:
        The coroutine's return value.
    """
    try:
        asyncio.get_running_loop()
        # If there's already a running loop, use nest_asyncio or
        # create a task. For simplicity, run in the existing loop.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop -- safe to use asyncio.run()
        return asyncio.run(coro)
