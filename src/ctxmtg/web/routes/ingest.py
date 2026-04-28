# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
HTTP Ingest Endpoint
====================

Provides POST /api/ingest for programmatic text ingestion over HTTP.
Any tool that can make an HTTP POST (browser extension, curl, Zapier,
n8n, the LLM proxy) can feed the knowledge system.

Localhost only, no auth required (same security model as /v1/*).
Can be enabled/disabled via settings.http_ingest_enabled or the
web UI toggle.

Depends on:
    - ctxmtg.ingestion.worker (IngestionWorker)
    - ctxmtg.web.deps (store access)
    - ctxmtg.config.settings (CtxMtgSettings)

Used by:
    - External tools (curl, browser extensions, webhooks)
    - ctxmtg.proxy (LLM proxy captures)
    - ctxmtg.web.app (registered as a router)
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ctxmtg.config.settings import CtxMtgSettings

logger = structlog.get_logger("ctxmtg.web.routes.ingest")

router = APIRouter(tags=["ingest"])

# Runtime toggle -- can be flipped from the web UI without restart.
_enabled: bool = True


def set_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def is_enabled() -> bool:
    return _enabled


class IngestRequest(BaseModel):
    text: str
    title: str | None = None
    source_type: str = "other"


class IngestResponse(BaseModel):
    status: str = "ok"
    entities_stored: int = 0
    facts_stored: int = 0
    embeddings_stored: int = 0


@router.post("/api/ingest", response_model=IngestResponse)
async def http_ingest(req: IngestRequest):
    """
    Ingest text via HTTP POST.

    Accepts JSON: {"text": "...", "title": "...", "source_type": "chat"}
    Returns ingestion statistics.
    """
    if not _enabled:
        raise HTTPException(status_code=403, detail="HTTP ingest is disabled")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text field is empty")

    try:
        import asyncio
        import concurrent.futures

        from ctxmtg.web.deps import get_sql_store, get_vector_store

        sql_store = get_sql_store()
        vector_store = get_vector_store()

        def _do_ingest() -> dict:
            """Run the synchronous ingestion worker in a thread."""
            from ctxmtg.ingestion.worker import IngestionWorker

            extraction = None
            embedder = None
            try:
                from ctxmtg.extraction.pipeline import BasicExtractionPipeline
                from ctxmtg.profile.loader import ProfileLoader

                profile = ProfileLoader.load("general")

                # Wire the extraction-role LLM verifier and abstractive
                # summariser if a provider is configured.  Mirrors the
                # CLI's _init_ingest_worker pattern so HTTP ingest --
                # used by the proxy, browser extensions, and the inbox
                # watcher -- gets the same LLM enhancement that the CLI
                # gets.  All paths fall back gracefully when no LLM is
                # configured.
                llm_verifier = None
                extraction_llm = None
                try:
                    from ctxmtg.llm.factory import create_provider
                    extraction_llm = create_provider("extraction")
                    if extraction_llm:
                        from ctxmtg.extraction.llm_verifier import (
                            LLMExtractionVerifier,
                        )
                        from ctxmtg.llm.prompt_assembler import PromptAssembler
                        assembler = PromptAssembler()
                        llm_verifier = LLMExtractionVerifier(
                            llm=extraction_llm,
                            prompt_assembler=assembler,
                            profile=profile,
                        )
                except Exception as exc:
                    logger.warning(
                        "http_ingest_llm_init_failed", error=str(exc)
                    )

                extraction = BasicExtractionPipeline(
                    profile,
                    llm_verifier=llm_verifier,
                    llm=extraction_llm,
                )
            except Exception:
                pass

            try:
                from ctxmtg.embedding.onnx_embedder import ONNXEmbeddingProvider

                embedder = ONNXEmbeddingProvider()
            except Exception:
                pass

            worker = IngestionWorker(
                sql_store=sql_store,
                vector_store=vector_store,
                extraction_pipeline=extraction,
                embedding_provider=embedder,
            )
            return worker.ingest_text(req.text, title=req.title)

        # Run in a thread to avoid event-loop conflicts with aiosqlite.
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            stats = await loop.run_in_executor(pool, _do_ingest)

        return IngestResponse(
            entities_stored=stats.get("entities_stored", 0),
            facts_stored=stats.get("facts_stored", 0),
            embeddings_stored=stats.get("embeddings_stored", 0),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("http_ingest_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
