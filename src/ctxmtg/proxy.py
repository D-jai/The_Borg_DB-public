# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
LLM Proxy for Live Chat Capture
================================

A transparent OpenAI-compatible proxy that sits between the user's chat
client and any LLM backend (Ollama, OpenAI, etc.). Intercepts each
exchange, captures user + assistant messages, and ingests them as
source_type=chat interactions.

Usage:
    ctxmtg proxy --port 11435 --upstream http://localhost:11434

Point your chat client at http://localhost:11435 instead of 11434.
All requests are forwarded to the upstream unchanged; responses are
forwarded back unchanged. The proxy silently captures the conversation.

Non-streaming only in this version. Streaming requests are forwarded
but not captured (the LLM response passes through without ingestion).

Depends on:
    - httpx (async HTTP client for upstream forwarding)
    - fastapi (ASGI framework)
    - uvicorn (ASGI server)
    - ctxmtg.ingestion.worker (IngestionWorker for capture)

Used by:
    - ctxmtg.cli (the `ctxmtg proxy` command)
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = structlog.get_logger("ctxmtg.proxy")

# Runtime toggle -- can be flipped from the web UI.
_capture_enabled: bool = True


def set_capture_enabled(enabled: bool) -> None:
    global _capture_enabled
    _capture_enabled = enabled


def is_capture_enabled() -> bool:
    return _capture_enabled


def _build_worker():
    """Build an IngestionWorker for capturing conversations."""
    from ctxmtg.config.settings import CtxMtgSettings
    from ctxmtg.ingestion.worker import IngestionWorker
    from ctxmtg.storage.lancedb_store import LanceDBStore
    from ctxmtg.storage.sqlite import SQLiteStore

    import asyncio

    settings = CtxMtgSettings()
    db_path = Path(settings.db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    vec_path = Path(settings.vector_path).expanduser()
    vec_path.mkdir(parents=True, exist_ok=True)

    sql_store = SQLiteStore(db_path=str(db_path))
    vector_store = LanceDBStore(db_path=str(vec_path))
    asyncio.run(sql_store.initialize())
    asyncio.run(vector_store.initialize())

    extraction = None
    embedder = None
    try:
        from ctxmtg.extraction.pipeline import BasicExtractionPipeline
        from ctxmtg.profile.loader import ProfileLoader

        profile = ProfileLoader.load("general")
        extraction = BasicExtractionPipeline(profile)
    except Exception:
        pass

    try:
        from ctxmtg.embedding.onnx_embedder import ONNXEmbeddingProvider

        embedder = ONNXEmbeddingProvider()
    except Exception:
        pass

    return IngestionWorker(
        sql_store=sql_store,
        vector_store=vector_store,
        extraction_pipeline=extraction,
        embedding_provider=embedder,
    )


def _capture_exchange(user_msg: str, assistant_msg: str, model: str, worker) -> None:
    """Format and ingest a user+assistant exchange."""
    if not _capture_enabled:
        return

    from ctxmtg.llm.api_provider import strip_thinking_tokens

    assistant_msg = strip_thinking_tokens(assistant_msg)
    content = f"[User]: {user_msg}\n\n[Assistant]: {assistant_msg}"
    title = f"LLM chat ({model})"

    try:
        worker.ingest_text(content, title=title)
        logger.info("proxy_captured", model=model, user_len=len(user_msg))
    except Exception as exc:
        logger.warning("proxy_capture_failed", error=str(exc))


def create_proxy_app(upstream: str) -> FastAPI:
    """
    Build the proxy FastAPI application.

    Args:
        upstream: The upstream LLM backend URL (e.g., http://localhost:11434).
    """
    worker = None

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal worker
        try:
            worker = _build_worker()
            logger.info("proxy_worker_ready")
        except Exception as exc:
            logger.warning("proxy_worker_init_failed", error=str(exc))
        yield

    app = FastAPI(title="ctxmtg LLM Proxy", lifespan=lifespan)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def proxy_all(request: Request, path: str):
        """Forward all requests to upstream and capture chat completions."""
        url = f"{upstream.rstrip('/')}/{path}"

        # Read the request body.
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)

        # Determine if this is a chat completions request.
        is_chat = path.rstrip("/") in (
            "v1/chat/completions",
            "api/chat",
        )

        # Parse the request to extract user message.
        user_msg = None
        model = "unknown"
        is_stream = False
        if is_chat and body:
            try:
                payload = json.loads(body)
                is_stream = payload.get("stream", False)
                model = payload.get("model", "unknown")
                messages = payload.get("messages", [])
                user_messages = [m for m in messages if m.get("role") == "user"]
                if user_messages:
                    user_msg = user_messages[-1].get("content", "")
            except (json.JSONDecodeError, AttributeError):
                pass

        async with httpx.AsyncClient(timeout=120.0) as client:
            upstream_resp = await client.request(
                method=request.method,
                url=url,
                content=body,
                headers=headers,
            )

        # For streaming, just forward the response without capture.
        if is_stream:
            return StreamingResponse(
                iter([upstream_resp.content]),
                status_code=upstream_resp.status_code,
                headers=dict(upstream_resp.headers),
            )

        # Try to capture non-streaming chat completions.
        if is_chat and user_msg and worker and upstream_resp.status_code == 200:
            try:
                resp_data = upstream_resp.json()
                choices = resp_data.get("choices", [])
                if choices:
                    assistant_msg = (
                        choices[0].get("message", {}).get("content", "")
                    )
                    if assistant_msg:
                        _capture_exchange(user_msg, assistant_msg, model, worker)
            except Exception as exc:
                logger.debug("proxy_capture_parse_failed", error=str(exc))

        return JSONResponse(
            content=upstream_resp.json() if upstream_resp.headers.get("content-type", "").startswith("application/json") else upstream_resp.text,
            status_code=upstream_resp.status_code,
        )

    return app


def run_proxy(
    port: int = 11435,
    upstream: str = "http://localhost:11434",
) -> None:
    """Start the proxy server. Called by `ctxmtg proxy`."""
    if "127.0.0.1" not in upstream and "localhost" not in upstream:
        logger.info("proxy_upstream_remote", upstream=upstream)

    app = create_proxy_app(upstream)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
