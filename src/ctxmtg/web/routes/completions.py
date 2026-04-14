# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
OpenAI-Compatible Chat Completions Endpoint
=============================================

Implements /v1/chat/completions following the OpenAI API schema so
that Open WebUI (or any OpenAI-compatible client) can use ctxmtg
as a knowledge backend.

When a user sends a chat message:
1. The last user message is treated as a query.
2. The query runs through the ctxmtg query pipeline.
3. Results are formatted as an assistant response.

Non-streaming only (streaming is a future enhancement).

Depends on:
    - ctxmtg.query.executor (QueryExecutor)
    - ctxmtg.query.interpreter (RuleBasedQueryInterpreter)
    - ctxmtg.query.planner (TemplateQueryPlanner)
    - ctxmtg.query.fusion (RRFFuser)
    - ctxmtg.query.reranker (TFIDFReranker)
    - ctxmtg.profile.loader (ProfileLoader)
    - ctxmtg.web.deps (store access)

Used by:
    - Open WebUI (configured with http://127.0.0.1:8080/v1 as API base)
    - Any OpenAI-compatible client
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from ctxmtg.web.deps import get_sql_store, get_vector_store

logger = structlog.get_logger("ctxmtg.web.routes.completions")

router = APIRouter(prefix="/v1", tags=["openai"])


# -- Request / Response models matching OpenAI API schema --

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "ctxmtg"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "ctxmtg"
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "ctxmtg"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


async def _build_executor():
    """Wire up a QueryExecutor from the shared stores."""
    from ctxmtg.profile.loader import ProfileLoader
    from ctxmtg.query.executor import QueryExecutor
    from ctxmtg.query.fusion import RRFFuser
    from ctxmtg.query.interpreter import RuleBasedQueryInterpreter
    from ctxmtg.query.planner import TemplateQueryPlanner
    from ctxmtg.query.quality_logger import QueryQualityLogger
    from ctxmtg.query.reranker import TFIDFReranker

    sql_store = get_sql_store()
    vector_store = get_vector_store()
    profile = ProfileLoader.load("general")

    interpreter = RuleBasedQueryInterpreter(sql_store)
    planner = TemplateQueryPlanner()
    fuser = RRFFuser()
    reranker = TFIDFReranker()
    quality_logger = QueryQualityLogger(sql_store)

    # Load embedding function for vector search.
    # ORIGINAL CODE (disabled 2026-04-07): embedding_fn was not passed,
    # so vector search was always skipped on web queries.
    # executor = QueryExecutor(...) # no embedding_fn
    embedding_fn = None
    try:
        from ctxmtg.embedding.onnx_embedder import ONNXEmbeddingProvider
        embedder = ONNXEmbeddingProvider()
        embedding_fn = lambda text: embedder.embed([text])[0]
    except Exception:
        pass

    executor = QueryExecutor(
        sql_store=sql_store,
        vector_store=vector_store,
        interpreter=interpreter,
        planner=planner,
        fuser=fuser,
        reranker=reranker,
        embedding_fn=embedding_fn,
        profile=profile,
        quality_logger=quality_logger,
    )
    return executor, profile


def _format_results(query_result) -> str:
    """Format QueryResult into a human-readable assistant message."""
    parts = []

    if query_result.synthesis:
        parts.append(query_result.synthesis)
    elif query_result.results:
        parts.append(
            f"Found {query_result.total_results} results "
            f"({query_result.sql_results_count} SQL, "
            f"{query_result.vector_results_count} vector):\n"
        )
        for i, r in enumerate(query_result.results[:10], 1):
            content = r.content[:200].replace("\n", " ")
            parts.append(f"{i}. [{r.source_store}] (score: {r.score:.3f}) {content}")
    else:
        parts.append("No results found for your query.")

    return "\n".join(parts)


@router.get("/models")
async def list_models():
    """List available models (Open WebUI discovery)."""
    return ModelListResponse(
        data=[
            ModelInfo(id="ctxmtg", owned_by="ctxmtg"),
            ModelInfo(id="ctxmtg-deep", owned_by="ctxmtg"),
        ]
    )


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
):
    """
    OpenAI-compatible chat completions endpoint.

    Extracts the last user message as a query, runs it through the
    ctxmtg query pipeline, and returns results in OpenAI format.
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming not yet supported. Set stream=false.",
        )

    # Extract the user's question from the last user message.
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=400,
            detail="No user message found in the request.",
        )
    question = user_messages[-1].content

    # Determine retrieval mode from model name.
    from ctxmtg.models.query import RetrievalMode

    mode = RetrievalMode.PARALLEL
    if request.model == "ctxmtg-deep":
        mode = RetrievalMode.BIDIRECTIONAL

    try:
        executor, profile = await _build_executor()
        result = await executor.execute(question, profile, mode=mode, top_k=10)
        answer = _format_results(result)
    except Exception as exc:
        logger.error("completions_query_failed", error=str(exc))
        answer = f"Error processing query: {exc}"

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            Choice(
                message=ChatMessage(role="assistant", content=answer),
            )
        ],
    )
