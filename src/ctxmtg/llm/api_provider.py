# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
API-based LLM Provider
=======================

Implements the LLMProvider interface using OpenAI-compatible HTTP APIs.
Works with OpenAI, Anthropic (via proxy), local servers (llama.cpp
server, vLLM, Ollama), and any endpoint that speaks the OpenAI chat
completions protocol.

Configuration is via environment variables:
    CTXMTG_LLM_API_KEY   - API key (required for cloud APIs)
    CTXMTG_LLM_BASE_URL  - Base URL (default: https://api.openai.com/v1)
    CTXMTG_LLM_MODEL     - Model name (default: gpt-4o-mini)

Depends on:
    - httpx (HTTP client for API calls)
    - ctxmtg.interfaces.llm (LLMProvider ABC)

Used by:
    - ctxmtg.llm.usage_tracker (records each call)
    - ctxmtg.extraction.llm_verifier (extraction enhancement)
    - ctxmtg.query.llm_interpreter (query understanding)
    - ctxmtg.query.synthesizer (answer synthesis)
"""

from __future__ import annotations

import json
import os
import re
import time

import httpx
import structlog

from ctxmtg.interfaces.llm import LLMProvider

logger = structlog.get_logger("ctxmtg.llm.api_provider")

# Patterns for LLM "thinking" blocks that should never reach storage or users.
_THINKING_PATTERNS = re.compile(
    r"<\|?thinking\|?>.*?<\|?/thinking\|?>|<think>.*?</think>",
    re.DOTALL,
)


def strip_thinking_tokens(text: str) -> str:
    """Remove LLM chain-of-thought / thinking blocks from output."""
    return _THINKING_PATTERNS.sub("", text).strip()


class APIProvider(LLMProvider):
    """
    LLM provider using OpenAI-compatible chat completions API.

    Supports any endpoint that speaks the OpenAI protocol: OpenAI,
    Azure OpenAI, local llama.cpp server, Ollama, vLLM, etc.

    Usage:
        provider = APIProvider(
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
        )
        response = provider.generate("Hello!", system_prompt="Be helpful.")
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        usage_tracker=None,
    ) -> None:
        self._api_key = api_key or os.environ.get("CTXMTG_LLM_API_KEY", "")
        self._base_url = (
            base_url
            or os.environ.get("CTXMTG_LLM_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self._model = model or os.environ.get("CTXMTG_LLM_MODEL", "gpt-4o-mini")
        self._timeout = timeout
        self._usage_tracker = usage_tracker
        self._available = bool(self._api_key or "localhost" in self._base_url)
        self._last_usage: dict | None = None
        self._stage: str = "unknown"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        json_mode: bool = False,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        if stop:
            body["stop"] = stop
        # ORIGINAL CODE (disabled 2026-04-07): LM Studio rejects json_object,
        # requires json_schema instead. Removed response_format entirely;
        # JSON structure is now prompt-driven for cross-provider compatibility.
        # TODO: Reintroduce response_format per-role after V2S/S2V split.
        # if json_mode:
        #     body["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        start = time.monotonic()
        error_msg = None
        success = True

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            self._last_usage = data.get("usage")
            content = strip_thinking_tokens(
                data["choices"][0]["message"]["content"] or ""
            )

            logger.info(
                "api_generate_success",
                model=self._model,
                prompt_tokens=self._last_usage.get("prompt_tokens", 0) if self._last_usage else 0,
                completion_tokens=self._last_usage.get("completion_tokens", 0) if self._last_usage else 0,
            )
            return content

        except Exception as exc:
            error_msg = str(exc)
            success = False
            logger.warning(
                "api_generate_failed",
                model=self._model,
                error=error_msg,
            )
            return ""

        finally:
            latency_ms = (time.monotonic() - start) * 1000
            if self._usage_tracker:
                from ctxmtg.llm.usage_tracker import UsageRecord

                usage = self._last_usage or {}
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._usage_tracker.record(UsageRecord(
                        model_name=self._model,
                        stage=self._stage,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                        latency_ms=latency_ms,
                        success=success,
                        error_message=error_msg,
                    )))
                except RuntimeError:
                    pass

    def is_available(self) -> bool:
        return self._available

    def get_model_name(self) -> str:
        return self._model

    @property
    def last_usage(self) -> dict | None:
        """Token usage from the last API call."""
        return self._last_usage
