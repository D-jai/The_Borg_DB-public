# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
LLM Provider Interface ABC
===========================

This module defines the abstract base class for local LLM providers --
the components that run large language models on the user's device
for enhanced extraction, query interpretation, and synthesis.

LLM support is optional. The system runs perfectly well without an
LLM (Tier 0-1), using spaCy + regex for extraction and rule-based
query interpretation. When a local LLM is available (Tier 2+), it
unlocks:
- Domain-specific NER (understanding "Judge Morrison" in legal contexts)
- Natural language synthesis (generating human-readable answers)
- Dynamic SQL generation (translating complex queries to SQL)
- Farming insight narratives (explaining discovered patterns)

Phase 1 defines this interface but does not implement it.
Phase 2 implements it with llama.cpp / MLX backends.

Depends on:
    - abc (Python's Abstract Base Class machinery)

Used by:
    - ctxmtg.llm.provider (implements LLMProvider with llama.cpp)
    - ctxmtg.extraction.pipeline (uses LLM for enhanced extraction)
    - ctxmtg.query.interpreter (uses LLM for query understanding)
    - ctxmtg.query.fusion (uses LLM for result synthesis)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# =====================================================================
# LLMProvider ABC -- Local Language Model Interface
# =====================================================================


class LLMProvider(ABC):
    """
    Local LLM provider (Tier 2+).

    Interface for running a large language model locally on the user's
    device. The provider handles model loading, prompt formatting, and
    text generation.

    The generate() method accepts all standard LLM parameters to give
    callers fine-grained control over generation behavior. Different
    pipeline stages use different parameters:
    - Extraction: low temperature (0.05-0.1) for precise, factual output
    - Synthesis: moderate temperature (0.3-0.5) for natural language
    - Farming: higher temperature (0.5-0.7) for creative pattern description

    The provider also reports availability (is_available) and the loaded
    model name (get_model_name) for runtime capability detection.

    Usage:
        llm = LlamaCppProvider(model_path="/path/to/model.gguf")
        if llm.is_available():
            response = llm.generate(
                prompt="Extract entities from: Alice proposed OAuth2.",
                system_prompt="You are a Named Entity Recognition system.",
                temperature=0.1,
                json_mode=True,
            )
    """

    @abstractmethod
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
        """
        Generate text from a prompt with given parameters.

        Sends a prompt (and optional system prompt) to the loaded LLM
        and returns the generated text. Parameters control the generation
        behavior: temperature for randomness, max_tokens for length,
        top_p for nucleus sampling, etc.

        Args:
            prompt: The user/input prompt to send to the LLM.
            system_prompt: Optional system prompt that sets the LLM's
                           behavior (e.g., "You are a NER system").
                           None means no system prompt.
            temperature: Controls randomness. 0.0 = deterministic,
                         1.0 = very random. Default 0.1 (precise).
            max_tokens: Maximum number of tokens to generate. Default 1024.
            top_p: Nucleus sampling threshold. Only tokens with
                   cumulative probability <= top_p are considered.
                   Default 0.9.
            frequency_penalty: Penalizes tokens based on frequency in
                               the generated text. 0.0 = no penalty.
                               Default 0.0.
            presence_penalty: Penalizes tokens that have appeared at all.
                              0.0 = no penalty. Default 0.0.
            stop: Optional list of stop sequences. Generation stops
                  when any of these sequences is produced. None means
                  no stop sequences.
            json_mode: If True, request the LLM to output valid JSON.
                       Useful for structured extraction tasks. Not all
                       models support this. Default False.

        Returns:
            The generated text as a string. Does not include the
            prompt or system prompt.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if a local LLM model is loaded and ready.

        Returns True if an LLM model is currently loaded and able
        to generate text. Returns False if no model is loaded,
        the model file is missing, or the model failed to initialize.

        This is used by the system to determine which tier of
        functionality is available (Tier 0-1 without LLM, Tier 2+
        with LLM).

        Returns:
            True if the LLM is ready to generate, False otherwise.
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the loaded model name.

        Reports the name of the currently loaded LLM model. Used
        for logging, metrics, and provenance tracking (which model
        generated a particular output).

        Returns:
            The model name as a string (e.g., "llama-3.2-3b-instruct"
            or "phi-3-mini"). Returns an empty string if no model
            is loaded.
        """
        ...
