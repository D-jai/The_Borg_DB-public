# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Model Chat Template Adapter
============================

This module adapts prompts for different model chat template formats.
Different LLM models expect their input formatted in specific ways --
ChatML (used by Phi, Qwen), Llama-3 format (used by Meta Llama models),
or plain concatenation (fallback for unknown models).

The adapter detects the right template from the model filename and
wraps system + user prompts in the correct special tokens so the
model actually follows instructions instead of just continuing text.

Why this matters:
    If you send a ChatML-formatted prompt to a Llama-3 model, it
    won't recognize the system/user/assistant boundaries. The model
    needs to see its own template tokens to activate its instruction-
    following behavior. Wrong template = garbage output.

Template formats supported:
    1. ChatML: <|im_start|>role\ncontent<|im_end|>
       Used by: Phi-3, Qwen, many fine-tunes
    2. Llama-3: <|start_header_id|>role<|end_header_id|>\ncontent<|eot_id|>
       Used by: Meta Llama 3, Llama 3.1, Llama 3.2
    3. Plain: Just concatenate system + user text with newlines
       Used by: Fallback for unknown models

Depends on:
    - nothing (leaf module, pure string formatting)

Used by:
    - ctxmtg.llm.provider (wraps prompts before sending to llama.cpp)
"""

from __future__ import annotations

import structlog

from ctxmtg.exceptions import ConfigError

# ---------------------------------------------------------------
# Module-level logger for template detection diagnostics.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.llm.model_adapter")

# =====================================================================
# Template ID constants -- the strings we use to identify templates.
# These are returned by detect_template() and consumed by format_prompt().
# =====================================================================

TEMPLATE_CHATML = "chatml"
TEMPLATE_LLAMA3 = "llama3"
TEMPLATE_PLAIN = "plain"

# ---------------------------------------------------------------
# All known template IDs, for validation in format_prompt().
# ---------------------------------------------------------------
KNOWN_TEMPLATES = {TEMPLATE_CHATML, TEMPLATE_LLAMA3, TEMPLATE_PLAIN}

# =====================================================================
# Filename heuristics for template detection.
# Each entry maps a substring (found in the model filename) to a
# template ID. Order matters -- first match wins.
# Based on common GGUF model naming conventions.
# =====================================================================

# Models that use the Llama-3 chat template format.
# "llama-3", "llama3", "Llama-3.2", etc.
_LLAMA3_INDICATORS = ["llama-3", "llama3", "llama_3"]

# Models that use the ChatML template format.
# Phi-3/4, Qwen, Mistral v0.2+, many community fine-tunes.
_CHATML_INDICATORS = ["phi", "qwen", "mistral", "chatml", "openhermes", "hermes"]


# =====================================================================
# ModelAdapter -- Static methods for template detection and formatting
# =====================================================================


class ModelAdapter:
    """
    Adapts prompts for different model chat template formats.

    Different models expect different chat formats (ChatML, Llama-3,
    plain, etc.). This adapter detects the model's preferred format
    from its filename and wraps system + user prompts accordingly.

    All methods are static -- no state needed. The adapter is a
    pure utility class for string formatting.

    Usage:
        template = ModelAdapter.detect_template("llama-3.2-3b-instruct.Q4_K_M.gguf")
        # Returns "llama3"

        prompt = ModelAdapter.format_prompt(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is 2+2?",
            template=template,
        )
        # Returns the prompt wrapped in Llama-3 chat tokens
    """

    # ---------------------------------------------------------------
    # detect_template: figure out which chat template a model expects
    # by scanning its filename for known patterns.
    # ---------------------------------------------------------------
    @staticmethod
    def detect_template(model_name: str) -> str:
        """
        Detect chat template from model name / filename.

        Scans the model name (case-insensitive) for known substrings
        that indicate which chat template the model was trained with.
        Returns a template ID string.

        Detection priority:
            1. Llama-3 indicators (llama-3, llama3, llama_3)
            2. ChatML indicators (phi, qwen, mistral, chatml, etc.)
            3. Default: ChatML (safest fallback for instruction models)

        Args:
            model_name: The model filename or identifier to analyze.
                        Examples: "llama-3.2-3b-instruct.Q4_K_M.gguf",
                        "phi-3-mini-4k-instruct.gguf", "mistral-7b.gguf"

        Returns:
            Template ID string: "chatml", "llama3", or "plain".
        """
        # Lowercase for case-insensitive matching.
        name_lower = model_name.lower()

        # Check Llama-3 indicators first -- Llama-3 has its own
        # distinct template that's different from older Llama models.
        for indicator in _LLAMA3_INDICATORS:
            if indicator in name_lower:
                logger.debug(
                    "template_detected",
                    model_name=model_name,
                    template=TEMPLATE_LLAMA3,
                    matched_indicator=indicator,
                )
                return TEMPLATE_LLAMA3

        # Check ChatML indicators -- a widely adopted format used by
        # Phi, Qwen, Mistral, and many community fine-tunes.
        for indicator in _CHATML_INDICATORS:
            if indicator in name_lower:
                logger.debug(
                    "template_detected",
                    model_name=model_name,
                    template=TEMPLATE_CHATML,
                    matched_indicator=indicator,
                )
                return TEMPLATE_CHATML

        # Default to ChatML -- it's the most common template for
        # instruction-tuned models, and a safe fallback. If the model
        # doesn't recognize ChatML tokens, it will mostly ignore them
        # and still produce output (just less reliably).
        logger.debug(
            "template_defaulted",
            model_name=model_name,
            template=TEMPLATE_CHATML,
        )
        return TEMPLATE_CHATML

    # ---------------------------------------------------------------
    # format_prompt: wrap system + user prompts in chat template tokens
    # ---------------------------------------------------------------
    @staticmethod
    def format_prompt(
        system_prompt: str,
        user_prompt: str,
        template: str = "chatml",
    ) -> str:
        """
        Format system + user prompts for the model's chat template.

        Takes a system prompt (the model's "personality" / task description)
        and a user prompt (the actual input), and wraps them in the
        appropriate chat template tokens so the model can distinguish
        between system instructions and user input.

        Args:
            system_prompt: The system-level instruction for the model.
                           Sets the model's behavior (e.g., "You are a
                           Named Entity Recognition system."). Can be
                           an empty string if no system prompt is needed.
            user_prompt: The user's actual input / question / data to
                         process. Must not be empty.
            template: The template ID to use. Must be one of:
                      "chatml", "llama3", "plain". Default: "chatml".

        Returns:
            The formatted prompt string with chat template tokens.

        Raises:
            ValueError: If the template ID is not recognized.
        """
        # Validate template ID to catch typos early.
        if template not in KNOWN_TEMPLATES:
            raise ConfigError(
                f"Unknown template '{template}'. "
                f"Must be one of: {sorted(KNOWN_TEMPLATES)}",
                error_code="CTXMTG-CFG-002",
            )

        # Dispatch to the appropriate formatter.
        if template == TEMPLATE_CHATML:
            return _format_chatml(system_prompt, user_prompt)
        elif template == TEMPLATE_LLAMA3:
            return _format_llama3(system_prompt, user_prompt)
        else:
            # Plain template: just concatenate with newlines.
            return _format_plain(system_prompt, user_prompt)


# =====================================================================
# Private formatting functions -- one per template type.
# These produce the actual token-wrapped prompt strings.
# =====================================================================


def _format_chatml(system_prompt: str, user_prompt: str) -> str:
    """
    Format prompts in ChatML template.

    ChatML is the most widely adopted chat template, used by OpenAI,
    Phi, Qwen, and many community models. Structure:

        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {user_prompt}<|im_end|>
        <|im_start|>assistant

    The trailing "assistant" header tells the model to start generating
    the assistant's response.

    Args:
        system_prompt: System instruction text.
        user_prompt: User input text.

    Returns:
        ChatML-formatted prompt string.
    """
    parts = []

    # Only include system block if there's a system prompt.
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

    # User block with the actual input.
    parts.append(f"<|im_start|>user\n{user_prompt}<|im_end|>")

    # Assistant header -- the model continues from here.
    parts.append("<|im_start|>assistant")

    # Join with newlines between blocks.
    return "\n".join(parts)


def _format_llama3(system_prompt: str, user_prompt: str) -> str:
    """
    Format prompts in Llama-3 template.

    Meta's Llama 3 uses a different set of special tokens:

        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Note the double newlines -- Llama-3's template has blank lines
    after each header, which is important for the model to parse.

    Args:
        system_prompt: System instruction text.
        user_prompt: User input text.

    Returns:
        Llama-3-formatted prompt string.
    """
    parts = []

    # Begin-of-text token starts the conversation.
    parts.append("<|begin_of_text|>")

    # System block (only if system prompt provided).
    if system_prompt:
        parts.append(
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
        )

    # User block with the actual input.
    parts.append(
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_prompt}<|eot_id|>"
    )

    # Assistant header -- the model starts generating from here.
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

    return "".join(parts)


def _format_plain(system_prompt: str, user_prompt: str) -> str:
    """
    Format prompts as plain text (no special tokens).

    This is a simple fallback for models that don't use special chat
    template tokens. Just concatenates system and user prompts with
    a blank line separator.

    This format is less reliable for instruction-following because the
    model can't distinguish "these are instructions" from "this is text
    to process". But it works as a last resort.

    Args:
        system_prompt: System instruction text.
        user_prompt: User input text.

    Returns:
        Plain text formatted prompt.
    """
    if system_prompt:
        return f"{system_prompt}\n\n{user_prompt}"
    return user_prompt
