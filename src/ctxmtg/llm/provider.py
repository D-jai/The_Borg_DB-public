# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
LlamaCpp LLM Provider
======================

This module implements the LLMProvider interface using llama-cpp-python,
which wraps the llama.cpp inference engine. This enables running GGUF-
format language models locally on the user's device -- no cloud API,
no internet required, all data stays on-device.

llama-cpp-python is an optional dependency (installed via `pip install
ctxmtg[llm]`). If it's not installed, __init__ raises ConfigError
with a helpful message telling the user how to install it.

Why llama.cpp?
    It supports quantized GGUF models that fit in 2-8 GB of RAM,
    runs on CPU (with optional GPU offloading), and works on all
    platforms including ARM (Raspberry Pi, Apple Silicon). It's the
    most portable local LLM backend available.
    (See research/round-3/architecture-proposal-v2.md for alternatives.)

Context window management:
    The provider creates the model with a fixed context window (n_ctx,
    default 4096 tokens). If a prompt + max_tokens exceeds n_ctx,
    llama.cpp will truncate or error. Callers are responsible for
    keeping prompts within bounds (the PromptAssembler handles this).

Error handling:
    - Model file not found → ConfigError
    - llama_cpp not installed → ConfigError
    - Out of memory during generation → logged warning, empty string returned
    - Generation timeout → logged warning, empty string returned

Depends on:
    - llama_cpp (optional: llama-cpp-python package)
    - ctxmtg.interfaces.llm (LLMProvider ABC)
    - ctxmtg.exceptions (ConfigError)
    - ctxmtg.llm.model_adapter (ModelAdapter for chat template formatting)

Used by:
    - ctxmtg.llm.prompt_assembler (passes prompts through this provider)
    - ctxmtg.extraction.llm_verifier (uses LLM for extraction enhancement)
    - ctxmtg.query.llm_interpreter (uses LLM for query understanding)
    - ctxmtg.query.synthesizer (uses LLM for answer synthesis)
"""

from __future__ import annotations

from pathlib import Path

import structlog

from ctxmtg.exceptions import ConfigError
from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.llm.api_provider import strip_thinking_tokens
from ctxmtg.llm.model_adapter import ModelAdapter

# ---------------------------------------------------------------
# Module-level logger -- structured JSON output, no PII in logs.
# Never log prompt content (could contain user data).
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.llm.provider")


# =====================================================================
# LlamaCppProvider -- Local LLM via llama.cpp
# =====================================================================


class LlamaCppProvider(LLMProvider):
    """
    Local LLM provider using llama-cpp-python.

    Loads a GGUF model file and provides text generation. Handles
    chat template detection (via ModelAdapter), context window
    management, and graceful error handling for OOM and timeouts.

    The provider is lazy about nothing -- the model is loaded
    immediately in __init__. This is intentional: we want to fail
    fast if the model file is missing or corrupted, rather than
    discovering the problem during the first query.

    Configuration:
        model_path: Path to a .gguf model file on disk.
        n_ctx: Context window size in tokens. Default 4096.
               Larger = more context but more RAM. 4096 is a safe
               default for 3B-7B models on 8GB+ RAM machines.
        n_gpu_layers: Number of model layers to offload to GPU.
                      Default 0 (CPU only). Set to -1 for all layers
                      on GPU, or a specific number for partial offload.
        verbose: Whether to print llama.cpp's internal debug output.
                 Default False (quiet). Set True for debugging.

    Usage:
        provider = LlamaCppProvider(
            model_path="/models/llama-3.2-3b-instruct.Q4_K_M.gguf",
            n_ctx=4096,
            n_gpu_layers=0,
        )
        if provider.is_available():
            response = provider.generate(
                prompt="What entities are in this text?",
                system_prompt="You are an NER system.",
                temperature=0.1,
                json_mode=True,
            )
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> None:
        """
        Load a GGUF model for local inference.

        Immediately loads the model file into memory. Raises ConfigError
        if the model file doesn't exist or if llama-cpp-python is not
        installed.

        Args:
            model_path: Absolute or relative path to a .gguf model file.
            n_ctx: Context window size in tokens. Default 4096.
            n_gpu_layers: GPU layers to offload. 0 = CPU only.
            verbose: Print llama.cpp debug output. Default False.

        Raises:
            ConfigError: If llama-cpp-python is not installed, or if
                         the model file does not exist on disk.
        """
        # ---------------------------------------------------------------
        # Step 1: Check that llama-cpp-python is installed.
        # It's an optional dependency -- users install it with
        # `pip install ctxmtg[llm]`. If it's missing, we raise a
        # ConfigError with a helpful installation message.
        # ---------------------------------------------------------------
        try:
            import llama_cpp  # noqa: F401 -- we just need to check importability
        except ImportError as exc:
            logger.error(
                "llama_cpp_not_installed",
                error_code="CTXMTG-CFG-005",
                error=str(exc),
            )
            raise ConfigError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install ctxmtg[llm]",
                error_code="CTXMTG-CFG-005",
            ) from exc

        # ---------------------------------------------------------------
        # Step 2: Verify the model file exists on disk.
        # We expand user home (~) and resolve symlinks to get the real
        # path, then check existence before attempting to load.
        # ---------------------------------------------------------------
        self._model_path = Path(model_path).expanduser().resolve()
        if not self._model_path.exists():
            logger.error(
                "llm_model_file_not_found",
                error_code="CTXMTG-EMB-001",
                model_path=str(self._model_path),
            )
            raise ConfigError(
                f"Model file not found: {self._model_path}. "
                f"Please download a GGUF model file.",
                error_code="CTXMTG-EMB-001",
            )

        # ---------------------------------------------------------------
        # Step 3: Store configuration parameters.
        # ---------------------------------------------------------------
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._verbose = verbose

        # ---------------------------------------------------------------
        # Step 4: Extract model name from filename (for logging and
        # provenance tracking). Strip the .gguf extension and any
        # quantization suffixes to get a clean name.
        # Example: "llama-3.2-3b-instruct.Q4_K_M.gguf" → "llama-3.2-3b-instruct.Q4_K_M"
        # ---------------------------------------------------------------
        self._model_name = self._model_path.stem

        # ---------------------------------------------------------------
        # Step 5: Detect the chat template from the model filename.
        # This determines how we format prompts before sending them
        # to the model. See model_adapter.py for template details.
        # ---------------------------------------------------------------
        self._template = ModelAdapter.detect_template(self._model_name)

        # ---------------------------------------------------------------
        # Step 6: Load the model via llama-cpp-python.
        # This allocates memory and loads the model weights. Can take
        # several seconds for large models (7B+).
        # ---------------------------------------------------------------
        logger.info(
            "loading_model",
            model_path=str(self._model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            template=self._template,
        )

        try:
            from llama_cpp import Llama

            self._model = Llama(
                model_path=str(self._model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )
            # Track that the model loaded successfully.
            self._available = True
            logger.info(
                "model_loaded",
                model_name=self._model_name,
                template=self._template,
            )
        except Exception as exc:
            # If model loading fails (corrupted file, OOM, etc.),
            # log the error but don't crash. The provider will report
            # is_available=False and callers can fall back.
            self._model = None
            self._available = False
            logger.error(
                "model_load_failed",
                error_code="CTXMTG-EMB-002",
                model_name=self._model_name,
                error=str(exc),
            )

    # ---------------------------------------------------------------
    # generate: send a prompt to the model and get text back
    # ---------------------------------------------------------------
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
        Generate text from a prompt using the loaded GGUF model.

        Formats the prompt using the detected chat template, sends it
        to the model, and returns the generated text. Handles errors
        gracefully -- OOM and timeouts return an empty string rather
        than crashing.

        Args:
            prompt: The user input / question / data to process.
            system_prompt: Optional system instruction. None = no system prompt.
            temperature: Controls randomness. 0.0 = deterministic. Default 0.1.
            max_tokens: Maximum tokens to generate. Default 1024.
            top_p: Nucleus sampling threshold. Default 0.9.
            frequency_penalty: Token frequency penalty. Default 0.0.
            presence_penalty: Token presence penalty. Default 0.0.
            stop: Optional stop sequences. Default None.
            json_mode: If True, request JSON output. Default False.

        Returns:
            Generated text string. Empty string if model unavailable or error.
        """
        # Can't generate if the model didn't load successfully.
        if not self._available or self._model is None:
            logger.warning(
                "generate_skipped",
                error_code="CTXMTG-EMB-002",
                reason="model_not_available",
            )
            return ""

        # ---------------------------------------------------------------
        # Handle json_mode: append instruction to the prompt if the model
        # doesn't support native JSON mode. Most GGUF models don't have
        # a built-in JSON mode, so we add "Respond with valid JSON." to
        # the prompt to encourage JSON output.
        # ---------------------------------------------------------------
        effective_prompt = prompt
        if json_mode:
            effective_prompt = prompt.rstrip() + "\n\nRespond with valid JSON."

        # ---------------------------------------------------------------
        # Format the prompt using the detected chat template.
        # This wraps system + user prompts in the correct special tokens
        # (ChatML, Llama-3, or plain) so the model can follow instructions.
        # ---------------------------------------------------------------
        formatted_prompt = ModelAdapter.format_prompt(
            system_prompt=system_prompt or "",
            user_prompt=effective_prompt,
            template=self._template,
        )

        # ---------------------------------------------------------------
        # Call the model to generate text.
        # We wrap this in try/except to handle OOM and other runtime errors
        # gracefully. A failed generation returns "" rather than crashing
        # the entire pipeline -- the caller can then fall back to Phase 1
        # behavior.
        # ---------------------------------------------------------------
        try:
            # Build generation kwargs. llama-cpp-python's __call__ method
            # accepts these parameters directly.
            result = self._model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop or [],
            )

            # Extract the generated text from the response dict.
            # llama-cpp-python returns a dict with "choices" list.
            choices = result.get("choices", [])
            if choices:
                generated_text: str = str(choices[0].get("text", ""))
                logger.debug(
                    "generation_complete",
                    model_name=self._model_name,
                    prompt_length=len(formatted_prompt),
                    output_length=len(generated_text),
                )
                return strip_thinking_tokens(generated_text)

            # No choices in response -- unusual but possible.
            logger.warning(
                "generation_empty",
                error_code="CTXMTG-EMB-003",
                model_name=self._model_name,
            )
            return ""

        except MemoryError:
            # Out of memory -- the model or generation is too large for
            # available RAM. Log and return empty string so callers can
            # fall back to non-LLM behavior.
            logger.error(
                "generation_oom",
                error_code="CTXMTG-EMB-003",
                model_name=self._model_name,
                prompt_length=len(formatted_prompt),
                max_tokens=max_tokens,
            )
            return ""

        except Exception as exc:
            logger.error(
                "generation_failed",
                error_code="CTXMTG-EMB-003",
                model_name=self._model_name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return ""

    # ---------------------------------------------------------------
    # is_available: check if the model is loaded and ready
    # ---------------------------------------------------------------
    def is_available(self) -> bool:
        """
        Check if the LLM model is loaded and ready to generate.

        Returns True if the model was successfully loaded in __init__.
        Returns False if loading failed (file missing, OOM, etc.).

        Used by the system to determine whether LLM-enhanced features
        (Tier 2+) are available, or if it should fall back to Phase 1
        rule-based behavior.

        Returns:
            True if the model is ready, False otherwise.
        """
        return self._available

    # ---------------------------------------------------------------
    # get_model_name: return the loaded model's name
    # ---------------------------------------------------------------
    def get_model_name(self) -> str:
        """
        Return the loaded model's name.

        Derived from the model filename (without extension). Used for
        logging, metrics, and provenance tracking -- when the LLM
        generates an entity or fact, we record which model did it.

        Returns:
            Model name string, e.g., "llama-3.2-3b-instruct.Q4_K_M".
            Empty string if no model is loaded.
        """
        if not self._available:
            return ""
        return self._model_name
