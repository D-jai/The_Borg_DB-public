# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Profile Assembler
=================

This module provides the ProfileAssembler class, which wires DomainProfile
stage parameters to LLM calls. Each pipeline stage (extraction, query
planning, retrieval, synthesis, farming) has its own temperature, top_p,
max_tokens, frequency_penalty, and presence_penalty defined in the
profile's ``stages`` section. This assembler provides a convenient way
to get the right parameters for each stage.

The primary use case is converting a profile's per-stage StageParams into
a dictionary that can be unpacked directly into LLMProvider.generate():

    assembler = ProfileAssembler(profile)
    kwargs = assembler.get_llm_kwargs("extraction")
    result = llm.generate(prompt, **kwargs)

If a stage is not defined in the profile's ``stages`` mapping, the
assembler returns default StageParams values (temperature=0.1, top_p=0.9,
max_tokens=1024, frequency_penalty=0.0, presence_penalty=0.0).

A convenience method ``get_system_prompt`` wraps PromptAssembler.assemble()
so callers can get both the prompt and the LLM kwargs from one object.

See research/parameter-control-analysis.md for the rationale behind
per-stage parameter tuning.

Depends on:
    - structlog (structured logging)
    - ctxmtg.models.profile (DomainProfile, StageParams, StageConfig)
    - ctxmtg.llm.prompt_assembler (PromptAssembler, for get_system_prompt)

Used by:
    - ctxmtg.extraction.llm_verifier (gets extraction params)
    - ctxmtg.query.llm_interpreter (gets query_planning params)
    - ctxmtg.query.synthesizer (gets synthesis params)
    - ctxmtg.query.llm_fusion (gets retrieval params)
    - ctxmtg.farming.insight_generator (gets farming params)
"""

from __future__ import annotations

from typing import Any

import structlog

from ctxmtg.models.profile import DomainProfile, StageParams

# ---------------------------------------------------------------
# Module-level logger: all log entries from this module are tagged
# with "ctxmtg.profile.assembler" for easy filtering.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.profile.assembler")


class ProfileAssembler:
    """
    Wires DomainProfile stage parameters to LLM calls.

    Each pipeline stage has its own temperature, top_p, max_tokens
    defined in the profile's stages section. This assembler provides
    a convenient way to get the right parameters for each stage,
    formatted as a dict ready to unpack into LLMProvider.generate().

    If a stage is not configured in the profile, default StageParams
    values are returned. This ensures graceful fallback when profiles
    don't define every stage explicitly.

    Usage:
        profile = ProfileLoader.load("legal")
        assembler = ProfileAssembler(profile)

        # Get LLM kwargs for extraction (low temperature, precise):
        kwargs = assembler.get_llm_kwargs("extraction")
        # kwargs == {"temperature": 0.05, "top_p": 0.85, ...}

        # Unpack directly into LLM call:
        result = llm.generate(prompt, **kwargs)
    """

    def __init__(self, profile: DomainProfile) -> None:
        """
        Initialize the ProfileAssembler with a DomainProfile.

        The profile provides per-stage LLM parameters via its
        ``stages`` dict. Each stage maps to a StageConfig which
        contains a StageParams with temperature, top_p, etc.

        Args:
            profile: The domain profile to read stage parameters from.
                     Must be a valid DomainProfile instance.
        """
        # Store the profile for parameter lookups. The profile is
        # immutable (Pydantic model) so it is safe to hold a reference.
        self._profile = profile

        logger.info(
            "profile_assembler_initialized",
            profile_name=profile.name,
            configured_stages=list(profile.stages.keys()),
        )

    # ---------------------------------------------------------------
    # get_llm_kwargs: convert stage params to LLM generation kwargs
    # ---------------------------------------------------------------
    def get_llm_kwargs(self, stage: str) -> dict[str, Any]:
        """
        Get LLM generation kwargs for a pipeline stage.

        Looks up the stage in the profile's ``stages`` dict and
        converts its StageParams into a dictionary that can be
        unpacked directly into LLMProvider.generate(**kwargs).

        The returned dict contains exactly these keys:
            - temperature (float): randomness of token selection
            - top_p (float): nucleus sampling cutoff
            - max_tokens (int): maximum generation length
            - frequency_penalty (float): repetition penalty
            - presence_penalty (float): topic diversity penalty

        If the stage is not defined in the profile, returns default
        StageParams values (temperature=0.1, top_p=0.9, max_tokens=1024,
        frequency_penalty=0.0, presence_penalty=0.0).

        Args:
            stage: Pipeline stage name (e.g., "extraction",
                   "query_planning", "retrieval", "synthesis",
                   "farming").

        Returns:
            Dictionary of LLM generation parameters, ready to unpack
            into LLMProvider.generate(**kwargs).
        """
        # Look up the stage config in the profile. If the profile
        # doesn't define this stage, fall back to default StageParams.
        stage_config = self._profile.stages.get(stage)

        if stage_config is not None:
            # Stage is configured -- use its params.
            params = stage_config.params
            logger.debug(
                "llm_kwargs_from_profile",
                stage=stage,
                profile_name=self._profile.name,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
            )
        else:
            # Stage not in profile -- use default params.
            params = StageParams()
            logger.debug(
                "llm_kwargs_default",
                stage=stage,
                profile_name=self._profile.name,
                reason="stage not configured in profile",
            )

        # Convert StageParams to a dict with exactly the keys that
        # LLMProvider.generate() accepts as keyword arguments.
        # We exclude structured_output because it maps to json_mode
        # in the provider, which is handled separately by each caller.
        return {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
        }

    # ---------------------------------------------------------------
    # get_system_prompt: convenience wrapper around PromptAssembler
    # ---------------------------------------------------------------
    def get_system_prompt(
        self,
        stage: str,
        prompt_assembler: Any,
        user_prefs: dict[str, str] | None = None,
    ) -> str:
        """
        Get the assembled system prompt for a pipeline stage.

        Convenience wrapper that delegates to PromptAssembler.assemble().
        Callers can get both the prompt and the LLM kwargs from a single
        ProfileAssembler instance without importing PromptAssembler.

        Args:
            stage: Pipeline stage name (e.g., "extraction").
            prompt_assembler: A PromptAssembler instance that handles
                              the 4-layer prompt composition. Typed as
                              Any to avoid circular imports (the real
                              type is ctxmtg.llm.prompt_assembler.PromptAssembler).
            user_prefs: Optional user preference overrides for Layer 4
                        of the prompt. Keys are preference names, values
                        are text to include. None means no user prefs.

        Returns:
            Complete system prompt string assembled from all 4 layers.

        Raises:
            ProfileError: If the stage name is unknown or a required
                          prompt template is missing.
        """
        # Delegate to the PromptAssembler for 4-layer composition.
        # The PromptAssembler reads domain overlay and slot values
        # from the profile stored in this assembler.
        result = prompt_assembler.assemble(
            stage=stage,
            profile=self._profile,
            user_prefs=user_prefs,
        )

        logger.debug(
            "system_prompt_assembled",
            stage=stage,
            profile_name=self._profile.name,
            prompt_length=len(result),
            has_user_prefs=user_prefs is not None and len(user_prefs) > 0,
        )

        return result

    # ---------------------------------------------------------------
    # profile property: read-only access to the underlying profile
    # ---------------------------------------------------------------
    @property
    def profile(self) -> DomainProfile:
        """
        The DomainProfile this assembler reads parameters from.

        Read-only property for callers that need to inspect the
        profile metadata (name, version, description, NER config, etc.).
        """
        return self._profile
