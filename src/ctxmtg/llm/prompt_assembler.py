# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
4-Layer Prompt Assembler
========================

This module provides the PromptAssembler class, which composes complete
system prompts for each LLM pipeline stage by combining four layers:

    Layer 1 (Base Identity):  Shared across all stages. Establishes the
        model's role as a local knowledge system component, sets output
        format rules, and defines safety constraints. (~100-200 tokens)

    Layer 2 (Stage Instructions):  Task-specific prompt for the current
        pipeline stage (extraction, query_planning, retrieval, synthesis,
        farming). Contains input/output schemas and reasoning instructions.
        (~300-800 tokens, domain-independent)

    Layer 3 (Domain Overlay):  Injected from the active DomainProfile.
        Contains domain-specific entity types, terminology mappings, and
        reasoning patterns (e.g., legal vs. medical vs. personal). (~100-400
        tokens, changes when profile switches)

    Layer 4 (User Preferences):  Optional user-specific customizations
        like preferred summary length, priority topics, or output language.
        (~50-150 tokens, user-editable)

The assembled prompt is: Base + Stage + Domain + UserPrefs, concatenated
with blank-line separators. Total prompt size: ~550-1550 tokens, well
within context limits for even small (1-3B parameter) local models.

Templates use {{slot_name}} syntax for slot injection. Slots are filled
from the DomainProfile at runtime. This is deterministic, auditable, and
debuggable -- the user can inspect the exact prompt used for any LLM call.

See research/system-prompt-architecture.md for the full 4-layer design.

Depends on:
    - pathlib (file path resolution)
    - structlog (structured logging)
    - ctxmtg.models.profile (DomainProfile, StageParams, StageConfig)
    - ctxmtg.exceptions (ProfileError for missing templates / unknown stages)

Used by:
    - ctxmtg.extraction.llm_verifier (assembles extraction prompts)
    - ctxmtg.query.llm_interpreter (assembles query planning prompts)
    - ctxmtg.query.synthesizer (assembles synthesis prompts)
    - ctxmtg.query.llm_fusion (assembles retrieval prompts)
    - ctxmtg.farming.insight_generator (assembles farming prompts)
    - ctxmtg.profile.assembler (convenience wrapper around this)
"""

from __future__ import annotations

import re
from pathlib import Path

import structlog

from ctxmtg.exceptions import ProfileError
from ctxmtg.models.profile import DomainProfile, StageParams

# ---------------------------------------------------------------
# Module-level logger: all log entries from this module are tagged
# with "ctxmtg.llm.prompt_assembler" for easy filtering.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.llm.prompt_assembler")

# ---------------------------------------------------------------
# Valid stage names: the 5 pipeline stages that the assembler
# knows about. Each maps to a prompt template subdirectory
# under prompts/stages/{stage_name}/. Unknown stage names raise
# ProfileError with a helpful message listing valid options.
# ---------------------------------------------------------------
VALID_STAGES: tuple[str, ...] = (
    "extraction",
    "query_planning",
    "retrieval",
    "synthesis",
    "farming",
)

# ---------------------------------------------------------------
# Slot pattern: matches {{slot_name}} in prompt templates.
# Used by _fill_slots() to find and replace template placeholders
# with actual values from the DomainProfile.
#
# Regex explanation:
#   \{\{   -- literal opening braces
#   \s*    -- optional whitespace inside braces
#   (...)  -- capture group for the slot name
#   [a-zA-Z_][a-zA-Z0-9_]*  -- valid Python identifier characters
#   \s*    -- optional whitespace before closing braces
#   \}\}   -- literal closing braces
# ---------------------------------------------------------------
_SLOT_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

# ---------------------------------------------------------------
# Default prompts directory: located at the project root under
# prompts/. This is where bundled prompt templates live. The path
# is resolved relative to this source file's location.
# ---------------------------------------------------------------
_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"


# =====================================================================
# PromptAssembler -- 4-Layer Prompt Composition Engine
# =====================================================================


class PromptAssembler:
    """
    4-layer prompt composition engine.

    Assembles prompts from: Base (Layer 1) + Stage (Layer 2) +
    Domain (Layer 3) + User Preferences (Layer 4).

    Templates are loaded from the prompts/ directory. Domain overlays
    come from the active DomainProfile. User preferences come from
    the profile's user_prefs section or are passed directly.

    See research/system-prompt-architecture.md for the full design.

    Why layered composition instead of monolithic prompts?
    Monolithic prompts create a combinatorial explosion: 5 stages ×
    N domains = 5N prompts to maintain. With 20 domains, that's 100
    separate prompts. A base change would require editing all 100.
    The layered approach means base changes propagate automatically.

    Usage:
        assembler = PromptAssembler()  # loads from default prompts/ dir
        profile = ProfileLoader.load("general")

        # Assemble a complete extraction prompt:
        prompt = assembler.assemble("extraction", profile)

        # Get LLM parameters for the extraction stage:
        params = assembler.get_stage_params("extraction", profile)
    """

    def __init__(self, prompts_dir: str | Path | None = None) -> None:
        """
        Load prompt templates from the specified directory.

        Scans the prompts/ directory structure for base and stage
        template files. Templates are cached in memory after loading
        so subsequent assemble() calls don't hit disk.

        The expected directory structure is:
            prompts_dir/
                base/
                    v1.0.0.txt    (Layer 1 base identity)
                stages/
                    extraction/
                        v1.0.0.txt    (Layer 2 extraction stage)
                    query_planning/
                        v1.0.0.txt    (Layer 2 query planning stage)
                    retrieval/
                        v1.0.0.txt    (Layer 2 retrieval stage)
                    synthesis/
                        v1.0.0.txt    (Layer 2 synthesis stage)
                    farming/
                        v1.0.0.txt    (Layer 2 farming stage)

        Args:
            prompts_dir: Path to the prompts directory. Defaults to
                         the bundled prompts/ directory at project root.
                         Accepts str or Path.

        Raises:
            ProfileError: If the prompts directory does not exist.
        """
        # Resolve the prompts directory path. Use default if not given.
        if prompts_dir is None:
            self._prompts_dir = _DEFAULT_PROMPTS_DIR
        else:
            self._prompts_dir = Path(prompts_dir)

        # ---------------------------------------------------------------
        # Verify the prompts directory exists. We check up front so that
        # callers get an immediate, clear error rather than a confusing
        # "file not found" later during assembly.
        # ---------------------------------------------------------------
        if not self._prompts_dir.is_dir():
            logger.error(
                "prompts_dir_not_found",
                error_code="CTXMTG-PRF-004",
                prompts_dir=str(self._prompts_dir),
            )
            raise ProfileError(
                f"Prompts directory not found: {self._prompts_dir}. "
                f"Expected a directory containing base/ and stages/ subdirectories "
                f"with prompt template files.",
                error_code="CTXMTG-PRF-004",
            )

        # ---------------------------------------------------------------
        # Template cache: stores loaded template text keyed by a tuple of
        # (category, name, version). Populated lazily by _load_template().
        # This avoids reading from disk on every assemble() call.
        # ---------------------------------------------------------------
        self._template_cache: dict[str, str] = {}

        logger.info(
            "prompt_assembler_initialized",
            prompts_dir=str(self._prompts_dir),
        )

    # ---------------------------------------------------------------
    # assemble: compose a complete 4-layer prompt for a stage + profile
    # ---------------------------------------------------------------
    def assemble(
        self,
        stage: str,
        profile: DomainProfile,
        user_prefs: dict[str, str] | None = None,
    ) -> str:
        """
        Assemble a complete system prompt for the given stage + profile.

        Combines all 4 layers into a single string:
            Layer 1: Base identity (from prompts/base/v{version}.txt)
            Layer 2: Stage instructions (from prompts/stages/{stage}/v{version}.txt)
            Layer 3: Domain overlay (from profile.stages[stage].domain_overlay)
            Layer 4: User preferences (from user_prefs dict)

        Template slots ({{slot_name}}) in Layers 1 and 2 are filled with
        values derived from the profile: entity_types from NER config,
        domain_description from the profile description, etc.

        Args:
            stage: Pipeline stage name. Must be one of: extraction,
                   query_planning, retrieval, synthesis, farming.
            profile: Active domain profile. Provides Layer 3 domain
                     overlay text and slot values (entity types, etc.).
            user_prefs: Optional user preference overrides for Layer 4.
                        Keys are preference names, values are text to
                        include. None means no user preferences layer.

        Returns:
            Complete system prompt string, all 4 layers concatenated
            with blank-line separators.

        Raises:
            ProfileError: If the stage name is unknown, or a required
                          template file is missing from the prompts dir.
        """
        # ---------------------------------------------------------------
        # Step 1: Validate the stage name.
        # Catch typos and unknown stages early with a clear error message
        # that lists all valid stage names.
        # ---------------------------------------------------------------
        self._validate_stage(stage)

        # ---------------------------------------------------------------
        # Step 2: Determine template versions to use.
        # The profile's stage config specifies which prompt version to use.
        # If the stage isn't configured in the profile, default to "1.0.0".
        # ---------------------------------------------------------------
        stage_config = profile.stages.get(stage)
        base_version = "1.0.0"
        stage_version = stage_config.prompt_version if stage_config else "1.0.0"

        # ---------------------------------------------------------------
        # Step 3: Load Layer 1 (base identity) template from disk/cache.
        # ---------------------------------------------------------------
        base_template = self._load_template("base", base_version)

        # ---------------------------------------------------------------
        # Step 4: Load Layer 2 (stage instructions) template from disk/cache.
        # ---------------------------------------------------------------
        stage_template = self._load_template(
            f"stages/{stage}",
            stage_version,
        )

        # ---------------------------------------------------------------
        # Step 5: Build the slot values dictionary from the profile.
        # These values replace {{slot_name}} placeholders in templates.
        # ---------------------------------------------------------------
        slot_values = self._build_slot_values(profile)

        # ---------------------------------------------------------------
        # Step 6: Fill slots in Layer 1 and Layer 2 templates.
        # Any remaining unfilled slots are left as-is -- they may be
        # filled later by the caller (e.g., {{query}}, {{sql_results}}).
        # ---------------------------------------------------------------
        filled_base = self._fill_slots(base_template, slot_values)
        filled_stage = self._fill_slots(stage_template, slot_values)

        # ---------------------------------------------------------------
        # Step 7: Build Layer 3 (domain overlay) from the profile.
        # The domain overlay is stored directly in the stage config
        # as a text string. If no stage config exists, Layer 3 is empty.
        # ---------------------------------------------------------------
        domain_overlay = ""
        if stage_config and stage_config.domain_overlay:
            domain_overlay = stage_config.domain_overlay.strip()

        # ---------------------------------------------------------------
        # Step 8: Build Layer 4 (user preferences) from the prefs dict.
        # Each preference key-value pair becomes a line in the prompt.
        # ---------------------------------------------------------------
        user_prefs_text = self._format_user_prefs(user_prefs)

        # ---------------------------------------------------------------
        # Step 9: Concatenate all layers with blank-line separators.
        # Only include layers that have content (skip empty layers).
        # ---------------------------------------------------------------
        layers = [filled_base, filled_stage]
        if domain_overlay:
            layers.append(domain_overlay)
        if user_prefs_text:
            layers.append(user_prefs_text)

        assembled = "\n\n".join(layers)

        logger.debug(
            "prompt_assembled",
            stage=stage,
            profile_name=profile.name,
            base_version=base_version,
            stage_version=stage_version,
            total_length=len(assembled),
            layer_count=len(layers),
        )

        return assembled

    # ---------------------------------------------------------------
    # get_stage_params: extract LLM parameters for a stage from profile
    # ---------------------------------------------------------------
    def get_stage_params(
        self,
        stage: str,
        profile: DomainProfile,
    ) -> StageParams:
        """
        Get LLM parameters for a specific stage from the profile.

        Each pipeline stage can have its own LLM parameters (temperature,
        top_p, max_tokens, etc.) defined in the profile's stages section.
        These control how the LLM behaves for that stage -- e.g., low
        temperature for extraction (precise) vs. higher temperature for
        farming (creative pattern discovery).

        If the stage is not defined in the profile, returns default
        StageParams (temperature=0.1, top_p=0.9, max_tokens=1024).

        Args:
            stage: Pipeline stage name. Must be one of the valid stages.
            profile: Domain profile to read parameters from.

        Returns:
            StageParams with the configured LLM parameters for this stage.

        Raises:
            ProfileError: If the stage name is unknown.
        """
        # Validate stage name first -- catch typos early.
        self._validate_stage(stage)

        # Look up the stage config in the profile. If the profile doesn't
        # define this stage, return default parameters.
        stage_config = profile.stages.get(stage)
        if stage_config is None:
            logger.debug(
                "stage_params_default",
                stage=stage,
                profile_name=profile.name,
                reason="stage not configured in profile",
            )
            return StageParams()

        # Return the params from the stage config.
        logger.debug(
            "stage_params_loaded",
            stage=stage,
            profile_name=profile.name,
            temperature=stage_config.params.temperature,
            max_tokens=stage_config.params.max_tokens,
        )
        return stage_config.params

    # =================================================================
    # Private helper methods
    # =================================================================

    # ---------------------------------------------------------------
    # _validate_stage: check that a stage name is one of the known stages
    # ---------------------------------------------------------------
    def _validate_stage(self, stage: str) -> None:
        """
        Validate that the stage name is one of the known pipeline stages.

        Raises ProfileError with a helpful message listing all valid
        stage names if the given name is unknown.

        Args:
            stage: Stage name to validate.

        Raises:
            ProfileError: If the stage name is not in VALID_STAGES.
        """
        if stage not in VALID_STAGES:
            logger.error(
                "invalid_pipeline_stage",
                error_code="CTXMTG-PRF-003",
                stage=stage,
                valid_stages=sorted(VALID_STAGES),
            )
            raise ProfileError(
                f"Unknown pipeline stage: '{stage}'. "
                f"Valid stages are: {', '.join(VALID_STAGES)}. "
                f"Check the stage name for typos.",
                error_code="CTXMTG-PRF-003",
            )

    # ---------------------------------------------------------------
    # _load_template: read a template file from disk (with caching)
    # ---------------------------------------------------------------
    def _load_template(self, subdirectory: str, version: str) -> str:
        """
        Load a prompt template file from the prompts directory.

        Templates are cached in memory after the first load, so
        subsequent calls for the same template don't hit disk.

        The file path is constructed as:
            prompts_dir / subdirectory / v{version}.txt

        For example:
            prompts_dir / "base" / "v1.0.0.txt"
            prompts_dir / "stages/extraction" / "v1.0.0.txt"

        Args:
            subdirectory: Relative path within prompts_dir
                          (e.g., "base", "stages/extraction").
            version: Semantic version string (e.g., "1.0.0").

        Returns:
            The template text as a string.

        Raises:
            ProfileError: If the template file does not exist or
                          cannot be read.
        """
        # Build a cache key from the subdirectory and version.
        cache_key = f"{subdirectory}/v{version}"

        # Return cached template if we've already loaded this one.
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        # Build the full file path.
        template_path = self._prompts_dir / subdirectory / f"v{version}.txt"

        # ---------------------------------------------------------------
        # Check that the file exists. Give a helpful error message
        # that includes the expected path and suggests corrective action.
        # ---------------------------------------------------------------
        if not template_path.exists():
            logger.error(
                "prompt_template_not_found",
                error_code="CTXMTG-PRF-004",
                template_path=str(template_path),
            )
            raise ProfileError(
                f"Prompt template not found: {template_path}. "
                f"Expected a prompt template file at this path. "
                f"Check that the prompts/ directory contains the correct "
                f"version files (e.g., v{version}.txt).",
                error_code="CTXMTG-PRF-004",
            )

        if not template_path.is_file():
            logger.error(
                "prompt_template_not_a_file",
                error_code="CTXMTG-PRF-004",
                template_path=str(template_path),
            )
            raise ProfileError(
                f"Prompt template path is not a file: {template_path}. "
                f"Expected a text file containing the prompt template.",
                error_code="CTXMTG-PRF-004",
            )

        try:
            template_text = template_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error(
                "prompt_template_read_failed",
                error_code="CTXMTG-PRF-004",
                template_path=str(template_path),
                error=str(e),
            )
            raise ProfileError(
                f"Cannot read prompt template {template_path}: {e}",
                error_code="CTXMTG-PRF-004",
            ) from e

        # Strip trailing whitespace/newlines for clean concatenation.
        template_text = template_text.strip()

        # Cache the loaded template for future use.
        self._template_cache[cache_key] = template_text

        logger.debug(
            "template_loaded",
            subdirectory=subdirectory,
            version=version,
            length=len(template_text),
            path=str(template_path),
        )

        return template_text

    # ---------------------------------------------------------------
    # _build_slot_values: extract slot values from a DomainProfile
    # ---------------------------------------------------------------
    def _build_slot_values(self, profile: DomainProfile) -> dict[str, str]:
        """
        Build a dictionary of slot values from the domain profile.

        These values replace {{slot_name}} placeholders in prompt
        templates. The slot names are designed to match the placeholders
        used in the bundled templates.

        Slots populated:
            - entity_types: comma-separated list from profile.ner.entity_types
            - domain_description: from profile.description
            - output_schema: standard extraction output schema (JSON)

        Args:
            profile: The domain profile to extract values from.

        Returns:
            Dictionary mapping slot names to their string values.
        """
        # ---------------------------------------------------------------
        # Build entity_types string from the NER configuration.
        # Join all entity types with ", " for readable inline display.
        # If no entity types are configured, use a helpful placeholder.
        # ---------------------------------------------------------------
        if profile.ner.entity_types:
            entity_types_str = ", ".join(profile.ner.entity_types)
        else:
            entity_types_str = "(no entity types configured)"

        # ---------------------------------------------------------------
        # Use the profile description as the domain description.
        # If no description is set, provide a generic fallback.
        # ---------------------------------------------------------------
        domain_description = profile.description or f"{profile.name} domain"

        # ---------------------------------------------------------------
        # Standard extraction output schema. This is the JSON schema
        # that the extraction stage should produce. It's the same
        # across all domains; domain-specific fields come from Layer 3.
        # ---------------------------------------------------------------
        output_schema = (
            '{"entities": [{"name": str, "type": str, "confidence": float}], '
            '"facts": [{"subject": str, "predicate": str, "object": str, '
            '"confidence": float}], "summary": str}'
        )

        return {
            "entity_types": entity_types_str,
            "domain_description": domain_description,
            "output_schema": output_schema,
        }

    # ---------------------------------------------------------------
    # _fill_slots: replace {{slot_name}} placeholders in a template
    # ---------------------------------------------------------------
    def _fill_slots(self, template: str, slot_values: dict[str, str]) -> str:
        """
        Replace {{slot_name}} placeholders in a template with actual values.

        Uses regex to find all {{slot_name}} patterns and replaces them
        with the corresponding value from slot_values. Slots that don't
        have a value in slot_values are left unchanged -- they may be
        runtime slots filled by the caller later (e.g., {{query}},
        {{sql_results}}, {{vector_results}}).

        Args:
            template: The template text with {{slot_name}} placeholders.
            slot_values: Dictionary mapping slot names to replacement values.

        Returns:
            The template with known slots replaced by their values.
        """
        def replace_slot(match: re.Match) -> str:
            """Replace a single slot match if the value is available."""
            slot_name = match.group(1)
            if slot_name in slot_values:
                return slot_values[slot_name]
            # Slot not in slot_values -- leave it as-is for later filling.
            return match.group(0)

        return _SLOT_PATTERN.sub(replace_slot, template)

    # ---------------------------------------------------------------
    # _format_user_prefs: format user preferences as prompt text
    # ---------------------------------------------------------------
    @staticmethod
    def _format_user_prefs(user_prefs: dict[str, str] | None) -> str:
        """
        Format user preferences as a prompt text section.

        Each preference is formatted as a "key: value" line under a
        "User Preferences:" header. Returns empty string if no
        preferences are provided.

        Args:
            user_prefs: Dictionary of user preference key-value pairs.
                        None or empty dict means no preferences.

        Returns:
            Formatted preference text, or empty string if none.
        """
        # No preferences → empty string (Layer 4 is skipped).
        if not user_prefs:
            return ""

        # Build a formatted text block with a header and one line
        # per preference. This makes the preferences visible and
        # structured within the assembled prompt.
        lines = ["User Preferences:"]
        for key, value in user_prefs.items():
            lines.append(f"- {key}: {value}")

        return "\n".join(lines)
