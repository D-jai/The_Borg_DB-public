# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Domain Profile Data Models
==========================

This module defines the Pydantic models for domain profiles -- the YAML
configuration files that control how ctxmtg behaves for different use cases
(legal, medical, engineering, personal, etc.).

A domain profile controls:
    - LLM parameters for each pipeline stage (temperature, top_p, etc.)
    - NER configuration (which entity types to extract, custom patterns)
    - Farming configuration (what patterns to look for, schedule)
    - Embedding configuration (model, chunk size, overlap)
    - Intake configuration (Traffic Cop rules for accept/defer/reject)

The profile system is what makes ctxmtg adaptable to any vertical.
A legal professional needs high-precision extraction with case-specific
entity types. A personal journal user needs discovery-oriented extraction
with emotion and goal entities. The same extraction pipeline handles
both -- the profile tells it what to look for and how aggressively.

Depends on:
    - pydantic (validation, serialization, default values)

Used by:
    - ctxmtg.profile.loader (loads YAML files into DomainProfile)
    - ctxmtg.extraction.pipeline (reads NER config, entity types)
    - ctxmtg.farming.pipeline (reads farming config, schedule)
    - ctxmtg.embedding.onnx_embedder (reads embedding config, model name)
    - ctxmtg.intake.rules (reads intake config, Traffic Cop rules)
    - ctxmtg.llm.prompt_assembler (reads stage configs for prompt params)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------
# StageParams model: LLM parameters for a single pipeline stage.
# These control how the LLM behaves when processing a specific task
# (extraction, query planning, synthesis, farming).
# ---------------------------------------------------------------
class StageParams(BaseModel):
    """
    LLM parameters for one pipeline stage.

    Each stage of the pipeline (extraction, query planning, synthesis,
    farming) can have its own LLM parameters. For example, extraction
    might use low temperature (precise) while farming uses higher
    temperature (creative pattern discovery).

    These parameters are only used when a local LLM is available (Tier 2+).
    In Tier 0-1 (no LLM), the extraction pipeline uses spaCy + regex
    and ignores these parameters.
    """

    # Controls randomness: 0.0 = deterministic, 1.0 = very random.
    # Lower values for extraction (precision), higher for farming (creativity).
    temperature: float = 0.1

    # Nucleus sampling: only tokens with cumulative probability <= top_p are considered.
    # 0.9 is a good balance between diversity and coherence.
    top_p: float = 0.9

    # Maximum number of tokens to generate in one response.
    max_tokens: int = 1024

    # Penalizes tokens that have already appeared, reducing repetition.
    # 0.0 = no penalty.
    frequency_penalty: float = 0.0

    # Penalizes tokens that have appeared at all, encouraging novelty.
    # 0.0 = no penalty.
    presence_penalty: float = 0.0

    # Whether to request structured (JSON) output from the LLM.
    # Useful for extraction stages where we need parseable results.
    structured_output: bool = False


# ---------------------------------------------------------------
# StageConfig model: configuration for one pipeline stage.
# Combines prompt versioning with LLM parameters for a stage.
# ---------------------------------------------------------------
class StageConfig(BaseModel):
    """
    Configuration for one pipeline stage within a domain profile.

    Each stage (extraction, query_planning, synthesis, farming) can have
    its own prompt version, domain-specific overlay text, and LLM parameters.
    The prompt_version determines which prompt template file to load from
    the prompts/ directory.
    """

    # Which version of the prompt template to use (e.g., "1.0.0")
    prompt_version: str = "1.0.0"

    # Domain-specific text to append to the base prompt (Layer 3 overlay)
    domain_overlay: str = ""

    # LLM parameters for this stage
    params: StageParams = Field(default_factory=StageParams)


# ---------------------------------------------------------------
# NERConfig model: controls the Named Entity Recognition pipeline.
# Specifies which entity types to extract and any custom patterns.
# ---------------------------------------------------------------
class EntityFilterConfig(BaseModel):
    """
    Entity rejection filters applied after NER + regex + merge.

    Entities matching any of these rules are dropped before storage.
    This prevents garbage entities (timestamp fragments, header labels,
    too-short names) from polluting the knowledge store.
    """

    # Minimum entity name length (drop shorter)
    min_name_length: int = 2

    # Maximum entity name length (drop longer — likely extraction errors)
    max_name_length: int = 200

    # Regex patterns — drop entity if full name matches any pattern
    reject_patterns: list[str] = Field(default_factory=list)

    # Exact names (case-insensitive) to always reject
    reject_names: list[str] = Field(default_factory=list)


class NERConfig(BaseModel):
    """
    NER configuration within a domain profile.

    Controls which entity types the extraction pipeline looks for and
    defines custom regex patterns for domain-specific entities. For
    example, a legal profile might add patterns for case numbers
    (e.g., "\\d{2}-CV-\\d{4}") and statute references.
    """

    # Which entity types to extract (e.g., ["person", "org", "project"])
    # Empty list means extract all supported types.
    entity_types: list[str] = Field(default_factory=list)

    # Custom regex patterns for domain-specific entities.
    # Each dict has "pattern" (regex) and "entity_type" (what to label matches as).
    custom_patterns: list[dict[str, str]] = Field(default_factory=list)

    # Entity rejection filters (applied post-extraction)
    entity_filters: EntityFilterConfig = Field(default_factory=EntityFilterConfig)


# ---------------------------------------------------------------
# FarmingConfig model: controls the meta-intelligence farming pipeline.
# Determines what patterns to look for, how often to run, and
# minimum thresholds for pattern detection.
# ---------------------------------------------------------------
class FarmingConfig(BaseModel):
    """
    Farming configuration within a domain profile.

    Controls the meta-intelligence mining pipeline: which patterns to
    prioritize, how often to run farming cycles, how far back to look,
    and minimum cluster sizes for grouping. Different domains may want
    different farming strategies -- legal might prioritize co-occurrence
    of case parties, while personal might prioritize emotional trends.
    """

    # Pattern types to prioritize (e.g., ["co-occurrence", "trend", "cluster"])
    priority_patterns: list[str] = Field(default_factory=list)

    # How often to run farming cycles (e.g., "daily", "weekly", "hourly")
    schedule: str = "daily"

    # How many days back to include in farming analysis
    lookback_window_days: int = 30

    # Minimum number of items to form a cluster (below this, no cluster is reported)
    cluster_min_size: int = 5


# ---------------------------------------------------------------
# EmbeddingConfig model: controls the text embedding pipeline.
# Specifies which model to use, chunking strategy, and fallback.
# ---------------------------------------------------------------
class EmbeddingConfig(BaseModel):
    """
    Embedding configuration within a domain profile.

    Controls which ONNX embedding model to use, with a fallback if the
    preferred model isn't available. Also controls text chunking: how
    large each chunk should be (in tokens) and how much overlap between
    adjacent chunks (to preserve context at chunk boundaries).
    """

    # Primary embedding model name (downloaded from HuggingFace)
    preferred_model: str = "all-MiniLM-L6-v2"

    # Fallback model if preferred isn't available
    fallback_model: str = "all-MiniLM-L6-v2"

    # Target chunk size in tokens for text splitting
    chunk_size: int = 256

    # Overlap between adjacent chunks (preserves context at boundaries)
    chunk_overlap: int = 32


# ---------------------------------------------------------------
# IntakeRule model: a single rule for the Traffic Cop.
# Each rule specifies conditions that, if matched, trigger an action
# (accept, defer, or reject) on the incoming interaction.
# ---------------------------------------------------------------
class IntakeRule(BaseModel):
    """
    A single Traffic Cop intake rule.

    Each rule defines conditions for matching incoming interactions.
    When evaluated, the Traffic Cop checks each condition that is set
    (non-None). If ALL set conditions match, the rule fires. Rules are
    evaluated in priority order: reject rules first, then defer, then accept.

    Fields are optional -- only set the conditions relevant to this rule.
    A rule with sender_pattern="*@noreply.*" only checks the sender;
    a rule with cc_only=True only checks if the user is CC'd.
    """

    # Glob pattern for sender address (e.g., "*@noreply.*" to reject auto-emails)
    sender_pattern: str | None = None

    # Glob pattern for email subject (e.g., "Weekly Digest*")
    subject_pattern: str | None = None

    # Source type to match (e.g., "automated_notification")
    source_type: str | None = None

    # If True, matches only when the user is CC'd (not in To:)
    cc_only: bool | None = None

    # Match threads deeper than this number of replies
    thread_depth_gt: int | None = None

    # If True, match when the sender is a known entity in the database
    sender_in_entities: bool | None = None

    # Match if the content contains ANY of these keywords
    contains_keywords: list[str] | None = None


# ---------------------------------------------------------------
# IntakeConfig model: the full Traffic Cop configuration.
# Groups rules by action (reject, defer, accept) and sets limits
# on content transformation (attachment stubs, code blocks, etc.).
# ---------------------------------------------------------------
class IntakeConfig(BaseModel):
    """
    Traffic Cop configuration within a domain profile.

    Defines the rules and content limits for the intake gateway.
    Rules are evaluated in order: reject first, then defer, then accept.
    If no rule matches, the default action is ACCEPT (permissive by default
    in Phase 1).

    Content transformation settings control how heavy content (attachments,
    code blocks, large text) is handled before extraction. This reduces
    noise and storage without losing metadata references.
    """

    # Operating mode: "passthrough" (no filtering), "rules" (YAML rules), "llm" (future)
    mode: str = "rules"

    # Rules that trigger REJECT action (first match wins)
    reject: list[IntakeRule] = Field(default_factory=list)

    # Rules that trigger DEFER action (first match wins)
    defer: list[IntakeRule] = Field(default_factory=list)

    # Rules that trigger ACCEPT action (first match wins)
    accept: list[IntakeRule] = Field(default_factory=list)

    # Whether to replace binary attachments with metadata stubs
    max_attachment_stub: bool = True

    # Max lines of code blocks to keep (0 = replace all with stubs)
    max_code_block_lines: int = 0

    # Max characters of inline content before truncation
    max_inline_content_chars: int = 10000


# ---------------------------------------------------------------
# DomainProfile model: a complete domain profile loaded from YAML.
# This is the top-level configuration object that controls all
# aspects of ctxmtg behavior for a specific domain/vertical.
# ---------------------------------------------------------------
class DomainProfile(BaseModel):
    """
    A complete domain profile loaded from YAML.

    This is the master configuration object for a specific use case
    (legal, medical, engineering, personal, etc.). It controls:
    - Pipeline stage parameters (LLM behavior per stage)
    - NER settings (entity types, custom patterns)
    - Farming settings (pattern priorities, schedule)
    - Embedding settings (model, chunk sizes)
    - Intake settings (Traffic Cop rules)

    Profiles are loaded from YAML files in the profiles/ directory.
    The profile loader validates the YAML against this model and fills
    in defaults for any missing fields.

    Usage:
        from ctxmtg.profile.loader import ProfileLoader
        profile = ProfileLoader.load("general")
        # profile.ner.entity_types → ["person", "org", "project", ...]
    """

    # Human-readable name of this profile (e.g., "General Engineering")
    name: str

    # Semantic version of this profile (e.g., "1.0.0").
    # Defaults to "1.0.0" for frictionless programmatic creation.
    version: str = "1.0.0"

    # Optional description of what this profile is for
    description: str = ""

    # LLM configuration per pipeline stage (keyed by stage name)
    stages: dict[str, StageConfig] = Field(default_factory=dict)

    # Named Entity Recognition configuration
    ner: NERConfig = Field(default_factory=NERConfig)

    # Meta-intelligence farming configuration
    farming: FarmingConfig = Field(default_factory=FarmingConfig)

    # Text embedding configuration
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    # Traffic Cop intake configuration
    intake: IntakeConfig = Field(default_factory=IntakeConfig)
