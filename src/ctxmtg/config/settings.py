# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Configuration Settings
======================

This module defines the CtxMtgSettings class, which provides a single
source of truth for all system configuration. Values are loaded from
multiple sources in priority order:

    1. Environment variables (highest priority, CTXMTG_ prefix)
    2. Config YAML file (if specified)
    3. Default values defined here (lowest priority)

Built on Pydantic BaseSettings, which gives automatic type validation,
environment variable binding, and clear documentation of all options.

Depends on:
    - pydantic_settings (BaseSettings for env var + config loading)
    - pathlib (file path handling)

Used by:
    - ctxmtg.cli (reads settings for all commands)
    - ctxmtg.ingestion.worker (reads storage paths, profile name)
    - ctxmtg.query.executor (reads query settings)
"""

from __future__ import annotations

from pathlib import Path

import structlog
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from ctxmtg import paths

# ---------------------------------------------------------------
# Module-level logger.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.config.settings")


# ---------------------------------------------------------------
# Hive configuration: controls how the local instance synchronises
# with the hive (the central knowledge repository).
# See research/notes/hive-sync-design.md for the full design.
# ---------------------------------------------------------------
class HiveSettings(BaseModel):
    """
    Hive sync configuration.

    The hive is a separate database that aggregates knowledge from
    one or more ctxmtg instances.  In "local" mode the hive is a
    second SQLite file on the same machine.  In "remote" mode it
    would be an HTTP server (implemented in Phase 4).

    Sync runs on a configurable schedule (default 30 minutes) and
    pushes interactions, entities, and facts that have not yet been
    synced (hive_synced_at IS NULL).
    """

    # Mode: "local" (second SQLite on same disk) or "remote" (HTTP, Phase 4)
    mode: str = Field(
        default="local",
        description="Hive mode: 'local' (SQLite) or 'remote' (HTTP, Phase 4)",
    )

    # Local-mode path to the hive SQLite database file
    local_db_path: str = Field(
        default_factory=lambda: str(paths.get_hive_db_path()),
        description="Path to the hive SQLite database (local mode)",
    )

    # Local-mode path to the hive vector store directory
    local_vector_path: str = Field(
        default_factory=lambda: str(paths.get_hive_vector_path()),
        description="Path to the hive vector store (local mode)",
    )

    # Remote-mode API URL (Phase 4 stub)
    remote_url: str | None = Field(
        default=None,
        description="Remote hive API URL (Phase 4)",
    )

    # Sync interval in minutes
    sync_interval_minutes: int = Field(
        default=30,
        description="Minutes between automatic sync cycles",
    )

    # Whether to include raw interaction content in the sync payload
    sync_interaction_content: bool = Field(
        default=True,
        description="Include interaction content in hive sync",
    )

    # ---------------------------------------------------------------
    # Outbox settings (2026-04-08: decoupled hive sync via outbox)
    # Local writes JSON manifests here; hive pulls from this path.
    # ---------------------------------------------------------------

    # Directory where this local writes outbox manifests for hive pull
    outbox_path: str = Field(
        default_factory=lambda: str(paths.get_outbox_path()),
        description="Directory for outbox manifest files (hive pulls from here)",
    )

    # Human-readable name for this local instance (used in manifests)
    instance_name: str = Field(
        default="local",
        description="Name of this local instance (appears in hive links and manifests)",
    )


class LLMRoleSettings(BaseModel):
    """
    Per-role LLM API configuration.

    Each pipeline stage (extraction, query_planning, retrieval,
    synthesis, farming, fusion) can have its own API provider.
    Empty strings mean "not configured" -- the role falls back
    to its non-LLM behavior.
    """

    api_key: str = Field(default="", description="API key for this role")
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API base URL",
    )
    model: str = Field(default="gpt-4o-mini", description="Model name")


class LLMSettings(BaseModel):
    """LLM configuration for all pipeline roles."""

    extraction: LLMRoleSettings = Field(default_factory=LLMRoleSettings)
    query_planning: LLMRoleSettings = Field(default_factory=LLMRoleSettings)
    retrieval: LLMRoleSettings = Field(default_factory=LLMRoleSettings)
    synthesis: LLMRoleSettings = Field(default_factory=LLMRoleSettings)
    farming: LLMRoleSettings = Field(default_factory=LLMRoleSettings)
    fusion: LLMRoleSettings = Field(default_factory=LLMRoleSettings)


class CtxMtgSettings(BaseSettings):
    """
    System-wide configuration for ctxmtg.

    Values are loaded in priority order (highest wins):
    1. Environment variables (CTXMTG_DB_PATH, CTXMTG_VECTOR_PATH, etc.)
    2. Default values defined here

    All paths support ~ expansion (replaced with user's home dir).

    Usage:
        settings = CtxMtgSettings()
        db_path = settings.resolve_db_path()
    """

    # ---------------------------------------------------------------
    # Storage paths
    # ---------------------------------------------------------------

    # Path to the SQLite knowledge database file
    db_path: str = Field(
        default_factory=lambda: str(paths.get_db_path()),
        description="Path to the SQLite database file",
    )

    # Directory for the LanceDB vector store
    vector_path: str = Field(
        default_factory=lambda: str(paths.get_vector_path()),
        description="Directory for vector store data",
    )

    # ---------------------------------------------------------------
    # Profile settings
    # ---------------------------------------------------------------

    # Which domain profile to load (must match a YAML in profiles/)
    profile_name: str = Field(
        default="general",
        description="Active domain profile name",
    )

    # Directory containing profile YAML files
    profile_dir: str = Field(
        default="",
        description="Custom profiles directory (empty = use bundled profiles)",
    )

    # ---------------------------------------------------------------
    # Extraction settings
    # ---------------------------------------------------------------

    # spaCy model name for NER
    spacy_model: str = Field(
        default="en_core_web_sm",
        description="spaCy model for Named Entity Recognition",
    )

    # ---------------------------------------------------------------
    # Embedding settings
    # ---------------------------------------------------------------

    # ONNX embedding model name
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="ONNX embedding model name",
    )

    # ---------------------------------------------------------------
    # Query settings
    # ---------------------------------------------------------------

    # Default retrieval mode
    retrieval_mode: str = Field(
        default="parallel",
        description="Default query retrieval mode (parallel or deep)",
    )

    # Default number of results
    top_k: int = Field(
        default=10,
        description="Default number of query results",
    )

    # ---------------------------------------------------------------
    # Inbox watcher settings (for ctxmtg watch)
    # ---------------------------------------------------------------

    inbox_path: str = Field(
        default_factory=lambda: str(paths.get_inbox_path()),
        description="Directory to watch for new files to ingest",
    )

    processed_path: str = Field(
        default_factory=lambda: str(paths.get_processed_path()),
        description="Directory where successfully ingested files are moved",
    )

    watch_interval_seconds: int = Field(
        default=30,
        description="Seconds between inbox polls (ctxmtg watch)",
    )

    # ---------------------------------------------------------------
    # HTTP ingest endpoint (on the web server)
    # ---------------------------------------------------------------

    http_ingest_enabled: bool = Field(
        default=True,
        description="Enable POST /api/ingest on the web server",
    )

    # ---------------------------------------------------------------
    # LLM proxy settings (ctxmtg proxy)
    # ---------------------------------------------------------------

    proxy_enabled: bool = Field(
        default=False,
        description="Enable the LLM proxy for live chat capture",
    )

    proxy_port: int = Field(
        default=11435,
        description="Port the LLM proxy listens on",
    )

    proxy_upstream: str = Field(
        default="http://localhost:11434",
        description="Upstream LLM backend URL (e.g., Ollama)",
    )

    # ---------------------------------------------------------------
    # Server settings (for ctxmtg start)
    # ---------------------------------------------------------------

    host: str = Field(
        default="127.0.0.1",
        description="Server bind address",
    )

    port: int = Field(
        default=8080,
        description="Server port",
    )

    # ---------------------------------------------------------------
    # Hive sync settings (see HiveSettings above)
    # ---------------------------------------------------------------
    hive: HiveSettings = Field(
        default_factory=HiveSettings,
        description="Hive sync configuration",
    )

    # ---------------------------------------------------------------
    # Per-role LLM API configuration (see LLMSettings above)
    # ---------------------------------------------------------------
    llm: LLMSettings = Field(
        default_factory=LLMSettings,
        description="Per-role LLM API configuration",
    )

    # ---------------------------------------------------------------
    # Pydantic Settings configuration
    # ---------------------------------------------------------------
    model_config = {
        "env_prefix": "CTXMTG_",
        "env_file": str(paths.get_env_file_path()),
        "env_file_encoding": "utf-8",
        "env_nested_delimiter": "__",
    }

    def resolve_db_path(self) -> Path:
        """
        Resolve the database path, expanding ~ to the home directory.

        Returns:
            An absolute Path to the SQLite database file.
        """
        return Path(self.db_path).expanduser()

    def resolve_vector_path(self) -> Path:
        """
        Resolve the vector store path, expanding ~ to home directory.

        Returns:
            An absolute Path to the vector store directory.
        """
        return Path(self.vector_path).expanduser()
