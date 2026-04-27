# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Runtime Path Resolution
========================

Single source of truth for every runtime artifact's filesystem
location.  All other modules MUST import their paths from here
rather than computing them locally; this guarantees that two
clones of the source tree (e.g. ``The_Borg_DB-public`` and
``The_Borg_DB-public2``) get cleanly separated runtime stores out
of the box, and that a single environment variable
(``CTXMTG_DATA_ROOT``) can override the location when needed.

Resolution precedence:
    1. ``CTXMTG_DATA_ROOT`` environment variable (expanded, resolved)
    2. ``<project_root>/.runtime/`` (default)

``project_root`` is computed as ``parents[2]`` of this file:
    paths.py -> ctxmtg/ -> src/ -> project_root

Tests that need to flip ``CTXMTG_DATA_ROOT`` mid-process can call
``get_data_root.cache_clear()`` to invalidate the lru_cache.

Depends on: nothing (leaf module -- safe to import from anywhere)

Used by:
    - ctxmtg.config.settings    (Pydantic field default factories)
    - ctxmtg.config.env_file    (.env file location)
    - ctxmtg.constants          (legacy constants now derive from here)
    - ctxmtg.web.auth           (web_auth.json)
    - ctxmtg.web.hive_auth      (hive_web_auth.json)
    - ctxmtg.sync.outbox_writer (outbox directory default)
    - ctxmtg.farming            (archive.db location)
    - ctxmtg.health.monitor     (knowledge.db default)
    - ctxmtg.query.evaluation   (evaluations directory)
    - ctxmtg.cli                (--data-root flag wiring)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_data_root() -> Path:
    """
    Resolve the runtime data root.

    Resolution order:
        1. ``CTXMTG_DATA_ROOT`` environment variable
           (expanded with ~ expansion, fully resolved).
        2. ``<project_root>/.runtime/`` -- where ``project_root``
           is two parents above this file's directory
           (``paths.py`` -> ``ctxmtg/`` -> ``src/`` -> root).

    Returns:
        Absolute ``Path`` to the runtime data directory.  The
        directory is **not** created here; callers create the
        specific subdirectories they need.
    """
    env = os.environ.get("CTXMTG_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / ".runtime").resolve()


# ---------------------------------------------------------------
# Per-artifact path helpers.  Each returns an absolute Path under
# the resolved data root.  Directories are created lazily by the
# specific subsystem that owns the artifact, not here.
# ---------------------------------------------------------------

def get_db_path() -> Path:
    """Path to the SQLite knowledge database file."""
    return get_data_root() / "knowledge.db"


def get_vector_path() -> Path:
    """Directory for the LanceDB vector store."""
    return get_data_root() / "vectors"


def get_inbox_path() -> Path:
    """Directory the inbox watcher polls for new files."""
    return get_data_root() / "inbox"


def get_processed_path() -> Path:
    """Directory the inbox watcher moves successfully ingested files to."""
    return get_data_root() / "processed"


def get_hive_db_path() -> Path:
    """Path to the local-mode hive SQLite database file."""
    return get_data_root() / "hive.db"


def get_hive_vector_path() -> Path:
    """Directory for the local-mode hive vector store."""
    return get_data_root() / "hive_vectors"


def get_outbox_path() -> Path:
    """Directory for hive sync outbox manifests."""
    return get_data_root() / "outbox"


def get_env_file_path() -> Path:
    """Path to the per-role LLM API key .env file."""
    return get_data_root() / ".env"


def get_archive_db_path() -> Path:
    """Path to the cold-storage archive database file."""
    return get_data_root() / "archive.db"


def get_eval_dir() -> Path:
    """Directory where ``ctxmtg query`` writes evaluation snapshots."""
    return get_data_root() / "evaluations"


def get_web_auth_path() -> Path:
    """Path to the web command center's bcrypt hash file."""
    return get_data_root() / "web_auth.json"


def get_hive_web_auth_path() -> Path:
    """Path to the hive web command center's bcrypt hash file."""
    return get_data_root() / "hive_web_auth.json"


def get_profile_dir() -> Path:
    """Directory for user-installed domain profile YAML files."""
    return get_data_root() / "profiles"


def get_config_yaml_path() -> Path:
    """Path to the user's optional config.yaml override file."""
    return get_data_root() / "config.yaml"
