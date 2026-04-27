# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
.env File Read/Write
=====================

Reads and writes the per-role LLM API key .env file (located under
the runtime data root; see ``ctxmtg.paths.get_env_file_path``).
The file uses standard KEY=VALUE format compatible with
pydantic-settings env_file loading.

Environment variable naming convention (nested delimiter __):
    CTXMTG_LLM__EXTRACTION__API_KEY=sk-...
    CTXMTG_LLM__EXTRACTION__BASE_URL=https://api.openai.com/v1
    CTXMTG_LLM__EXTRACTION__MODEL=gpt-4o

Depends on:
    - ctxmtg.paths (resolves the .env file location)

Used by:
    - ctxmtg.web.routes.local (save LLM config from UI)
    - ctxmtg.config.settings (loads .env at startup)
"""

from __future__ import annotations

import structlog

from ctxmtg import paths

logger = structlog.get_logger("ctxmtg.config.env_file")


def _env_path():
    """Resolve the .env file path at call time so CTXMTG_DATA_ROOT
    overrides are honoured even after import."""
    return paths.get_env_file_path()

# The 6 pipeline roles that can have LLM config.
LLM_ROLES = ("extraction", "query_planning", "retrieval", "synthesis", "farming", "fusion")

# Fields per role.
ROLE_FIELDS = ("api_key", "base_url", "model")


def _env_key(role: str, field: str) -> str:
    """Build the env var name for a role field."""
    return f"CTXMTG_LLM__{role.upper()}__{field.upper()}"


def read_env() -> dict[str, str]:
    """Read the .env file into a dict of KEY=VALUE pairs.

    Returns an empty dict if the file does not exist.
    Lines starting with # and blank lines are skipped.
    """
    env_path = _env_path()
    if not env_path.exists():
        return {}

    result: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip surrounding quotes if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value
    return result


def write_env(values: dict[str, str]) -> None:
    """Write KEY=VALUE pairs to the .env file.

    Preserves non-LLM keys and comments. Overwrites LLM keys with
    new values. Creates the file and parent directory if needed.
    """
    env_path = _env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing lines (preserve comments and non-LLM settings).
    existing_lines: list[str] = []
    written_keys: set[str] = set()

    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                existing_lines.append(line)
                continue
            if "=" not in stripped:
                existing_lines.append(line)
                continue
            key, _, _ = stripped.partition("=")
            key = key.strip()
            if key in values:
                # Replace with new value.
                val = values[key]
                existing_lines.append(f"{key}={val}")
                written_keys.add(key)
            else:
                existing_lines.append(line)

    # Append any new keys not in the original file.
    for key, val in values.items():
        if key not in written_keys:
            existing_lines.append(f"{key}={val}")

    env_path.write_text("\n".join(existing_lines) + "\n", encoding="utf-8")
    logger.info("env_file_written", path=str(env_path), key_count=len(values))


def get_role_config(role: str) -> dict[str, str]:
    """Read a single role's LLM config from the .env file."""
    env = read_env()
    return {
        field: env.get(_env_key(role, field), "")
        for field in ROLE_FIELDS
    }


def get_all_role_configs() -> dict[str, dict[str, str]]:
    """Read all role configs from the .env file."""
    return {role: get_role_config(role) for role in LLM_ROLES}


def save_role_config(role: str, api_key: str, base_url: str, model: str) -> None:
    """Save a single role's LLM config to the .env file."""
    env = read_env()
    env[_env_key(role, "api_key")] = api_key
    env[_env_key(role, "base_url")] = base_url
    env[_env_key(role, "model")] = model
    write_env(env)


def save_all_role_configs(configs: dict[str, dict[str, str]]) -> None:
    """Save all role configs to the .env file at once."""
    env = read_env()
    for role, cfg in configs.items():
        for field in ROLE_FIELDS:
            if field in cfg:
                env[_env_key(role, field)] = cfg[field]
    write_env(env)
