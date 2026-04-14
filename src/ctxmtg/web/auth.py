# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Authentication Module
=====================

Provides bcrypt-based password hashing and session cookie management
for the web command center. A single admin password is set on first
run and stored as a bcrypt hash in ~/.ctxmtg/web_auth.json.

Security model:
    - localhost only (no network exposure)
    - bcrypt hash with 12 rounds
    - signed session cookie (secrets.token_hex)
    - session expires after 24 hours of inactivity

Depends on:
    - bcrypt (password hashing)
    - json, pathlib (hash file storage)
    - secrets (session token generation)

Used by:
    - ctxmtg.web.app (middleware and login routes)
    - ctxmtg.web.deps (session validation dependency)
"""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

import bcrypt
import structlog

logger = structlog.get_logger("ctxmtg.web.auth")

# Session expiry: 24 hours in seconds.
SESSION_MAX_AGE = 86400

# In-memory session store: token -> expiry timestamp.
_sessions: dict[str, float] = {}


def _auth_file() -> Path:
    """Path to the bcrypt hash file."""
    return Path("~/.ctxmtg/web_auth.json").expanduser()


def is_password_set() -> bool:
    """Check whether an admin password has been configured."""
    path = _auth_file()
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        return bool(data.get("password_hash"))
    except (json.JSONDecodeError, OSError):
        return False


def set_password(password: str) -> None:
    """
    Hash and persist the admin password.

    Args:
        password: Plaintext password (minimum 8 characters).

    Raises:
        ValueError: If password is shorter than 8 characters.
    """
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
    path = _auth_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"password_hash": hashed.decode("utf-8")}))
    logger.info("admin_password_set", path=str(path))


def verify_password(password: str) -> bool:
    """
    Verify a plaintext password against the stored bcrypt hash.

    Returns False if no password is set or if the password is wrong.
    """
    path = _auth_file()
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        stored_hash = data.get("password_hash", "")
        return bcrypt.checkpw(
            password.encode("utf-8"), stored_hash.encode("utf-8")
        )
    except (json.JSONDecodeError, OSError, ValueError):
        return False


def create_session() -> str:
    """Create a new session token and store it with an expiry time."""
    token = secrets.token_hex(32)
    _sessions[token] = time.time() + SESSION_MAX_AGE
    return token


def validate_session(token: str | None) -> bool:
    """Check whether a session token is valid and not expired."""
    if not token:
        return False
    expiry = _sessions.get(token)
    if expiry is None:
        return False
    if time.time() > expiry:
        _sessions.pop(token, None)
        return False
    return True


def destroy_session(token: str | None) -> None:
    """Remove a session token."""
    if token:
        _sessions.pop(token, None)
