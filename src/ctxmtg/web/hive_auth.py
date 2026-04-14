# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Hive Authentication Module
============================

Thin wrapper around the standard auth module but using a separate
auth file (hive_web_auth.json) so the hive UI has its own password
independent of any local instance.

Same security model: bcrypt hash, signed session cookie, 24h expiry.
"""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

import bcrypt
import structlog

logger = structlog.get_logger("ctxmtg.web.hive_auth")

SESSION_MAX_AGE = 86400

_sessions: dict[str, float] = {}


def _auth_file() -> Path:
    """Path to the hive bcrypt hash file (separate from local)."""
    return Path("~/.ctxmtg/hive_web_auth.json").expanduser()


def is_password_set() -> bool:
    path = _auth_file()
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        return bool(data.get("password_hash"))
    except (json.JSONDecodeError, OSError):
        return False


def set_password(password: str) -> None:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
    path = _auth_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"password_hash": hashed.decode("utf-8")}))
    logger.info("hive_admin_password_set", path=str(path))


def verify_password(password: str) -> bool:
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
    token = secrets.token_hex(32)
    _sessions[token] = time.time() + SESSION_MAX_AGE
    return token


def validate_session(token: str | None) -> bool:
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
    if token:
        _sessions.pop(token, None)
