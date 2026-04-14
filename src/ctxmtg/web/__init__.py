# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Web Command Center Package
==========================

This package provides the FastAPI-based web interface for ctxmtg.
It serves a dashboard (Jinja2 + htmx) on localhost for managing
the knowledge system: viewing locals, hive status, triggering
push/farm operations, editing profiles, and exposing an
OpenAI-compatible chat endpoint for Open WebUI integration.

Security: binds to 127.0.0.1 only. Session-based bcrypt auth
with a single admin password stored as a hash.

Submodules:
    - app.py     : FastAPI application factory
    - auth.py    : bcrypt password hashing and session auth
    - deps.py    : FastAPI dependency injection (stores, settings)
    - routes/    : Route modules (dashboard, local, chat completions)
    - templates/ : Jinja2 HTML templates with htmx

Used by:
    - ctxmtg.cli (the `ctxmtg serve` command)
"""
