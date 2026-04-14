# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Configuration Package
=====================

This package manages all system configuration for ctxmtg. It
provides a single source of truth for settings, with values loaded
from multiple sources in priority order:

    1. Environment variables (highest priority, e.g., CTXMTG_DB_PATH)
    2. Config YAML file (~/.ctxmtg/config.yaml)
    3. Default values defined in code (lowest priority)

Configuration is built on Pydantic BaseSettings, which gives us
automatic type validation, environment variable binding, and
clear documentation of all available settings.

Submodules:
    - settings.py : Pydantic BaseSettings class with all config options
"""
