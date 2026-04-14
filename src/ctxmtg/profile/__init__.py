# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Profile Package
===============

This package manages domain profiles -- YAML configuration files
that customize the system's behavior for specific professional
verticals (legal, medical, engineering, personal, etc.).

A domain profile controls:
    - Which entity types to extract (e.g., legal: STATUTE, CASE_NUMBER)
    - Extraction parameters (temperature, structured output, etc.)
    - Prompt overlays for LLM interactions
    - Query planning strategies
    - Farming focus areas

Profiles are the key mechanism for making ctxmtg useful across
many different domains without hardcoding domain-specific logic.
See research/round-2/refinement-01.md for the profile design.

Submodules:
    - loader.py    : YAML profile loader + validator
    - assembler.py : Runtime prompt assembly from profile
    - switcher.py  : Profile switching logic (time-based, source-based)
"""
