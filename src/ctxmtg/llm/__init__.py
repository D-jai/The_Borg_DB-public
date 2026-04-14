# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
LLM Package
============

This package provides local Large Language Model (LLM) integration
for Tier 2+ deployments (notebooks and above with sufficient RAM).
LLM capabilities enhance extraction quality and enable more
sophisticated query synthesis.

The LLM subsystem is entirely optional -- the system works without
it using rule-based extraction and template-based query planning.
When available, the LLM enhances:
    - Entity and fact extraction accuracy
    - Query answer synthesis (natural language responses)
    - Farming insight generation

The 4-layer prompt architecture:
    Layer 1: Base identity prompt (who the system is)
    Layer 2: Stage-specific prompt (what task to perform)
    Layer 3: Domain overlay (vertical-specific terminology)
    Layer 4: Dynamic context (current query, relevant data)

Submodules:
    - provider.py         : llama.cpp / MLX provider
    - prompt_assembler.py : 4-layer prompt composition
    - model_adapter.py    : Model-specific chat template formatting
"""
