# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Intake Package
==============

This package handles input handling and format normalization for
content entering the ctxmtg system. It is the first stage of the
ingestion pipeline, responsible for:

    1. Accepting content from various sources (files, clipboard,
       API calls, CLI input)
    2. Detecting and validating input formats
    3. Normalizing content encoding (UTF-8 normalization)
    4. Applying initial content validation (non-empty, within
       size limits, supported format)

The intake package sits between external input and the ingestion
worker. It ensures that only clean, validated content reaches the
extraction pipeline.

Depends on:
    - ctxmtg.exceptions (IntakeError for validation failures)
    - ctxmtg.constants (size limits, supported formats)

Used by:
    - ctxmtg.ingestion.worker (receives normalized content)
    - ctxmtg.cli (CLI ingest command feeds through intake)
"""
