# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
JSON File Loader
================

This module loads .json files and converts them into Interaction
objects. It supports two formats:
    1. A single dict with Interaction-shaped fields.
    2. An array of dicts, each representing one Interaction.

The loader validates each dict against the Interaction Pydantic model
and fills in defaults for missing fields. This allows users to batch-
import structured data (e.g., exported meeting notes, CRM exports).

Depends on:
    - json (Python stdlib)
    - pathlib (file path handling)
    - ctxmtg.models.interaction (Interaction, SourceType)
    - ctxmtg.storage.id_gen (generate_interaction_id)
    - ctxmtg.exceptions (IntakeError)

Used by:
    - ctxmtg.ingestion.loaders (registered for .json extension)
    - ctxmtg.ingestion.worker (loads files before extraction)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from pydantic import ValidationError

from ctxmtg.exceptions import IntakeError
from ctxmtg.models.interaction import Interaction, SourceType
from ctxmtg.storage.id_gen import generate_interaction_id

# ---------------------------------------------------------------
# Module-level logger -- logs file metadata, never content.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.ingestion.loaders.json_loader")


def load_json_file(file_path: Path) -> list[Interaction]:
    """
    Load a JSON file and produce one or more Interaction objects.

    Supports both a single dict (one interaction) and an array of
    dicts (batch of interactions). Each dict is validated against
    the Interaction model. Missing fields like 'id' and 'created_at'
    are auto-generated.

    Args:
        file_path: Path to the .json file.

    Returns:
        A list of Interaction objects (one element if input is a dict).

    Raises:
        IntakeError: If the file cannot be read, is not valid JSON,
                     or contains invalid interaction data.
    """
    # Validate file existence
    if not file_path.exists():
        logger.error(
            "json_file_not_found",
            error_code="CTXMTG-ING-001",
            file_path=str(file_path),
        )
        raise IntakeError(
            f"JSON file not found: {file_path}",
            error_code="CTXMTG-ING-001",
        )

    if not file_path.is_file():
        logger.error(
            "json_path_not_a_file",
            error_code="CTXMTG-ING-001",
            file_path=str(file_path),
        )
        raise IntakeError(
            f"Path is not a file: {file_path}",
            error_code="CTXMTG-ING-001",
        )

    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error(
            "json_read_failed",
            error_code="CTXMTG-ING-001",
            file_path=str(file_path),
            error=str(exc),
        )
        raise IntakeError(
            f"Cannot read JSON file {file_path}: {exc}",
            error_code="CTXMTG-ING-001",
        ) from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error(
            "json_parse_failed",
            error_code="CTXMTG-ING-002",
            file_path=str(file_path),
            error=str(exc),
        )
        raise IntakeError(
            f"Invalid JSON in {file_path}: {exc}",
            error_code="CTXMTG-ING-002",
        ) from exc

    if isinstance(data, dict):
        interactions = [_dict_to_interaction(data, file_path)]
    elif isinstance(data, list):
        interactions = [
            _dict_to_interaction(item, file_path)
            for item in data
            if isinstance(item, dict)
        ]
        if not interactions:
            logger.error(
                "json_array_empty",
                error_code="CTXMTG-ING-003",
                file_path=str(file_path),
            )
            raise IntakeError(
                f"JSON array in {file_path} contains no valid interaction dicts",
                error_code="CTXMTG-ING-003",
            )
    else:
        logger.error(
            "json_unexpected_type",
            error_code="CTXMTG-ING-003",
            file_path=str(file_path),
            data_type=type(data).__name__,
        )
        raise IntakeError(
            f"JSON file {file_path} must contain a dict or array of dicts, "
            f"got {type(data).__name__}",
            error_code="CTXMTG-ING-003",
        )

    logger.info(
        "json_file_loaded",
        file_path=str(file_path),
        interaction_count=len(interactions),
    )

    return interactions


def _dict_to_interaction(data: dict[str, Any], source_path: Path) -> Interaction:
    """
    Convert a dict to an Interaction, filling in missing fields.

    Required field: 'content' (the text to process). All other fields
    have sensible defaults:
        - id: auto-generated from content hash
        - source_type: defaults to "other"
        - created_at: defaults to current time
        - participants: defaults to empty list
        - metadata: defaults to empty dict

    Args:
        data: The dict to convert. Must have at least a 'content' key.
        source_path: The source file path (for error messages).

    Returns:
        A validated Interaction object.

    Raises:
        IntakeError: If required fields are missing or validation fails.
    """
    # Ensure the dict has content (the only truly required field)
    if "content" not in data or not data["content"]:
        logger.error(
            "json_missing_content",
            error_code="CTXMTG-ING-003",
            source_path=str(source_path),
        )
        raise IntakeError(
            f"JSON interaction dict in {source_path} missing required 'content' field",
            error_code="CTXMTG-ING-003",
        )

    content = str(data["content"]).strip()
    if not content:
        logger.error(
            "json_empty_content",
            error_code="CTXMTG-ING-003",
            source_path=str(source_path),
        )
        raise IntakeError(
            f"Empty content in JSON interaction from {source_path}",
            error_code="CTXMTG-ING-003",
        )

    # Ensure source_type has a default value
    if "source_type" not in data:
        data["source_type"] = "other"
    source_type_str = data.get("source_type", "other")

    # Validate source_type early with a helpful error message.
    try:
        SourceType(source_type_str)
    except ValueError:
        valid = ", ".join(st.value for st in SourceType)
        raise IntakeError(
            f"Unknown source_type '{source_type_str}' in {source_path}. "
            f"Valid values: {valid}",
            error_code="CTXMTG-ING-003",
        )

    # Auto-generate ID if not provided
    if "id" not in data:
        data["id"] = generate_interaction_id(source_type_str, content)

    # Ensure created_at is set
    if "created_at" not in data:
        data["created_at"] = datetime.now(timezone.utc)

    # Add source file metadata
    if "metadata" not in data:
        data["metadata"] = {}
    data["metadata"]["source_file"] = str(source_path.name)

    # Validate via Pydantic model
    try:
        interaction = Interaction.model_validate(data)
    except ValidationError as exc:
        logger.error(
            "json_validation_failed",
            error_code="CTXMTG-ING-003",
            source_path=str(source_path),
            error=str(exc),
        )
        raise IntakeError(
            f"Invalid interaction data in {source_path}: {exc}",
            error_code="CTXMTG-ING-003",
        ) from exc

    return interaction
