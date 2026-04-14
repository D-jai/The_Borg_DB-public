# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Text File Loader
================

This module loads plain .txt files and converts them into Interaction
objects for the ingestion pipeline. It's the simplest loader -- it reads
the file content as-is, with minimal transformation.

The loader handles encoding detection (falls back to UTF-8) and strips
leading/trailing whitespace. The source_type defaults to OTHER unless
the content contains heuristic markers suggesting a specific type
(e.g., "From:" header suggesting an email).

Depends on:
    - pathlib (file path handling)
    - ctxmtg.models.interaction (Interaction, SourceType)
    - ctxmtg.storage.id_gen (generate_interaction_id)
    - ctxmtg.exceptions (IntakeError)

Used by:
    - ctxmtg.ingestion.loaders (registered for .txt extension)
    - ctxmtg.ingestion.worker (loads files before extraction)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import structlog

from ctxmtg.exceptions import IntakeError
from ctxmtg.models.interaction import Interaction, SourceType
from ctxmtg.storage.id_gen import generate_interaction_id

# ---------------------------------------------------------------
# Module-level logger -- logs file metadata, never file content.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.ingestion.loaders.text_loader")


def load_text_file(file_path: Path) -> Interaction:
    """
    Load a plain text file and produce an Interaction object.

    Reads the file as UTF-8 text, assigns source_type=OTHER by default,
    and generates a deterministic interaction ID from the content.

    Args:
        file_path: Path to the .txt file to load.

    Returns:
        An Interaction object containing the file's text content.

    Raises:
        IntakeError: If the file cannot be read or is empty.
    """
    # Ensure the file exists before attempting to read it
    if not file_path.exists():
        logger.error(
            "text_file_not_found",
            error_code="CTXMTG-ING-001",
            file_path=str(file_path),
        )
        raise IntakeError(
            f"Text file not found: {file_path}",
            error_code="CTXMTG-ING-001",
        )

    if not file_path.is_file():
        logger.error(
            "text_path_not_a_file",
            error_code="CTXMTG-ING-001",
            file_path=str(file_path),
        )
        raise IntakeError(
            f"Path is not a file: {file_path}",
            error_code="CTXMTG-ING-001",
        )

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = file_path.read_text(encoding="latin-1")
        except OSError as exc:
            logger.error(
                "text_read_failed",
                error_code="CTXMTG-ING-001",
                file_path=str(file_path),
                error=str(exc),
            )
            raise IntakeError(
                f"Cannot read text file {file_path}: {exc}",
                error_code="CTXMTG-ING-001",
            ) from exc
    except OSError as exc:
        logger.error(
            "text_read_failed",
            error_code="CTXMTG-ING-001",
            file_path=str(file_path),
            error=str(exc),
        )
        raise IntakeError(
            f"Cannot read text file {file_path}: {exc}",
            error_code="CTXMTG-ING-001",
        ) from exc

    content = content.strip()
    if not content:
        logger.error(
            "text_file_empty",
            error_code="CTXMTG-ING-003",
            file_path=str(file_path),
        )
        raise IntakeError(
            f"Text file is empty: {file_path}",
            error_code="CTXMTG-ING-003",
        )

    # Generate a deterministic interaction ID from the content.
    # Same file content always produces the same ID (idempotent).
    interaction_id = generate_interaction_id("other", content)

    # Use the filename (without extension) as the title
    title = file_path.stem

    logger.info(
        "text_file_loaded",
        file_path=str(file_path),
        content_length=len(content),
    )

    return Interaction(
        id=interaction_id,
        source_type=SourceType.OTHER,
        title=title,
        content=content,
        participants=[],
        metadata={"source_file": str(file_path.name)},
        created_at=datetime.now(timezone.utc),
    )


def load_text_string(text: str, title: str | None = None) -> Interaction:
    """
    Create an Interaction from a raw text string (no file).

    Used when the user passes text directly via CLI instead of a file.

    Args:
        text: The raw text content to ingest.
        title: Optional title for the interaction.

    Returns:
        An Interaction object containing the text.

    Raises:
        IntakeError: If the text is empty.
    """
    # Validate non-empty input
    text = text.strip()
    if not text:
        logger.error(
            "text_string_empty",
            error_code="CTXMTG-ING-003",
        )
        raise IntakeError(
            "Cannot ingest empty text",
            error_code="CTXMTG-ING-003",
        )

    # Generate deterministic ID from content
    interaction_id = generate_interaction_id("other", text)

    return Interaction(
        id=interaction_id,
        source_type=SourceType.OTHER,
        title=title or "Direct text input",
        content=text,
        participants=[],
        metadata={"source": "direct_input"},
        created_at=datetime.now(timezone.utc),
    )
