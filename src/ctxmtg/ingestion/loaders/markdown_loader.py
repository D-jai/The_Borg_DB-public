# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Markdown File Loader
====================

Loads .md files as plain text Interactions. Markdown is already
human-readable text -- the NER and SPO extraction pipelines work
directly on the raw content without stripping formatting.

Depends on:
    - ctxmtg.ingestion.loaders.text_loader (reuses load_text_file)

Used by:
    - ctxmtg.ingestion.loaders (registered for .md extension)
"""

from __future__ import annotations

from pathlib import Path

from ctxmtg.models.interaction import Interaction

from .text_loader import load_text_file


def load_markdown_file(file_path: Path) -> Interaction:
    """
    Load a Markdown file and produce an Interaction object.

    Delegates entirely to the text loader -- Markdown is text.
    The source_type is set to OTHER and the title is the filename stem.

    Args:
        file_path: Path to the .md file.

    Returns:
        An Interaction object containing the raw Markdown content.
    """
    interaction = load_text_file(file_path)
    interaction.metadata["source_format"] = "markdown"
    return interaction
