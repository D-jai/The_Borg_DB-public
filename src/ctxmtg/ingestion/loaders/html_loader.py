# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
HTML File Loader
================

Loads .html/.htm files by stripping HTML tags and extracting the
visible text content. Uses Python's stdlib html.parser -- no
external dependencies.

The <title> tag content is used as the Interaction title. Script
and style blocks are discarded entirely.

Depends on:
    - html.parser (Python stdlib)
    - ctxmtg.models.interaction (Interaction, SourceType)
    - ctxmtg.storage.id_gen (generate_interaction_id)
    - ctxmtg.exceptions (IntakeError)

Used by:
    - ctxmtg.ingestion.loaders (registered for .html and .htm)
"""

from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import structlog

from ctxmtg.exceptions import IntakeError
from ctxmtg.models.interaction import Interaction, SourceType
from ctxmtg.storage.id_gen import generate_interaction_id

logger = structlog.get_logger("ctxmtg.ingestion.loaders.html_loader")


class _TextExtractor(HTMLParser):
    """Strips HTML tags and collects visible text."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._title_parts: list[str] = []
        self._in_title = False
        self._skip = False  # True inside <script> or <style>

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        elif tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self._title_parts.append(data)
        self._parts.append(data)

    @property
    def text(self) -> str:
        raw = " ".join(self._parts)
        # Collapse whitespace runs.
        return " ".join(raw.split())

    @property
    def title(self) -> str:
        return " ".join(self._title_parts).strip()


def load_html_file(file_path: Path) -> Interaction:
    """
    Load an HTML file, strip tags, and produce an Interaction.

    Args:
        file_path: Path to the .html or .htm file.

    Returns:
        An Interaction containing the extracted text.

    Raises:
        IntakeError: If the file cannot be read or has no text content.
    """
    if not file_path.exists():
        raise IntakeError(f"HTML file not found: {file_path}", error_code="CTXMTG-ING-001")
    if not file_path.is_file():
        raise IntakeError(f"Path is not a file: {file_path}", error_code="CTXMTG-ING-001")

    try:
        raw = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = file_path.read_text(encoding="latin-1")
    except OSError as exc:
        raise IntakeError(
            f"Cannot read HTML file {file_path}: {exc}", error_code="CTXMTG-ING-001"
        ) from exc

    if not raw.strip():
        raise IntakeError(f"HTML file is empty: {file_path}", error_code="CTXMTG-ING-003")

    extractor = _TextExtractor()
    extractor.feed(raw)
    content = extractor.text

    if not content.strip():
        raise IntakeError(
            f"HTML file has no extractable text: {file_path}",
            error_code="CTXMTG-ING-003",
        )

    title = extractor.title or file_path.stem
    interaction_id = generate_interaction_id("other", content)

    logger.info("html_file_loaded", file_path=str(file_path), content_length=len(content))

    return Interaction(
        id=interaction_id,
        source_type=SourceType.OTHER,
        title=title,
        content=content,
        participants=[],
        metadata={"source_file": file_path.name, "source_format": "html"},
        created_at=datetime.now(timezone.utc),
    )
