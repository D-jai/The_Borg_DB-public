# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
File Loaders Subpackage
=======================

This subpackage provides file format loaders for the ingestion
pipeline. Each loader handles a specific file format (.txt, .json,
.eml, .ics, .vcf) and converts it into Interaction objects that the
extraction pipeline can process.

Loaders are registered by file extension in the FileLoaderRegistry
and selected automatically based on the input file type. New formats
can be supported by adding a loader module and registering it.

The registry maps file extensions (e.g., ".txt") to loader functions.
Each loader function takes a Path and returns one or more Interaction
objects (or loader-specific result types for calendar/contact which
include pre-built entities and facts).

Depends on:
    - ctxmtg.ingestion.loaders.text_loader (plain text files)
    - ctxmtg.ingestion.loaders.json_loader (JSON files)
    - ctxmtg.ingestion.loaders.eml_loader (email files)
    - ctxmtg.ingestion.loaders.calendar_loader (ICS calendar files)
    - ctxmtg.ingestion.loaders.contact_loader (VCF contact files)

Used by:
    - ctxmtg.ingestion.worker (dispatches file loading by extension)
    - ctxmtg.cli (auto-detects format from file extension)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

from ctxmtg.exceptions import IntakeError

# ---------------------------------------------------------------
# Module-level logger for the registry.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.ingestion.loaders")


class FileLoaderRegistry:
    """
    Maps file extensions to loader functions.

    The registry is the central dispatcher for file ingestion.
    When a file comes in, the CLI or worker looks up the extension
    in the registry to find the right loader.

    Extensions are stored with the leading dot (e.g., ".txt", ".json").
    Lookup is case-insensitive (".TXT" matches ".txt").

    Usage:
        registry = FileLoaderRegistry()
        loader = registry.get_loader(".txt")
        result = loader(Path("meeting_notes.txt"))

        # Check if a format is supported:
        if registry.can_load(".pdf"):
            ...

        # List all supported extensions:
        extensions = registry.supported_extensions()
    """

    def __init__(self) -> None:
        """Initialize the registry with all built-in loaders."""
        # Map of lowercase extension → loader callable
        self._loaders: dict[str, Callable[..., Any]] = {}

        # Register all built-in loaders
        self._register_builtin_loaders()

    def _register_builtin_loaders(self) -> None:
        """Register all built-in file format loaders."""
        # Import loaders here to avoid circular imports
        from ctxmtg.ingestion.loaders.calendar_loader import load_ics_file
        from ctxmtg.ingestion.loaders.contact_loader import load_vcf_file
        from ctxmtg.ingestion.loaders.csv_loader import load_csv_file
        from ctxmtg.ingestion.loaders.docx_loader import load_docx_file
        from ctxmtg.ingestion.loaders.eml_loader import load_eml_file
        from ctxmtg.ingestion.loaders.html_loader import load_html_file
        from ctxmtg.ingestion.loaders.json_loader import load_json_file
        from ctxmtg.ingestion.loaders.markdown_loader import load_markdown_file
        from ctxmtg.ingestion.loaders.pdf_loader import load_pdf_file
        from ctxmtg.ingestion.loaders.text_loader import load_text_file

        # Plain text files
        self._loaders[".txt"] = load_text_file

        # Markdown files (treated as text)
        self._loaders[".md"] = load_markdown_file

        # CSV files (rows formatted as key-value text)
        self._loaders[".csv"] = load_csv_file

        # HTML files (tags stripped, text extracted)
        self._loaders[".html"] = load_html_file
        self._loaders[".htm"] = load_html_file

        # Word documents (paragraph and table text)
        self._loaders[".docx"] = load_docx_file

        # PDF documents (text-based, no OCR)
        self._loaders[".pdf"] = load_pdf_file

        # JSON files (single or batch interactions)
        self._loaders[".json"] = load_json_file

        # Email files (MIME messages)
        self._loaders[".eml"] = load_eml_file

        # Calendar files (iCalendar/ICS events)
        self._loaders[".ics"] = load_ics_file

        # Contact files (vCard/VCF contacts)
        self._loaders[".vcf"] = load_vcf_file

        logger.debug(
            "loaders_registered",
            extensions=list(self._loaders.keys()),
        )

    def register(self, extension: str, loader: Callable[..., Any]) -> None:
        """
        Register a custom loader for a file extension.

        Allows users or plugins to add support for new file formats
        without modifying the built-in loader modules.

        Args:
            extension: The file extension including the dot (e.g., ".md").
            loader: A callable that takes a Path and returns Interaction(s).
        """
        # Normalize the extension to lowercase with leading dot
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"

        self._loaders[ext] = loader
        logger.info("loader_registered", extension=ext)

    def get_loader(self, extension: str) -> Callable[..., Any]:
        """
        Get the loader function for a file extension.

        Args:
            extension: The file extension (e.g., ".txt", ".json").

        Returns:
            The loader callable for this extension.

        Raises:
            IntakeError: If no loader is registered for this extension.
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"

        loader = self._loaders.get(ext)
        if loader is None:
            supported = ", ".join(sorted(self._loaders.keys()))
            logger.error(
                "no_loader_for_extension",
                error_code="CTXMTG-ING-005",
                extension=extension,
                supported=supported,
            )
            raise IntakeError(
                f"No loader registered for extension '{extension}'. "
                f"Supported formats: {supported}",
                error_code="CTXMTG-ING-005",
            )

        return loader

    def can_load(self, extension: str) -> bool:
        """
        Check if a file extension has a registered loader.

        Args:
            extension: The file extension (e.g., ".txt").

        Returns:
            True if a loader exists for this extension.
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        return ext in self._loaders

    def supported_extensions(self) -> list[str]:
        """
        List all file extensions that have registered loaders.

        Returns:
            A sorted list of supported extensions (e.g., [".eml", ".json", ".txt"]).
        """
        return sorted(self._loaders.keys())
