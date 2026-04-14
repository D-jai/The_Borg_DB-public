# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
PDF File Loader
===============

Loads .pdf files using pdfplumber. Extracts text from each page and
joins into a single content block. Tables are extracted as structured
text when detected.

Limitations:
    - Text-based PDFs only. Scanned/image PDFs produce empty content.
    - No OCR. Add Tesseract integration in a future phase if needed.

Depends on:
    - pdfplumber (PDF text extraction)
    - ctxmtg.models.interaction (Interaction, SourceType)
    - ctxmtg.storage.id_gen (generate_interaction_id)
    - ctxmtg.exceptions (IntakeError)

Used by:
    - ctxmtg.ingestion.loaders (registered for .pdf extension)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import structlog

from ctxmtg.exceptions import IntakeError
from ctxmtg.models.interaction import Interaction, SourceType
from ctxmtg.storage.id_gen import generate_interaction_id

logger = structlog.get_logger("ctxmtg.ingestion.loaders.pdf_loader")


def load_pdf_file(file_path: Path) -> Interaction:
    """
    Load a PDF file and produce an Interaction object.

    Iterates each page, extracts text content, and joins with
    double newlines. Tables detected by pdfplumber are formatted
    as pipe-separated rows.

    Args:
        file_path: Path to the .pdf file.

    Returns:
        An Interaction containing the document text.

    Raises:
        IntakeError: If pdfplumber is not installed, the file cannot
            be read, or the PDF has no extractable text.
    """
    if not file_path.exists():
        raise IntakeError(f"PDF file not found: {file_path}", error_code="CTXMTG-ING-001")
    if not file_path.is_file():
        raise IntakeError(f"Path is not a file: {file_path}", error_code="CTXMTG-ING-001")

    try:
        import pdfplumber
    except ImportError as exc:
        raise IntakeError(
            "pdfplumber is required for .pdf files: pip install pdfplumber",
            error_code="CTXMTG-ING-006",
        ) from exc

    try:
        pdf = pdfplumber.open(str(file_path))
    except Exception as exc:
        raise IntakeError(
            f"Cannot open PDF file {file_path}: {exc}", error_code="CTXMTG-ING-001"
        ) from exc

    page_texts: list[str] = []
    total_pages = len(pdf.pages)

    for page in pdf.pages:
        text = page.extract_text()
        if text and text.strip():
            page_texts.append(text.strip())

        # Extract tables as structured text.
        for table in page.extract_tables():
            rows = []
            for row in table:
                cells = [str(c).strip() if c else "" for c in row]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                page_texts.append("\n".join(rows))

    pdf.close()

    content = "\n\n".join(page_texts)
    if not content.strip():
        raise IntakeError(
            f"PDF has no extractable text (scanned/image PDF?): {file_path}",
            error_code="CTXMTG-ING-003",
        )

    title = file_path.stem
    interaction_id = generate_interaction_id("doc", content)

    logger.info(
        "pdf_file_loaded",
        file_path=str(file_path),
        pages=total_pages,
        content_length=len(content),
    )

    return Interaction(
        id=interaction_id,
        source_type=SourceType.DOC,
        title=title,
        content=content,
        participants=[],
        metadata={
            "source_file": file_path.name,
            "source_format": "pdf",
            "page_count": total_pages,
        },
        created_at=datetime.now(timezone.utc),
    )
