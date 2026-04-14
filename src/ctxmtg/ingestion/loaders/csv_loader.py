# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
CSV File Loader
===============

Loads .csv files into Interaction objects. Each row is formatted as
a line of text so the extraction pipeline can process it.

Strategy: read all rows, format each as "col1: val1 | col2: val2 | ...",
and join into a single content block. This preserves column semantics
while producing text the NER pipeline can parse.

Depends on:
    - csv (Python stdlib)
    - ctxmtg.models.interaction (Interaction, SourceType)
    - ctxmtg.storage.id_gen (generate_interaction_id)
    - ctxmtg.exceptions (IntakeError)

Used by:
    - ctxmtg.ingestion.loaders (registered for .csv extension)
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path

import structlog

from ctxmtg.exceptions import IntakeError
from ctxmtg.models.interaction import Interaction, SourceType
from ctxmtg.storage.id_gen import generate_interaction_id

logger = structlog.get_logger("ctxmtg.ingestion.loaders.csv_loader")


def load_csv_file(file_path: Path) -> Interaction:
    """
    Load a CSV file and produce an Interaction object.

    Reads the CSV with header row auto-detection. Each data row is
    formatted as "header1: value1 | header2: value2 | ..." and all
    rows are joined with newlines into a single content block.

    Args:
        file_path: Path to the .csv file.

    Returns:
        An Interaction containing the formatted CSV content.

    Raises:
        IntakeError: If the file cannot be read, is empty, or has no data rows.
    """
    if not file_path.exists():
        raise IntakeError(f"CSV file not found: {file_path}", error_code="CTXMTG-ING-001")
    if not file_path.is_file():
        raise IntakeError(f"Path is not a file: {file_path}", error_code="CTXMTG-ING-001")

    try:
        raw = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = file_path.read_text(encoding="latin-1")
    except OSError as exc:
        raise IntakeError(
            f"Cannot read CSV file {file_path}: {exc}", error_code="CTXMTG-ING-001"
        ) from exc

    raw = raw.strip()
    if not raw:
        raise IntakeError(f"CSV file is empty: {file_path}", error_code="CTXMTG-ING-003")

    reader = csv.reader(io.StringIO(raw))
    rows = list(reader)

    if len(rows) < 2:
        raise IntakeError(
            f"CSV file has no data rows (header only or empty): {file_path}",
            error_code="CTXMTG-ING-003",
        )

    headers = rows[0]
    lines = []
    for row in rows[1:]:
        parts = []
        for i, val in enumerate(row):
            col = headers[i] if i < len(headers) else f"col{i}"
            if val.strip():
                parts.append(f"{col}: {val.strip()}")
        if parts:
            lines.append(" | ".join(parts))

    if not lines:
        raise IntakeError(
            f"CSV file has no non-empty data rows: {file_path}",
            error_code="CTXMTG-ING-003",
        )

    content = "\n".join(lines)
    interaction_id = generate_interaction_id("other", content)

    logger.info("csv_file_loaded", file_path=str(file_path), rows=len(lines))

    return Interaction(
        id=interaction_id,
        source_type=SourceType.OTHER,
        title=file_path.stem,
        content=content,
        participants=[],
        metadata={"source_file": file_path.name, "source_format": "csv", "row_count": len(lines)},
        created_at=datetime.now(timezone.utc),
    )
