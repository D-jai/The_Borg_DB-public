# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Outbox Writer (Local Side)
===========================

Writes JSON manifest files to the local outbox directory.  Each
manifest contains distilled intelligence (distiller summaries and
meta insights) ready for a hive to pull.

The local instance writes to its own outbox -- it has no knowledge
of where (or whether) a hive exists.  The hive discovers locals
via user-configured "links" that point to each local's outbox path.

Manifest format:
    - manifest_version: integer schema version for forward compat
    - instance_name: human-readable name of this local
    - created_at: ISO-8601 timestamp of manifest creation
    - batch_id: unique sortable identifier for this batch
    - high_water_marks: what was already sent (for debugging gaps)
    - payload: distiller_summaries + meta_insights dicts

File naming: {ISO_timestamp}_{instance_name}_{4hex}.json
    Lexicographically sortable by time; random suffix prevents
    collisions if two pushes happen in the same second.

Atomic writes: write to .tmp file, then os.rename() to final path.
    This prevents the hive from reading a half-written manifest.

Depends on:
    - structlog (structured logging)
    - pathlib (path operations)

Used by:
    - ctxmtg.sync.hive_sync (HiveSyncWorker calls write_manifest)
    - ctxmtg.cli (ctxmtg hive push command)
    - ctxmtg.web.routes.dashboard (Push to Hive button)
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger("ctxmtg.sync.outbox_writer")


class OutboxWriter:
    """
    Writes intelligence manifests to the local outbox directory.

    Each call to write_manifest() creates a self-describing JSON file
    that a hive can later pull and ingest.  The payload matches the
    format expected by HiveDatabase.push_intelligence() exactly,
    so the hive needs zero transformation.

    Usage:
        writer = OutboxWriter(
            outbox_path=Path("~/.ctxmtg/outbox").expanduser(),
            instance_name="tickets",
        )
        manifest_path = writer.write_manifest(
            summaries=[...],
            insights=[...],
            high_water_marks={...},
        )
    """

    def __init__(
        self,
        outbox_path: Path,
        instance_name: str,
    ) -> None:
        """
        Configure the outbox writer.

        Args:
            outbox_path:   Directory to write manifest files into.
                           Created automatically if it doesn't exist.
            instance_name: Human-readable name of this local instance.
                           Embedded in manifests and filenames.
        """
        self._outbox_path = outbox_path
        self._instance_name = instance_name

    @property
    def outbox_path(self) -> Path:
        """Return the outbox directory path."""
        return self._outbox_path

    @property
    def instance_name(self) -> str:
        """Return this instance's name."""
        return self._instance_name

    def write_manifest(
        self,
        summaries: list[dict[str, Any]],
        insights: list[dict[str, Any]],
        high_water_marks: dict[str, Any],
        local_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write a JSON manifest to the outbox directory.

        Creates the outbox directory if it doesn't exist.  Uses
        atomic write (write .tmp then rename) to prevent the hive
        from reading a partial file.

        Args:
            summaries:        Distiller summary dicts from the local DB.
            insights:         Meta insight dicts from the local DB.
            high_water_marks: Dict recording the starting point of this
                              batch (for debugging / gap detection).
            local_metadata:   Snapshot of the local instance's state at
                              push time.  Includes schema version, table
                              counts, profile, platform info, etc.

        Returns:
            Path to the written manifest file.
        """
        # Ensure outbox directory exists
        self._outbox_path.mkdir(parents=True, exist_ok=True)

        # Generate sortable, collision-safe filename
        now = datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%dT%H%M%SZ")
        hex_suffix = secrets.token_hex(2)  # 4 hex chars
        batch_id = f"{timestamp_str}_{self._instance_name}_{hex_suffix}"
        filename = f"{batch_id}.json"

        # Build the manifest
        manifest: dict[str, Any] = {
            "manifest_version": 2,
            "instance_name": self._instance_name,
            "created_at": now.isoformat(),
            "batch_id": batch_id,
            "local_metadata": local_metadata or {},
            "high_water_marks": high_water_marks,
            "payload": {
                "distiller_summaries": summaries,
                "meta_insights": insights,
            },
        }

        # Atomic write: .tmp → rename
        final_path = self._outbox_path / filename
        tmp_path = self._outbox_path / f".{filename}.tmp"

        try:
            tmp_path.write_text(
                json.dumps(manifest, indent=2, default=str),
                encoding="utf-8",
            )
            os.rename(tmp_path, final_path)
        except Exception:
            # Clean up temp file on failure
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        logger.info(
            "outbox_manifest_written",
            path=str(final_path),
            batch_id=batch_id,
            summaries=len(summaries),
            insights=len(insights),
        )

        return final_path

    def list_pending(self) -> list[Path]:
        """
        List unprocessed manifest files in the outbox (sorted by name).

        Returns:
            Sorted list of .json file paths in the outbox directory.
            Excludes the processed/ subdirectory.
        """
        if not self._outbox_path.exists():
            return []

        manifests = sorted(
            p for p in self._outbox_path.glob("*.json")
            if p.is_file()
        )
        return manifests
