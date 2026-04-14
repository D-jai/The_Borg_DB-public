# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Outbox Reader (Hive Side)
==========================

Reads JSON manifest files from a local instance's outbox directory
and ingests them into the hive database.  Used by the hive UI when
the user clicks "Pull" on a configured link.

The reader:
    1. Lists all .json files in the outbox (sorted = chronological).
    2. Validates each manifest's version and structure.
    3. Calls HiveDatabase.push_intelligence() with the payload.
    4. On success, moves the file to outbox/processed/.
    5. On failure, logs the error and leaves the file in place.

This mirrors the InboxWatcher pattern used for local ingestion:
poll a directory, process files, move to processed/.

Depends on:
    - structlog (structured logging)
    - ctxmtg.sync.hive_db (HiveDatabase for push_intelligence)

Used by:
    - ctxmtg.web.routes.hive_dashboard (Pull button action)
    - ctxmtg.cli (future: ctxmtg hive pull command)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import structlog

from ctxmtg.sync.hive_db import HiveDatabase

logger = structlog.get_logger("ctxmtg.sync.outbox_reader")

# Supported manifest schema versions
# v1: original (no local_metadata)
# v2: added local_metadata snapshot
SUPPORTED_VERSIONS = {1, 2}


class OutboxReader:
    """
    Reads and ingests manifest files from a local's outbox into the hive.

    Each manifest is a self-describing JSON file written by an
    OutboxWriter on the local side.  The payload matches what
    HiveDatabase.push_intelligence() expects, so no transformation
    is needed.

    Usage:
        reader = OutboxReader(hive_db)
        result = await reader.pull_from_link(link_dict)
    """

    def __init__(self, hive_db: HiveDatabase) -> None:
        self._hive_db = hive_db

    async def pull_from_link(self, link: dict[str, Any]) -> dict[str, Any]:
        """
        Pull and ingest all pending manifests from a link's outbox.

        Args:
            link: Dict with at least 'link_id', 'local_name',
                  and 'outbox_path' keys (from hive_links table).

        Returns:
            Summary dict:
            {
                "batches": int,      -- manifests processed
                "profiles": int,     -- entity profiles upserted
                "insights": int,     -- insights inserted
                "errors": list[str], -- error messages for failed manifests
            }
        """
        outbox_path = Path(link["outbox_path"])
        local_name = link["local_name"]
        link_id = link["link_id"]

        result: dict[str, Any] = {
            "batches": 0,
            "profiles": 0,
            "insights": 0,
            "errors": [],
        }

        # Check that the outbox directory exists
        if not outbox_path.exists():
            msg = f"Outbox path does not exist: {outbox_path}"
            logger.warning("outbox_path_missing", path=str(outbox_path))
            result["errors"].append(msg)
            return result

        if not outbox_path.is_dir():
            msg = f"Outbox path is not a directory: {outbox_path}"
            logger.warning("outbox_path_not_dir", path=str(outbox_path))
            result["errors"].append(msg)
            return result

        # List pending manifest files, sorted lexicographically.
        # Filenames start with ISO timestamps (YYYYMMDDTHHMMSSZ_...),
        # so sorted() gives FIFO order: oldest first, newest last.
        # This ensures the hive processes batches in the order they
        # were created, preserving temporal consistency.
        manifests = sorted(
            p for p in outbox_path.glob("*.json")
            if p.is_file()
        )

        if not manifests:
            logger.info(
                "outbox_pull_noop",
                link=local_name,
                reason="no_pending_manifests",
            )
            return result

        logger.info(
            "outbox_pull_start",
            link=local_name,
            pending_manifests=len(manifests),
        )

        # Ensure processed/ subdirectory exists
        processed_dir = outbox_path / "processed"
        processed_dir.mkdir(exist_ok=True)

        # Process each manifest
        for manifest_path in manifests:
            try:
                counts = await self._process_manifest(
                    manifest_path, local_name, processed_dir
                )
                result["batches"] += 1
                result["profiles"] += counts.get("profiles", 0)
                result["insights"] += counts.get("insights", 0)
            except FileNotFoundError:
                # Another process already moved this file (concurrent pull)
                logger.debug(
                    "outbox_manifest_already_processed",
                    path=str(manifest_path),
                )
            except Exception as exc:
                msg = f"{manifest_path.name}: {exc}"
                logger.error(
                    "outbox_manifest_failed",
                    path=str(manifest_path),
                    error=str(exc),
                )
                result["errors"].append(msg)

        # Update link stats in hive DB
        if result["batches"] > 0:
            await self._hive_db.update_link_pulled(
                link_id, result["batches"]
            )

        logger.info(
            "outbox_pull_complete",
            link=local_name,
            batches=result["batches"],
            profiles=result["profiles"],
            insights=result["insights"],
            errors=len(result["errors"]),
        )

        return result

    async def _process_manifest(
        self,
        manifest_path: Path,
        local_name: str,
        processed_dir: Path,
    ) -> dict[str, int]:
        """
        Read, validate, ingest one manifest, and move to processed/.

        Args:
            manifest_path: Path to the JSON manifest file.
            local_name:    Name of the local (from the link config).
            processed_dir: Directory to move processed files into.

        Returns:
            Dict with counts: {"profiles": n, "insights": n}

        Raises:
            FileNotFoundError: If the file was already moved.
            ValueError: If the manifest is invalid.
            Exception: If push_intelligence fails.
        """
        # Read and parse the manifest
        raw = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(raw)

        # Validate version
        version = manifest.get("manifest_version")
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported manifest version: {version} "
                f"(supported: {SUPPORTED_VERSIONS})"
            )

        # Extract payload
        payload = manifest.get("payload", {})
        summaries = payload.get("distiller_summaries", [])
        insights = payload.get("meta_insights", [])

        if not summaries and not insights:
            logger.debug(
                "outbox_manifest_empty",
                path=str(manifest_path),
            )
            # Still move to processed (it's been handled)
            self._move_to_processed(manifest_path, processed_dir)
            return {"profiles": 0, "insights": 0}

        # Push to hive via existing push_intelligence()
        # Use local_name from the link (user-chosen), not from the
        # manifest's instance_name (which is the local's self-ID).
        counts = await self._hive_db.push_intelligence(
            distiller_summaries=summaries,
            meta_insights=insights,
            local_name=local_name,
        )

        # Move to processed/
        self._move_to_processed(manifest_path, processed_dir)

        logger.debug(
            "outbox_manifest_processed",
            path=str(manifest_path),
            profiles=counts.get("profiles", 0),
            insights=counts.get("insights", 0),
        )

        return counts

    @staticmethod
    def _move_to_processed(
        manifest_path: Path,
        processed_dir: Path,
    ) -> None:
        """
        Move a manifest file to the processed/ subdirectory.

        Handles name collisions by appending _N suffix (same pattern
        as the local InboxWatcher).

        Args:
            manifest_path: Source file to move.
            processed_dir: Target directory.
        """
        dest = processed_dir / manifest_path.name

        # Handle collision (shouldn't happen with random suffix, but safe)
        if dest.exists():
            stem = manifest_path.stem
            suffix = manifest_path.suffix
            counter = 1
            while dest.exists():
                dest = processed_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        shutil.move(str(manifest_path), str(dest))
