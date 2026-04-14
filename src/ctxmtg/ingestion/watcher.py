# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Inbox Watcher
=============

Polls ~/.ctxmtg/inbox/ for new files and ingests them through the
standard IngestionWorker pipeline. Successfully processed files are
moved to ~/.ctxmtg/processed/. Failed files stay in inbox with an
error logged.

Two modes:
    - Continuous: polls every N seconds (ctxmtg watch)
    - Single-pass: processes once and exits (ctxmtg watch --once)

Depends on:
    - ctxmtg.ingestion.worker (IngestionWorker)
    - ctxmtg.ingestion.loaders (FileLoaderRegistry for extension check)
    - ctxmtg.config.settings (inbox_path, processed_path, interval)

Used by:
    - ctxmtg.cli (the `ctxmtg watch` command)
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import structlog

from ctxmtg.ingestion.loaders import FileLoaderRegistry

logger = structlog.get_logger("ctxmtg.ingestion.watcher")


class InboxWatcher:
    """
    Watches the inbox directory and ingests new files.

    Files are identified by extension using the FileLoaderRegistry.
    Unsupported files are left in place (logged as skipped).
    """

    def __init__(
        self,
        inbox_path: Path,
        processed_path: Path,
        worker: Any,
        interval_seconds: int = 30,
    ) -> None:
        self._inbox = inbox_path
        self._processed = processed_path
        self._worker = worker
        self._interval = interval_seconds
        self._registry = FileLoaderRegistry()
        self._running = False

    def scan_once(self) -> list[dict[str, Any]]:
        """
        Process all supported files currently in the inbox.

        Returns:
            List of per-file result dicts with keys: file, status,
            and optional error or stats fields.
        """
        self._inbox.mkdir(parents=True, exist_ok=True)
        self._processed.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        supported = set(self._registry.supported_extensions())

        files = sorted(f for f in self._inbox.iterdir() if f.is_file())
        if not files:
            return results

        for file_path in files:
            ext = file_path.suffix.lower()
            if ext not in supported:
                logger.debug("watcher_skip_unsupported", file=file_path.name, ext=ext)
                results.append({
                    "file": file_path.name,
                    "status": "skipped",
                    "reason": f"unsupported extension {ext}",
                })
                continue

            result = self._ingest_and_move(file_path)
            results.append(result)

        return results

    def run(self) -> None:
        """
        Poll the inbox continuously until stopped.

        Catches KeyboardInterrupt for clean shutdown.
        """
        self._running = True
        logger.info(
            "watcher_started",
            inbox=str(self._inbox),
            interval=self._interval,
        )

        try:
            while self._running:
                results = self.scan_once()
                if results:
                    ok = sum(1 for r in results if r["status"] == "ok")
                    failed = sum(1 for r in results if r["status"] == "error")
                    skipped = sum(1 for r in results if r["status"] == "skipped")
                    logger.info(
                        "watcher_cycle",
                        ok=ok,
                        failed=failed,
                        skipped=skipped,
                    )
                time.sleep(self._interval)
        except KeyboardInterrupt:
            logger.info("watcher_stopped_by_user")
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the watcher to stop after the current cycle."""
        self._running = False

    def _ingest_and_move(self, file_path: Path) -> dict[str, Any]:
        """Ingest a single file and move it to processed/ on success."""
        try:
            stats = self._worker.ingest_file(file_path)

            dest = self._processed / file_path.name
            # Avoid overwriting: append a counter if name exists.
            if dest.exists():
                stem = file_path.stem
                ext = file_path.suffix
                counter = 1
                while dest.exists():
                    dest = self._processed / f"{stem}_{counter}{ext}"
                    counter += 1

            shutil.move(str(file_path), str(dest))
            logger.info(
                "watcher_ingested",
                file=file_path.name,
                entities=stats.get("entities_stored", 0),
                facts=stats.get("facts_stored", 0),
            )
            return {
                "file": file_path.name,
                "status": "ok",
                "entities": stats.get("entities_stored", 0),
                "facts": stats.get("facts_stored", 0),
            }
        except Exception as exc:
            logger.error(
                "watcher_ingest_failed",
                file=file_path.name,
                error=str(exc),
            )
            return {
                "file": file_path.name,
                "status": "error",
                "error": str(exc),
            }
