# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Health Monitor
==============

This module provides system health monitoring for ctxmtg. It collects
metrics about resource usage (RAM, database size), data counts
(interactions, entities, vectors), and pipeline statistics (intake
accept/defer/reject counts).

Health data is displayed by the `ctxmtg health` CLI command and can
be written to a JSONL metrics file for historical tracking.

This is especially important for edge deployments (Raspberry Pi) where
resource constraints are tight and monitoring helps prevent out-of-
memory crashes or disk-full conditions.

Depends on:
    - os (file size checks)
    - psutil or resource (RAM usage -- graceful fallback if unavailable)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)

Used by:
    - ctxmtg.cli (health command)
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from ctxmtg.interfaces.storage import SQLStore, VectorStore

# ---------------------------------------------------------------
# Module-level logger.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.health.monitor")


class HealthMonitor:
    """
    Collects and reports system health metrics.

    Gathers information from multiple sources:
    - Process memory usage (via resource module)
    - Database file size on disk
    - Record counts from SQL store
    - Vector count from vector store
    - Intake statistics (from the Traffic Cop gateway)

    Usage:
        monitor = HealthMonitor(
            sql_store=sqlite_store,
            vector_store=lancedb_store,
            db_path=Path("~/.ctxmtg/knowledge.db"),
        )
        health = monitor.get_health()
        # health = {"ram_mb": 45.2, "db_size_mb": 12.5, ...}
    """

    def __init__(
        self,
        sql_store: SQLStore | None = None,
        vector_store: VectorStore | None = None,
        db_path: Path | None = None,
        intake_stats: dict[str, int] | None = None,
    ) -> None:
        """
        Initialize the health monitor with data sources.

        Args:
            sql_store: The SQL store for record counts.
            vector_store: The vector store for vector counts.
            db_path: Path to the SQLite database file for size checks.
            intake_stats: Intake statistics dict from the Traffic Cop.
        """
        self._sql_store = sql_store
        self._vector_store = vector_store
        self._db_path = db_path
        self._intake_stats = intake_stats or {}

    def ensure_directories(self) -> None:
        """
        Create the inbox/ and processed/ directories if they don't exist.

        Called by `ctxmtg health` so the data directory is always ready
        for the watcher daemon.
        """
        from ctxmtg.config.settings import CtxMtgSettings

        settings = CtxMtgSettings()
        for attr in ("inbox_path", "processed_path"):
            p = Path(getattr(settings, attr)).expanduser()
            p.mkdir(parents=True, exist_ok=True)

    def get_health(self) -> dict[str, Any]:
        """
        Collect and return all health metrics.

        Returns:
            A dict of health metrics including RAM usage, DB size,
            record counts, vector counts, and intake stats.
        """
        self.ensure_directories()

        health: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "healthy",
        }

        # RAM usage
        health["ram_mb"] = self._get_ram_usage_mb()

        # Database file size
        health["db_size_mb"] = self._get_db_size_mb()

        # Record counts from SQL store
        if self._sql_store:
            health["record_counts"] = self._get_record_counts()

        # Vector count from vector store
        if self._vector_store:
            health["vector_count"] = self._get_vector_count()

        # Intake statistics
        health["intake_stats"] = dict(self._intake_stats)

        return health

    def _get_ram_usage_mb(self) -> float:
        """
        Get current process RAM usage in megabytes.

        Uses the resource module (Unix) for memory info. Falls back
        to a rough estimate if the module is not available.

        Returns:
            RAM usage in MB, rounded to 1 decimal place.
        """
        try:
            import resource

            # getrusage returns maxrss in KB on Linux, bytes on macOS
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                # macOS: maxrss is in bytes
                ram_mb = usage.ru_maxrss / (1024 * 1024)
            else:
                # Linux: maxrss is in KB
                ram_mb = usage.ru_maxrss / 1024
            return round(ram_mb, 1)
        except (ImportError, AttributeError):
            return 0.0

    def _get_db_size_mb(self) -> float:
        """
        Get the SQLite database file size in megabytes.

        Returns:
            File size in MB, or 0.0 if the file doesn't exist.
        """
        if self._db_path and self._db_path.exists():
            size_bytes = os.path.getsize(self._db_path)
            return round(size_bytes / (1024 * 1024), 2)
        return 0.0

    def _get_record_counts(self) -> dict[str, int]:
        """
        Get record counts from the SQL store.

        Queries the SQL store for counts of interactions, entities,
        and facts. Returns 0 for any count that fails.

        Returns:
            A dict of table_name → record_count.
        """
        counts: dict[str, int] = {
            "interactions": 0,
            "entities": 0,
            "facts": 0,
        }

        if not self._sql_store:
            return counts

        try:
            # Query each table count
            for table in counts:
                try:
                    results = _run_async(
                        self._sql_store.execute_sql(f"SELECT COUNT(*) as cnt FROM {table}")
                    )
                    if results:
                        counts[table] = int(results[0].get("cnt", 0))
                except Exception:
                    pass  # Table may not exist yet
        except Exception as exc:
            logger.warning(
                "record_count_failed",
                error_code="CTXMTG-HLT-003",
                error=str(exc),
            )

        return counts

    def _get_vector_count(self) -> int:
        """
        Get the total number of vectors in the vector store.

        Returns:
            The vector count, or 0 if the store is unavailable.
        """
        if not self._vector_store:
            return 0

        try:
            return _run_async(self._vector_store.count())
        except Exception as exc:
            logger.warning(
                "vector_count_failed",
                error_code="CTXMTG-HLT-003",
                error=str(exc),
            )
            return 0


def _run_async(coro: Any) -> Any:
    """
    Run an async coroutine from synchronous code.

    Args:
        coro: The coroutine to execute.

    Returns:
        The coroutine's return value.
    """
    try:
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)
