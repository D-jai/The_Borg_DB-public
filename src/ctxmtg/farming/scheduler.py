# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Farming Scheduler
==================

Idle-time scheduler for farming cycles.  Monitors system idle state
and triggers a farming cycle when the CPU has been idle above a
threshold for a sustained period.

On Linux: reads /proc/stat for CPU idle percentage.
Fallback: time-based scheduling (always eligible after min_interval).

The scheduler runs as an asyncio loop.  Call start() to begin
monitoring and stop() to shut down gracefully.

Depends on:
    - asyncio (event loop for scheduling)
    - time (interval tracking)
    - ctxmtg.farming.pipeline (FarmingPipeline to trigger)

Used by:
    - ctxmtg.cli (``ctxmtg farm schedule start/stop``)
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from ctxmtg.farming.pipeline import FarmingPipeline

# ---------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.scheduler")


class FarmingScheduler:
    """
    Idle-time scheduler for farming cycles.

    Monitors CPU idle percentage (Linux) or falls back to pure
    time-based scheduling.  When idle conditions are sustained
    for ``sustained_seconds`` and at least ``min_interval_seconds``
    have elapsed since the last cycle, triggers a farming cycle.

    Usage:
        scheduler = FarmingScheduler(pipeline, idle_threshold=80.0)
        await scheduler.start()   # blocks until stop() is called
        await scheduler.stop()    # from another coroutine / signal handler
    """

    def __init__(
        self,
        pipeline: FarmingPipeline,
        idle_threshold: float = 80.0,
        sustained_seconds: int = 30,
        min_interval_seconds: int = 3600,
    ) -> None:
        """
        Configure the scheduler.

        Args:
            pipeline:              The FarmingPipeline to trigger.
            idle_threshold:        Minimum CPU idle % to consider "idle".
            sustained_seconds:     How long idle must be sustained.
            min_interval_seconds:  Minimum seconds between farming cycles.
        """
        self._pipeline = pipeline
        self._idle_threshold = idle_threshold
        self._sustained_seconds = sustained_seconds
        self._min_interval_seconds = min_interval_seconds

        # Runtime state
        self._running = False
        self._last_cycle_time: float = 0.0

    # =================================================================
    # Public API
    # =================================================================

    async def start(self) -> None:
        """
        Start the scheduler loop.  Blocks until stop() is called.

        Polls CPU idle percentage every 5 seconds.  When conditions
        are met (sustained idle + interval elapsed), triggers a
        farming cycle via the pipeline.
        """
        self._running = True
        logger.info(
            "scheduler_started",
            idle_threshold=self._idle_threshold,
            sustained_seconds=self._sustained_seconds,
            min_interval=self._min_interval_seconds,
        )

        while self._running:
            try:
                if self.should_run():
                    logger.info("scheduler_triggering_cycle")
                    result = await self._pipeline.run_cycle(trigger="idle")
                    self._last_cycle_time = time.monotonic()
                    logger.info(
                        "scheduler_cycle_complete",
                        cycle_id=result.get("cycle_id"),
                        status=result.get("status"),
                    )
            except Exception as exc:
                # Log but don't crash the scheduler loop
                logger.error(
                    "scheduler_cycle_error",
                    error_code="CTXMTG-FRM-008",
                    error=str(exc),
                )

            # Sleep before next check (5 second poll interval)
            await asyncio.sleep(5.0)

        logger.info("scheduler_stopped")

    async def stop(self) -> None:
        """Stop the scheduler loop gracefully."""
        self._running = False

    def should_run(self) -> bool:
        """
        Check if conditions are met for a farming cycle.

        Returns True when:
        1. min_interval_seconds have elapsed since the last cycle
        2. CPU has been idle above threshold for sustained_seconds

        If /proc/stat is unavailable (macOS, CI), the idle check
        is skipped and only the interval check applies.
        """
        # Check interval
        elapsed = time.monotonic() - self._last_cycle_time
        if elapsed < self._min_interval_seconds:
            return False

        # Check sustained idle
        checks_needed = max(1, self._sustained_seconds // 5)
        for _ in range(checks_needed):
            idle = self.get_cpu_idle_percent(sample_interval=2.0)
            if idle < self._idle_threshold:
                return False

        return True

    # =================================================================
    # CPU idle detection
    # =================================================================

    @staticmethod
    def get_cpu_idle_percent(sample_interval: float = 1.0) -> float:
        """
        Read /proc/stat to compute CPU idle percentage.

        Samples /proc/stat twice with ``sample_interval`` seconds
        between samples, then computes the idle proportion of the
        elapsed CPU time.

        Falls back to 100.0 (always idle) if /proc/stat is not
        available (macOS, Windows, CI containers without /proc).
        This causes the scheduler to fall back to pure time-based
        scheduling.

        Args:
            sample_interval: Seconds between the two /proc/stat reads.

        Returns:
            CPU idle percentage as a float (0.0 to 100.0).
        """
        proc_stat = Path("/proc/stat")
        if not proc_stat.exists():
            # Not on Linux -- fall back to time-based scheduling
            return 100.0

        try:
            # Read CPU times: user nice system idle iowait irq softirq steal
            def _read_cpu() -> list[int]:
                line = proc_stat.read_text().splitlines()[0]
                parts = line.split()
                return [int(p) for p in parts[1:9]]

            t1 = _read_cpu()
            time.sleep(sample_interval)
            t2 = _read_cpu()

            deltas = [t2[i] - t1[i] for i in range(len(t1))]
            total = sum(deltas)
            # idle (index 3) + iowait (index 4)
            idle = deltas[3] + deltas[4]

            return (idle / total * 100) if total > 0 else 100.0

        except Exception:
            # Any read error -- fall back to time-based
            return 100.0
