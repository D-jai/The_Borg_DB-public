# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Farming Pipeline Orchestrator
===============================

Orchestrates all farming stages in sequence.  Manages the farming
window: suspends hive pull, runs stages with per-stage timeouts,
saves checkpoints, resumes pull.  Handles crash recovery by checking
farming_checkpoints for interrupted cycles.

The pipeline is the single entry point for farming -- both the CLI
(``ctxmtg farm run``) and the idle-time scheduler call run_cycle().

Design:
    1. Suspend hive pull (if configured) to avoid write contention.
    2. Create a farming_cycles record in the SQL store.
    3. For each registered stage, in registration order:
       a. Skip if already completed in this cycle (crash recovery).
       b. Create a farming_checkpoints record (status = 'running').
       c. Build a FarmingContext with budget, checkpoint, and config.
       d. Run the stage in a thread (it's CPU-bound / sync) with a
          hard timeout as a safety net.
       e. Store returned insights; mark checkpoint completed.
       f. On failure/timeout: mark checkpoint failed, continue.
    4. Mark the cycle completed (or partial/failed).
    5. Resume hive pull.

Depends on:
    - asyncio (timeout enforcement, thread execution)
    - time (duration measurement)
    - structlog (structured logging)
    - ctxmtg.interfaces.farming (FarmingStage, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.farming.checkpoint (SQLiteCheckpointStore)
    - ctxmtg.sync.hive_pull (HivePullWorker -- optional)

Used by:
    - ctxmtg.farming.scheduler (triggers cycles during idle time)
    - ctxmtg.cli (``ctxmtg farm run`` command)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import structlog

from ctxmtg.farming.checkpoint import SQLiteCheckpointStore
from ctxmtg.interfaces.farming import FarmingContext, FarmingStage
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.models.farming import FarmingInsight

# ---------------------------------------------------------------
# Module-level logger -- logs cycle and stage progress.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.pipeline")

# ---------------------------------------------------------------
# Default configuration values.
# ---------------------------------------------------------------
DEFAULT_STAGE_BUDGET_SECONDS = 60


class FarmingPipeline:
    """
    Orchestrates all farming stages in sequence.

    Manages the farming window: suspends hive pull, runs stages
    with per-stage timeouts, saves checkpoints, resumes pull.
    Handles crash recovery by checking farming_checkpoints for
    interrupted cycles.

    Usage:
        pipeline = FarmingPipeline(sql_store, vector_store)
        pipeline.register_stage(EntityAnalyticsStage())
        pipeline.register_stage(TrendDetectionStage())
        result = await pipeline.run_cycle(trigger="manual")
    """

    def __init__(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        hive_pull_worker: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Prepare the pipeline orchestrator.

        Args:
            sql_store:         The SQL store for reading data and
                               storing insights + checkpoints.
            vector_store:      The vector store for semantic data.
            hive_pull_worker:  Optional HivePullWorker to suspend/resume
                               during farming windows.
            config:            Pipeline configuration dict.  Keys:
                               - default_stage_budget_seconds (int)
                               - stage-specific configs keyed by name.
        """
        self._sql_store = sql_store
        self._vector_store = vector_store
        self._hive_pull = hive_pull_worker
        self._config = config or {}
        self._stages: list[FarmingStage] = []

    def register_stage(self, stage: FarmingStage) -> None:
        """
        Register a farming stage.  Stages run in registration order.

        Args:
            stage: A FarmingStage implementation to include in cycles.
        """
        self._stages.append(stage)
        logger.debug("stage_registered", stage=stage.get_name())

    async def run_cycle(self, trigger: str = "manual") -> dict[str, Any]:
        """
        Run one full farming cycle.

        Executes every registered stage in order, with per-stage
        timeouts and checkpointing.  Hive pull is suspended for
        the duration and resumed in a finally block.

        Args:
            trigger: What initiated this cycle ("manual", "idle",
                     "scheduled").

        Returns:
            Dict with cycle results:
                - cycle_id (int)
                - status ("completed", "partial", "failed")
                - stages_run (int)
                - stages_succeeded (int)
                - stages_failed (int)
                - insights_produced (int)
                - duration_ms (float)
        """
        start_time = time.monotonic()

        # ---------------------------------------------------------------
        # Step 1: Suspend hive pull to avoid write contention.
        # ---------------------------------------------------------------
        if self._hive_pull is not None:
            self._hive_pull.suspend()

        try:
            # -----------------------------------------------------------
            # Step 2: Create a farming_cycles record.
            # -----------------------------------------------------------
            cycle_id = await self._create_cycle(trigger)

            # -----------------------------------------------------------
            # Step 3: Run each registered stage.
            # -----------------------------------------------------------
            stages_succeeded = 0
            stages_failed = 0
            total_insights = 0

            for stage in self._stages:
                stage_name = stage.get_name()

                # Check if this stage already completed (crash recovery)
                if await self._is_stage_completed(cycle_id, stage_name):
                    logger.info(
                        "stage_already_completed",
                        cycle_id=cycle_id,
                        stage=stage_name,
                    )
                    stages_succeeded += 1
                    continue

                # Run the stage
                success, insight_count = await self._run_stage(
                    cycle_id, stage, stage_name
                )

                if success:
                    stages_succeeded += 1
                    total_insights += insight_count
                else:
                    stages_failed += 1

            # -----------------------------------------------------------
            # Step 4: Determine cycle status and update record.
            # -----------------------------------------------------------
            stages_run = stages_succeeded + stages_failed

            if stages_failed == 0:
                status = "completed"
            elif stages_succeeded == 0:
                status = "failed"
            else:
                status = "partial"

            await self._complete_cycle(cycle_id, status, stages_run)

            duration_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "farming_cycle_complete",
                cycle_id=cycle_id,
                status=status,
                stages_run=stages_run,
                stages_succeeded=stages_succeeded,
                stages_failed=stages_failed,
                insights_produced=total_insights,
                duration_ms=round(duration_ms, 2),
            )

            return {
                "cycle_id": cycle_id,
                "status": status,
                "stages_run": stages_run,
                "stages_succeeded": stages_succeeded,
                "stages_failed": stages_failed,
                "insights_produced": total_insights,
                "duration_ms": round(duration_ms, 2),
            }

        finally:
            # -----------------------------------------------------------
            # Step 5: ALWAYS resume hive pull, even on exception.
            # -----------------------------------------------------------
            if self._hive_pull is not None:
                self._hive_pull.resume()

    # =================================================================
    # Private: stage execution
    # =================================================================

    async def _run_stage(
        self,
        cycle_id: int,
        stage: FarmingStage,
        stage_name: str,
    ) -> tuple[bool, int]:
        """
        Run a single stage with timeout and checkpointing.

        Creates a checkpoint record, runs the stage in a thread
        (sync → async bridge), stores returned insights, and marks
        the checkpoint completed or failed.

        Args:
            cycle_id:    The current farming cycle ID.
            stage:       The FarmingStage implementation to run.
            stage_name:  The stage's name (for logging and DB).

        Returns:
            (success: bool, insight_count: int) tuple.
        """
        stage_start = time.monotonic()

        # Read stage budget from config, fall back to default
        budget = self._config.get(
            stage_name, {}
        ).get("budget_seconds", self._config.get(
            "default_stage_budget_seconds", DEFAULT_STAGE_BUDGET_SECONDS
        ))

        # Create checkpoint record (status = 'running')
        await self._create_checkpoint(cycle_id, stage_name)

        # Build the FarmingContext for this stage
        checkpoint_store = SQLiteCheckpointStore(
            self._sql_store, cycle_id, stage_name
        )
        context = FarmingContext(
            cycle_id=cycle_id,
            budget_seconds=budget,
            checkpoint=checkpoint_store,
            config=self._config.get(stage_name, {}),
        )

        try:
            # Run sync stage in a thread with async timeout.
            # FarmingStage.run() is synchronous (CPU-bound), so we
            # use run_in_executor to avoid blocking the event loop.
            loop = asyncio.get_event_loop()
            insights: list[FarmingInsight] = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    stage.run,
                    self._sql_store,
                    self._vector_store,
                    context,
                ),
                timeout=budget + 5,  # 5s grace period over stage budget
            )

            # Store each insight in the SQL store
            insight_count = 0
            for insight in insights:
                await self._sql_store.store_insight(insight)
                insight_count += 1

            # Mark checkpoint as completed
            elapsed = time.monotonic() - stage_start
            await self._complete_checkpoint(
                cycle_id, stage_name, insight_count
            )

            logger.info(
                "stage_completed",
                cycle_id=cycle_id,
                stage=stage_name,
                insights=insight_count,
                duration_ms=round(elapsed * 1000, 2),
            )
            return (True, insight_count)

        except asyncio.TimeoutError:
            await self._fail_checkpoint(
                cycle_id, stage_name, "Stage timed out"
            )
            logger.warning(
                "stage_timed_out",
                error_code="CTXMTG-FRM-001",
                cycle_id=cycle_id,
                stage=stage_name,
                budget_seconds=budget,
            )
            return (False, 0)

        except Exception as exc:
            await self._fail_checkpoint(
                cycle_id, stage_name, str(exc)
            )
            logger.error(
                "stage_failed",
                error_code="CTXMTG-FRM-001",
                cycle_id=cycle_id,
                stage=stage_name,
                error=str(exc),
            )
            return (False, 0)

    # =================================================================
    # Private: database operations for cycle/checkpoint management
    # =================================================================

    async def _exec_and_commit(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute SQL via the store AND commit the transaction.

        SQLiteStore.execute_sql() doesn't auto-commit writes, which
        leaves an open transaction.  Subsequent calls to store_insight()
        (which uses BEGIN IMMEDIATE) would then fail with "cannot start
        a transaction within a transaction".  This helper commits after
        every write to keep the connection clean.
        """
        result = await self._sql_store.execute_sql(sql, params)
        # Access the underlying connection to commit.
        # This is safe because we're the only writer during a farming
        # window (hive pull is suspended).
        db = self._sql_store._ensure_db()
        await db.commit()
        return result

    async def _create_cycle(self, trigger: str) -> int:
        """Create a farming_cycles record and return the cycle_id."""
        await self._exec_and_commit(
            "INSERT INTO farming_cycles (status, trigger) "
            "VALUES (:status, :trigger)",
            {"status": "running", "trigger": trigger},
        )
        # Fetch the auto-incremented cycle_id (read -- no commit needed)
        rows = await self._sql_store.execute_sql(
            "SELECT MAX(cycle_id) as cid FROM farming_cycles", {}
        )
        return rows[0]["cid"]

    async def _complete_cycle(
        self, cycle_id: int, status: str, stages_done: int
    ) -> None:
        """Mark a farming cycle as completed/partial/failed."""
        now = datetime.now(timezone.utc).isoformat()
        await self._exec_and_commit(
            "UPDATE farming_cycles "
            "SET status = :status, completed_at = :ts, stages_done = :done "
            "WHERE cycle_id = :cid",
            {"status": status, "ts": now, "done": stages_done, "cid": cycle_id},
        )

    async def _is_stage_completed(self, cycle_id: int, stage: str) -> bool:
        """Check if a stage is already completed for this cycle."""
        rows = await self._sql_store.execute_sql(
            "SELECT status FROM farming_checkpoints "
            "WHERE cycle_id = :cid AND stage = :stage",
            {"cid": cycle_id, "stage": stage},
        )
        return bool(rows) and rows[0]["status"] == "completed"

    async def _create_checkpoint(self, cycle_id: int, stage: str) -> None:
        """Create a farming_checkpoints record (status = 'running')."""
        await self._exec_and_commit(
            "INSERT OR REPLACE INTO farming_checkpoints "
            "(cycle_id, stage, status) VALUES (:cid, :stage, 'running')",
            {"cid": cycle_id, "stage": stage},
        )

    async def _complete_checkpoint(
        self, cycle_id: int, stage: str, items: int
    ) -> None:
        """Mark a checkpoint as completed."""
        now = datetime.now(timezone.utc).isoformat()
        await self._exec_and_commit(
            "UPDATE farming_checkpoints "
            "SET status = 'completed', completed_at = :ts, "
            "items_processed = :items "
            "WHERE cycle_id = :cid AND stage = :stage",
            {"ts": now, "items": items, "cid": cycle_id, "stage": stage},
        )

    async def _fail_checkpoint(
        self, cycle_id: int, stage: str, error_msg: str
    ) -> None:
        """Mark a checkpoint as failed with an error message."""
        now = datetime.now(timezone.utc).isoformat()
        await self._exec_and_commit(
            "UPDATE farming_checkpoints "
            "SET status = 'failed', completed_at = :ts, "
            "error_message = :err "
            "WHERE cycle_id = :cid AND stage = :stage",
            {"ts": now, "err": error_msg, "cid": cycle_id, "stage": stage},
        )
