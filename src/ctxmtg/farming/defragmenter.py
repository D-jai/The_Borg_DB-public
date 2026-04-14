# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Defragmenter Maintenance Stage
================================

Performs SQLite housekeeping and vector store compaction to keep the
knowledge database performant as data accumulates.  This is the
system's "janitor" -- it runs periodically to optimise storage,
checkpoint WAL files, and collect metrics.

Operations performed:
    1. ``PRAGMA optimize`` -- updates index statistics so the query
       planner makes better decisions.
    2. ``PRAGMA wal_checkpoint(TRUNCATE)`` -- forces the WAL file
       to be merged into the main database file and then truncated.
    3. ``VACUUM`` (optional, disabled by default) -- rebuilds the
       entire database file to reclaim space from deleted rows.
       Expensive: should only run monthly or on demand.
    4. Storage metrics collection -- counts of entities, facts, and
       meta-insights; database file size.
    5. Vector store ``compact()`` -- delegates to the vector store's
       own compaction (tombstone cleanup, file merging).

All PRAGMAs are wrapped in try/except because not every SQLite build
or environment supports every PRAGMA.  The stage degrades gracefully
when a PRAGMA fails.

Results are logged to ``maintenance_defragmenter`` and returned as
a single :class:`FarmingInsight` with type ``meta``.

Depends on:
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.interfaces.llm (LLMProvider -- reserved, unused)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync→async bridge)
    - structlog (structured logging)
    - uuid (unique IDs for maintenance log entries)
    - json (serialisation of log details)

Used by:
    - ctxmtg.farming.pipeline (registered as maintenance stage 14)
"""

from __future__ import annotations

import json
from uuid import uuid4

import structlog

from ctxmtg.farming.checkpoint import _run_async
from ctxmtg.interfaces.farming import FarmingContext, FarmingStage
from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.models.farming import FarmingInsight

# ---------------------------------------------------------------
# Module-level logger -- logs defragmentation events.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.farming.defragmenter")


class DefragmenterStage(FarmingStage):
    """
    Maintenance stage that optimises storage performance.

    Runs SQLite PRAGMAs (optimize, wal_checkpoint, optional VACUUM),
    collects storage metrics (row counts, database size), and triggers
    vector store compaction.  Designed to run every farming cycle for
    lightweight operations; the expensive VACUUM should only be
    enabled for monthly full-maintenance runs.

    Usage:
        defragmenter = DefragmenterStage(vacuum_enabled=False)
        insights = defragmenter.run(sql_store, vector_store, context)
    """

    def __init__(
        self,
        vacuum_enabled: bool = False,
        wal_max_size_mb: int = 100,
        llm: LLMProvider | None = None,
    ) -> None:
        """
        Configure the defragmenter.

        Args:
            vacuum_enabled: Whether to run ``VACUUM`` after the
                lightweight optimizations.  Defaults to False because
                VACUUM rewrites the entire database file and can be
                very slow on large stores.  Enable it for periodic
                deep-maintenance runs (e.g., monthly cron).
            wal_max_size_mb: Reserved for future WAL-size monitoring.
                If the WAL exceeds this threshold, the defragmenter
                could trigger an aggressive checkpoint.  Currently
                informational only.
            llm: Optional LLM provider.  Reserved for future use.
        """
        self._vacuum_enabled = vacuum_enabled
        self._wal_max_size_mb = wal_max_size_mb
        self._llm = llm

    # -----------------------------------------------------------------
    # FarmingStage interface -- stage name for logging/checkpointing.
    # -----------------------------------------------------------------
    def get_name(self) -> str:
        """Return the stage name used for logging and checkpointing."""
        return "defragmenter"

    # =================================================================
    # Main entry point
    # =================================================================
    def run(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        """
        Run storage optimisation and collect metrics.

        Steps:
        1. Execute SQLite maintenance PRAGMAs (optimize, WAL
           checkpoint, optional VACUUM).
        2. Collect storage metrics (row counts, DB size).
        3. Run vector store compaction.
        4. Log results to maintenance_defragmenter.
        5. Return a FarmingInsight with the metrics.

        Args:
            sql_store:    SQL store for PRAGMA execution and metrics.
            vector_store: Vector store for compaction.
            context:      Farming context with cycle_id and budget.

        Returns:
            List containing exactly one FarmingInsight with
            storage metrics and optimisation results.
        """
        return _run_async(
            self._run_impl(sql_store, vector_store, context)
        )

    # =================================================================
    # Async implementation
    # =================================================================
    async def _run_impl(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        """
        Async implementation of the defragmentation logic.

        Separated from run() so we can ``await`` on sql_store methods.
        The sync run() method bridges to this via _run_async().
        """
        # Accumulate metrics from each step into a single dict.
        metrics: dict[str, object] = {}

        # ---------------------------------------------------------
        # STEP 1a: PRAGMA optimize -- update index statistics.
        # This is cheap and safe to run on every cycle.  It tells
        # SQLite to refresh ANALYZE data for any indexes whose stats
        # are stale.
        # ---------------------------------------------------------
        try:
            await sql_store.execute_sql("PRAGMA optimize")
            metrics["pragma_optimize"] = "ok"
            logger.debug("defragmenter_pragma_optimize_ok")
        except Exception as exc:
            # Some SQLite versions or environments do not support
            # PRAGMA optimize.  Log and continue gracefully.
            metrics["pragma_optimize"] = f"error: {exc}"
            logger.warning(
                "defragmenter_pragma_optimize_failed",
                error_code="CTXMTG-FRM-001",
                error=str(exc),
            )

        # ---------------------------------------------------------
        # STEP 1b: PRAGMA wal_checkpoint(TRUNCATE) -- merge WAL
        # into the main DB file and truncate the WAL.
        # ---------------------------------------------------------
        try:
            await sql_store.execute_sql(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            )
            metrics["wal_checkpoint"] = "ok"
            logger.debug("defragmenter_wal_checkpoint_ok")
        except Exception as exc:
            # WAL checkpoint can fail if another connection holds a
            # read transaction.  Not critical -- will retry next cycle.
            metrics["wal_checkpoint"] = f"error: {exc}"
            logger.warning(
                "defragmenter_wal_checkpoint_failed",
                error_code="CTXMTG-FRM-001",
                error=str(exc),
            )

        # ---------------------------------------------------------
        # STEP 1c: VACUUM -- rebuild the database file (optional).
        # Only runs when vacuum_enabled is True because VACUUM is
        # expensive: it copies the entire database to a temp file
        # and rewrites it.  Suitable for monthly maintenance only.
        # ---------------------------------------------------------
        if self._vacuum_enabled:
            try:
                await sql_store.execute_sql("VACUUM")
                metrics["vacuum"] = "ok"
                logger.info("defragmenter_vacuum_ok")
            except Exception as exc:
                metrics["vacuum"] = f"error: {exc}"
                logger.warning(
                    "defragmenter_vacuum_failed",
                    error_code="CTXMTG-FRM-001",
                    error=str(exc),
                )
        else:
            # Record that VACUUM was intentionally skipped.
            metrics["vacuum"] = "skipped"

        # ---------------------------------------------------------
        # STEP 2: Collect storage metrics -- row counts.
        # Each query is wrapped in try/except for resilience.
        # ---------------------------------------------------------

        # 2a: Entity count
        try:
            rows = await sql_store.execute_sql(
                "SELECT COUNT(*) as entity_count FROM entities"
            )
            metrics["entity_count"] = rows[0]["entity_count"]
        except Exception as exc:
            metrics["entity_count"] = f"error: {exc}"
            logger.warning(
                "defragmenter_entity_count_failed",
                error_code="CTXMTG-FRM-001",
                error=str(exc),
            )

        # 2b: Fact count
        try:
            rows = await sql_store.execute_sql(
                "SELECT COUNT(*) as fact_count FROM facts"
            )
            metrics["fact_count"] = rows[0]["fact_count"]
        except Exception as exc:
            metrics["fact_count"] = f"error: {exc}"
            logger.warning(
                "defragmenter_fact_count_failed",
                error_code="CTXMTG-FRM-001",
                error=str(exc),
            )

        # 2c: Meta-insight count
        try:
            rows = await sql_store.execute_sql(
                "SELECT COUNT(*) as insight_count FROM meta_insights"
            )
            metrics["insight_count"] = rows[0]["insight_count"]
        except Exception as exc:
            metrics["insight_count"] = f"error: {exc}"
            logger.warning(
                "defragmenter_insight_count_failed",
                error_code="CTXMTG-FRM-001",
                error=str(exc),
            )

        # 2d: Database file size (page_count * page_size).
        # This PRAGMA syntax is not available on all SQLite versions,
        # so we guard it carefully.
        try:
            rows = await sql_store.execute_sql(
                "SELECT page_count * page_size as db_size "
                "FROM pragma_page_count(), pragma_page_size()"
            )
            metrics["db_size_bytes"] = rows[0]["db_size"]
        except Exception as exc:
            # Older SQLite versions may not support table-valued
            # PRAGMA functions.  Fall back gracefully.
            metrics["db_size_bytes"] = f"error: {exc}"
            logger.warning(
                "defragmenter_db_size_failed",
                error_code="CTXMTG-FRM-001",
                error=str(exc),
            )

        # ---------------------------------------------------------
        # STEP 3: Vector store compaction.
        # Delegates to the VectorStore.compact() method.  The method
        # may not be fully implemented in all store backends, so
        # we wrap it in try/except.
        # ---------------------------------------------------------
        try:
            compact_result = await vector_store.compact()
            metrics["vector_compact"] = compact_result
            logger.info(
                "defragmenter_vector_compact_ok",
                result=compact_result,
            )
        except Exception as exc:
            # compact() may not be implemented or may raise if the
            # vector store is empty.  Not critical.
            metrics["vector_compact"] = f"error: {exc}"
            logger.warning(
                "defragmenter_vector_compact_failed",
                error_code="CTXMTG-FRM-001",
                error=str(exc),
            )

        # ---------------------------------------------------------
        # STEP 4: Log to maintenance_defragmenter table.
        # ---------------------------------------------------------
        log_id = str(uuid4())
        # Serialise metrics for storage; convert non-serialisable
        # values to strings.
        detail = json.dumps(metrics, default=str)
        await sql_store.execute_sql(
            "INSERT INTO maintenance_defragmenter "
            "(id, cycle_id, action, target_ids, detail) "
            "VALUES (:id, :cycle, 'optimize', :targets, :detail)",
            {
                "id": log_id,
                "cycle": context.cycle_id,
                "targets": json.dumps([]),  # no specific targets
                "detail": detail,
            },
        )

        # Commit the log entry
        db = sql_store._ensure_db()  # type: ignore[attr-defined]
        await db.commit()

        logger.info(
            "defragmenter_complete",
            metrics=metrics,
            cycle_id=context.cycle_id,
        )

        # ---------------------------------------------------------
        # STEP 5: Build a FarmingInsight summarising the run.
        # ---------------------------------------------------------
        insight = FarmingInsight(
            id=str(uuid4()),
            insight_type="meta",
            title="Storage optimized",
            description=(
                "Ran SQLite maintenance PRAGMAs, collected storage "
                "metrics, and triggered vector store compaction."
            ),
            evidence=[],
            confidence=1.0,
            parameters=metrics,
        )

        return [insight]
