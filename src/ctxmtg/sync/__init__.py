# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Sync Package (Phase 5 Intelligence Aggregator Model)
=====================================================

This package implements the hive intelligence sync system.

Phase 5 redesign: the hive is an intelligence aggregator, not a
data mirror.  Locals push distilled intelligence (distiller summaries
+ meta insights) to the hive.  The hive merges them into unified
entity profiles and mines the collective for cross-stream patterns.

Modules:
    - hive_db.py            : HiveDatabase -- intelligence schema + CRUD
    - hive_sync.py          : HiveSyncWorker -- pushes intelligence to hive
    - intelligence_merger.py: IntelligenceMerger -- merges entity profiles
    - intelligence_pull.py  : IntelligencePullWorker -- pulls hive intelligence
    - hive_farming.py       : HiveFarmingPipeline -- 3-stage hive farming
    - hive_pull.py          : HivePullWorker -- legacy pull model (kept for
                              backward compatibility with farming pipeline)
    - stages/               : Hive farming stage implementations

Depends on:
    - ctxmtg.storage.sqlite (SQLiteStore for local store access)
    - ctxmtg.storage.schema (DDL and migration)
    - ctxmtg.exceptions (SyncError for error reporting)

Used by:
    - ctxmtg.cli (exposes hive commands to the user)
"""

from ctxmtg.sync.hive_db import HiveDatabase
from ctxmtg.sync.hive_pull import HivePullWorker
from ctxmtg.sync.hive_sync import HiveSyncWorker
from ctxmtg.sync.intelligence_pull import IntelligencePullWorker

__all__ = [
    "HiveDatabase",
    "HivePullWorker",
    "HiveSyncWorker",
    "IntelligencePullWorker",
]
