# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Sync Provider Interface ABC
============================

This module defines the abstract base class for multi-device
synchronization providers. Sync enables the user to run ctxmtg
on multiple devices (laptop, desktop, Pi) and keep their knowledge
base consistent across all of them.

Synchronization is a Phase 4 feature. The interface is defined now
so that the rest of the system can be designed with sync awareness
(e.g., source_instance fields on models, CRDT-compatible ID generation).

The sync strategy uses:
- cr-sqlite (CRDT-based SQLite sync) for structured data
- Vector event logs with delta replay for vector store sync

All sync methods are async because synchronization involves network
I/O (pushing/pulling data between devices).

Depends on:
    - abc (Python's Abstract Base Class machinery)

Used by:
    - ctxmtg.sync.crdt_sync (will implement SyncProvider in Phase 4)
    - ctxmtg.cli (exposes sync commands to the user)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# =====================================================================
# SyncProvider ABC -- Multi-Device Synchronization Interface
# =====================================================================


class SyncProvider(ABC):
    """
    Multi-device synchronization (Phase 4).

    Handles bidirectional synchronization of knowledge data between
    multiple ctxmtg instances running on different devices. The sync
    provider manages both pushing local changes to a remote store
    and pulling remote changes to the local store.

    Synchronization is conflict-aware: when the same record is modified
    on multiple devices, the provider uses CRDT (Conflict-free Replicated
    Data Types) merge semantics to resolve conflicts automatically.

    This interface is defined in Phase 1 but implemented in Phase 4.
    The source_instance fields on data models (Interaction, Entity, Fact)
    are sync-aware: they track which instance created each record.

    Usage:
        sync = CRDTSyncProvider(remote_url="https://sync.example.com")
        pushed = await sync.push()    # Push local changes
        pulled = await sync.pull()    # Pull remote changes
        status = await sync.get_sync_status()  # Check sync state
    """

    @abstractmethod
    async def push(self) -> int:
        """
        Push local changes to the remote. Returns count of records pushed.

        Identifies all records that have been created or modified locally
        since the last sync, and transmits them to the remote store.
        Records include interactions, entities, facts, and embedding
        metadata.

        Implementations should handle:
        - Tracking which records are "dirty" (modified since last push)
        - Batching records for efficient network transmission
        - Handling network failures gracefully (retry, resume)
        - Updating sync watermarks after successful push

        Returns:
            Number of records successfully pushed to the remote.
            Returns 0 if there are no local changes to push.
        """
        ...

    @abstractmethod
    async def pull(self) -> int:
        """
        Pull remote changes to local. Returns count of records pulled.

        Fetches all records from the remote store that have been
        created or modified since the last sync, and merges them
        into the local store. Uses CRDT merge semantics to resolve
        any conflicts (same record modified on multiple devices).

        Implementations should handle:
        - Fetching only new/modified records (delta sync)
        - CRDT conflict resolution for concurrent edits
        - Updating local indexes after merge
        - Updating sync watermarks after successful pull

        Returns:
            Number of records successfully pulled and merged locally.
            Returns 0 if there are no remote changes to pull.
        """
        ...

    @abstractmethod
    async def get_sync_status(self) -> dict[str, Any]:
        """
        Return sync status: last sync time, pending changes, etc.

        Reports the current state of synchronization, including when
        the last sync occurred, how many local changes are pending,
        and whether there are known remote changes to pull.

        This is used by the health monitor and the CLI to show the
        user their sync status.

        Returns:
            A dict with at least these keys:
                - last_push_at: str | None -- ISO datetime of last push
                - last_pull_at: str | None -- ISO datetime of last pull
                - pending_push_count: int -- local records awaiting push
                - pending_pull_count: int | None -- remote records
                  awaiting pull (None if unknown until next check)
                - connected: bool -- whether the remote is reachable
        """
        ...
