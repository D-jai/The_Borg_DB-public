# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Rationalizer Maintenance Stage
================================

Scans entities in progressive batches and marks garbage entities by
setting their confidence to 0.1.  This is a non-destructive, reversible
operation -- downstream stages (particularly the Archivist) decide
whether to archive or delete low-confidence entities.

Garbage detection rules (regex / heuristic):
    - Embedded newlines or carriage returns in name
    - URL fragments (http://, www., ://)
    - Markdown link artifacts ( ]( , [, <http )
    - Bare decimal numbers that look like phone fragments (1.855.4)
    - Truncation markers (exact match "TRUNCATED")
    - Pure punctuation / symbols (* - --- *** etc.)
    - Names shorter than 2 characters after stripping whitespace
    - Excessive internal whitespace (3+ consecutive spaces)

The stage does NOT downgrade entities that have protected facts
(responsible_for, leads, reports_to, decided, committed_to) since
these represent real relationships even if the entity name is messy.

Depends on:
    - ctxmtg.interfaces.farming (FarmingStage ABC, FarmingContext)
    - ctxmtg.interfaces.storage (SQLStore, VectorStore)
    - ctxmtg.models.farming (FarmingInsight output model)
    - ctxmtg.farming.checkpoint (_run_async helper for sync->async bridge)
    - ctxmtg.farming.progress (progressive offset scanning)
    - structlog (structured logging)

Used by:
    - ctxmtg.farming.__init__ (registered as first maintenance stage)
"""

from __future__ import annotations

import json
import re
from uuid import uuid4

import structlog

from ctxmtg.farming.checkpoint import _run_async
from ctxmtg.farming.progress import get_offset_with_wrap, update_offset
from ctxmtg.interfaces.farming import FarmingContext, FarmingStage
from ctxmtg.interfaces.storage import SQLStore, VectorStore
from ctxmtg.models.farming import FarmingInsight

logger = structlog.get_logger("ctxmtg.farming.rationalizer")

# -------------------------------------------------------------------
# Garbage detection configuration
# -------------------------------------------------------------------
GARBAGE_CONFIDENCE = 0.1
BATCH_SIZE = 100

# Predicates that indicate real relationships -- entities with these
# facts are protected even if their name looks garbage-like.
PROTECTED_PREDICATES = frozenset({
    "responsible_for",
    "leads",
    "reports_to",
    "decided",
    "committed_to",
})

# Compiled regex patterns for garbage detection
_GARBAGE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[\n\r]"), "embedded_newline"),
    (re.compile(r"https?://|www\.|://"), "url_fragment"),
    (re.compile(r"\]\(|<http|\[http"), "markdown_link"),
    (re.compile(r"^\d{1,3}\.\d{1,4}(\.\d{1,4})?$"), "bare_decimal"),
    (re.compile(r"^\s*[*\-=_]{1,5}(\s+[*\-=_]{1,5})*\s*$"), "pure_punctuation"),
    (re.compile(r"\s{3,}"), "excessive_whitespace"),
]

# Exact-match names that are always garbage
_GARBAGE_EXACT: frozenset[str] = frozenset({
    "TRUNCATED",
    "truncated",
    "Additional comments",
    "AdditionalAdditional comments",
})


def _is_garbage(name: str) -> str | None:
    """
    Test whether an entity name is garbage.

    Returns:
        The rule name that matched, or None if the entity is clean.
    """
    stripped = name.strip()

    if len(stripped) < 2:
        return "too_short"

    if stripped in _GARBAGE_EXACT:
        return "exact_match"

    for pattern, rule_name in _GARBAGE_PATTERNS:
        if pattern.search(name):
            return rule_name

    return None


class RationalizerStage(FarmingStage):
    """
    Maintenance stage that marks garbage entities with low confidence.

    Scans entities progressively (BATCH_SIZE per cycle) and sets
    confidence = 0.1 on entities whose names match garbage detection
    rules.  The Archivist stage (which runs later in the same cycle)
    can then archive these entities regardless of age.
    """

    def get_name(self) -> str:
        return "rationalizer"

    def run(
        self,
        sql_store: SQLStore,
        vector_store: VectorStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        return _run_async(self._run_impl(sql_store, context))

    async def _run_impl(
        self,
        sql_store: SQLStore,
        context: FarmingContext,
    ) -> list[FarmingInsight]:
        # Get total entity count for offset wrapping
        total_rows = await sql_store.execute_sql(
            "SELECT COUNT(*) as cnt FROM entities WHERE confidence > :threshold",
            {"threshold": GARBAGE_CONFIDENCE},
        )
        total_count = total_rows[0]["cnt"] if total_rows else 0

        offset = await get_offset_with_wrap(
            sql_store, "rationalizer", total_count, BATCH_SIZE,
        )

        # Fetch batch of entities not already marked as garbage
        rows = await sql_store.execute_sql(
            "SELECT id, name FROM entities "
            "WHERE confidence > :threshold "
            "ORDER BY created_at ASC "
            "LIMIT :limit OFFSET :offset",
            {"threshold": GARBAGE_CONFIDENCE, "limit": BATCH_SIZE, "offset": offset},
        )

        if not rows:
            logger.info("rationalizer_no_entities", cycle_id=context.cycle_id)
            return []

        logger.info(
            "rationalizer_start",
            cycle_id=context.cycle_id,
            batch_size=len(rows),
            offset=offset,
        )

        # Check each entity against garbage rules
        garbage_ids: list[str] = []
        garbage_details: list[dict] = []

        for row in rows:
            entity_id = row["id"]
            name = row["name"]
            rule = _is_garbage(name)
            if rule is None:
                continue

            # Check for protected facts before downgrading
            protected = await sql_store.execute_sql(
                "SELECT COUNT(*) as cnt FROM facts "
                "WHERE subject_entity_id = :eid "
                "AND predicate IN ({})".format(
                    ", ".join(f"'{p}'" for p in PROTECTED_PREDICATES)
                ),
                {"eid": entity_id},
            )
            if protected and protected[0]["cnt"] > 0:
                logger.debug(
                    "rationalizer_protected",
                    entity_id=entity_id,
                    name=name,
                    rule=rule,
                )
                continue

            garbage_ids.append(entity_id)
            garbage_details.append({
                "id": entity_id,
                "name": name,
                "rule": rule,
            })

        # Mark garbage entities with low confidence
        if garbage_ids:
            db = sql_store._ensure_db()  # type: ignore[attr-defined]
            for eid in garbage_ids:
                await sql_store.execute_sql(
                    "UPDATE entities SET confidence = :conf WHERE id = :eid",
                    {"conf": GARBAGE_CONFIDENCE, "eid": eid},
                )
            await db.commit()

            # Log to maintenance table
            log_id = str(uuid4())
            await sql_store.execute_sql(
                "INSERT INTO maintenance_rationalizer "
                "(id, cycle_id, action, target_ids, detail) "
                "VALUES (:id, :cycle, 'marked_garbage', :targets, :detail)",
                {
                    "id": log_id,
                    "cycle": context.cycle_id,
                    "targets": json.dumps(garbage_ids),
                    "detail": json.dumps(garbage_details),
                },
            )
            await db.commit()

        await update_offset(
            sql_store, "rationalizer", offset + BATCH_SIZE, len(rows),
        )

        logger.info(
            "rationalizer_complete",
            cycle_id=context.cycle_id,
            scanned=len(rows),
            garbage_marked=len(garbage_ids),
            offset=offset,
        )

        if not garbage_ids:
            return []

        # Build summary insight
        rule_counts: dict[str, int] = {}
        for d in garbage_details:
            rule_counts[d["rule"]] = rule_counts.get(d["rule"], 0) + 1

        insight = FarmingInsight(
            id=str(uuid4()),
            insight_type="verification",
            title=f"{len(garbage_ids)} garbage entities marked (confidence → {GARBAGE_CONFIDENCE})",
            description=(
                f"Scanned {len(rows)} entities, marked {len(garbage_ids)} as garbage. "
                f"Rules triggered: {rule_counts}"
            ),
            evidence=garbage_ids,
            confidence=1.0,
            parameters={
                "scanned": len(rows),
                "garbage_count": len(garbage_ids),
                "rule_counts": rule_counts,
                "garbage_details": garbage_details[:20],  # cap for storage
            },
        )

        return [insight]
