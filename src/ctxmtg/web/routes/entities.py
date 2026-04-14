# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Entity Resolution Routes
=========================

Web UI for reviewing and merging duplicate entities. Shows groups
of entities with similar names and lets users merge them (choosing
a canonical name) or dismiss them as distinct.

The resolution page queries entities grouped by normalized name
and presents candidates for manual merge.

Depends on:
    - ctxmtg.web.deps (auth, store access)
    - ctxmtg.storage.sqlite (SQLiteStore)

Used by:
    - ctxmtg.web.app (included as a router)
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ctxmtg.web.deps import get_sql_store, require_auth

logger = structlog.get_logger("ctxmtg.web.routes.entities")

router = APIRouter(tags=["entities"])
templates: Jinja2Templates | None = None


def set_templates(t: Jinja2Templates) -> None:
    global templates
    templates = t


async def _find_duplicate_candidates(sql_store, min_group: int = 2) -> list[dict]:
    """Find entity names with actual name variants or mixed types.

    Only shows groups where there are genuinely different name spellings
    (e.g., "Alice" vs "alice") or mixed entity types that need resolution.
    Single-name entities appearing across multiple interactions are normal
    and not shown.
    """
    # ORIGINAL QUERY (disabled 2026-04-07): showed all groups with
    # COUNT(DISTINCT name) >= 2 OR COUNT(DISTINCT entity_type) >= 2.
    # This surfaced normal per-interaction entities as false merge candidates.
    rows = await sql_store.execute_sql(
        """SELECT LOWER(name) AS norm_name,
                  COUNT(*) AS instance_count,
                  COUNT(DISTINCT interaction_id) AS interaction_count,
                  GROUP_CONCAT(DISTINCT name) AS name_variants,
                  GROUP_CONCAT(DISTINCT entity_type) AS types,
                  GROUP_CONCAT(DISTINCT id) AS entity_ids,
                  MAX(confidence) AS max_confidence
           FROM entities
           GROUP BY LOWER(name)
           HAVING COUNT(DISTINCT name) >= :min_group
           ORDER BY instance_count DESC
           LIMIT 100""",
        {"min_group": min_group},
    )
    groups = []
    for row in rows:
        groups.append({
            "norm_name": row["norm_name"],
            "instance_count": row["instance_count"],
            "interaction_count": row["interaction_count"],
            "variants": row["name_variants"].split(",") if row["name_variants"] else [],
            "types": row["types"].split(",") if row["types"] else [],
            "entity_ids": row["entity_ids"].split(",") if row["entity_ids"] else [],
            "max_confidence": row["max_confidence"],
        })
    return groups


async def _find_similar_names(sql_store) -> list[dict]:
    """Find entities with names that differ only by case or whitespace.

    Uses a self-join on LOWER(TRIM(name)) to find near-duplicates.
    Only shows pairs where the actual name strings differ (not just
    the same entity from different interactions).
    """
    # ORIGINAL QUERY (disabled 2026-04-07): showed all pairs where
    # LOWER(TRIM(name)) matched across interactions, even when names
    # were identical. This surfaced normal per-interaction entities.
    rows = await sql_store.execute_sql(
        """SELECT a.name AS name_a, b.name AS name_b,
                  a.entity_type AS type_a, b.entity_type AS type_b,
                  a.id AS id_a, b.id AS id_b,
                  a.confidence AS conf_a, b.confidence AS conf_b
           FROM entities a
           JOIN entities b ON LOWER(TRIM(a.name)) = LOWER(TRIM(b.name))
                          AND a.name != b.name
                          AND a.id < b.id
                          AND a.interaction_id != b.interaction_id
           GROUP BY LOWER(TRIM(a.name))
           ORDER BY a.name
           LIMIT 200"""
    )
    pairs = []
    for row in rows:
        pairs.append({
            "name_a": row["name_a"],
            "name_b": row["name_b"],
            "type_a": row["type_a"],
            "type_b": row["type_b"],
            "id_a": row["id_a"],
            "id_b": row["id_b"],
            "conf_a": row["conf_a"],
            "conf_b": row["conf_b"],
        })
    return pairs


@router.get("/entities", response_class=HTMLResponse)
async def entities_page(
    request: Request,
    _auth=Depends(require_auth),
    sql_store=Depends(get_sql_store),
):
    """Render the entity resolution page."""
    groups = await _find_duplicate_candidates(sql_store)
    pairs = await _find_similar_names(sql_store)

    # Get total entity stats.
    stats_rows = await sql_store.execute_sql(
        """SELECT COUNT(*) AS total,
                  COUNT(DISTINCT LOWER(name)) AS unique_names,
                  COUNT(DISTINCT entity_type) AS type_count
           FROM entities"""
    )
    stats = stats_rows[0] if stats_rows else {"total": 0, "unique_names": 0, "type_count": 0}

    assert templates is not None
    return templates.TemplateResponse(
        request=request,
        name="entities.html",
        context={
            "groups": groups,
            "pairs": pairs,
            "stats": stats,
        },
    )


@router.post("/entities/merge", response_class=HTMLResponse)
async def merge_entities(
    request: Request,
    canonical_name: str = Form(...),
    entity_ids: str = Form(...),
    _auth=Depends(require_auth),
    sql_store=Depends(get_sql_store),
):
    """Merge selected entities: update all to use the canonical name."""
    ids = [eid.strip() for eid in entity_ids.split(",") if eid.strip()]

    if not ids or not canonical_name.strip():
        return HTMLResponse(
            '<div class="alert alert-error">Invalid merge request.</div>'
        )

    # BUG FIX (2026-04-07): execute_sql does not commit DML.
    # Also: update by NAME (all instances across interactions), not just by ID.
    db = sql_store._ensure_db()
    merged_count = 0

    # Get distinct names from the provided IDs
    names_to_merge = set()
    for eid in ids:
        name_rows = await sql_store.execute_sql(
            "SELECT name FROM entities WHERE id = :id", {"id": eid}
        )
        if name_rows:
            names_to_merge.add(name_rows[0]["name"])

    # Update ALL entities with those names to the canonical name
    for old_name in names_to_merge:
        if old_name == canonical_name.strip():
            continue
        cursor = await db.execute(
            "UPDATE entities SET name = :new WHERE LOWER(name) = LOWER(:old)",
            {"new": canonical_name.strip(), "old": old_name},
        )
        merged_count += cursor.rowcount
    await db.commit()

    logger.info(
        "entities_merged",
        canonical_name=canonical_name,
        merged_count=merged_count,
        names_merged=list(names_to_merge),
    )

    return HTMLResponse(
        f'<tr><td colspan="7" class="alert alert-success">'
        f'Merged {merged_count} entities to "{canonical_name}".'
        f'</td></tr>'
    )


@router.post("/entities/dismiss", response_class=HTMLResponse)
async def dismiss_group(
    request: Request,
    norm_name: str = Form(...),
    _auth=Depends(require_auth),
):
    """Dismiss a group as not duplicates (no-op, just removes from view)."""
    logger.info("entity_group_dismissed", norm_name=norm_name)
    return HTMLResponse(
        '<div class="alert alert-success">Group dismissed.</div>'
    )


@router.post("/entities/delete", response_class=HTMLResponse)
async def delete_entities(
    request: Request,
    entity_ids: str = Form(...),
    _auth=Depends(require_auth),
    sql_store=Depends(get_sql_store),
):
    """Delete selected entities and their related facts, insights, and distiller summaries."""
    ids = [eid.strip() for eid in entity_ids.split(",") if eid.strip()]

    if not ids:
        return HTMLResponse(
            '<div class="alert alert-error">No entities selected for deletion.</div>'
        )

    deleted_entities = 0
    db = sql_store._ensure_db()

    # Resolve IDs to names, then delete ALL instances by name
    names_to_delete = set()
    for eid in ids:
        name_rows = await sql_store.execute_sql(
            "SELECT name FROM entities WHERE id = :id", {"id": eid}
        )
        if name_rows:
            names_to_delete.add(name_rows[0]["name"])

    for entity_name in names_to_delete:
        try:
            # Get all IDs for this name to clean up facts
            id_rows = await sql_store.execute_sql(
                "SELECT id FROM entities WHERE LOWER(name) = LOWER(:name)",
                {"name": entity_name},
            )
            for row in id_rows:
                await db.execute(
                    "DELETE FROM facts WHERE subject_entity_id = :id OR object_entity_id = :id",
                    {"id": row["id"]},
                )

            # Delete all entities with this name
            cursor = await db.execute(
                "DELETE FROM entities WHERE LOWER(name) = LOWER(:name)",
                {"name": entity_name},
            )
            deleted_entities += cursor.rowcount

            # Clean up insights and distiller
            await db.execute(
                "DELETE FROM meta_insights WHERE title LIKE :pattern",
                {"pattern": f"%{entity_name}%"},
            )
            await db.execute(
                "DELETE FROM distiller_summaries WHERE entity_name = :name",
                {"name": entity_name},
            )

        except Exception as exc:
            logger.warning("entity_delete_failed", entity_id=eid, error=str(exc))

    await db.commit()

    logger.info(
        "entities_deleted",
        deleted_entities=deleted_entities,
        names=list(names_to_delete),
    )

    return HTMLResponse(
        f'<tr><td colspan="7" class="alert alert-success">'
        f'Deleted {deleted_entities} entities ({len(names_to_delete)} names) and related data.'
        f'</td></tr>'
    )


@router.post("/entities/merge-batch", response_class=HTMLResponse)
async def merge_batch(
    request: Request,
    _auth=Depends(require_auth),
    sql_store=Depends(get_sql_store),
):
    """Batch merge: update ALL entities matching the names in each checked group
    to use the chosen canonical name."""
    form = await request.form()
    logger.info("merge_batch_form_received", form_keys=list(form.keys()), form_items={k: v for k, v in form.items()})
    merge_ids_list = form.getlist("merge_ids")
    logger.info("merge_batch_ids", merge_ids_list=merge_ids_list)

    if not merge_ids_list:
        return HTMLResponse('<div class="alert alert-error">No rows selected for merge.</div>')

    db = sql_store._ensure_db()
    merged_total = 0

    # Build map: for each checked row, find the original names from the entity IDs,
    # then update ALL entities with those names (not just the specific IDs).
    for idx, ids_str in enumerate(merge_ids_list):
        ids = [eid.strip() for eid in ids_str.split(",") if eid.strip()]
        canonical = form.get(f"canonical_{idx}", "").strip()
        if not canonical or not ids:
            continue

        # Get the distinct names from these entity IDs
        for eid in ids:
            name_rows = await sql_store.execute_sql(
                "SELECT DISTINCT name FROM entities WHERE id = :id", {"id": eid}
            )
            for row in name_rows:
                old_name = row["name"]
                if old_name == canonical:
                    continue
                # Update ALL entities with this name, not just this ID
                cursor = await db.execute(
                    "UPDATE entities SET name = :new WHERE LOWER(name) = LOWER(:old)",
                    {"new": canonical, "old": old_name},
                )
                merged_total += cursor.rowcount

    await db.commit()
    logger.info("batch_merge_complete", merged_total=merged_total)
    return HTMLResponse(
        f'<div class="alert alert-success">Merged {merged_total} entities. Refresh to see changes.</div>'
    )


@router.post("/entities/delete-batch", response_class=HTMLResponse)
async def delete_batch(
    request: Request,
    _auth=Depends(require_auth),
    sql_store=Depends(get_sql_store),
):
    """Batch delete: delete ALL entities matching the names in each checked group,
    plus related facts, insights, and distiller summaries."""
    form = await request.form()
    logger.info("delete_batch_form_received", form_keys=list(form.keys()), form_items={k: v for k, v in form.items()})
    delete_ids_list = form.getlist("delete_ids")
    logger.info("delete_batch_ids", delete_ids_list=delete_ids_list)

    if not delete_ids_list:
        return HTMLResponse('<div class="alert alert-error">No rows selected for deletion.</div>')

    db = sql_store._ensure_db()
    deleted_total = 0

    # Collect all entity IDs, then resolve to names, then delete by name
    all_ids = []
    for ids_str in delete_ids_list:
        all_ids.extend([eid.strip() for eid in ids_str.split(",") if eid.strip()])

    # Get distinct names from the selected IDs
    names_to_delete = set()
    for eid in all_ids:
        name_rows = await sql_store.execute_sql(
            "SELECT name FROM entities WHERE id = :id", {"id": eid}
        )
        if name_rows:
            names_to_delete.add(name_rows[0]["name"])

    # Delete ALL instances of each name (across all interactions)
    for entity_name in names_to_delete:
        try:
            # Get all IDs for this name (to clean up facts by FK)
            id_rows = await sql_store.execute_sql(
                "SELECT id FROM entities WHERE LOWER(name) = LOWER(:name)",
                {"name": entity_name},
            )
            for row in id_rows:
                await db.execute(
                    "DELETE FROM facts WHERE subject_entity_id = :id OR object_entity_id = :id",
                    {"id": row["id"]},
                )

            # Delete all entities with this name
            cursor = await db.execute(
                "DELETE FROM entities WHERE LOWER(name) = LOWER(:name)",
                {"name": entity_name},
            )
            deleted_total += cursor.rowcount

            # Clean up insights and distiller
            await db.execute(
                "DELETE FROM meta_insights WHERE title LIKE :pattern",
                {"pattern": f"%{entity_name}%"},
            )
            await db.execute(
                "DELETE FROM distiller_summaries WHERE entity_name = :name",
                {"name": entity_name},
            )
        except Exception as exc:
            logger.warning("batch_delete_failed", name=entity_name, error=str(exc))

    await db.commit()
    logger.info("batch_delete_complete", deleted_total=deleted_total, names=len(names_to_delete))
    return HTMLResponse(
        f'<div class="alert alert-success">Deleted {deleted_total} entities ({len(names_to_delete)} names) and related data. Refresh to see changes.</div>'
    )


@router.get("/entities/list", response_class=HTMLResponse)
async def entity_list_fragment(
    request: Request,
    _auth=Depends(require_auth),
    sql_store=Depends(get_sql_store),
):
    """htmx fragment: refreshable entity duplicate list."""
    groups = await _find_duplicate_candidates(sql_store)

    assert templates is not None
    return templates.TemplateResponse(
        request=request,
        name="fragments/entity_groups.html",
        context={"groups": groups},
    )
