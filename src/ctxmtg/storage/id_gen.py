# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
UUIDv5 Deterministic ID Generation
====================================

This module generates deterministic IDs for every row stored in the
SQL database. All IDs are UUIDv5 strings computed from a project-
specific namespace UUID and a content-derived name string.

Why UUIDv5 instead of random UUIDs (v4)?
    - **Deterministic:** Processing the same content twice produces
      the same ID → idempotent writes and natural deduplication.
    - **Cross-node consistency:** Two edge devices ingesting the
      same Slack message generate identical IDs.
    - **Collision-safe:** UUIDv5 with SHA-1 gives effectively zero
      collision risk for our scale (< 10 M records).
    (See research/round-2/03-unified-schema-design.md § 3.1)

ID strategies differ per table:
    - **interactions** — hash source_type + content → same document
      re-ingested gets the same ID (idempotent).
    - **entities** — hash interaction_id + name + entity_type →
      per-interaction IDs. The same real-world person mentioned
      in two different meetings gets DIFFERENT entity IDs.
      Entity merging is deferred to the Consolidator (Phase 3).
    - **facts** — hash subject_entity_id + predicate + object_value
      → deterministic per triple.
    - **embeddings** — hash source_id + chunk_index → deterministic
      per text chunk.
    - **insights** — hash insight_type + title → deterministic per
      farming discovery.

Depends on:
    - uuid (Python stdlib — UUID generation)
    - hashlib (Python stdlib — SHA-256 content hashing)

Used by:
    - ctxmtg.storage.sqlite (generates IDs before INSERT)
    - ctxmtg.ingestion.worker (pre-assigns IDs during ingestion)
    - tests/test_storage/test_id_gen.py (verifies determinism)
"""

from __future__ import annotations

import hashlib
import uuid

# =====================================================================
# Project namespace UUID — the fixed "salt" for all UUIDv5 generation.
# Generated once and stored here as a constant. Changing this would
# invalidate every existing ID in every database, so treat it as
# immutable for the lifetime of the project.
# (Matches the constant in research/round-2/03-unified-schema-design.md)
# =====================================================================
CTXMTG_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


# =====================================================================
# Per-table ID generators
# =====================================================================


def generate_interaction_id(source_type: str, content: str) -> str:
    """
    Generate a deterministic ID for an interaction.

    The ID is derived from the source type and a SHA-256 hash of the
    content. This means re-ingesting the exact same document produces
    the same ID, enabling idempotent writes (INSERT OR IGNORE).

    Args:
        source_type: The interaction source type (e.g., "meeting", "email").
        content:     The full text content of the interaction.

    Returns:
        A UUIDv5 string (e.g., "3fa85f64-5717-4562-b3fc-2c963f66afa6").
    """
    # Hash the content with SHA-256 first to normalise length.
    # Two identical documents always produce the same hash regardless
    # of encoding details.
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Combine source type and content hash into the UUIDv5 name string
    name = f"{source_type}:{content_hash}"
    return str(uuid.uuid5(CTXMTG_NAMESPACE, name))


def generate_entity_id(interaction_id: str, name: str, entity_type: str) -> str:
    """
    Generate a deterministic ID for an entity within an interaction.

    IMPORTANT: Entity IDs are per-interaction, NOT globally unique per
    real-world entity. The same person "Alice" appearing in two separate
    meetings will have two different entity IDs. This is intentional —
    entity merging across interactions is deferred to the Consolidator
    micro-agent (Phase 3). Queries find entities by name (COLLATE
    NOCASE), not by entity ID.

    Args:
        interaction_id: The ID of the parent interaction.
        name:           The entity's canonical name (e.g., "Alice").
        entity_type:    The entity type string (e.g., "person").

    Returns:
        A UUIDv5 string unique within this interaction.
    """
    # Lower-case + strip the name so "Alice" and " alice " produce
    # the same ID within the same interaction.
    normalised_name = name.lower().strip()
    name_str = f"entity:{interaction_id}:{entity_type}:{normalised_name}"
    return str(uuid.uuid5(CTXMTG_NAMESPACE, name_str))


def generate_fact_id(
    subject_entity_id: str, predicate: str, object_value: str
) -> str:
    """
    Generate a deterministic ID for a fact triple.

    The object_value should be either the object_entity_id or the
    object_literal, whichever is populated. Deterministic so that
    re-extracting the same fact from the same text is idempotent.

    Args:
        subject_entity_id: The ID of the subject entity.
        predicate:         The verb or relationship string.
        object_value:      The object entity ID or literal text.

    Returns:
        A UUIDv5 string unique to this subject-predicate-object triple.
    """
    name = f"fact:{subject_entity_id}:{predicate}:{object_value}"
    return str(uuid.uuid5(CTXMTG_NAMESPACE, name))


def generate_embedding_id(source_id: str, chunk_index: int) -> str:
    """
    Generate a deterministic ID for an embedding chunk.

    Each chunk of a source record gets its own embedding and its own
    metadata row. The ID is derived from the source record's ID and
    the chunk index so that re-embedding the same text produces the
    same metadata ID.

    Args:
        source_id:   The ID of the source SQL record (interaction or entity).
        chunk_index: The 0-based index of the chunk within the source.

    Returns:
        A UUIDv5 string unique to this source + chunk combination.
    """
    name = f"emb:{source_id}:{chunk_index}"
    return str(uuid.uuid5(CTXMTG_NAMESPACE, name))


def generate_insight_id(insight_type: str, title: str) -> str:
    """
    Generate a deterministic ID for a farming insight.

    Using the type + title as the name means the same discovery
    (e.g., "OAuth2 mentions increasing") always gets the same ID,
    allowing the farming pipeline to update rather than duplicate
    insights across runs.

    Args:
        insight_type: The insight category (e.g., "trend", "cluster").
        title:        Human-readable title of the insight.

    Returns:
        A UUIDv5 string unique to this type + title combination.
    """
    name = f"insight:{insight_type}:{title}"
    return str(uuid.uuid5(CTXMTG_NAMESPACE, name))
