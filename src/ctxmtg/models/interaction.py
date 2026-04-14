# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Interaction Data Models
=======================

This module defines the core data models for the ctxmtg knowledge system:
interactions (meetings, emails, documents), entities (people, places,
topics), facts (subject-predicate-object triples), embedding metadata,
and extraction results.

These models are the lingua franca of the system -- every module reads
and writes data using these structures. They flow from ingestion through
extraction, into storage, and back out through queries.

In the system architecture, these models sit at the center. The extraction
pipeline produces them, the storage layer persists them, the query engine
retrieves them, and the farming pipeline mines them for patterns.

Depends on:
    - pydantic (validation, serialization, field constraints)
    - datetime (timestamp fields)
    - enum (source types, entity types, intake actions)

Used by:
    - ctxmtg.extraction.pipeline (produces Entity, Fact, ExtractionResult)
    - ctxmtg.storage.sqlite (stores and retrieves all models)
    - ctxmtg.query.executor (returns SearchResult containing these models)
    - ctxmtg.farming.pipeline (reads entities and facts for pattern mining)
    - ctxmtg.ingestion.worker (orchestrates creation of Interaction objects)
    - ctxmtg.intake.rules (classifies interactions using IntakeAction)
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------
# SourceType enum: identifies where an interaction came from.
# Each value maps to a different ingestion loader and may trigger
# different extraction strategies (e.g., calendar/contact bypass NLP).
# ---------------------------------------------------------------
class SourceType(str, Enum):
    """
    Types of interaction sources the system can ingest.

    Each source type may have a dedicated file loader (e.g., .eml for EMAIL,
    .ics for CALENDAR) and may use different extraction strategies. For example,
    CALENDAR and CONTACT sources bypass NLP extraction entirely because their
    data is already structured.
    """

    SLACK = "slack"          # Slack messages and threads
    EMAIL = "email"          # Email messages (.eml files)
    DOC = "doc"              # Document snippets (policy docs, specs)
    MEETING = "meeting"      # Meeting transcripts
    CHAT = "chat"            # Generic chat messages
    CALENDAR = "calendar"    # .ics events -- bypass NLP, direct entity/fact creation
    CONTACT = "contact"      # .vcf contacts -- bypass NLP, direct entity creation
    OTHER = "other"          # Catch-all for unrecognized sources


# ---------------------------------------------------------------
# EntityType enum: the kinds of entities the NER pipeline can find.
# These map to spaCy NER labels (PERSON, ORG, GPE, etc.) plus
# domain-specific types (PROJECT, TOPIC, TOOL, EVENT).
# ---------------------------------------------------------------
class EntityType(str, Enum):
    """
    Types of entities the NER pipeline can extract.

    These cover both standard NLP entity types (PERSON, ORG, LOCATION)
    and domain-specific types (PROJECT, TOPIC, TOOL, EVENT) that are
    useful for knowledge management contexts.
    """

    PERSON = "person"        # People (names, roles)
    ORG = "org"              # Organizations (companies, teams)
    PROJECT = "project"      # Projects (codenames, product names)
    TOPIC = "topic"          # Topics (concepts, themes)
    TOOL = "tool"            # Tools (software, hardware)
    LOCATION = "location"    # Locations (cities, offices, rooms)
    EVENT = "event"          # Events (meetings, milestones, deadlines)
    OTHER = "other"          # Catch-all for unrecognized entity types


# ---------------------------------------------------------------
# IntakeAction enum: the Traffic Cop's classification of inbound data.
# This determines whether an interaction proceeds to extraction (ACCEPT),
# is queued for later (DEFER), discarded (REJECT), or sent elsewhere (ROUTE).
# ---------------------------------------------------------------
class IntakeAction(str, Enum):
    """
    Traffic Cop classification of inbound data.

    The intake gateway (Traffic Cop) classifies every incoming interaction
    into one of these actions before any extraction happens. Only ACCEPT
    interactions proceed to the extraction pipeline.
    """

    ACCEPT = "accept"        # Process immediately through extraction
    DEFER = "defer"          # Queue for later processing
    REJECT = "reject"        # Discard (log reason but don't extract)
    ROUTE = "route"          # Forward to another system or profile


# ---------------------------------------------------------------
# Interaction model: the fundamental unit of data in ctxmtg.
# Every piece of content the system ingests becomes an Interaction.
# This is the starting point for the entire extraction pipeline.
# ---------------------------------------------------------------
class Interaction(BaseModel):
    """
    A single interaction (meeting, email, document, etc.).

    This is the fundamental data unit in ctxmtg. Every piece of content
    the system ingests -- whether it's a meeting transcript, an email,
    a Slack message, or a calendar event -- becomes an Interaction object.

    The Interaction flows through the system:
    1. Ingestion creates it from raw input
    2. Traffic Cop classifies it (intake_action)
    3. Extraction pipeline processes it → entities, facts, summary
    4. Storage persists it and its extracted data
    5. Query engine retrieves it when users ask questions
    """

    # Unique identifier for this interaction (UUIDv5, content-derived)
    id: str

    # What kind of source produced this interaction
    source_type: SourceType

    # Optional external ID from the originating system (e.g., Slack message ID)
    source_id: str | None = None

    # Human-readable title (e.g., email subject, meeting name)
    title: str | None = None

    # The actual text content of the interaction
    content: str

    # List of participant names (people involved in this interaction)
    participants: list[str] = Field(default_factory=list)

    # Arbitrary metadata from the source system (e.g., channel name, labels)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Which instance created this interaction (for future hive sync)
    source_instance: str = "local"

    # Traffic Cop's classification decision (future: intake gateway sets this)
    intake_action: IntakeAction = IntakeAction.ACCEPT

    # When the original interaction occurred
    created_at: datetime

    # When this record was last modified (e.g., re-extracted)
    updated_at: datetime | None = None

    # When this record was last synced to the hive (None = not yet synced).
    # Set by HiveSyncWorker after a successful push to the hive database.
    # See research/notes/hive-sync-design.md for the two-phase sync design.
    hive_synced_at: datetime | None = None


# ---------------------------------------------------------------
# Entity model: a named entity extracted from an interaction.
# Entities are the building blocks of knowledge -- people, places,
# topics, tools, etc. that appear in the user's interactions.
# ---------------------------------------------------------------
class Entity(BaseModel):
    """
    An entity extracted from an interaction.

    Entities represent real-world objects (people, organizations, projects,
    topics, etc.) mentioned in interactions. Each entity belongs to exactly
    one interaction -- the same real-world person appearing in two meetings
    produces two separate Entity records with different IDs. This is
    intentional: entity merging is deferred to the Consolidator (Phase 3).

    The 'context' dict provides rich unstructured context for vector embedding
    (semantic search on entities), while 'tags' provides structured key-value
    pairs for SQL filtering (precise lookups).
    """

    # Unique identifier (UUIDv5, derived from interaction_id + name + type)
    id: str

    # Which interaction this entity was extracted from
    interaction_id: str

    # The entity's canonical name (e.g., "Alice", "OAuth2", "New York")
    name: str

    # Classification of what kind of entity this is
    entity_type: EntityType

    # Alternative names or spellings (e.g., ["Al", "Alice Smith"])
    aliases: list[str] = Field(default_factory=list)

    # How confident the extractor is in this entity (0.0 to 1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    # Which extraction system found this (e.g., "spacy:en_core_web_sm:3.7")
    provenance: str | None = None

    # Rich unstructured context for vector embedding.
    # Example: {"summary": "proposed OAuth2 migration, backend team",
    #           "co_entities": ["Bob", "OAuth2"], "role": "tech lead"}.
    # Max 500 chars when serialized to JSON string.
    context: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Rich unstructured context for vector embedding. "
            "E.g. {'summary': 'proposed OAuth2 migration, backend team', "
            "'co_entities': ['Bob', 'OAuth2'], 'role': 'tech lead'}. "
            "Max 500 chars when serialized."
        ),
    )

    # Structured key-value pairs for SQL filtering.
    # Example: {"department": "engineering", "team": "backend"}.
    # Max 20 key-value pairs.
    tags: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Structured key-value pairs for SQL filtering. "
            "E.g. {'department': 'engineering', 'team': 'backend'}. "
            "Max 20 key-value pairs."
        ),
    )

    # Which instance created this entity (for future hive sync)
    source_instance: str = "local"

    # When this entity record was created
    created_at: datetime | None = None

    # When this entity was last synced to the hive (None = not yet synced).
    # Set by HiveSyncWorker after a successful push to the hive database.
    hive_synced_at: datetime | None = None

    # ---------------------------------------------------------------
    # Validator: ensure serialized context dict does not exceed 500 chars.
    # This limit prevents bloated embeddings and keeps storage manageable.
    # ---------------------------------------------------------------
    @field_validator("context")
    @classmethod
    def validate_context_size(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure serialized context does not exceed 500 characters."""
        serialized = json.dumps(v)
        if len(serialized) > 500:
            raise ValueError(
                f"Context dict exceeds 500 chars when serialized "
                f"(got {len(serialized)} chars)"
            )
        return v

    # ---------------------------------------------------------------
    # Validator: ensure tags dict has at most 20 key-value pairs.
    # This limit prevents excessive metadata that would slow SQL queries.
    # ---------------------------------------------------------------
    @field_validator("tags")
    @classmethod
    def validate_tags_count(cls, v: dict[str, str]) -> dict[str, str]:
        """Ensure tags dict has at most 20 key-value pairs."""
        if len(v) > 20:
            raise ValueError(
                f"Tags dict exceeds 20 key-value pairs (got {len(v)} pairs)"
            )
        return v


# ---------------------------------------------------------------
# Fact model: a subject-predicate-object triple extracted from text.
# Facts are the structured knowledge atoms -- "Alice proposed OAuth2",
# "deadline is April 15", "Charlie leads implementation".
# ---------------------------------------------------------------
class Fact(BaseModel):
    """
    A subject-predicate-object triple extracted from an interaction.

    Facts represent structured knowledge in the form of triples:
    - Subject: an entity (referenced by ID)
    - Predicate: a verb or relationship (e.g., "proposed", "leads")
    - Object: another entity (by ID) or a literal string

    Facts can be superseded when newer information contradicts them.
    The Pruner micro-agent (Phase 3) sets superseded_by to point to
    the replacement fact.
    """

    # Unique identifier (UUIDv5, derived from subject + predicate + object)
    id: str

    # Which interaction this fact was extracted from
    interaction_id: str

    # The entity that is the subject of this fact
    subject_entity_id: str

    # The relationship or action (e.g., "proposed", "leads", "scheduled_at")
    predicate: str

    # The entity that is the object (if the object is an entity)
    object_entity_id: str | None = None

    # The literal string object (if the object is not an entity)
    object_literal: str | None = None

    # How confident the extractor is in this fact (0.0 to 1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    # The original text span this fact was extracted from
    source_span: str | None = None

    # Which instance created this fact (inherited from interaction, future: hive sync)
    source_instance: str = "local"

    # If this fact has been replaced by a newer one, this points to it.
    # The Pruner micro-agent (Phase 3) sets this field.
    superseded_by: str | None = None

    # When this fact record was created
    created_at: datetime | None = None

    # When this fact was last synced to the hive (None = not yet synced).
    # Set by HiveSyncWorker after a successful push to the hive database.
    hive_synced_at: datetime | None = None


# ---------------------------------------------------------------
# EmbeddingMetadata model: links SQL records to their vector
# representations in the vector store. This is the bridge between
# the structured (SQL) and semantic (vector) halves of the dual-store.
# ---------------------------------------------------------------
class EmbeddingMetadata(BaseModel):
    """
    Metadata about a stored embedding (links SQL to vector store).

    When text is embedded into vectors for semantic search, this metadata
    record tracks which SQL record the embedding came from, which chunk
    of the text was embedded, and which model produced the embedding.
    This allows the system to re-embed content when models change and
    to trace vector search results back to their source records.
    """

    # Unique identifier for this embedding metadata record
    id: str

    # Which SQL table the source record lives in (e.g., "interactions", "entities")
    source_table: str

    # The ID of the source record in the SQL table
    source_id: str

    # Which chunk of the source text this embedding covers (0 = first/only)
    chunk_index: int = 0

    # Character offset where this chunk starts in the source text
    chunk_start: int | None = None

    # Character offset where this chunk ends in the source text
    chunk_end: int | None = None

    # Name of the embedding model (e.g., "all-MiniLM-L6-v2")
    model_name: str

    # Version of the embedding model
    model_version: str

    # Dimensionality of the embedding vector (e.g., 384 for MiniLM)
    dimensions: int

    # When this embedding was created
    created_at: datetime | None = None


# ---------------------------------------------------------------
# ExtractionResult model: the complete output of running the
# extraction pipeline on one interaction. Bundles together all
# the entities, facts, summary, and text chunks that were extracted.
# ---------------------------------------------------------------
class ExtractionResult(BaseModel):
    """
    Complete output of the extraction pipeline for one interaction.

    After the extraction pipeline processes an Interaction, it produces
    this result containing all extracted entities, facts, an optional
    summary, and text chunks ready for embedding. This is the handoff
    between extraction and storage -- the ingestion worker takes this
    result and persists everything to the dual stores.
    """

    # Which interaction this result came from
    interaction_id: str

    # All entities extracted from the interaction
    entities: list[Entity]

    # All facts (subject-predicate-object triples) extracted
    facts: list[Fact]

    # An optional summary of the interaction content
    summary: str | None = None

    # Text chunks ready for embedding (split from the original content)
    chunks: list[str] = Field(default_factory=list)
