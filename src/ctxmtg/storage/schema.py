# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Schema DDL and Migration Logic
===============================

This module holds every CREATE TABLE / CREATE INDEX / CREATE TRIGGER
statement for the SQLite edge database as string constants, plus a
lightweight migration function driven by SQLite's user_version pragma.

The schema design comes from the unified schema research document
(research/round-2/03-unified-schema-design.md) with the following
additions specified in the Phase 1 plan:

    - interactions: source_instance, intake_action columns
    - entities:     context (JSON), tags (JSON), source_instance columns
    - facts:        source_instance column
    - source_type CHECK constraint includes 'calendar' and 'contact'

The migration system is intentionally simple: a monotonically
increasing integer (user_version) stored in the SQLite header.
At startup the code reads the current version and applies any
pending migration steps in order.

Depends on:
    - aiosqlite (async database connection for apply_schema / migrate)

Used by:
    - ctxmtg.storage.sqlite (calls apply_schema on initialise)
    - tests/test_storage/test_schema.py (validates DDL + migrations)
"""

from __future__ import annotations

import aiosqlite

# =====================================================================
# Schema version: bumped every time a migration is added.
# The very first schema creation sets user_version to this value.
# =====================================================================
SCHEMA_VERSION = 5

# =====================================================================
# PRAGMA statements -- applied on every connection (not just first run).
# WAL mode enables concurrent readers + single writer without blocking.
# Foreign keys must be re-enabled per-connection in SQLite.
# busy_timeout gives the writer up to 5 s to acquire the lock before
# raising SQLITE_BUSY (see countermeasures 2.2).
# =====================================================================
PRAGMAS = """\
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
"""

# =====================================================================
# DDL Constants -- one string per table (plus indexes and triggers).
# Using IF NOT EXISTS everywhere so the DDL is idempotent.
# =====================================================================

# -----------------------------------------------------------------
# interactions: the fundamental data unit in ctxmtg.
# Every ingested piece of content becomes one row here.
# -----------------------------------------------------------------
CREATE_INTERACTIONS = """\
CREATE TABLE IF NOT EXISTS interactions (
    id              TEXT PRIMARY KEY,                    -- UUIDv5
    source_type     TEXT NOT NULL CHECK(source_type IN (
        'slack','email','doc','meeting','chat','calendar','contact','other'
    )),
    source_id       TEXT,                                -- external system identifier
    title           TEXT,
    content         TEXT NOT NULL,
    participants    TEXT DEFAULT '[]',                    -- JSON array
    metadata        TEXT DEFAULT '{}',                    -- JSON object
    source_instance TEXT NOT NULL DEFAULT 'local',       -- which instance created this (hive sync)
    intake_action   TEXT NOT NULL DEFAULT 'accept' CHECK(intake_action IN (
        'accept','defer','reject','route'
    )),
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    is_deleted      INTEGER NOT NULL DEFAULT 0,
    hive_synced_at  TEXT                                 -- NULL = not yet synced to hive
);
"""

CREATE_INTERACTIONS_INDEXES = """\
CREATE INDEX IF NOT EXISTS idx_interactions_source
    ON interactions(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_interactions_created
    ON interactions(created_at);
"""

# -----------------------------------------------------------------
# entities: people, orgs, projects, topics, etc. extracted from
# interactions by the NER pipeline. Each entity row belongs to
# exactly one interaction (per-interaction IDs, not global dedup).
# -----------------------------------------------------------------
CREATE_ENTITIES = """\
CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,
    interaction_id  TEXT NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    entity_type     TEXT NOT NULL CHECK(entity_type IN (
        'person','org','project','topic','tool','location','event','other'
    )),
    aliases         TEXT DEFAULT '[]',                    -- JSON array
    confidence      REAL NOT NULL DEFAULT 1.0 CHECK(confidence BETWEEN 0.0 AND 1.0),
    provenance      TEXT,                                -- model:version string
    context         TEXT DEFAULT '{}',                    -- JSON dict, rich context
    tags            TEXT DEFAULT '{}',                    -- JSON dict, KV pairs
    source_instance TEXT NOT NULL DEFAULT 'local',       -- which instance created this (hive sync)
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    hive_synced_at  TEXT                                 -- NULL = not yet synced to hive
);
"""

CREATE_ENTITIES_INDEXES = """\
CREATE INDEX IF NOT EXISTS idx_entities_interaction
    ON entities(interaction_id);
CREATE INDEX IF NOT EXISTS idx_entities_type_name
    ON entities(entity_type, name);
"""

# -----------------------------------------------------------------
# facts: subject-predicate-object triples linking entities.
# The CHECK constraint ensures every fact has at least one object
# representation (either an entity reference or a literal string).
# -----------------------------------------------------------------
CREATE_FACTS = """\
CREATE TABLE IF NOT EXISTS facts (
    id                  TEXT PRIMARY KEY,
    interaction_id      TEXT NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    subject_entity_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate           TEXT NOT NULL,
    object_entity_id    TEXT REFERENCES entities(id) ON DELETE SET NULL,
    object_literal      TEXT,
    confidence          REAL NOT NULL DEFAULT 1.0 CHECK(confidence BETWEEN 0.0 AND 1.0),
    source_span         TEXT,
    source_instance     TEXT NOT NULL DEFAULT 'local',   -- inherited from interaction (hive sync)
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    superseded_by       TEXT REFERENCES facts(id) ON DELETE SET NULL,
    hive_synced_at      TEXT,                            -- NULL = not yet synced to hive
    CHECK (object_entity_id IS NOT NULL OR object_literal IS NOT NULL)
);
"""

CREATE_FACTS_INDEXES = """\
CREATE INDEX IF NOT EXISTS idx_facts_interaction
    ON facts(interaction_id);
CREATE INDEX IF NOT EXISTS idx_facts_subject
    ON facts(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_facts_object_entity
    ON facts(object_entity_id);
"""

# -----------------------------------------------------------------
# embeddings_metadata: links SQL records to their vector embeddings.
# This is the bridge between the structured and semantic halves
# of the dual-store architecture.
# -----------------------------------------------------------------
CREATE_EMBEDDINGS_METADATA = """\
CREATE TABLE IF NOT EXISTS embeddings_metadata (
    id              TEXT PRIMARY KEY,
    source_table    TEXT NOT NULL CHECK(source_table IN ('interactions','entities')),
    source_id       TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL DEFAULT 0,
    chunk_start     INTEGER,
    chunk_end       INTEGER,
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    dimensions      INTEGER NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""

CREATE_EMBEDDINGS_METADATA_INDEXES = """\
CREATE INDEX IF NOT EXISTS idx_embeddings_source
    ON embeddings_metadata(source_table, source_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model
    ON embeddings_metadata(model_name, model_version);
"""

# -----------------------------------------------------------------
# meta_insights: patterns discovered by the farming pipeline.
# Insights have an optional expiration so stale patterns can be
# filtered out.
# -----------------------------------------------------------------
CREATE_META_INSIGHTS = """\
CREATE TABLE IF NOT EXISTS meta_insights (
    id              TEXT PRIMARY KEY,
    insight_type    TEXT NOT NULL CHECK(insight_type IN (
        'cluster','trend','anomaly','relationship'
    )),
    title           TEXT NOT NULL,
    description     TEXT,
    evidence        TEXT DEFAULT '[]',                    -- JSON array of source IDs
    confidence      REAL NOT NULL DEFAULT 1.0 CHECK(confidence BETWEEN 0.0 AND 1.0),
    parameters      TEXT DEFAULT '{}',                    -- JSON: algorithm, hyperparams
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    expires_at      TEXT                                  -- NULL = never expires
);
"""

CREATE_META_INSIGHTS_INDEXES = """\
CREATE INDEX IF NOT EXISTS idx_meta_insights_type
    ON meta_insights(insight_type);
"""

# -----------------------------------------------------------------
# sync_log: tracks row-level changes for multi-device replication.
# Rows stay with synced_at = NULL until successfully pushed.
# -----------------------------------------------------------------
CREATE_SYNC_LOG = """\
CREATE TABLE IF NOT EXISTS sync_log (
    id              TEXT PRIMARY KEY,
    table_name      TEXT NOT NULL,
    row_id          TEXT NOT NULL,
    operation       TEXT NOT NULL CHECK(operation IN ('insert','update','delete')),
    payload         TEXT,                                 -- JSON snapshot of the row
    synced_at       TEXT,                                 -- NULL until pushed to server
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    sync_generation INTEGER NOT NULL DEFAULT 0            -- monotonic counter
);
"""

CREATE_SYNC_LOG_INDEXES = """\
CREATE INDEX IF NOT EXISTS idx_sync_pending
    ON sync_log(synced_at) WHERE synced_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_sync_table_row
    ON sync_log(table_name, row_id);
"""

# -----------------------------------------------------------------
# FTS5 virtual table: full-text search index over interactions.
# Uses porter stemming + unicode61 for broad keyword matching.
# The content= and content_rowid= options make it a "content-less"
# external-content table that mirrors the interactions table.
# -----------------------------------------------------------------
CREATE_FTS = """\
CREATE VIRTUAL TABLE IF NOT EXISTS interactions_fts USING fts5(
    title,
    content,
    content='interactions',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
"""

# -----------------------------------------------------------------
# FTS triggers: keep the FTS index in sync with the interactions
# table automatically on INSERT / DELETE / UPDATE.
# -----------------------------------------------------------------
CREATE_FTS_TRIGGERS = """\
CREATE TRIGGER IF NOT EXISTS interactions_ai AFTER INSERT ON interactions BEGIN
    INSERT INTO interactions_fts(rowid, title, content)
        VALUES (new.rowid, new.title, new.content);
END;

CREATE TRIGGER IF NOT EXISTS interactions_ad AFTER DELETE ON interactions BEGIN
    INSERT INTO interactions_fts(interactions_fts, rowid, title, content)
        VALUES ('delete', old.rowid, old.title, old.content);
END;

CREATE TRIGGER IF NOT EXISTS interactions_au AFTER UPDATE ON interactions BEGIN
    INSERT INTO interactions_fts(interactions_fts, rowid, title, content)
        VALUES ('delete', old.rowid, old.title, old.content);
    INSERT INTO interactions_fts(rowid, title, content)
        VALUES (new.rowid, new.title, new.content);
END;
"""

# =====================================================================
# ALL_DDL: a list of every DDL string in dependency order.
# Tables must be created before their foreign-key dependents, and
# FTS / triggers come last because they reference the interactions table.
# =====================================================================
ALL_DDL: list[str] = [
    CREATE_INTERACTIONS,
    CREATE_INTERACTIONS_INDEXES,
    CREATE_ENTITIES,
    CREATE_ENTITIES_INDEXES,
    CREATE_FACTS,
    CREATE_FACTS_INDEXES,
    CREATE_EMBEDDINGS_METADATA,
    CREATE_EMBEDDINGS_METADATA_INDEXES,
    CREATE_META_INSIGHTS,
    CREATE_META_INSIGHTS_INDEXES,
    CREATE_SYNC_LOG,
    CREATE_SYNC_LOG_INDEXES,
    CREATE_FTS,
    CREATE_FTS_TRIGGERS,
    # LLM usage tracking (v5)
    """CREATE TABLE IF NOT EXISTS llm_usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    model_name      TEXT NOT NULL,
    stage           TEXT NOT NULL,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    latency_ms      REAL NOT NULL DEFAULT 0.0,
    success         INTEGER NOT NULL DEFAULT 1,
    error_message   TEXT
);""",
    "CREATE INDEX IF NOT EXISTS idx_llm_usage_date ON llm_usage(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_llm_usage_model ON llm_usage(model_name);",
    "CREATE INDEX IF NOT EXISTS idx_llm_usage_stage ON llm_usage(stage);",
    # Outbox progress tracking (2026-04-08: decoupled hive sync)
    # Local tracks what intelligence was already written to outbox,
    # so next push only sends changes since last batch.
    """CREATE TABLE IF NOT EXISTS outbox_progress (
    table_name       TEXT PRIMARY KEY,
    last_synced_cycle INTEGER DEFAULT 0,
    last_synced_at   TEXT,
    records_sent     INTEGER DEFAULT 0,
    last_batch_id    TEXT
);""",
]

# =====================================================================
# Migration Registry
# =====================================================================
# Each entry is (target_version, list_of_SQL_statements).
# Migrations are applied in order when user_version < target_version.
# The initial schema creation (version 0 → 1) is handled by
# apply_schema(), not by the migration list.
# Future migrations (e.g., adding columns) go here.
# =====================================================================
MIGRATIONS: list[tuple[int, list[str]]] = [
    # ---------------------------------------------------------------
    # v1 → v2: Add hive_synced_at column to interactions, entities,
    # and facts tables for hive sync tracking.  NULL means the record
    # has not yet been synced to the hive.  The column is set to an
    # ISO-8601 timestamp after a successful hive push.
    # See research/notes/hive-sync-design.md for the full design.
    # ---------------------------------------------------------------
    (2, [
        "ALTER TABLE interactions ADD COLUMN hive_synced_at TEXT;",
        "ALTER TABLE entities ADD COLUMN hive_synced_at TEXT;",
        "ALTER TABLE facts ADD COLUMN hive_synced_at TEXT;",
    ]),
    # ---------------------------------------------------------------
    # v2 → v3: Phase 3 – farming pipeline, maintenance agents, and
    # self-learning infrastructure.
    #
    # This migration adds:
    #   - farming_cycles / farming_checkpoints: orchestration tables
    #     for the periodic farming pipeline that mines patterns.
    #   - query_quality_log: captures query performance metrics so
    #     the self-learning loop can tune retrieval strategies.
    #   - idx_facts_predicate: speeds up predicate lookups for the
    #     farming pipeline's relationship-analysis stage.
    #   - meta_insights expansion: new insight types for maintenance
    #     agents (causal, consolidation, supersession, verification,
    #     gap, archive, topic, meta) plus an entity_ids column.
    #   - Seven maintenance agent log tables: one per agent
    #     (consolidator, pruner, verifier, completionist, linker,
    #     archivist, defragmenter) to track actions taken.
    #   - farming_clustering_progress: resumable clustering state.
    #   - entity_interactions: junction table linking entities to
    #     interactions for many-to-many queries.
    #   - hive_pull_progress: tracks per-table pull cursors for
    #     incremental hive synchronisation.
    # ---------------------------------------------------------------
    (3, [
        # -- farming_cycles: one row per farming pipeline run ------
        # Tracks overall cycle status and how many stages completed.
        """CREATE TABLE IF NOT EXISTS farming_cycles (
    cycle_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    completed_at    TEXT,
    status          TEXT NOT NULL CHECK(status IN ('running','completed','partial','failed')),
    trigger         TEXT,
    stages_done     INTEGER DEFAULT 0
);""",

        # -- farming_checkpoints: per-stage progress within a cycle
        # Allows the pipeline to resume from the last completed stage
        # after a crash or partial run.  state_blob stores opaque
        # serialised state for the stage's algorithm.
        """CREATE TABLE IF NOT EXISTS farming_checkpoints (
    cycle_id        INTEGER NOT NULL,
    stage           TEXT NOT NULL,
    status          TEXT NOT NULL CHECK(status IN ('running','completed','failed')),
    started_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    completed_at    TEXT,
    items_processed INTEGER DEFAULT 0,
    items_total     INTEGER DEFAULT 0,
    state_blob      BLOB,
    error_message   TEXT,
    PRIMARY KEY (cycle_id, stage)
);""",

        # -- query_quality_log: self-learning feedback loop --------
        # Records every query and its retrieval metrics so the
        # system can detect poor-quality patterns and adjust.
        """CREATE TABLE IF NOT EXISTS query_quality_log (
    id              TEXT PRIMARY KEY,
    query_text      TEXT NOT NULL,
    mode            TEXT,
    result_ids      TEXT DEFAULT '[]',
    sql_result_count INTEGER DEFAULT 0,
    vector_result_count INTEGER DEFAULT 0,
    latency_ms      REAL,
    refined_within_60s INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);""",

        # -- facts.predicate index: speeds up relationship mining --
        # The farming pipeline's relationship-analysis stage queries
        # facts by predicate to discover causal and temporal patterns.
        "CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate);",

        # -- meta_insights expansion: widen insight_type CHECK -----
        # Step 1: rename the old table to a backup.
        "ALTER TABLE meta_insights RENAME TO meta_insights_v2_backup;",

        # Step 2: create the new meta_insights with expanded types
        # and an entity_ids column for linking insights to entities.
        """CREATE TABLE IF NOT EXISTS meta_insights (
    id              TEXT PRIMARY KEY,
    insight_type    TEXT NOT NULL CHECK(insight_type IN (
        'cluster','trend','anomaly','relationship',
        'causal','consolidation','supersession','verification',
        'gap','archive','topic','meta'
    )),
    title           TEXT NOT NULL,
    description     TEXT,
    evidence        TEXT DEFAULT '[]',
    confidence      REAL NOT NULL DEFAULT 1.0 CHECK(confidence BETWEEN 0.0 AND 1.0),
    parameters      TEXT DEFAULT '{}',
    entity_ids      TEXT DEFAULT '[]',
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    expires_at      TEXT
);""",

        # Step 3: copy existing rows from the backup into the new
        # table.  Migrated rows get entity_ids = '[]'.
        """INSERT INTO meta_insights (
    id, insight_type, title, description, evidence,
    confidence, parameters, entity_ids, created_at, expires_at
) SELECT
    id, insight_type, title, description, evidence,
    confidence, parameters, '[]', created_at, expires_at
FROM meta_insights_v2_backup;""",

        # Step 4: drop the backup table now that all data is moved.
        "DROP TABLE meta_insights_v2_backup;",

        # Step 5: recreate indexes on the new meta_insights table.
        "CREATE INDEX IF NOT EXISTS idx_meta_insights_type ON meta_insights(insight_type);",
        "CREATE INDEX IF NOT EXISTS idx_meta_insights_created ON meta_insights(created_at);",

        # -- Maintenance agent tracking tables ---------------------
        # One table per maintenance agent.  Each logs the actions it
        # took during a farming cycle (what it merged, pruned, etc.).
        # All seven share the same schema for consistency.
        """CREATE TABLE IF NOT EXISTS maintenance_consolidator (
    id              TEXT PRIMARY KEY,
    cycle_id        INTEGER NOT NULL,
    action          TEXT NOT NULL,
    target_ids      TEXT NOT NULL,
    canonical_id    TEXT,
    detail          TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);""",

        """CREATE TABLE IF NOT EXISTS maintenance_pruner (
    id              TEXT PRIMARY KEY,
    cycle_id        INTEGER NOT NULL,
    action          TEXT NOT NULL,
    target_ids      TEXT NOT NULL,
    canonical_id    TEXT,
    detail          TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);""",

        """CREATE TABLE IF NOT EXISTS maintenance_verifier (
    id              TEXT PRIMARY KEY,
    cycle_id        INTEGER NOT NULL,
    action          TEXT NOT NULL,
    target_ids      TEXT NOT NULL,
    canonical_id    TEXT,
    detail          TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);""",

        """CREATE TABLE IF NOT EXISTS maintenance_completionist (
    id              TEXT PRIMARY KEY,
    cycle_id        INTEGER NOT NULL,
    action          TEXT NOT NULL,
    target_ids      TEXT NOT NULL,
    canonical_id    TEXT,
    detail          TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);""",

        """CREATE TABLE IF NOT EXISTS maintenance_linker (
    id              TEXT PRIMARY KEY,
    cycle_id        INTEGER NOT NULL,
    action          TEXT NOT NULL,
    target_ids      TEXT NOT NULL,
    canonical_id    TEXT,
    detail          TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);""",

        """CREATE TABLE IF NOT EXISTS maintenance_archivist (
    id              TEXT PRIMARY KEY,
    cycle_id        INTEGER NOT NULL,
    action          TEXT NOT NULL,
    target_ids      TEXT NOT NULL,
    canonical_id    TEXT,
    detail          TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);""",

        """CREATE TABLE IF NOT EXISTS maintenance_defragmenter (
    id              TEXT PRIMARY KEY,
    cycle_id        INTEGER NOT NULL,
    action          TEXT NOT NULL,
    target_ids      TEXT NOT NULL,
    canonical_id    TEXT,
    detail          TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);""",

        """CREATE TABLE IF NOT EXISTS maintenance_rationalizer (
    id              TEXT PRIMARY KEY,
    cycle_id        INTEGER NOT NULL,
    action          TEXT NOT NULL,
    target_ids      TEXT NOT NULL,
    canonical_id    TEXT,
    detail          TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);""",

        # -- farming_clustering_progress: resumable clustering -----
        # Stores the last entity processed so clustering can resume
        # from where it left off after an interruption.
        """CREATE TABLE IF NOT EXISTS farming_clustering_progress (
    id              TEXT PRIMARY KEY,
    cycle_id        INTEGER NOT NULL,
    last_entity     TEXT NOT NULL,
    entities_done   INTEGER NOT NULL,
    batch_size      INTEGER NOT NULL,
    completed_at    TEXT NOT NULL
);""",

        # -- entity_interactions: many-to-many junction table ------
        # Links entities to interactions for queries that need to
        # find all entities mentioned in an interaction (or all
        # interactions that mention a given entity).
        """CREATE TABLE IF NOT EXISTS entity_interactions (
    entity_id       TEXT NOT NULL,
    interaction_id  TEXT NOT NULL,
    PRIMARY KEY (entity_id, interaction_id)
);""",

        # -- hive_pull_progress: incremental hive sync cursors -----
        # Tracks per-table cursors so hive pulls only fetch rows
        # added since the last successful pull.
        """CREATE TABLE IF NOT EXISTS hive_pull_progress (
    table_name      TEXT PRIMARY KEY,
    last_pulled_at  TEXT NOT NULL,
    last_row_id     TEXT,
    records_pulled  INTEGER DEFAULT 0
);""",
    ]),
    # ---------------------------------------------------------------
    # v3 → v4: Phase 4 – distiller farming stage.
    #
    # This migration adds:
    #   - distiller_summaries: stores per-entity summaries produced
    #     by the DistillerStage, including relevance scores, top
    #     co-entities, and top predicates.  The distiller condenses
    #     raw entity data into compact, query-friendly summaries.
    #   - idx_distiller_relevance: speeds up relevance-ranked lookups
    #     so the query engine can quickly find the most important
    #     entities.
    # ---------------------------------------------------------------
    (4, [
        # -- distiller_summaries: per-entity distilled intelligence --
        # One row per unique entity name.  Updated each farming cycle
        # with aggregated stats, co-entity relationships, and a
        # natural-language summary.  Relevance score combines mention
        # frequency, interaction spread, and recency.
        """CREATE TABLE IF NOT EXISTS distiller_summaries (
    entity_name       TEXT PRIMARY KEY,
    entity_type       TEXT NOT NULL,
    summary           TEXT NOT NULL,
    source_instances  TEXT NOT NULL DEFAULT '[]',
    top_co_entities   TEXT NOT NULL DEFAULT '[]',
    top_predicates    TEXT NOT NULL DEFAULT '[]',
    relevance_score   REAL NOT NULL DEFAULT 0.0,
    updated_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    cycle_id          INTEGER NOT NULL DEFAULT 0
);""",

        # -- relevance index: enables fast "most important entities" queries
        "CREATE INDEX IF NOT EXISTS idx_distiller_relevance ON distiller_summaries(relevance_score DESC);",

        # -- local_intelligence_cache: caches distilled hints pulled from
        # the hive for use by the extraction pipeline.  One row per entity
        # name, keyed by PRIMARY KEY entity_name.  The pull worker reads
        # distiller_summaries from the hive and copies relevant hints here
        # so the extraction pipeline can enrich its prompts with cross-
        # instance intelligence without hitting the hive during extraction.
        """CREATE TABLE IF NOT EXISTS local_intelligence_cache (
    entity_name       TEXT PRIMARY KEY,
    entity_type       TEXT NOT NULL,
    summary           TEXT NOT NULL,
    top_co_entities   TEXT NOT NULL DEFAULT '[]',
    top_predicates    TEXT NOT NULL DEFAULT '[]',
    relevance_score   REAL NOT NULL DEFAULT 0.0,
    source_instances  TEXT NOT NULL DEFAULT '[]',
    fetched_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);""",
    ]),
    # ---------------------------------------------------------------
    # v4 → v5: LLM Usage Tracking
    #
    # Stores every LLM API call for cost monitoring and usage stats.
    # One row per generate() invocation with model name, pipeline
    # role (stage), token counts, and latency.
    # ---------------------------------------------------------------
    (5, [
        """CREATE TABLE IF NOT EXISTS llm_usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    model_name      TEXT NOT NULL,
    stage           TEXT NOT NULL,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    latency_ms      REAL NOT NULL DEFAULT 0.0,
    success         INTEGER NOT NULL DEFAULT 1,
    error_message   TEXT
);""",
        "CREATE INDEX IF NOT EXISTS idx_llm_usage_date ON llm_usage(created_at);",
        "CREATE INDEX IF NOT EXISTS idx_llm_usage_model ON llm_usage(model_name);",
        "CREATE INDEX IF NOT EXISTS idx_llm_usage_stage ON llm_usage(stage);",
    ]),
]


# =====================================================================
# Schema application and migration functions
# =====================================================================


async def apply_pragmas(db: aiosqlite.Connection) -> None:
    """
    Apply per-connection PRAGMA settings.

    These must be set on every new connection because SQLite's
    PRAGMAs are connection-scoped, not database-scoped. WAL mode
    is the exception (it persists), but setting it again is harmless.
    """
    for pragma in PRAGMAS.strip().splitlines():
        # Skip blank lines
        pragma = pragma.strip()
        if pragma:
            await db.execute(pragma)


async def apply_schema(db: aiosqlite.Connection) -> None:
    """
    Create all tables, indexes, and triggers if they don't exist.

    This is called once at startup by SQLiteStore.initialize().
    Because every DDL statement uses IF NOT EXISTS, running it
    against an already-initialised database is a safe no-op.

    On a brand-new database (user_version == 0) the version is
    stamped to SCHEMA_VERSION so the migration system knows the
    current state.  Already-initialised databases keep their
    existing version so that future migrations applied by
    migrate() are not accidentally skipped.

    Uses executescript() because some DDL blocks (triggers) contain
    internal semicolons that would break naive split-by-semicolon
    parsing.
    """
    # Read the current version BEFORE running DDL.  A brand-new
    # database returns 0; an existing one returns whatever version
    # was previously stamped.
    current_version = await get_schema_version(db)

    # Concatenate all DDL blocks into a single script.
    # executescript() handles multiple statements and trigger bodies
    # correctly because it uses SQLite's internal parser.
    full_script = "\n".join(ALL_DDL)
    await db.executescript(full_script)

    # Only stamp the version on a fresh database (version 0).
    # Existing databases keep their version so that migrate() can
    # detect pending migrations and apply them correctly.
    if current_version == 0:
        await db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    await db.commit()


async def get_schema_version(db: aiosqlite.Connection) -> int:
    """
    Read the current schema version from the database.

    SQLite stores user_version as an integer in the database header.
    A brand-new database returns 0.
    """
    cursor = await db.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    return row[0] if row else 0


async def migrate(
    db: aiosqlite.Connection,
    current_version: int | None = None,
    target_version: int | None = None,
) -> int:
    """
    Apply pending schema migrations sequentially.

    Reads the current user_version from the database (or accepts it
    as a parameter), then walks the MIGRATIONS list applying every
    step whose target_version exceeds the current version.

    Args:
        db:              Open aiosqlite connection.
        current_version: Override for the current schema version.
                         If None, reads from PRAGMA user_version.
        target_version:  Stop after reaching this version.
                         If None, applies all available migrations.

    Returns:
        The schema version after all applicable migrations have run.
    """
    # Read current version from the database if not supplied
    if current_version is None:
        current_version = await get_schema_version(db)

    # Default target: latest version available
    if target_version is None:
        target_version = SCHEMA_VERSION
        if MIGRATIONS:
            target_version = max(target_version, MIGRATIONS[-1][0])

    # Walk through migrations in order and apply those that are needed
    for migration_version, statements in MIGRATIONS:
        # Skip migrations we have already applied
        if current_version >= migration_version:
            continue
        # Stop if we have reached the requested target
        if migration_version > target_version:
            break

        # Apply every SQL statement in this migration step
        for stmt in statements:
            stmt = stmt.strip()
            if stmt:
                await db.execute(stmt)

        # Update the version stamp after each successful step
        await db.execute(f"PRAGMA user_version = {migration_version}")
        await db.commit()
        current_version = migration_version

    return current_version
