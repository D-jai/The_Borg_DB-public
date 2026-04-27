# Changelog

All notable changes to The_Borg_DB will be documented in this file.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.1] — 2026-04-27

Project-local runtime layout, a fresh-install schema bug fix, and the
post-release Phase C documentation deep-dive. The runtime / schema
changes are surface-only: existing installations continue to work
without data migration. The Phase C deliverables are pure documentation
plus a docstring fix in `farming/__init__.py`.

### Documentation

- **`audit_findings_v0.7.1.md`** — consolidated Phase C audit covering
  the 18-stage farming pipeline (keying analysis), LLM wiring across
  the codebase, and the `/entities/merge` defect catalog re-walk.
- **`runtimechange.md`** — canonical record of the runtime relocation:
  why, the resolver design, the three wiring patterns used, the
  configuration changes, the migration paths, and the deliberate
  non-goals.
- **`autonomous_farming.md`** — working design doc for the farming
  subsystem: what each stage does, why some take an `llm` and don't
  call it, the keying question, the merge-as-rename model, the
  proposed Phase 4.0 quick-win batch, and three open design questions.
- **`README.md`** — adds a "For LLMs and AI Agents" section that
  tells AI coding assistants what to load on first contact, what to
  skip, the project's hard invariants, and common task shapes.

### Added

- **`src/ctxmtg/paths.py`** — single source of truth for every runtime
  artifact's filesystem location. Resolves to `<project_root>/.runtime/`
  by default; override the entire root with the `CTXMTG_DATA_ROOT`
  environment variable. Each clone of the source tree gets a cleanly
  separated runtime store out of the box, supporting multi-instance
  deployments (Local_Tickets / Local_Emails / Hive) on the same machine
  without `CTXMTG_HOME` gymnastics.

### Changed

- **Runtime data root** — moved from `~/.ctxmtg/` to
  `<project_root>/.runtime/`. Affected files: SQLite knowledge database,
  LanceDB vector store, hive database + vectors, inbox/processed/outbox
  directories, `.env` file, `web_auth.json`, `hive_web_auth.json`,
  `archive.db`, evaluation snapshots. `~/.ctxmtg/` is no longer touched.
- **`config/settings.py`** — every path field now uses a `default_factory`
  pointing at the matching `paths.get_*()` helper. YAML / env-var
  overrides still win.
- **`config/env_file.py`** — drops the module-level `ENV_PATH` constant in
  favour of `paths.get_env_file_path()` evaluated at call time, so
  `CTXMTG_DATA_ROOT` overrides take effect even after import.
- **`web/auth.py`, `web/hive_auth.py`** — `_auth_file()` now returns
  `paths.get_web_auth_path()` / `paths.get_hive_web_auth_path()`.
- **`farming/__init__.py`** — `archive.db` derivation no longer reads the
  `CTXMTG_HOME` environment variable. `CTXMTG_DB_PATH` still wins for
  custom multi-instance setups; otherwise the file lands at
  `<runtime_root>/archive.db`.
- **`query/evaluation.py`** — `DEFAULT_EVAL_DIR` constant removed.
  `get_eval_dir()` falls through to `paths.get_eval_dir()`.
- **`constants.py`** — `DEFAULT_DATA_DIR`, `DEFAULT_DB_PATH`,
  `DEFAULT_VECTOR_PATH`, `DEFAULT_PROFILE_DIR` removed (none were
  imported by any module). `paths.py` is the single source of truth now.
- **`configs/default.yaml`** — `storage.db_path`, `storage.vector_path`,
  `hive.local_db_path`, `hive.local_vector_path` now default to empty
  strings; the runtime resolver fills them in. Set them only to override.
- **`.env.example`** — documents the new `CTXMTG_DATA_ROOT` knob.
- **`.gitignore`** — adds `.runtime/` and `.venv/`.

### Fixed

- **Fresh-install schema gaps (P0)** — `apply_schema()` stamped fresh
  databases with `PRAGMA user_version = SCHEMA_VERSION (=5)` immediately
  after running `ALL_DDL`, which short-circuited `migrate()` and skipped
  every v3 / v4 / v5 migration on a clean install. Net effect: the
  farming pipeline did not run end-to-end on a freshly cloned repository
  because `farming_cycles`, `farming_checkpoints`, `farming_progress`,
  `query_quality_log`, all `maintenance_*` tables, `distiller_summaries`,
  `local_intelligence_cache`, `farming_clustering_progress`,
  `entity_interactions`, and `hive_pull_progress` were never created.
  `ALL_DDL` is now the complete v5 schema. The migration list is
  unchanged so existing v1 / v2 / v3 / v4 databases still upgrade
  through the same paths.
- **`CREATE_META_INSIGHTS` CHECK constraint** — was the original 4-type
  list (`cluster, trend, anomaly, relationship`) on fresh installs, so
  any insight typed as `causal`, `consolidation`, `verification`, etc.
  would have been rejected before reaching disk. Now the full 12-type
  v3 schema with the `entity_ids` column.
- **`farming_progress` table** — referenced by `farming/progress.py`
  (Post-Install fix from 2026-04-07) but had no `CREATE TABLE`
  statement anywhere in the codebase. Now declared in `schema.py`
  alongside the other Phase 3 tables.
- **`idx_meta_insights_created`** — referenced by the v3 migration but
  absent from `ALL_DDL`. Added.

### Migration notes

- **Existing `~/.ctxmtg/` installations:** your data is untouched. To
  carry it forward, copy the directory contents to
  `<project_root>/.runtime/` (or set `CTXMTG_DATA_ROOT` to the old
  `~/.ctxmtg` path).
- **Multi-instance setups:** `CTXMTG_HOME` is no longer read by
  `farming/__init__.py`. Use `CTXMTG_DATA_ROOT` instead, or
  `CTXMTG_DB_PATH` for the legacy archive-next-to-knowledge.db
  arrangement.

## [0.7.0] — 2024-04-13

First public source-available release under the Business Source License 1.1.
This release consolidates a large block of post-installation fixes and
architectural changes made during production hardening on the reference
hardware (DGX Spark, dual-local + single-hive deployment).

### Added

- **Rationalizer farming stage** — new first maintenance stage that tests each
  entity name against eight garbage-detection rules (embedded newlines, URL
  fragments, markdown link artifacts, bare decimal phone fragments, truncation
  markers, pure punctuation, sub-2-char names, excessive whitespace). Matches
  get `confidence = 0.1`. Entities with important facts
  (`responsible_for`, `leads`, `reports_to`, `decided`, `committed_to`) are
  protected. Non-destructive and reversible. The farming pipeline is now
  **18 stages** (was 17).
- **Archivist garbage bypass** — any entity with `confidence ≤ 0.1` is
  archived to `archive.db` immediately regardless of age, implementing the
  second phase of the rationalizer → archivist → (future) trim lifecycle.
- **Hive outbox/inbox sync** — replaces direct-database hive push with a
  file-based manifest protocol. `ctxmtg hive push` writes atomic JSON
  manifests to `~/.ctxmtg/outbox/` with a full local metadata snapshot;
  `ctxmtg hive serve` starts a separate Hive Command Center web UI
  (port 8082, amber/gold theme) where the user creates per-local "links" and
  pulls manifests on demand.
- **Local metadata v2** — each outbox manifest now carries identity, DB
  state (counts + full table list), farming state, profile, embedding model,
  LLM role models, platform, and cumulative outbox history. The hive has
  full visibility into each local at every batch.
- **Profile-driven entity filters** — new `EntityFilterConfig` model with
  `min_name_length`, `max_name_length`, `reject_patterns`, and `reject_names`.
  Applied at Step 3a of the extraction pipeline (after NER + regex merge,
  before ID assignment). Default rules reject timestamp fragments, timezone
  offsets, ISO dates, bare integers, and header labels.
- **Entity delete endpoint + web UI button** — cascades cleanup to facts,
  meta_insights, and distiller_summaries.
- **Thinking-token stripping** — removes `<think>`, `<thinking>`, and
  `<|thinking|>` blocks from all LLM responses before they reach the
  extraction pipeline, stored facts, synthesis output, or the LLM proxy's
  captured transcripts.
- **Progressive farming scans** — new `farming_progress` table and helper
  module. Each farming stage now reads its last offset, processes the next
  batch, and saves the new offset. Running N cycles covers N × batch_size
  entities instead of repeating the same top slice. Nine stages updated:
  entity_analytics, graph_analysis, linker, completionist, verifier, pruner,
  consolidator, archivist, causal_miner.
- **`transformers` and `aiosqlite` added to core runtime dependencies.**
  Both were required at runtime but previously assumed to be installed as
  side effects of other packages. `jinja2` added to the `[web]` extra.
- **`.env.example`** shipped in the repo root with the complete LLM role
  layout and instance-name variable.
- **Business Source License 1.1** — `LICENSE`, `NOTICE`, and the README
  license section are now aligned around BSL 1.1 with Change Date 2029-01-01
  and Change License GPL-3.0-or-later.

### Changed

- **Farming insight deduplication** — removed `cycle_id` from the insight ID
  format in six stages (entity_analytics, graph_analysis, causal_miner,
  trend_detection, topic_modeling, clustering). `INSERT OR REPLACE` now
  naturally updates the same co-occurrence / graph / trend row instead of
  duplicating it per cycle.
- **Causal miner scope** — now excludes facts generated by farming stages
  themselves (`source_span LIKE 'linker:%'`, etc.), preventing the miner
  from learning temporal patterns in its own output.
- **V2S retrieval prompt** — updated to guide the LLM toward
  `interactions.content` / `interactions.title` LIKE patterns instead of
  entity-only SQL. Previous prompt saved as `v0.9.0.txt`.
- **JSON response parsing for V2S/S2V bridges** — now strips markdown code
  fences before JSON parsing. Debug logging added for both bridges.
- **Query executor wiring** — CLI and web endpoints now load `OnnxEmbedder`
  at query time and pass `embedding_fn` to `QueryExecutor` so vector search
  is actually used. Semantic half of the dual-store architecture is now
  wired for queries, not just ingestion.
- **Web UI branding** from `ctxmtg` to `The_Borg_DB`.
- **Entity merge / delete endpoints** now use `db.execute()` +
  `await db.commit()` directly instead of `execute_sql()`, fixing silent
  drop of DML changes.
- **CLI async consolidation** — all per-query async operations (main
  retrieval, hive query, store cleanup) now run inside a single
  `asyncio.run()` call to avoid orphaned aiosqlite background threads and
  the intermittent "Event loop is closed" error in `--mode deep`.
- **Project manifest cleanup** — stripped test/dev-only sections from
  `pyproject.toml` (pytest, ruff, mypy, coverage) and test-tool patterns
  from `.gitignore`. The package now installs cleanly with
  `pip install -e ".[web]"` for runtime use.

### Fixed

- **Fresh-install schema gaps** — `farming_cycles`, `farming_checkpoints`,
  `query_quality_log`, all `maintenance_*` tables, `farming_clustering_progress`,
  `entity_interactions`, `hive_pull_progress`, `distiller_summaries`,
  `local_intelligence_cache`, `farming_progress`, `maintenance_rationalizer`,
  and `outbox_progress` are now created on fresh install paths. Indexes on
  `facts(predicate)`, `distiller_summaries(relevance_score DESC)`,
  `meta_insights(insight_type)`, and `meta_insights(created_at)` added.
- **Foreign-key constraint failures** in LLM fact creation — `_merge_results`
  now auto-creates an entity of type `OTHER` when the LLM references a
  subject name that is not present in the entity lookup, instead of failing
  the entire fact batch.
- **Farming runs via the web UI no longer degrade to math-only** — the
  `trigger_farm` endpoint now loads the farming-role LLM the same way the
  CLI does.
- **Version-number regex** in the `general` profile — no longer matches
  timestamp fragments (e.g. `20.000`, `11.000`) from ISO date headers.
  Requires either a `v` prefix or a three-part version.

### Removed

- **Direct-database hive sync** — replaced by the outbox/inbox architecture
  described above. Old hive code is retained commented-out for rollback.
- **Dev-only dependency groups** — `[dev]` extra, `[tool.ruff]`, `[tool.mypy]`,
  `[tool.pytest.ini_options]`, and `[tool.coverage.*]` removed from
  `pyproject.toml`. The repository ships runtime-only.
- **Unused `mock_provider.py`** — test-only helper not imported by any
  runtime code path.
