# Baseline smoke test -- The_Borg_DB-public (2026-04-27)

> First-run smoke test of `github.com/D-jai/The_Borg_DB-public` on
> Windows after fresh clone + venv install. Establishes the
> behavioral baseline before any further changes.

---

## Environment

- **Host:** Windows (win32 10.0.26200), PowerShell.
- **Python:** 3.14.3 (system Python).
- **Repo HEAD:** `ba2ed7f` (Update whitepaper with various corrections).
- **Branch:** `main`.
- **Tag at release:** `v0.7.0`.
- **Venv:** `C:\Stage\The_Borg_DB-public\.venv\`.
- **Default data root:** `C:\Users\Deejr\.ctxmtg\` (home dir; not yet
  relocated to `<project>/.runtime/`).

## Install

| Step | Status | Notes |
|---|---|---|
| `python -m venv .venv` | OK | Python 3.14.3 inside venv |
| `pip install --upgrade pip` | OK | 25.3 -> 26.1 |
| `pip install -e ".[dev,web]"` | OK | 87 packages including `transformers-5.6.2`, `aiosqlite-0.22.1`. **Both already in deps -- Post-Inst TODO #5 is fixed in the public release.** |
| `pip install <bridge>/en_core_web_sm-3.8.0-py3-none-any.whl` | OK | Cached wheel from bridge (avoided github.com 502) |

## Smoke tests

### `ctxmtg health` -- PASS

Stores initialize cleanly at `C:\Users\Deejr\.ctxmtg\`. Health
report: status `healthy`, 0 interactions / 0 entities / 0 facts /
0 vectors. Stores close cleanly.

### `ctxmtg ingest "Alice proposed migrating auth to OAuth2..."` -- PASS

- spaCy NER: 3 entities (Stage got 2; the public version's profile
  may extract more, e.g., dates).
- Regex: 0 entities.
- Fact extraction: 1 fact.
- LLM verifier: skipped (no API key configured -- expected).
- Embedding: ONNX `all-MiniLM-L6-v2`, 384-dim, CPUExecutionProvider.
- Stored: 3 entities, 1 fact, 1 embedding.
- Final: `Done: 3 entities, 1 facts, 1 embeddings (0ms)`.

### `ctxmtg query "What did Alice propose?"` -- PASS with one bug

**Major improvement vs Stage:** `has_vector=True`. Vector search
actually executed (`vector_count=1`); RRF fused 1 SQL + 1 vector
result. **Post-Inst #11 (embedding_fn wiring) is confirmed
applied** in the public release.

- 2 final results: SQL match (`Alice | migrate | auth`, score 0.098)
  and vector match (the original sentence, score 0.037).
- Latency: 4.1 s (model loading; subsequent queries should be fast).

**Real code bug found (documented, not fixed):**
`error='no such table: query_quality_log'` -- the
`query_quality_log` table is not created on fresh install. Quality
logging silently fails. **Post-Inst #7/#21 is only PARTIALLY
applied:** the `outbox_progress` table is created, but
`query_quality_log` and most farming tables are not. See "Tables on
fresh install" below.

### `ctxmtg farm run` -- BLOCKED (timed out at 300 s)

**Real code bug confirmed: same root cause as Stage.** Fresh
install does not create the farming tables (`farming_cycles`,
`farming_checkpoints`, `farming_progress`,
`maintenance_rationalizer`, `distiller_summaries`,
`query_quality_log`). The pipeline's first action attempts
`INSERT INTO farming_cycles` which fails / hangs on retries.

The 18-stage pipeline (including Rationalizer) cannot be exercised
until those tables are created. Either:

- (a) The DDL exists in code but is not invoked on fresh init.
  Likely candidate: `storage/migrations.py` or wherever Post-Inst
  #21's DDL consolidation lives. **Inspect after Phase B.**
- (b) The DDL was never folded into the consolidated set. Larger
  fix.

**Status:** documented per user directive ("fix only
Windows-specific issues; document anything that looks like a real
code bug"). This is squarely a code bug.

### `ctxmtg serve --port 8081` -- PASS

- Server bound to `127.0.0.1:8081` cleanly.
- `GET /login` -> 200.
- Stopped cleanly via `Stop-Process`.

---

## Tables on fresh install

```
14 tables created:
  embeddings_metadata
  entities
  facts
  interactions
  interactions_fts (+ _config, _data, _docsize, _idx)
  llm_usage
  meta_insights
  outbox_progress
  sqlite_sequence
  sync_log

Missing critical tables:
  farming_cycles            (blocks farming)
  farming_checkpoints       (blocks farming)
  farming_progress          (blocks per-stage progressive scan, Post-Inst #17)
  query_quality_log         (blocks query quality logging, Post-Inst #7/#21)
  distiller_summaries       (blocks Distiller stage)
  maintenance_rationalizer  (blocks Rationalizer stage, Post-Inst #23)
```

Compared to Stage (which had 13 tables on fresh install): the
public release has **only one additional table** (`outbox_progress`,
Post-Inst #20). This means the bulk of the DDL fixes from
Post-Installation Updates were either NOT folded into
`storage/migrations.py` (or wherever the consolidated DDL lives)
or were folded in but are gated by something the fresh-install
codepath doesn't hit.

**Action item for follow-up:** locate where the public release
intends `farming_cycles` etc. to be created and verify.

---

## What this confirms about the 24 Post-Installation items

| # | Item | Public-release status |
|---|---|---|
| 2 | aiosqlite + pytorch detection | aiosqlite in deps; transformers stderr noise unchanged |
| 5 (TODO) | transformers in pyproject | **Applied** -- `transformers-5.6.2` installed via deps |
| 7 / 21 | DDL consolidation | **Partial** -- `outbox_progress` exists; farming tables and `query_quality_log` still missing |
| 11 | embedding_fn wiring | **Applied** -- vector search actually runs in query path |
| 17 | farming_progress + progress.py | File exists; table does NOT (Post-Inst #21 didn't fold it in) |
| 18 | cycle_id removed from insight IDs | Untested (farming blocked) |
| 19 | CausalMiner skip linker facts | Untested (farming blocked) |
| 20 | outbox/inbox decoupling | Files exist; `outbox_progress` table exists |
| 23 | Rationalizer stage | File exists; cannot exercise (table missing) |

**Net:** the public release is substantially better than Stage at
the **code level** (more files, more wiring) but the **schema level
fixes did not all land**. Farming remains broken until the missing
DDL is restored on fresh init.

---

## Compared to Stage smoke test (`bridge/smoke_test_stage.md`)

| Behavior | Stage | Public |
|---|---|---|
| `transformers` install | manual extra step | bundled in deps |
| `aiosqlite` install | manual extra step | bundled in deps |
| Query vector search | skipped (no embedding_fn) | **runs** |
| Query result count | 1 | **2** |
| Farming | hangs (missing tables) | hangs (still missing tables) |
| `query_quality_log` warning | yes | yes (unchanged) |
| `outbox_progress` table | missing | **present** |
| Web server | works | works |

---

## Phase A verdict

PASS for install + most smoke tests. Two real code bugs found and
documented; both subsequently fixed in Phase A.5 (see below).

## Phase A.5 -- DDL bug fix (2026-04-27)

Root cause: `apply_schema()` stamped fresh databases with
`PRAGMA user_version = SCHEMA_VERSION (=5)` immediately after
running `ALL_DDL`. The `migrate()` runner then read v5, decided
nothing was pending, and skipped v3 / v4 / v5 migrations entirely.
Net effect: fresh installs never received the farming, maintenance,
distiller, or query-quality tables defined in those migrations.

Three secondary bugs fell out of the same audit:

1. `CREATE_META_INSIGHTS` in `ALL_DDL` still carried the original
   four-type CHECK constraint. Fresh installs would have rejected
   inserts with `insight_type = 'causal'`, `'consolidation'`,
   `'verification'`, etc. (eight types from v3).
2. `farming_progress` (Post-Inst #17) was referenced by
   `farming/progress.py` but had no `CREATE TABLE` statement
   anywhere in the codebase.
3. `idx_meta_insights_created` was referenced by the v3 migration
   but absent from `ALL_DDL`.

Fix applied in `src/ctxmtg/storage/schema.py`:

- Expanded `CREATE_META_INSIGHTS` to the 12-type schema with
  `entity_ids` column.
- Added `idx_meta_insights_created` to `CREATE_META_INSIGHTS_INDEXES`.
- Added 14 new DDL constants to the module: `CREATE_FARMING_CYCLES`,
  `CREATE_FARMING_CHECKPOINTS`, `CREATE_FARMING_PROGRESS`,
  `CREATE_QUERY_QUALITY_LOG`, `CREATE_FACTS_PREDICATE_INDEX`,
  `CREATE_MAINTENANCE_LOGS` (8 maintenance tables in one block),
  `CREATE_FARMING_CLUSTERING_PROGRESS`, `CREATE_ENTITY_INTERACTIONS`,
  `CREATE_HIVE_PULL_PROGRESS`, `CREATE_DISTILLER_SUMMARIES`,
  `CREATE_DISTILLER_RELEVANCE_INDEX`,
  `CREATE_LOCAL_INTELLIGENCE_CACHE`.
- Appended all of them to `ALL_DDL`.

The v3 / v4 / v5 `MIGRATIONS` were left untouched, so existing
v1 / v2 databases still upgrade through the same paths. The new
`ALL_DDL` entries use `IF NOT EXISTS` so applying them on already-
upgraded databases is idempotent.

### Verification on fresh install (post-fix)

| Check | Before | After |
|---|---|---|
| Tables on fresh init | 14 | **31** |
| Critical farming/quality tables | 1 of 18 | **18 of 18** |
| `ctxmtg health` | PASS | PASS |
| `ctxmtg ingest` | PASS | PASS |
| `ctxmtg query` -- vector search | PASS | PASS |
| `ctxmtg query` -- quality log warning | yes | **none** |
| `ctxmtg farm run` | hung (300 s timeout) | **completed in 272 ms, 18 of 18 stages succeeded, 5 insights produced** |
| `ctxmtg serve` -- HTTP 200 on /login | PASS | PASS |

Single-file change: `src/ctxmtg/storage/schema.py`, +239 / -1.
