# Audit findings

> Living audit of the codebase.  Originally consolidated at commit
> `e460c74` (2026-04-27, v0.7.1) and updated as items land.  Three
> audits in scope:
>
>   1. The 18-stage farming pipeline -- name-keyed vs id-keyed
>   2. LLM wiring across the codebase
>   3. The `/entities/merge` defect catalog re-walk
>
> Each section ends with a short, actionable list of open items.

---

## 1. 18-stage farming pipeline -- keying audit

The persistent design tension here is the **per-interaction entity ID**
model documented in `storage/id_gen.py`.  Every interaction that
mentions "Alice" stores a fresh entity row with a fresh deterministic
ID, so "Alice" appearing in 10 interactions is 10 entity rows.  Some
stages are cleanly designed against this (they group by `name`) and
some are not (they group by or join on `subject_entity_id`).  The
latter category cannot connect facts about the same logical entity
across interactions.

The merge UI route (`/entities/merge`, see Section 3) is a
**name rename**, not an ID merge -- it updates rows so that all
variants of "alice" share the canonical name "Alice".  This means
the keying lens applies post-rename: name-keyed stages naturally
benefit from the rename; id-keyed stages do not.

### 1.1 Stage-by-stage assessment

| # | Stage | Keying | Cross-interaction safe? | Notes |
|---|---|---|---|---|
| 1 | EntityAnalyticsStage | **name** | Yes | `GROUP BY name, entity_type`; `GROUP BY a.name, b.name` for co-occurrence. |
| 2 | TrendDetectionStage | **name** | Yes | `GROUP BY name, DATE(created_at)`. |
| 3 | ClusteringStage | name (via embeddings) | Yes | Operates on the LanceDB vector store keyed by entity name, not id. |
| 4 | TopicModelingStage | name | Yes | Reads `entities.name`. |
| 5 | GraphAnalysisStage | **name** | Yes | `GROUP BY a.name, b.name`; PageRank keyed on names. |
| 6 | InsightGeneratorStage | mixed | N/A | Writes insights; doesn't query entities by id or name in a way that matters here. |
| 7 | CausalMinerStage | **id** | **No -- structural blind spot** | `JOIN facts f2 ON f1.subject_entity_id = f2.subject_entity_id`. Same logical entity in two different interactions has two different ids; the join silently misses every cross-interaction temporal pattern. |
| 8 | FeedbackLoopStage | n/a | N/A | Operates on `query_quality_log`, not entities. |
| 9 | RationalizerStage | **id** (by design) | Yes | Marks individual entity rows with `confidence = 0.1`. Per-row action; id is the correct key. |
| 10 | ConsolidatorStage | **id** | **Partial blind spot** | `GROUP BY subject_entity_id, predicate, object_literal` deduplicates within one entity row. Cannot dedup the same fact stated in multiple interactions about the same logical person. |
| 11 | PrunerStage | id (by design) | Yes | Supersedes individual fact rows; id is the correct key. |
| 12 | CompletionistStage | **name** | Yes | `GROUP BY name, entity_type`; gap detection by name. |
| 13 | LinkerStage | **name** | Yes | Looks up `WHERE name = :name`; cross-interaction by design. |
| 14 | VerifierStage | name (via JOIN) | Yes | `JOIN entities ... WHERE e2.name = :entity_name`. |
| 15 | CalibratorStage | n/a | N/A | Operates on insights and quality signals. |
| 16 | DistillerStage | **name** | Yes | `GROUP BY name COLLATE NOCASE`; the public-facing "what does the system know about Alice" stage. |
| 17 | ArchivistStage | id (by design) | Yes | Archives individual entity rows + their facts. Id is the correct key. |
| 18 | DefragmenterStage | n/a | N/A | Operates on storage, not entities. |

### 1.2 Findings

- **Eight pure name-keyed stages** (1, 2, 4, 5, 12, 13, 14, 16) work
  correctly across interactions and benefit from the merge UI's
  rename behaviour.
- **Three "by design" id-keyed stages** (9 Rationalizer, 11 Pruner,
  17 Archivist) need ids because their unit of work is an individual
  row (mark, supersede, archive). Correct as is.
- **Two structural blind spots** (7 CausalMiner, 10 Consolidator) use
  ids but operate on logical entities, not row lifecycle. They miss
  cross-interaction patterns and duplicate facts about the same
  logical entity stated in different interactions. Both are real
  quality losses and both are fixable -- see Section 1.3.
- **Five n/a stages** (3, 6, 8, 15, 18) don't have a keying problem
  because they don't query the `entities` table by name or id in a
  way that exposes the design tension.

### 1.3 Open items (farming keying)

1. **CausalMinerStage** -- replace
   `JOIN facts f2 ON f1.subject_entity_id = f2.subject_entity_id`
   with a name-aware join: `JOIN entities e1 ON f1.subject_entity_id = e1.id`,
   `JOIN entities e2 ON f2.subject_entity_id = e2.id`,
   `WHERE e1.name = e2.name COLLATE NOCASE`. Filed to plan; **not**
   fixed in v0.7.1 because it is a behavioural change.
2. **ConsolidatorStage** -- same shape: change
   `GROUP BY subject_entity_id, predicate, object_literal` to
   `GROUP BY e.name, predicate, object_literal` after joining
   through entities. Same behavioural-change reasoning.
3. Add a unit test that ingests "Alice did X" in two separate
   interactions and asserts CausalMiner / Consolidator handle them
   as one logical Alice. Currently no such test exists.

---

## 2. LLM wiring audit

### 2.1 Live wiring

Searched `src/ctxmtg/` for `self._llm.generate(`, `self.llm.generate(`,
and `llm.generate(`. Eleven hits across nine modules:

| File | Line | Role |
|---|---|---|
| `extraction/llm_verifier.py` | 211 | Extraction role -- LLM disambiguation of NER candidates. |
| `query/llm_interpreter.py` | 184 | Query-planning role -- intent parsing. |
| `query/synthesizer.py` | 204 | Synthesis role -- final answer composition. |
| `query/llm_fusion.py` | 186 | Fusion role -- multi-source result blending. |
| `query/informed_retrieval.py` | 368, 836 | Retrieval role -- two SQL/vector bridge prompts. |
| `sync/context_enricher.py` | 457 | Sync (extraction-adjacent) -- enrich pulled context. |
| `profile/assembler.py` | 19, 81 | Profile role -- assemble domain profile context. |
| `cli.py` | 862 | Evaluation -- `ctxmtg evaluate` LLM judge. |
| `interfaces/llm.py` | 63 | Reference call in the interface docstring. |

The query stack is fully wired, extraction is wired, profile is wired,
sync is wired, evaluation is wired.

### 2.2 Dead wiring (the big finding)

`grep -r 'self._llm.generate' src/ctxmtg/farming/` returns **zero
matches**.  Every farming stage that accepts an `llm` parameter
(`__init__(self, llm: LLMProvider | None = None)`) stores it as
`self._llm` but never invokes it.  Sixteen stages are in this state:

| Stage | Stores `self._llm` | Calls `.generate()` |
|---|---|---|
| EntityAnalyticsStage | yes | no |
| TrendDetectionStage | yes | no |
| TopicModelingStage | yes | no |
| GraphAnalysisStage | yes | no |
| InsightGeneratorStage | yes | no |
| CausalMinerStage | yes | no |
| FeedbackLoopStage | yes | no |
| ConsolidatorStage | yes | no |
| PrunerStage | yes | no |
| CompletionistStage | yes | no |
| LinkerStage | yes | no |
| VerifierStage | yes | no |
| CalibratorStage | yes | no |
| DistillerStage | yes | no |
| ArchivistStage | yes | no |
| DefragmenterStage | yes | no |
| ClusteringStage | n/a (doesn't accept) | no |
| RationalizerStage | n/a (doesn't accept) | no |

**Net:** the entire farming subsystem is LLM-aware on the surface
and LLM-blind in practice. Adding LLM-augmented logic to even one
stage (e.g. Distiller's natural-language summary) would unlock real
quality wins without changing any interface. The plumbing is
already there.

### 2.3 Open items (LLM wiring)

1. **Distiller** is the highest-value first target. The stage
   already produces a summary string; today it's a simple
   formula over `top_predicates` and `top_co_entities`. Wiring
   `self._llm.generate()` with a small prompt would produce a
   one-line natural summary that the query stack could surface
   directly when answering "What do you know about X?".
2. **CausalMiner** as a second target. After the keying fix
   (Section 1.3), an LLM call could verify each mined causal
   pair before it is written -- removing the bulk of the false
   positives that motivated the "skip-linker-output" exclusion
   added in Post-Install #19.
3. **Web UI -- Test Connection** button per role in the LLM
   roles page. Today users save credentials and find out they
   are wrong only when the relevant stage runs. A 1-shot
   `generate("hello")` round-trip would close the gap. (Not a
   farming concern but lives in the same wiring spec.)

---

## 3. `/entities/merge` defect catalog -- re-walk

Stage's earlier audit catalogued 13 defects in the merge route. The
current code (commit `e460c74`, `web/routes/entities.py`) has been
re-read end to end. Status as of v0.7.1:

| # | Stage's original finding | v0.7.1 status |
|---|---|---|
| 1 | `execute_sql` did not commit DML | **FIXED** -- now uses `db.execute()` + `await db.commit()` (Post-Install #13). |
| 2 | No transaction wrapping (no `BEGIN IMMEDIATE`) | Open. Multiple UPDATEs followed by a single commit. Crash partway through leaves the rename half-applied. |
| 3 | Update by id only, missed other rows with same name | **FIXED** -- now `UPDATE entities SET name = :new WHERE LOWER(name) = LOWER(:old)`. |
| 4 | No fact repointing | Reframed -- not a defect for a name-rename merge. Facts still reference the same entity rows; only the name changed. |
| 5 | No vector deletion / re-embedding | Open. Embeddings are tied to `entities.id`, not name, so old vectors stay correct but their associated text is now stale relative to the new canonical name. Low-impact; semantic search still works. |
| 6 | No idempotency guard | Open. Running merge with the same canonical name twice is a no-op in practice (the `WHERE LOWER(name) = LOWER(:old)` skips the canonical row), but no explicit guard. |
| 7 | No persistent audit log | Open. `logger.info("entities_merged", ...)` writes to structlog only. A `merge_audit_log` table would let users see and reverse merges. |
| 8 | No reverse operation | Open. Direct consequence of #7. |
| 9 | No CSRF protection | Open. FastAPI doesn't add CSRF by default. The localhost-only deployment model (per `web/auth.py`) softens this. |
| 10 | No rate limiting | Open. Same softening. |
| 11 | Weak request validation | Open. Only `if not ids or not canonical_name.strip()`. No length cap, no character class check, no max-IDs cap. |
| 12 | Some adjacent routes missed `_auth=Depends(require_auth)` | **FIXED** -- `entities/delete`, `entities/merge-batch`, `entities/delete-batch` now all gate on `require_auth`. |
| 13 | `DELETE FROM meta_insights WHERE title LIKE :pattern` matches unrelated insights | Open. The pattern is `%{entity_name}%` which over-matches when the name is a substring of unrelated titles. A targeted `entity_ids JSON_CONTAINS` would be more precise. |

**Score:** 3 fixed, 10 open. Most opens are localhost-deployment
soft (CSRF, rate limit) or feature-shaped (audit log, reverse).

### 3.1 Other things the re-read surfaced

- **`entity_list_fragment`** (the htmx refresh route) reloads
  `_find_duplicate_candidates` but not `_find_similar_names`, so
  refreshing the page after a merge updates one section and
  silently drops the other. Not a correctness bug but visible UX.
- **`merge_batch` and `delete_batch` log the full form payload**
  via `logger.info(..., form_items={k: v for k, v in form.items()})`.
  Today the form fields are non-sensitive (entity ids and a
  canonical-name string) but the pattern would leak if any new
  sensitive field is added.

### 3.2 Open items (entity merge)

1. Wrap the merge UPDATEs in `BEGIN IMMEDIATE ... COMMIT` (defect 2).
2. Add a `merge_audit_log` table and write one row per merge with
   the old names, the canonical name, the row count, and the
   timestamp (defect 7). Enables a reverse operation (defect 8).
3. Tighten `entities/delete` insight cleanup: replace the
   `LIKE :pattern` match with a JSON-contains check against
   `meta_insights.entity_ids` (defect 13).
4. Update `entity_list_fragment` to also refresh the similar-names
   pairs section.

---

## 4. Cross-cutting

- **Stale module docstring** in `farming/__init__.py` ("17 stages",
  "9 maintenance stages", "16-stage orchestrator") was fixed
  alongside this audit.
- **No tests in the public release.** Several of the open items
  above (the keying fix, the merge audit log, the LLM wiring on
  Distiller) would each ideally land with a regression test. The
  release ships runtime-only by design; a future decision is
  whether to ship a minimal smoke-test suite alongside.

## 5. Audit verdict

The codebase is in better shape than Stage's earlier audit
implied. Most of the high-impact "Stage finding" items either
turned out to be already fixed (merge commit, embedding wiring,
DDL completeness in v0.7.1, web LLM loading), softened by the
rename-based merge model (fact repointing, vector deletion), or
deferred to a separate behavioural-change PR (CausalMiner /
Consolidator keying, Distiller LLM wiring).

The remaining open items cluster into three buckets:

- **Localhost-deployment soft items** (CSRF, rate limit,
  request validation): defer until a non-localhost mode appears.
- **Feature-shaped items** (merge audit log, reverse merge,
  Distiller LLM wiring, Test Connection button): plan into the
  next minor release as a coherent batch.
- **Behavioural fixes** (CausalMiner / Consolidator name keying,
  insight LIKE-match precision): each is a one-file change with
  a clear regression-test shape; do them next, gated on adding
  a small smoke-test suite.
