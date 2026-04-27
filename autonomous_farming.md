# Autonomous farming -- design notes (v0.7.1)

> Working design doc for the farming subsystem. The codebase is
> the ground truth; this file explains what the code does, why it
> does it that way, and what the open questions are.

## 1. What "farming" means here

The Borg DB ingests human-in-the-loop conversation: the user
talks, the extractor pulls entities and facts, the storage layer
persists them. Everything up to that point is **synchronous,
single-interaction** work.

Farming is the **asynchronous, cross-interaction** work that
sits on top. It periodically scans the accumulated knowledge
store and produces:

- *Meta-insights* -- statements that describe the whole knowledge
  store, not any single interaction. ("Alice and Bob co-occur
  more often than Bob and Carol." "Mentions of `auth refactor`
  are trending up week over week." "When Alice proposes X, a
  decision tends to follow within two interactions.")
- *Health actions* -- maintenance work that keeps the store
  small, accurate, and queryable. (Mark garbage entities; merge
  duplicate insights; prune stale facts; archive cold rows;
  re-index storage.)

Farming runs as one driven pipeline of 18 stages. The pipeline
is idempotent and resumable: each stage is checkpointed, and a
crashed run can be resumed mid-pipeline without redoing
completed stages.

## 2. The 18 stages

Three groups, ordered:

```
Intelligence (7)            Self-learning (1)    Maintenance (10)
1. EntityAnalytics          8. FeedbackLoop      9.  Rationalizer
2. TrendDetection                                10. Consolidator
3. Clustering                                    11. Pruner
4. TopicModeling                                 12. Completionist
5. GraphAnalysis                                 13. Linker
6. InsightGenerator                              14. Verifier
7. CausalMiner                                   15. Calibrator
                                                 16. Distiller
                                                 17. Archivist
                                                 18. Defragmenter
```

Stage interface (in `interfaces/farming.py`):

```python
class FarmingStage(Protocol):
    def get_name(self) -> str: ...
    async def run(self, sql_store, context) -> StageResult: ...
```

`StageResult` carries `insights`, `actions_taken`, `errors`, and
per-stage `stats`. The pipeline aggregates these into a
`PipelineResult` for the run.

### 2.1 Intelligence stages

These produce new knowledge. None of them mutate facts or
entities.

- **EntityAnalytics** -- `GROUP BY name, entity_type` to count
  mention frequency and `GROUP BY a.name, b.name` for
  co-occurrence. Writes "frequent entity" and "frequently
  co-occurring pair" insights.
- **TrendDetection** -- `GROUP BY name, DATE(created_at)` over a
  configurable window. Linear regression on the per-day count
  determines "rising" vs "falling". Writes "trending entity"
  insights.
- **Clustering** -- pulls entity embeddings from LanceDB, runs
  K-Means or HDBSCAN, writes one insight per non-trivial
  cluster.
- **TopicModeling** -- LDA or BERTopic over interaction text.
  Writes one insight per discovered topic.
- **GraphAnalysis** -- builds a co-occurrence graph (`name`-keyed),
  runs PageRank, writes "central entity" insights for the top
  nodes.
- **InsightGenerator** -- not really intelligence; it's the
  storage adapter that the other intelligence stages share.
  Does deterministic id assignment, idempotent INSERT-OR-IGNORE,
  the schema-checked write.
- **CausalMiner** -- looks for fact pairs separated by a small
  time delta where the second fact references concepts present
  in the first. Currently keys on `subject_entity_id`, which
  prevents cross-interaction discovery; see `audit_findings_v0.7.1.md`
  Section 1.3.

### 2.2 Self-learning

- **FeedbackLoop** -- reads `query_quality_log` for low-confidence
  or low-recall queries. Each such row becomes an "intelligence
  gap" insight tagged with the query phrase, so the next farming
  cycle's Completionist can target it.

### 2.3 Maintenance stages

These mutate facts, entities, and insights.

- **Rationalizer** (Stage 9) -- regex-driven detection of garbage
  entity names: empty strings after `strip()`, non-alphanumeric-
  only, embedded newlines, single-character names, etc. Garbage
  entities get `confidence = 0.1` rather than being deleted; the
  Archivist later moves them to `archive.db`.
- **Consolidator** -- groups duplicate facts within an entity
  (`GROUP BY subject_entity_id, predicate, object_literal`) and
  replaces N copies with one. Cross-interaction blind spot
  noted in the audit.
- **Pruner** -- soft-supersedes facts whose `created_at` is older
  than the configured window. Writes a row to `facts_history`
  before doing the supersede. Time-based; no LLM.
- **Completionist** -- finds entities with too-thin profiles
  (few facts, few co-entities) and queues them for next-cycle
  enrichment. Writes "gap" insights tagged on the entity row.
- **Linker** -- the cross-interaction entity-resolution stage.
  Looks for entities that should be the same logical thing
  across interactions (case variants, leading-article variants,
  whitespace variants) and inserts pseudo-facts on the canonical
  spelling that point to the variants. Name-keyed.
- **Verifier** -- replays each fact through the schema /
  predicate vocabulary check. Anything that fails gets
  `confidence` halved and a `verifier_failed` flag.
- **Calibrator** -- reads recent quality signals and adjusts the
  per-stage weights stored in `farming_calibration`. The next
  cycle uses the adjusted weights.
- **Distiller** (Stage 16) -- the public-facing summarisation
  stage. For each entity with a name, gathers top predicates,
  top co-entities, recent facts, and writes a row to
  `distiller_summaries`. Today the "summary" is a deterministic
  formula; this is where LLM wiring would land first (see
  audit Section 2.3).
- **Archivist** -- moves rows whose confidence has dropped below
  the threshold (Rationalizer's garbage marks, plus aged-out
  cold rows) into `archive.db`, then deletes them from the live
  store.
- **Defragmenter** -- runs SQLite `VACUUM`, rebuilds covering
  indexes, recomputes ANALYZE statistics. Always last.

## 3. Pipeline orchestrator

`farming/pipeline.py` runs the 18 stages in order. Per-stage:

1. Read the resume checkpoint from `farming_checkpoints`. If the
   stage has already succeeded for the current cycle, skip it.
2. Read the per-stage progressive scan offset from
   `farming_progress`. Some stages (Distiller, Linker) walk
   their input table in batches; the offset tells them where to
   resume.
3. Run the stage.
4. Write a per-stage row to `farming_cycles` with `success`,
   `error`, `duration_ms`, `insights_produced`, `actions_taken`.
5. Update the checkpoint and the progress offset atomically.

If any stage raises, the orchestrator records the error,
marks the stage failed in the checkpoint, and moves on. The
next cycle picks up failed stages first.

## 4. Why some stages take an `llm` and some don't

The interface shape is:

```python
class SomeStage:
    def __init__(self, llm: LLMProvider | None = None) -> None:
        ...
        self._llm = llm
```

Sixteen of the 18 stages take an `llm` parameter. The audit
(Section 2.2) confirms zero of them currently call
`self._llm.generate()`. The pattern is a deliberate
"plumbing-first" decision:

- The pipeline orchestrator already passes `llm` everywhere.
- The stage already stores it.
- Adding LLM-driven logic to a stage is a one-method change.

This is not "dead code" in the bad sense. It's an intentional
pre-wired hook so that each stage can adopt LLM augmentation
without touching the pipeline contract. The current state is
"all hooks installed, none called".

The two stages that don't take `llm` -- `ClusteringStage` and
`RationalizerStage` -- are deterministic by design. Clustering
is K-Means / HDBSCAN over embeddings; Rationalizer is regex
over a string. Adding an LLM would change the contract.

## 5. The keying question

(Mirrors `audit_findings_v0.7.1.md` Section 1; restated here so
this doc is readable on its own.)

Entity ids are deterministically derived per interaction (see
`storage/id_gen.py`). Same logical "Alice" mentioned in 10
interactions has 10 entity rows with 10 different ids. Stages
that join on `entities.id` therefore cannot connect the same
logical entity across interactions; stages that group on
`entities.name` can.

Of the 18 stages:

- **8 are name-keyed and cross-interaction safe**:
  EntityAnalytics, TrendDetection, TopicModeling, GraphAnalysis,
  Completionist, Linker, Verifier, Distiller. (Plus Clustering
  via the embedding store.)
- **3 are id-keyed by design** because the unit of work is a
  single row's lifecycle: Rationalizer, Pruner, Archivist.
- **2 are id-keyed structural blind spots**: CausalMiner and
  Consolidator. Both should move to name keying. Both are
  one-file fixes guarded behind a regression test.

## 6. The merge UI is a name rename, not an ID merge

`/entities/merge` runs:

```sql
UPDATE entities SET name = :new WHERE LOWER(name) = LOWER(:old)
```

It does not consolidate entity rows; it does not repoint facts.
The per-interaction id model already keeps entity rows
disjoint, so a merge that renamed all variants of "alice" to
"Alice" is sufficient: every name-keyed downstream stage
now treats them as the same logical entity, and every
id-keyed stage continues to operate on the per-row lifecycle
correctly.

This was a non-obvious design choice and is explicitly the
right one. The merge UI's job is "make the canonical name
everywhere consistent", not "merge ten entity rows into one".

## 7. Idle-time scheduling

`farming/scheduler.py` is the wrapper around the pipeline
orchestrator that decides *when* to run. The `serve` command
starts a background scheduler that wakes every N seconds,
checks whether the user has been idle (no requests for M
seconds), and starts a farming cycle if so. A cycle that
crosses a non-idle boundary completes its current stage, then
parks until idle returns.

`ctxmtg farm run` bypasses the scheduler and runs the pipeline
synchronously. This is the "force a cycle now" entry point used
by tests and by users who want to confirm the pipeline still
works.

## 8. Phase 4.0 -- proposed quick wins (revised against v0.7.1)

The Stage-era list of "Phase 4.0 quick wins" needed re-derivation
because several items have already shipped. Revised list:

| # | Item | Status / Plan |
|---|---|---|
| 1 | Web LLM wiring (dashboard + completions) | **Already shipped** in Post-Install #8. |
| 2 | Embedding fn wiring through query stack | **Already shipped** in Post-Install #11; v0.7.1 smoke test confirms vector search active and `query_quality_log` populated. |
| 3 | Wire Distiller to LLM | Open. One stage, one method. Highest ROI of the LLM-wiring opportunities. |
| 4 | CausalMiner / Consolidator name keying | Open. Two one-file fixes; behavioural change so should ship with regression tests. |
| 5 | LLM Test-Connection button per role | Open. Round-trip `generate("hello")` from the role-config UI. |
| 6 | Merge audit log + reverse merge | Open. New table + two routes. |
| 7 | Tighten `entities/delete` insight cleanup (defect 13) | Open. Replace `LIKE` match with JSON-contains. |
| 8 | Refresh `entity_list_fragment` similar-names section | Open. UI consistency. |
| 9 | Minimal smoke test suite in the public release | Open. Required before #4 ships. |

Items 3, 4, 7, 8 are the true near-term batch. Items 5, 6 are
features. Item 9 is the prerequisite.

## 9. Open design questions

### 9.1 Should farming write back to facts at all?

Today four maintenance stages mutate facts: Pruner (supersede),
Verifier (confidence half), Consolidator (delete duplicates),
Archivist (move to cold). All four are non-destructive --
supersede preserves history, Verifier just adjusts confidence,
Consolidator's deletions are deduplications, Archivist moves
rather than drops. Even so, the "farming touches the live store"
contract makes farming dependent on the same ACID guarantees as
ingest, and a buggy farming run can corrupt the live store.

Open: should the maintenance stages write to a *staging* table
that ingest reads through a view, so a buggy farming run is
recoverable by truncating staging? Probably yes for v0.8.

### 9.2 Should the pipeline be a DAG?

Today the 18 stages run as a strict linear sequence. Some
stages have real dependencies (Defragmenter must be last;
Archivist must be after Rationalizer); most do not.

Open: model the pipeline as a DAG, parallelise independent
stages, target a 5-10x runtime reduction. Probably yes for
v0.9; the orchestrator rewrite is non-trivial.

### 9.3 How should agentic stages plug in?

The "accept-but-ignore" `llm` hook is the staging area for
agentic stages: a stage that calls `self._llm.generate()` with
a tool-use prompt, lets the model issue SQL through a
guarded executor, iterates. The Distiller wiring (item 3 in
Section 8) is the smallest possible first agentic stage.

Open: what's the safety contract for SQL-issuing stages? At
minimum: read-only role, statement timeout, query budget per
stage, structured logging. Probably codify in
`interfaces/farming.py` as a separate `AgenticFarmingStage`
protocol that extends `FarmingStage`.

## 10. Glossary

- **Cycle** -- one complete pass of the 18-stage pipeline.
- **Stage** -- one of the 18 sub-units within a cycle.
- **Checkpoint** -- per-stage row in `farming_checkpoints`
  recording last-success cycle id.
- **Progress offset** -- per-stage row in `farming_progress`
  recording last-scanned input id, used by stages that walk
  their input table in batches.
- **Garbage entity** -- entity row whose name failed the
  Rationalizer's regex checks. Marked `confidence = 0.1` and
  later archived.
- **Cold row** -- entity or fact row whose `last_seen` is older
  than the configured cold threshold. Eligible for archival.
- **Distiller summary** -- the per-entity row in
  `distiller_summaries` that the query stack reads when
  answering "what do you know about X".
