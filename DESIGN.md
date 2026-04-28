# The_Borg_DB design -- WORKING SPEC

> Working design spec for the system.  The codebase is the ground
> truth; this file explains what the code does, why it does it
> that way, and what is planned next.  Edit, argue with, replace.
>
> Two halves, one document, one load:
>
>   - **Part I -- Farming subsystem** describes the 18-stage
>     idle-time pipeline that produces meta-insights and keeps
>     the knowledge store healthy.
>   - **Part II -- LLM strategy** maps every LLM call site,
>     enumerates the prompt-experimentation landscape, and
>     sketches the autonomy roadmap.
>
> Companion to `audit_findings.md` (the live audit of known
> issues and deliberate non-goals) and `CHANGELOG.md` (what
> has actually shipped).  This doc is the "what we're building
> toward" thread.

---

## Status board (as of `[Unreleased]`)

| Item | State |
|---|---|
| Per-interaction entity-id model | Stable. Documented invariant. |
| 18-stage farming pipeline | Stable. Idempotent + resumable. |
| 4-layer prompt assembler + 5 role templates | Stable. Live in production. |
| LLM extraction verifier (CLI ingest) | Stable. |
| LLM extraction verifier (HTTP ingest) | **Shipped in `[Unreleased]`** -- parity with CLI. |
| Abstractive interaction summary | **Shipped in `[Unreleased]`** -- TextRank fallback. |
| **Phase 4.1 -- Distiller LLM wiring** | **Shipped.** First live farming hook. |
| Phase 4.2 -- TopicModeling + InsightGenerator wiring | Not started. |
| Phase 4.3 -- CausalMiner LLM verification | Blocked on keying fix. |
| Eval harness (Section II.7) | Not started. **Highest-leverage gate.** |
| L2 self-tuning (Calibrator → weights table) | Skeleton only. |
| L3 self-improving prompts | Not started; needs eval harness. |
| L4 agentic farming | Not started; needs L3. |

After `[Unreleased]` ships, the farming pipeline has 1 live LLM
hook (Distiller) and 15 still-dead hooks; the ingest pipeline
has 2 live LLM call sites (verifier + abstractive summary)
exposed equally on CLI and HTTP.

---

# Part I -- Farming subsystem

## I.1 What "farming" means here

The Borg DB ingests human-in-the-loop conversation: the user
talks, the extractor pulls entities and facts, the storage layer
persists them.  Everything up to that point is **synchronous,
single-interaction** work.

Farming is the **asynchronous, cross-interaction** work that
sits on top.  It periodically scans the accumulated knowledge
store and produces:

- *Meta-insights* -- statements that describe the whole knowledge
  store, not any single interaction.  ("Alice and Bob co-occur
  more often than Bob and Carol."  "Mentions of `auth refactor`
  are trending up week over week."  "When Alice proposes X, a
  decision tends to follow within two interactions.")
- *Health actions* -- maintenance work that keeps the store
  small, accurate, and queryable.  (Mark garbage entities; merge
  duplicate insights; prune stale facts; archive cold rows;
  re-index storage.)

Farming runs as one driven pipeline of 18 stages.  The pipeline
is idempotent and resumable: each stage is checkpointed, and a
crashed run can be resumed mid-pipeline without redoing
completed stages.

## I.2 The 18 stages

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
                                                 16. Distiller   *LLM-wired*
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
per-stage `stats`.  The pipeline aggregates these into a
`PipelineResult` for the run.

### I.2.1 Intelligence stages

These produce new knowledge.  None of them mutate facts or
entities.

- **EntityAnalytics** -- `GROUP BY name, entity_type` to count
  mention frequency and `GROUP BY a.name, b.name` for
  co-occurrence.  Writes "frequent entity" and "frequently
  co-occurring pair" insights.
- **TrendDetection** -- `GROUP BY name, DATE(created_at)` over a
  configurable window.  Linear regression on the per-day count
  determines "rising" vs "falling".  Writes "trending entity"
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
  in the first.  Currently keys on `subject_entity_id`, which
  prevents cross-interaction discovery; see `audit_findings.md`
  Section 1.3.

### I.2.2 Self-learning

- **FeedbackLoop** -- reads `query_quality_log` for low-confidence
  or low-recall queries.  Each such row becomes an "intelligence
  gap" insight tagged with the query phrase, so the next farming
  cycle's Completionist can target it.

### I.2.3 Maintenance stages

These mutate facts, entities, and insights.

- **Rationalizer** (Stage 9) -- regex-driven detection of garbage
  entity names: empty strings after `strip()`, non-alphanumeric-
  only, embedded newlines, single-character names, etc.  Garbage
  entities get `confidence = 0.1` rather than being deleted; the
  Archivist later moves them to `archive.db`.
- **Consolidator** -- groups duplicate facts within an entity
  (`GROUP BY subject_entity_id, predicate, object_literal`) and
  replaces N copies with one.  Cross-interaction blind spot
  noted in the audit.
- **Pruner** -- soft-supersedes facts whose `created_at` is older
  than the configured window.  Writes a row to `facts_history`
  before doing the supersede.  Time-based; no LLM.
- **Completionist** -- finds entities with too-thin profiles
  (few facts, few co-entities) and queues them for next-cycle
  enrichment.  Writes "gap" insights tagged on the entity row.
- **Linker** -- the cross-interaction entity-resolution stage.
  Looks for entities that should be the same logical thing
  across interactions (case variants, leading-article variants,
  whitespace variants) and inserts pseudo-facts on the canonical
  spelling that point to the variants.  Name-keyed.
- **Verifier** -- replays each fact through the schema /
  predicate vocabulary check.  Anything that fails gets
  `confidence` halved and a `verifier_failed` flag.
- **Calibrator** -- reads recent quality signals and adjusts the
  per-stage weights stored in `farming_calibration`.  The next
  cycle uses the adjusted weights.
- **Distiller** (Stage 16) -- the public-facing summarisation
  stage.  For each entity with a name, gathers top predicates,
  top co-entities, recent facts, and writes a row to
  `distiller_summaries`.  As of Phase 4.1 (`[Unreleased]`),
  when a `farming`-role LLM is configured, the deterministic
  summary is replaced by a one-sentence natural-language
  summary; otherwise the deterministic baseline is retained.
  See Section II.6.2.1 for the wiring details.
- **Archivist** -- moves rows whose confidence has dropped below
  the threshold (Rationalizer's garbage marks, plus aged-out
  cold rows) into `archive.db`, then deletes them from the live
  store.
- **Defragmenter** -- runs SQLite `VACUUM`, rebuilds covering
  indexes, recomputes ANALYZE statistics.  Always last.

## I.3 Pipeline orchestrator

`farming/pipeline.py` runs the 18 stages in order.  Per-stage:

1. Read the resume checkpoint from `farming_checkpoints`.  If the
   stage has already succeeded for the current cycle, skip it.
2. Read the per-stage progressive scan offset from
   `farming_progress`.  Some stages (Distiller, Linker) walk
   their input table in batches; the offset tells them where to
   resume.
3. Run the stage.
4. Write a per-stage row to `farming_cycles` with `success`,
   `error`, `duration_ms`, `insights_produced`, `actions_taken`.
5. Update the checkpoint and the progress offset atomically.

If any stage raises, the orchestrator records the error,
marks the stage failed in the checkpoint, and moves on.  The
next cycle picks up failed stages first.

## I.4 Why some stages take an `llm` and some don't

The interface shape is:

```python
class SomeStage:
    def __init__(self, llm: LLMProvider | None = None) -> None:
        ...
        self._llm = llm
```

Sixteen of the 18 stages take an `llm` parameter.  As of Phase
4.1, **one** of them (`DistillerStage`) actually calls
`self._llm.generate()`; the other 15 still hold the parameter
for future wirings.  The pattern is a deliberate
"plumbing-first" decision:

- The pipeline orchestrator already passes `llm` everywhere.
- The stage already stores it.
- Adding LLM-driven logic to a stage is a one-method change
  (see `farming/distiller.py::_maybe_llm_summary` for the
  reference shape).

This is not "dead code" in the bad sense.  It's an intentional
pre-wired hook so that each stage can adopt LLM augmentation
without touching the pipeline contract.

The two stages that don't take `llm` -- `ClusteringStage` and
`RationalizerStage` -- are deterministic by design.  Clustering
is K-Means / HDBSCAN over embeddings; Rationalizer is regex
over a string.  Adding an LLM would change the contract.

## I.5 The keying question

(Mirrors `audit_findings.md` Section 1; restated here so this
doc is readable on its own.)

Entity ids are deterministically derived per interaction (see
`storage/id_gen.py`).  Same logical "Alice" mentioned in 10
interactions has 10 entity rows with 10 different ids.  Stages
that join on `entities.id` therefore cannot connect the same
logical entity across interactions; stages that group on
`entities.name` can.

Of the 18 stages:

- **8 are name-keyed and cross-interaction safe**:
  EntityAnalytics, TrendDetection, TopicModeling, GraphAnalysis,
  Completionist, Linker, Verifier, Distiller.  (Plus Clustering
  via the embedding store.)
- **3 are id-keyed by design** because the unit of work is a
  single row's lifecycle: Rationalizer, Pruner, Archivist.
- **2 are id-keyed structural blind spots**: CausalMiner and
  Consolidator.  Both should move to name keying.  Both are
  one-file fixes guarded behind a regression test.

## I.6 The merge UI is a name rename, not an ID merge

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
right one.  The merge UI's job is "make the canonical name
everywhere consistent", not "merge ten entity rows into one".

## I.7 Idle-time scheduling

`farming/scheduler.py` is the wrapper around the pipeline
orchestrator that decides *when* to run.  The `serve` command
starts a background scheduler that wakes every N seconds,
checks whether the user has been idle (no requests for M
seconds), and starts a farming cycle if so.  A cycle that
crosses a non-idle boundary completes its current stage, then
parks until idle returns.

`ctxmtg farm run` bypasses the scheduler and runs the pipeline
synchronously.  This is the "force a cycle now" entry point used
by tests and by users who want to confirm the pipeline still
works.

## I.8 Phase 4.0 -- proposed quick wins (revised against `[Unreleased]`)

| # | Item | Status / Plan |
|---|---|---|
| 1 | Web LLM wiring (dashboard + completions) | **Shipped** in Post-Install #8. |
| 2 | Embedding fn wiring through query stack | **Shipped** in Post-Install #11. |
| 3 | Wire Distiller to LLM | **Shipped** in `[Unreleased]` (Phase 4.1). |
| 3b | HTTP ingest LLM parity (verifier + abstractive summary) | **Shipped** in `[Unreleased]`. |
| 4 | CausalMiner / Consolidator name keying | Open.  Two one-file fixes; behavioural change so should ship with regression tests. |
| 5 | LLM Test-Connection button per role | Open.  Round-trip `generate("hello")` from the role-config UI. |
| 6 | Merge audit log + reverse merge | Open.  New table + two routes. |
| 7 | Tighten `entities/delete` insight cleanup (defect 13) | Open.  Replace `LIKE` match with JSON-contains. |
| 8 | Refresh `entity_list_fragment` similar-names section | Open.  UI consistency. |
| 9 | Minimal smoke test suite in the public release | Open.  Required before #4 ships. |

Items 4, 7, 8 are the true near-term batch.  Items 5, 6 are
features.  Item 9 is the prerequisite.

## I.9 Open design questions

### I.9.1 Should farming write back to facts at all?

Today four maintenance stages mutate facts: Pruner (supersede),
Verifier (confidence half), Consolidator (delete duplicates),
Archivist (move to cold).  All four are non-destructive --
supersede preserves history, Verifier just adjusts confidence,
Consolidator's deletions are deduplications, Archivist moves
rather than drops.  Even so, the "farming touches the live store"
contract makes farming dependent on the same ACID guarantees as
ingest, and a buggy farming run can corrupt the live store.

Open: should the maintenance stages write to a *staging* table
that ingest reads through a view, so a buggy farming run is
recoverable by truncating staging?  Probably yes for v0.8.

### I.9.2 Should the pipeline be a DAG?

Today the 18 stages run as a strict linear sequence.  Some
stages have real dependencies (Defragmenter must be last;
Archivist must be after Rationalizer); most do not.

Open: model the pipeline as a DAG, parallelise independent
stages, target a 5-10x runtime reduction.  Probably yes for
v0.9; the orchestrator rewrite is non-trivial.

### I.9.3 How should agentic stages plug in?

The "accept-but-ignore" `llm` hook is the staging area for
agentic stages: a stage that calls `self._llm.generate()` with
a tool-use prompt, lets the model issue SQL through a
guarded executor, iterates.  The Distiller wiring (Phase 4.1)
is the smallest possible first agentic step -- but it is a
narrator, not an actor.  A real agentic stage is several phases
out; see Section II.8.

Open: what's the safety contract for SQL-issuing stages?  At
minimum: read-only role, statement timeout, query budget per
stage, structured logging.  Probably codify in
`interfaces/farming.py` as a separate `AgenticFarmingStage`
protocol that extends `FarmingStage`.

## I.10 Glossary

- **Cycle** -- one complete pass of the 18-stage pipeline.
- **Stage** -- one of the 18 sub-units within a cycle.
- **Checkpoint** -- per-stage row in `farming_checkpoints`
  recording last-success cycle id.
- **Progress offset** -- per-stage row in `farming_progress`
  recording last-scanned input id, used by stages that walk
  their input table in batches.
- **Garbage entity** -- entity row whose name failed the
  Rationalizer's regex checks.  Marked `confidence = 0.1` and
  later archived.
- **Cold row** -- entity or fact row whose `last_seen` is older
  than the configured cold threshold.  Eligible for archival.
- **Distiller summary** -- the per-entity row in
  `distiller_summaries` that the query stack reads when
  answering "what do you know about X".  Phase 4.1 wired the
  LLM rewrite path; the deterministic formula is the fallback.

---

# Part II -- LLM strategy

## II.1 Frame -- and the surprise

The system already has a **four-layer versioned prompt assembler**
(`src/ctxmtg/llm/prompt_assembler.py`) with templates on disk
under `prompts/`:

```
prompts/
  base/
    v1.0.0.txt              <- shared safety + format rules
  stages/
    extraction/
      v0.9.0.txt            <- previous prompt
      v1.0.0.txt            <- current prompt (adds anti-markdown rule)
    query_planning/
    retrieval/
    synthesis/
    farming/
      v0.9.0.txt
      v1.0.0.txt
```

Five of the six user-facing roles already load their prompt
through this assembler.  The version diff between
`extraction/v0.9.0.txt` and `extraction/v1.0.0.txt` is a real
prompt experiment that already shipped: v1.0.0 added "Output ONLY
a single valid JSON object.  No markdown, no backticks."

**We are not designing a prompt-experimentation system from
scratch.  We are under-using one that already exists.  The piece
we're missing is measurement, not authoring.**

Three implications drive the rest of this part:

- Prompt experimentation is **a configuration change, not a code
  change**.  New prompt strategy = new file at
  `prompts/stages/<role>/vX.Y.Z.txt`.
- The 4-layer model lets us hold base safety constant while
  varying stage instructions; vary stage instructions while
  holding domain overlay constant; etc.  Each layer has its own
  test cycle.
- Every claim of the form "prompt A is better than prompt B" is
  vibes until we have an eval harness.  Building that harness is
  the highest-value next move on this whole roadmap.

## II.2 The 4-layer architecture

| Layer | What it controls | Who edits it | Test cycle |
|---|---|---|---|
| 1 -- Base | Identity, safety, format invariants.  ~100-200 tokens. | Project maintainer.  Touch rarely. | Released; evaluate against full corpus when changed. |
| 2 -- Stage | Per-role task instructions.  ~300-800 tokens.  **Most experiments live here.** | Project maintainer.  Versioned per role. | A/B test per role on a fixture corpus. |
| 3 -- Domain | Slot-injected from active `DomainProfile`.  ~100-400 tokens.  Includes entity types, terminology, reasoning patterns. | User (per profile). | A/B test per domain on a domain-specific corpus. |
| 4 -- User prefs | Per-user customization (preferred summary length, priority topics, output language).  ~50-150 tokens. | End user. | Opt-in only; no system-wide test cycle. |

Three categories of "different prompts -> different outcomes":

- **Layer 2 variants** -- "what does this role do better with a
  chain-of-thought prefix?"
- **Layer 3 variants** -- "what entity types should the legal
  domain extract that the medical domain shouldn't?"
- **Layer 4 variants** -- "what does this user want answers to
  look like?"

The first two are systematic experimentation.  The third is
personalisation and is downstream of #1 and #2 working well.

## II.3 Map of the live LLM surface

| Role | Live in | Loaded by PromptAssembler? | Today's status |
|---|---|---|---|
| Extraction | `extraction/llm_verifier.py:211` | Yes | Live.  v0.9.0 -> v1.0.0 already shipped.  Now also exposed on HTTP ingest (`web/routes/ingest.py`). |
| Abstractive summary | `extraction/pipeline.py::_maybe_llm_summary` | Module constants (not assembler yet) | **Live as of `[Unreleased]`.**  TextRank is the deterministic fallback. |
| Query Planning | `query/llm_interpreter.py:184` | Yes | Live.  v1.0.0 only. |
| Retrieval | `query/informed_retrieval.py:368, 836` | Yes | Live, two call sites.  v1.0.0 only. |
| Synthesis | `query/synthesizer.py:204` | Yes | Live.  v1.0.0 only. |
| Fusion | `query/llm_fusion.py:186` | Yes | Live.  v1.0.0 only. |
| Sync (extraction-adjacent) | `sync/context_enricher.py:457` | (uses extraction template) | Live. |
| Profile | `profile/assembler.py:19, 81` | Yes (wrapper) | Live. |
| Evaluation (LLM-as-judge) | `cli.py:862` | Yes | Live, used by `ctxmtg evaluate`. |
| Farming -- Distiller | `farming/distiller.py::_maybe_llm_summary` | Module constants (not assembler yet) | **Live as of `[Unreleased]`.**  Phase 4.1.  Deterministic `_build_summary` is the fallback. |
| Farming -- 15 other stages | None | Template exists, no caller | **Dead** -- 15 stages store `self._llm`, none call `.generate()`. |

Eleven call sites today, six user-facing roles, with the
farming role partially live (1 of 16 hooks wired).

## II.4 Per-role prompt landscape (5 active roles)

For each role: today's prompt (v1.0.0), the hypothesis space, and
3-5 concrete variant ideas ready to draft as `vX.Y.Z.txt` files.

### II.4.1 Extraction

**Today (v1.0.0):** "Verify each entity is real, add missing ones,
verify each fact, output JSON only, no markdown."

**Hypothesis space:**
- *Conservatism dial* -- how aggressively should we reject NER candidates?
- *Schema strictness* -- accept any predicate or only profile-allowed predicates?
- *Confidence calibration* -- does asking for explicit confidence change false-positive rate?
- *Multi-pass refinement* -- one prompt that does both verify and add, vs two prompts in series.

**Concrete variants to draft:**

- `v1.1.0-cot.txt` -- chain-of-thought: "First list each candidate
  with a one-line rationale, THEN emit the JSON."
- `v1.1.0-conservative.txt` -- "When in doubt, REJECT.  Better to
  miss an entity than fabricate one."
- `v1.1.0-liberal.txt` -- "Add entities the NER missed
  aggressively.  Wrong adds get caught downstream."
- `v1.1.0-fewshot.txt` -- inject 3 worked examples (one
  obvious-yes, one obvious-no, one ambiguous).
- `v1.1.0-twopass.txt` -- "(Pass A) verify only.  (Pass B, separate
  call) propose new entities."

**Measurable outcomes:**
- Precision: of entities the model returned, how many are correct?
- Recall: of true entities in the gold set, how many did the model return?
- F1 on entities; F1 on (subject, predicate, object) triples.
- Token cost per correct entity.

### II.4.2 Query Planning

**Today (v1.0.0):** Parse the user's natural-language query into
intent + entities + predicates.

**Hypothesis space:**
- *Decomposition* -- one-shot intent vs multi-step (decompose into sub-queries).
- *Ambiguity surfacing* -- "did you mean X or Y?" vs commit silently.
- *Profile injection* -- how much of the active DomainProfile to include in the planning prompt.

**Concrete variants:**

- `v1.1.0-decompose.txt` -- "If the query has multiple intents,
  output them as a list."
- `v1.1.0-ambiguity.txt` -- "If two interpretations are
  defensible, emit a clarification question instead of a plan."
- `v1.1.0-profile-rich.txt` -- inject the full DomainProfile
  vocabulary into the prompt.
- `v1.1.0-profile-thin.txt` -- inject only entity types, no
  vocabulary.

**Measurable outcomes:**
- Intent-classification accuracy (vs. gold).
- Downstream synthesis judge score (does better planning help the final answer?).
- Latency (decomposition adds passes).

### II.4.3 Retrieval

**Today (v1.0.0):** Formulate V2S (vector→SQL) and S2V (SQL→vector)
bridge queries from the planning output.

**Hypothesis space:**
- *Query expansion* -- generate 3 paraphrases of the user query for the vector side.
- *Negative-example mining* -- "and AVOID these terms".
- *Result-conditioned re-querying* -- after first retrieval pass, let the LLM see results and reformulate.

**Concrete variants:**

- `v1.1.0-expand.txt` -- "Generate 3 paraphrases for the vector query.  Combine results."
- `v1.1.0-negative.txt` -- "List 3 terms that would surface irrelevant results, exclude them."
- `v1.1.0-iterative.txt` -- two-pass: retrieve, then "given these results, what's missing?  Issue one more query."

**Measurable outcomes:**
- Recall@10 on a fixture query set.
- Precision@10.
- Token cost per relevant result.

### II.4.4 Synthesis

**Today (v1.0.0):** Combine SQL + vector results into a cited answer.

**Hypothesis space:**
- *Citation density* -- cite every claim vs cite at end of paragraph.
- *Hedging language* -- "likely", "appears to", vs flat assertions.
- *Multi-perspective* -- "if SQL and vector results disagree, present both sides".
- *Output format* -- prose vs structured JSON vs markdown table.

**Concrete variants:**

- `v1.1.0-hedged.txt` -- explicit hedging rules.
- `v1.1.0-structured.txt` -- "Output JSON: {answer, citations, caveats, follow_up_questions}".
- `v1.1.0-contradiction-first.txt` -- "If sources contradict, lead with that, then synthesise."
- `v1.1.0-concise.txt` -- "Answer in 30 words or fewer."
- `v1.1.0-detailed.txt` -- "Include every relevant cited claim."

**Measurable outcomes:**
- LLM-judge score: faithfulness, completeness, citation quality.
- User-rated helpfulness (when we have user feedback).
- Token cost per answer.

### II.4.5 Fusion

**Today (v1.0.0):** Re-rank fused SQL + vector results by semantic relevance.

**Hypothesis space:**
- *Novelty bias* -- prefer results that introduce new entities.
- *Recency bias* -- prefer fresher facts.
- *Entity-coverage* -- maximise distinct entities in the top-K.
- *Confidence-weighted* -- prefer high-confidence facts.

**Concrete variants:**

- `v1.1.0-novelty.txt` -- "Prefer results introducing new entities."
- `v1.1.0-recency.txt` -- "When two results are equal, prefer the more recent."
- `v1.1.0-coverage.txt` -- "Maximise the number of distinct entities in the top 5 results."

**Measurable outcomes:**
- nDCG@10 vs gold ranking.
- Downstream synthesis judge score.

## II.5 Cross-cutting prompt techniques

These are the levers we can pull at any role.  Each is a tool;
we'll use whichever fits the role's hypothesis.

| Technique | What it does | Cost | Buys |
|---|---|---|---|
| **Persona / role** | "You are a medical knowledge extractor." | Free | Tighter domain alignment. |
| **Structured output (JSON schema)** | Specify exact shape; reject anything else. | Free, sometimes 1-2 retries on parse fail. | Determinism for downstream parsing. |
| **Few-shot examples** | Inject 2-3 representative input->output pairs. | Adds 200-500 input tokens. | Calibration on edge cases the prose can't capture. |
| **Chain-of-thought** | "Reason step by step before answering." | Adds 50-200 output tokens. | Better accuracy on reasoning-heavy tasks; can hurt on pattern-matching tasks. |
| **Self-verification** | "Review your draft and identify mistakes." | Doubles output tokens. | Catches obvious errors; diminishing returns. |
| **Multi-pass** | Two LLM calls, second sees first's output. | 2x cost. | Decoupled tasks (e.g. extract, then verify). |
| **Negative prompting** | "Do NOT include X, Y, Z." | Free. | Rules out specific failure modes seen in practice. |
| **Calibrated confidence** | "Rate confidence 0-1 with rationale." | Adds 30-50 output tokens. | Lets downstream stages filter on confidence. |
| **Profile-conditioned** | Inject Layer 3 domain overlay. | Already happens via PromptAssembler. | Domain-tuned behaviour. |
| **Sentinel-on-thin-input** | Emit `INSUFFICIENT` when the input cannot be summarised meaningfully; caller falls back to deterministic. | Free. | Hallucination floor.  Used by Distiller (II.6.2.1) and the abstractive summariser. |

A reasonable rule of thumb: try **structured output** + **calibrated
confidence** on every role first.  They're cheap, robust, and they
make every other experiment easier to measure.

## II.6 Farming wiring -- 1 live, 15 dead hooks

The audit established that 16 of 18 farming stages accept `llm`
and never call `.generate()`.  As of Phase 4.1, the **Distiller**
hook is live (Section II.6.2.1); the other 15 are still
plumbing-only.

### II.6.1 The wiring pattern (reference)

The contract is already in place.  A wired stage does not change
its interface or its caller.  It only adds a `_maybe_llm_*`
helper that returns the deterministic value on every failure
path.  The reference implementation lives at
`farming/distiller.py::_maybe_llm_summary`:

```python
def _maybe_llm_summary(self, *, ..., fallback: str) -> str:
    if self._llm is None:
        return fallback
    try:
        if not self._llm.is_available():
            return fallback
    except Exception:
        return fallback
    user_prompt = TEMPLATE.format(...)
    try:
        raw = self._llm.generate(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=80,
        )
    except Exception as exc:
        logger.warning("..._llm_failed", error=str(exc))
        return fallback
    text = (raw or "").strip()
    if not text or text.upper() == INSUFFICIENT_SENTINEL:
        return fallback
    return text  # cap to column-length contract before return
```

Three properties this gives us for free:

1. **No regressions when `llm is None`** -- the deterministic path is
   the fallback, so existing installs keep working unchanged.
2. **No new failure modes for LLM downtime** -- `try/except` falls
   back to deterministic on any error.
3. **No interface churn** -- stage signature, pipeline wiring,
   tests, and CLI commands all stay identical.

The same shape is used at ingest time by
`extraction/pipeline.py::_maybe_llm_summary` for the abstractive
interaction summary.

### II.6.2 Tier 1 -- high-ROI farming stages

These three are the highest-value first wirings: each produces
output the user actually reads, the deterministic baseline is
visibly mechanical, and a single LLM call per output unit is
plenty.

#### II.6.2.1 Distiller (Stage 16) -- per-entity natural summary [SHIPPED]

**Status:** Live as of Phase 4.1 (`[Unreleased]`).  Implementation
in `farming/distiller.py`; system / user prompts as module
constants `DISTILLER_SYSTEM_PROMPT` / `DISTILLER_USER_PROMPT_TEMPLATE`.

**What the stage already has at the call point:**

```
entity_name        = "Alice Johnson"
entity_type        = "PERSON"
mention_count      = 14
interaction_count  = 7
top_predicates     = ["proposed", "raised_concerns", "approved", "reviewed"]
co_entities        = ["Bob Smith", "OAuth2 Migration", "Q2 Roadmap"]
recent_facts       = [<5 most recent fact rows>]
```

**Today's deterministic output** (`_build_summary`, used as
fallback):

> "Alice Johnson (PERSON, 14 mentions across 7 interactions);
> top predicates: proposed, raised_concerns, approved, reviewed;
> co-entities: Bob Smith, OAuth2 Migration, Q2 Roadmap"

**Shipped system prompt:**

> You are a knowledge-base summariser.  Given a structured profile
> of an entity (its name, type, frequent predicates, and
> co-occurring entities), produce ONE sentence (under 30 words)
> describing what the knowledge base knows about this entity.
> Use active voice.  Do not invent facts or relationships not
> shown in the profile.  Do not editorialise.  Do not mention
> "the knowledge base" or "the system".  If the profile is too
> thin to summarise meaningfully, output the literal string
> INSUFFICIENT and nothing else.

**Expected output:**

> "Alice Johnson is a PERSON who has proposed and approved
> migrations -- most prominently OAuth2 -- while raising concerns
> reviewed alongside Bob Smith on the Q2 Roadmap."

**Downstream effect:** the string lands in
`distiller_summaries.summary`, which the query stack reads when
answering "what do you know about Alice?".  The entity-page
"Top Distilled Entities" card in `[Unreleased]` exposes it
directly to the user.

**Cost per cycle:** ~80 output tokens, ~120 input tokens per
entity.  For a 5,000-entity store with daily farming, that's
~25,000 input / ~16,667 output tokens per cycle.  At hosted
gpt-4o-mini prices (~$0.15/$0.60 per 1M tokens) this is
~$0.014 per cycle, or ~$0.42/month.

**Risk:** lowest of the Tier 1 stages -- the LLM gets structured
input, the output is a single sentence, and the fallback is
identical to the pre-Phase-4.1 behaviour.

#### II.6.2.2 InsightGenerator (Stage 6) -- narrative meta-insights

**Status:** Not started.  Phase 4.2 candidate.

**What the stage has at the call point:**

```
current_cycle_counts   = {"relationship": 5, "trend": 3, "cluster": 2, "topic": 1}
previous_cycle_counts  = {"relationship": 3, "trend": 5, "cluster": 2}
new_types              = {"topic"}
disappeared_types      = set()
delta_per_type         = {"relationship": +2, "trend": -2, "cluster": 0, "topic": +1}
```

**Today's output:**

> "Cycle delta: relationship +2, trend -2, topic appeared, cluster
> unchanged.  Total insights: 11 (was 10)."

**Proposed system prompt:**

> You write one-sentence cycle summaries for a knowledge-mining
> pipeline.  Given the per-type delta between two cycles, describe
> what shifted in plain language.  Mention only types whose count
> changed by 2 or more, OR types that newly appeared, OR types
> that disappeared.  Avoid jargon.  Do not speculate about causes.

**Cost:** one call per cycle.  Negligible.

#### II.6.2.3 CausalMiner (Stage 7) -- pair verifier

**Status:** Blocked on the keying fix (`audit_findings.md`
Section 1.3).  Phase 4.3 candidate.

**Proposed system prompt:**

> You evaluate whether a statistically-correlated event pair
> represents plausible cause-and-effect, not mere co-occurrence.
> Given the predicate names of two events and the typical time
> lag, respond with a JSON object:
> `{"causal": true|false, "confidence": 0.0-1.0, "rationale": "<one
> short sentence>"}`.  Be conservative: if either predicate is
> too generic to support causation, say `false`.  Do not invent
> context not in the input.

**Risk:** higher than the first two -- the LLM is now a gating
filter, not a narrator.  False negatives become a quality
regression.  Mitigations: log every rejected pair, expose them
in the dashboard, gate behind a config flag
(`farming.causal_miner.use_llm`).

### II.6.3 Tier 2 -- medium-ROI farming stages

- **TopicModeling** (Stage 4) -- topic labels.  Today's output is
  `"topic-7"` (cluster id); wired output is `"OAuth2 / SAML
  migration rollout"`.  Phase 4.2 candidate.
- **Linker** (Stage 13) -- alias adjudicator.  Output: `{"same":
  true|false, "confidence": 0.0-1.0}`.  Gate on `confidence >=
  0.85` AND require human confirmation through merge UI before
  applying.  Phase 4.4 candidate.
- **Verifier** (Stage 14) -- plausibility checker for
  (subject, predicate, object) triples beyond the schema check.
  Implausible triples get `confidence` halved.  Phase 4.4
  candidate.

### II.6.4 Tier 3 -- lower-ROI / advisory farming stages

- **Completionist** -- generate specific gap-filling questions for
  thin entity profiles.  Output queues for next-cycle ingest.
- **TrendDetection** -- explain the likely *driver* of a detected
  trend.
- **GraphAnalysis** -- explain centrality of top PageRank nodes.
- **FeedbackLoop** -- refine extracted gap queries.

These are good batched additions in a "narrative pass" PR once
Tier 1 + Tier 2 are in place.

### II.6.5 Where LLM doesn't help

EntityAnalytics, Clustering, Pruner, Consolidator (after keying
fix), Calibrator, Rationalizer, Archivist, Defragmenter -- all
deterministic by design.  LLM adds cost without quality.

### II.6.6 Aggregate cost & latency budget

Daily cycle on a 5,000-entity / 20,000-fact store, all wirings
shipped:

| Stage | Calls/cycle | Tokens in | Tokens out | Latency (hosted) |
|---|---|---|---|---|
| Distiller (live) | ~5,000 | ~600k | ~400k | ~5 min |
| InsightGenerator | 1 | ~80 | ~30 | <1 s |
| CausalMiner | ~50 | ~2.5k | ~4k | ~10 s |
| TopicModeling | ~15 | ~3k | ~600 | ~3 s |
| Linker | ~30 | ~3k | ~1.5k | ~5 s |
| Verifier | ~2,000 | ~200k | ~80k | ~2 min |
| Tier 3 (4 stages) | ~200 total | ~30k | ~15k | ~30 s |
| **Total** | ~7,300 | ~840k | ~500k | ~8 min |

At gpt-4o-mini hosted prices: **~$0.43 per daily cycle, ~$13/month.**
For an 8B-class local model: ~30-60 ms per call → ~4-8 minutes
wall time.  Fits inside an idle window.

Distiller and Verifier dominate (per-entity / per-fact).  The
other six stages combined are <1% of cost.

### II.6.7 Farming wiring sequence

- **Phase 4.1 -- Distiller alone.** [SHIPPED in `[Unreleased]`.]
- **Phase 4.2 -- TopicModeling + InsightGenerator.**  Two small
  narrative additions.
- **Phase 4.3 -- CausalMiner with LLM verification, gated behind
  the keying fix.**  First stage where the LLM is a *filter*,
  not a narrator.
- **Phase 4.4 -- Verifier + Linker.**  The judgement-style
  stages.  Linker requires merge-UI confirmation before
  applying.
- **Phase 4.5 -- Tier 3 batch.**  Narrative polish pass.

## II.7 Systematic exploration methodology

### II.7.1 What we have

- `prompts/` versioned template tree.
- `PromptAssembler` to load any version on demand.
- `cli.py:862` already calls an LLM-as-judge for the
  `ctxmtg evaluate` command.
- Per-role LLM config so we can point different roles at
  different models.

### II.7.2 What we need to build -- the eval harness

A small CLI command.  Sketch:

```bash
ctxmtg eval prompts \
  --role extraction \
  --variants v1.0.0,v1.1.0-cot,v1.1.0-fewshot \
  --corpus tests/fixtures/extraction_corpus.jsonl \
  --judge gpt-4o-mini \
  --n-samples 30 \
  --output evaluations/extraction-2026-04-30.json
```

What it does internally:

1. Load the corpus (gold inputs + gold outputs in JSONL).
2. For each variant, swap the prompt by symlinking the chosen
   version into `prompts/stages/<role>/active.txt`, then run the
   role's normal call site against the corpus.
3. Score each variant's output against gold using two paths:
   - **Mechanical** -- F1 on entities, F1 on triples, recall@K on
     retrieval, latency, token cost.  No LLM needed.
   - **Judged** -- LLM-as-judge score for synthesis quality,
     answer relevance.
4. Output a comparison report:

```
Role: extraction
Corpus: 30 samples

Variant          Precision  Recall  F1     Latency(ms)  Cost($)
v1.0.0           0.91       0.78    0.84   850          0.0021
v1.1.0-cot       0.93       0.88    0.90   1240         0.0034
v1.1.0-fewshot   0.96       0.85    0.90   980          0.0027
v1.1.0-conserv.  0.97       0.71    0.82   870          0.0022

Judge confidence interval: 95% (n=30)
Recommendation: v1.1.0-cot or v1.1.0-fewshot
```

### II.7.3 Statistical rigor checklist

- N >= 20 per variant (more for noisy roles like synthesis).
- Each variant evaluated by **at least two judges** when judge
  scores are decisive (mitigates LLM-judge bias).
- Confidence intervals reported alongside means.
- Cost-per-quality-unit (token cost / F1 point) for budgeting.
- Recompute when the model changes (judge or actor).
- Identify "tied" variants (within CI) and report all ties.

### II.7.4 Promotion criterion

A variant `vX.Y.Z` becomes the role's default when:

1. It dominates the previous default on at least one primary
   metric and is not worse on any other primary metric (within CI).
2. It has been judged by two distinct judge models.
3. The corpus is large enough (n >= 30) for the CI to be tight.

When dominance is partial, keep both versions in the registry
and let users opt-in via `config/settings.yaml`.  Promotion is a
real release event with a CHANGELOG entry.

### II.7.5 Versioning

- `vMAJOR.MINOR.PATCH.txt`
- MAJOR -- breaking change to slot syntax.
- MINOR -- new prompt strategy.
- PATCH -- typo / wording fix.
- "Active" version = whichever the assembler is configured to
  load; default = "highest-v1 in the directory"; config can pin
  per role.

## II.8 Autonomy roadmap (parallel thread)

The same body of work that lets us test prompts also drives the
system toward autonomy.  Both threads share infrastructure.

### II.8.1 Levels

| Level | Description | This codebase today |
|---|---|---|
| L0 | User-driven only. | Yes (CLI). |
| L1 | Idle-time pipeline.  Inbox watcher ingests; scheduler runs farming. | **Partial** -- `farming/scheduler.py` and `cli.py watch` exist. |
| L2 | Self-tuning weights.  Calibrator stage measures quality and adjusts farming weights. | **Skeleton** -- CalibratorStage runs but doesn't write to a weights table. |
| L3 | Self-improving prompts.  System measures prompt variants, picks winners, rolls out new defaults. | **Not yet** -- needs the eval harness from Section II.7. |
| L4 | Agentic farming.  LLM is a decision-maker with tool use, not just a narrator. | **Not yet** -- needs L3 plus a tool-use protocol plus an action sandbox. |
| L5 | Multi-agent.  Specialised agents (extractor, planner, retriever, synthesiser) coordinate via the knowledge store as shared memory. | **Not yet** -- needs L4 plus inter-agent coordination. |

### II.8.2 The dependency chain

```
Eval harness (II.7)  -->  L2  -->  L3  -->  L4  -->  L5
```

L2 needs deterministic metrics; the eval harness produces them.

L3 = "the system runs the eval harness on itself, proposes new
prompts, ships the winners".  Concretely, an LLM-as-author loop:
take the current best prompt + the corpus where it underperforms,
ask the LLM to draft three variants targeting those failure modes,
evaluate, promote.

L4 = "the system uses an LLM to decide what action to take next":
which file to ingest first, which entity to enrich, when to farm,
what to ask the user.  Requires L3 (tested prompts) plus a
sandboxed action layer.

L5 = "specialised agents handing off via the knowledge store".
Each existing farming stage is already a narrowly scoped agent
shape -- just without LLM decision-making.

### II.8.3 Where to start

The right next move is **the eval harness in Section II.7**.  It
unlocks both threads:

- For prompt experimentation, it's the missing measurement piece.
- For autonomy, it's the L2 gate, the L3 gate, and the L4
  guardrail.

Without it, every claim about "this prompt is better" is a vibe.
With it, claims are testable.

### II.8.4 Scoping autonomy with user trust

A separate axis from the L0-L5 ladder: how much agency the system
has over the user's data.

| Scope | What the system can do without asking |
|---|---|
| Read-only | Ingest, query, farm, write insights to its own tables.  (Today.) |
| Suggest | Surface "I think we should merge these entities" but require user click. |
| Apply-with-undo | Auto-apply changes that have a one-click undo. |
| Apply-irrevocably | Delete, archive, prune without asking. |

Today the system is mostly read-only with a few apply-irrevocable
maintenance stages (Pruner, Archivist).  A serious autonomy push
should add a Suggest scope first -- the merge UI is exactly that
shape.  Apply-with-undo is the long-game default.

## II.9 Concrete first experiments

Ten experiments, ordered by ROI.  Each is one new file in
`prompts/stages/<role>/` plus an entry in the eval harness.

1. **Extraction `v1.1.0-cot.txt`** -- chain-of-thought before JSON.
   Hypothesis: improves F1 on triples by 3-5%.
2. **Synthesis `v1.1.0-structured.txt`** -- JSON output with
   explicit citations, caveats, follow-ups.  Hypothesis:
   downstream parsing reliability up >> any quality drop.
3. **Synthesis `v1.1.0-hedged.txt`** -- enforced hedging language.
   Hypothesis: judge faithfulness score up.
4. **Query planning `v1.1.0-ambiguity.txt`** -- surface ambiguity.
   Hypothesis: planning accuracy roughly flat, user-rated
   helpfulness up.
5. **Farming `distiller_v1.1.0.txt`** -- variant of the shipped
   Phase 4.1 prompt.  Move from module constant to versioned
   file under `prompts/stages/farming/`.  Hypothesis: enables
   first head-to-head comparison once eval harness is in place.
6. **Extraction `v1.1.0-fewshot.txt`** -- 3 worked examples.
   Hypothesis: F1 up on edge cases (timestamps, fragments).
7. **Retrieval `v1.1.0-expand.txt`** -- query paraphrase
   expansion.  Hypothesis: recall@10 up at modest cost.
8. **Fusion `v1.1.0-coverage.txt`** -- entity-coverage maximisation
   in top-K.  Hypothesis: synthesis judge score up.
9. **Farming `causal_miner_v1.0.0.txt`** (after the keying fix
   AND Phase 4.3 wiring) -- LLM verification of mined pairs.
   Hypothesis: false positives cut by half or more.
10. **Layer 1 base `v1.1.0.txt`** -- tightened safety + format
    rules with explicit "if you don't know, say UNKNOWN, never
    invent".  Hypothesis: hallucination rate down across all
    roles.

## II.10 Recommended sequence

The actual plan, given everything above:

1. **Land Phase 4.1 -- Distiller LLM wiring.** [SHIPPED in
   `[Unreleased]`.]  Smallest possible first step.  One stage,
   one method, one prompt template.  Established the
   `_maybe_llm_*` shape every later wiring will follow.
2. **Build the minimal eval harness** (Section II.7).  One CLI
   command, one fixture corpus, mechanical metrics first, judge
   metrics second.  The unlock.
3. **Run baseline eval** on v1.0.0 of every role + the new
   Distiller prompt.  Establish the numbers we're trying to beat.
4. **A/B test the top-3 variants** from Section II.9 (likely
   experiments 1, 2, 5).  Promote the winners as v1.1.0.
5. **Wire Calibrator to a weights table** (L2 autonomy).  Eval
   harness produces data; Calibrator writes adjustments; next
   farming cycle reads them.
6. **Add the Suggest scope to the merge UI** (autonomy with user
   trust).  System surfaces "merge candidates" backed by
   evaluated prompt confidence; user clicks accept.
7. **Build the prompt-author LLM loop** (L3).  LLM proposes new
   variants targeting failure modes; eval harness judges; winners
   promoted.  Closed loop.
8. **Design the action sandbox** (L4 prereq).  Read-only by
   default, allow-list of write tools, audit log of every action.

Items 1-3: weeks (item 1 done).  Items 4-5: weeks-to-months.
Items 6-7: months.  Item 8: the start of a longer thread.

## II.11 Open design questions

- **Where do prompts physically live?**  Filesystem (today) is
  versionable in git, easy to diff, easy to load.  DB storage
  buys per-user customisation but loses git history.  Probably
  keep filesystem; store user-edited Layer 4 separately.  The
  Phase 4.1 module-constant prompts are a deliberate transitional
  state -- they should migrate into `prompts/stages/farming/` as
  versioned templates as part of Section II.9 experiment 5.
- **Should the eval harness ship in the public release?**  Pro:
  users verify their own prompts, trust the system.  Con: needs a
  corpus, which adds maintenance.  Probably yes, with a small
  public corpus and a hook for user-supplied corpora.
- **Local vs hosted for eval?**  Probably support both with the
  recommendation: local actor, hosted judge during development;
  local actor + local judge in production.
- **LLM-as-judge bias.**  Self-judging is biased; same-family
  judging less so but still biased.  Mitigation: at least two
  judges from different families for promotion decisions.
- **How does Layer 4 (user prefs) get edited?**  Web UI?  Config
  file?  CLI?  The 4-layer model needs a clear authoring surface
  for end-users.
- **Self-improving prompts (L3) before user-facing autonomy
  (L4)?**  Probably yes -- L3 affects only the system's own
  internal prompts, no user-visible action surface.

## II.12 What this part deliberately does NOT do

- Does not pick winners.  The whole point is "measure, don't
  speculate".
- Does not write the eval harness.  That's the next concrete
  deliverable after Phase 4.1 (which has now shipped).
- Does not enumerate every prompt variant per role.  The 50
  combinations of (role) x (technique) are not all equal; the
  ones in Sections II.4 and II.9 are the high-information ones.
- Does not commit to L4/L5.  Those are downstream of L3 results
  we don't have yet.
- Does not commit to a UI for Layer 4 user prefs.  Authoring
  surface is its own design problem.
