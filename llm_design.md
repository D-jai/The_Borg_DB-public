# LLM design -- WORKING SPEC

> Working spec, not a release doc. Edit, argue with, replace.
>
> Scope: every LLM-touching surface in the system, the prompt
> strategies we can vary at each one, a methodology for exploring
> them systematically, the 16 dead farming hooks and what to do
> with them, and a parallel thread on how this same body of work
> drives the system toward autonomy.
>
> Companion to `audit_findings_v0.7.1.md`,
> `autonomous_farming.md`, `runtimechange.md`. This doc is the
> "what's next for the LLM surface" thread.

---

## Table of contents

1. [Frame -- and the surprise](#1-frame----and-the-surprise)
2. [The 4-layer architecture](#2-the-4-layer-architecture)
3. [Map of the live LLM surface](#3-map-of-the-live-llm-surface)
4. [Per-role prompt landscape (5 active roles)](#4-per-role-prompt-landscape-5-active-roles)
5. [Cross-cutting prompt techniques](#5-cross-cutting-prompt-techniques)
6. [Farming wiring -- the 16 dead hooks](#6-farming-wiring----the-16-dead-hooks)
7. [Systematic exploration methodology](#7-systematic-exploration-methodology)
8. [Autonomy roadmap (parallel thread)](#8-autonomy-roadmap-parallel-thread)
9. [Concrete first experiments](#9-concrete-first-experiments)
10. [Recommended sequence](#10-recommended-sequence)
11. [Open design questions](#11-open-design-questions)
12. [What this doc deliberately does NOT do](#12-what-this-doc-deliberately-does-not-do)

---

## 1. Frame -- and the surprise

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
through this assembler. The version diff between
`extraction/v0.9.0.txt` and `extraction/v1.0.0.txt` is a real
prompt experiment that already shipped: v1.0.0 added "Output ONLY
a single valid JSON object. No markdown, no backticks."

**We are not designing a prompt-experimentation system from
scratch. We are under-using one that already exists. The piece
we're missing is measurement, not authoring.**

Three implications drive the rest of this doc:

- Prompt experimentation is **a configuration change, not a code
  change**. New prompt strategy = new file at
  `prompts/stages/<role>/vX.Y.Z.txt`.
- The 4-layer model lets us hold base safety constant while
  varying stage instructions; vary stage instructions while
  holding domain overlay constant; etc. Each layer has its own
  test cycle.
- Every claim of the form "prompt A is better than prompt B" is
  vibes until we have an eval harness. Building that harness is
  the highest-value next move on this whole roadmap.

---

## 2. The 4-layer architecture

| Layer | What it controls | Who edits it | Test cycle |
|---|---|---|---|
| 1 -- Base | Identity, safety, format invariants. ~100-200 tokens. | Project maintainer. Touch rarely. | Released; evaluate against full corpus when changed. |
| 2 -- Stage | Per-role task instructions. ~300-800 tokens. **Most experiments live here.** | Project maintainer. Versioned per role. | A/B test per role on a fixture corpus. |
| 3 -- Domain | Slot-injected from active `DomainProfile`. ~100-400 tokens. Includes entity types, terminology, reasoning patterns. | User (per profile). | A/B test per domain on a domain-specific corpus. |
| 4 -- User prefs | Per-user customization (preferred summary length, priority topics, output language). ~50-150 tokens. | End user. | Opt-in only; no system-wide test cycle. |

Three categories of "different prompts -> different outcomes":

- **Layer 2 variants** -- "what does this role do better with a
  chain-of-thought prefix?"
- **Layer 3 variants** -- "what entity types should the legal
  domain extract that the medical domain shouldn't?"
- **Layer 4 variants** -- "what does this user want answers to
  look like?"

The first two are systematic experimentation. The third is
personalisation and is downstream of #1 and #2 working well.

---

## 3. Map of the live LLM surface

| Role | Live in | Loaded by PromptAssembler? | Today's status |
|---|---|---|---|
| Extraction | `extraction/llm_verifier.py:211` | Yes | Live. v0.9.0 -> v1.0.0 already shipped. |
| Query Planning | `query/llm_interpreter.py:184` | Yes | Live. v1.0.0 only. |
| Retrieval | `query/informed_retrieval.py:368, 836` | Yes | Live, two call sites. v1.0.0 only. |
| Synthesis | `query/synthesizer.py:204` | Yes | Live. v1.0.0 only. |
| Fusion | `query/llm_fusion.py:186` | Yes | Live. v1.0.0 only. |
| Sync (extraction-adjacent) | `sync/context_enricher.py:457` | (uses extraction template) | Live. |
| Profile | `profile/assembler.py:19, 81` | Yes (wrapper) | Live. |
| Evaluation (LLM-as-judge) | `cli.py:862` | Yes | Live, used by `ctxmtg evaluate`. |
| Farming (16 stages) | None of `farming/*.py` | Template exists, no caller | **Dead** -- 16 stages store `self._llm`, none call `.generate()`. |

Nine call sites, six user-facing roles, one of which (farming)
is wired in plumbing only.

---

## 4. Per-role prompt landscape (5 active roles)

For each role: today's prompt (v1.0.0), the hypothesis space, and
3-5 concrete variant ideas ready to draft as `vX.Y.Z.txt` files.

### 4.1 Extraction

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
- `v1.1.0-conservative.txt` -- "When in doubt, REJECT. Better to
  miss an entity than fabricate one."
- `v1.1.0-liberal.txt` -- "Add entities the NER missed
  aggressively. Wrong adds get caught downstream."
- `v1.1.0-fewshot.txt` -- inject 3 worked examples (one
  obvious-yes, one obvious-no, one ambiguous).
- `v1.1.0-twopass.txt` -- "(Pass A) verify only. (Pass B, separate
  call) propose new entities."

**Measurable outcomes:**
- Precision: of entities the model returned, how many are correct?
- Recall: of true entities in the gold set, how many did the model return?
- F1 on entities; F1 on (subject, predicate, object) triples.
- Token cost per correct entity.

### 4.2 Query Planning

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

### 4.3 Retrieval

**Today (v1.0.0):** Formulate V2S (vector→SQL) and S2V (SQL→vector)
bridge queries from the planning output.

**Hypothesis space:**
- *Query expansion* -- generate 3 paraphrases of the user query for the vector side.
- *Negative-example mining* -- "and AVOID these terms".
- *Result-conditioned re-querying* -- after first retrieval pass, let the LLM see results and reformulate.

**Concrete variants:**

- `v1.1.0-expand.txt` -- "Generate 3 paraphrases for the vector query. Combine results."
- `v1.1.0-negative.txt` -- "List 3 terms that would surface irrelevant results, exclude them."
- `v1.1.0-iterative.txt` -- two-pass: retrieve, then "given these results, what's missing? Issue one more query."

**Measurable outcomes:**
- Recall@10 on a fixture query set.
- Precision@10.
- Token cost per relevant result.

### 4.4 Synthesis

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

### 4.5 Fusion

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

---

## 5. Cross-cutting prompt techniques

These are the levers we can pull at any role. Each is a tool;
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

A reasonable rule of thumb: try **structured output** + **calibrated
confidence** on every role first. They're cheap, robust, and they
make every other experiment easier to measure.

---

## 6. Farming wiring -- the 16 dead hooks

The audit established that 16 of 18 farming stages accept `llm`
and never call `.generate()`. This section is the concrete
"what to wire and how" subset of the broader strategy.

### 6.1 The wiring pattern

The contract is already in place. A wired stage does not change
its interface or its caller. It only adds an `if self._llm:`
branch around the deterministic logic:

```python
def run(self, sql_store, vector_store, context):
    # ... gather data deterministically ...
    deterministic_summary = _build_summary(top_predicates, co_entities, ...)

    if self._llm is not None:
        try:
            llm_summary = self._llm.generate(
                system_prompt=DISTILLER_SYSTEM_PROMPT,
                user_prompt=DISTILLER_USER_PROMPT.format(
                    entity_name=entity_name,
                    top_predicates=", ".join(top_predicates),
                    co_entities=", ".join(co_entities),
                    interaction_count=interaction_count,
                ),
                max_tokens=80,
                temperature=0.2,
            )
            summary = llm_summary.strip() or deterministic_summary
        except Exception as exc:
            logger.warning("distiller_llm_failed", error=str(exc))
            summary = deterministic_summary
    else:
        summary = deterministic_summary
```

Three properties this gives us for free:

1. **No regressions when `llm is None`** -- the deterministic path is
   the fallback, so existing installs keep working unchanged.
2. **No new failure modes for LLM downtime** -- `try/except` falls
   back to deterministic on any error.
3. **No interface churn** -- stage signature, pipeline wiring,
   tests, and CLI commands all stay identical.

The reference call site for this pattern is
`src/ctxmtg/query/synthesizer.py:204`.

### 6.2 Tier 1 -- high-ROI farming stages

These three are the highest-value first wirings: each produces
output the user actually reads, the deterministic baseline is
visibly mechanical, and a single LLM call per output unit is
plenty.

#### 6.2.1 Distiller (Stage 16) -- per-entity natural summary

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

**Today's output** (`_build_summary`):

> "Alice Johnson (PERSON, 14 mentions across 7 interactions);
> top predicates: proposed, raised_concerns, approved, reviewed;
> co-entities: Bob Smith, OAuth2 Migration, Q2 Roadmap"

**Proposed system prompt:**

> You are a knowledge-base summariser. Given a structured profile
> of an entity (its name, type, frequent predicates, co-occurring
> entities, and recent facts), produce ONE sentence (under 30
> words) describing what the knowledge base knows about this
> entity. Use active voice. Do not invent facts. Do not editorialise.
> Do not mention "the knowledge base" or "the system". If the
> profile is too thin to summarise meaningfully, output the literal
> string `INSUFFICIENT`.

**Proposed user prompt template:**

```
Entity: {entity_name} ({entity_type})
Mentions: {mention_count} across {interaction_count} interactions
Top predicates: {top_predicates}
Frequently co-occurs with: {co_entities}
Recent facts:
{recent_facts_bulleted}

Write the one-sentence summary now.
```

**Expected output:**

> "Alice Johnson is a PERSON who has proposed and approved
> migrations -- most prominently OAuth2 -- while raising concerns
> reviewed alongside Bob Smith on the Q2 Roadmap."

**Downstream effect:** the string lands in
`distiller_summaries.summary`, which the query stack reads when
answering "what do you know about Alice?". Today the answer
returns the mechanical string; after wiring, the answer returns
human language.

**Cost per cycle:** ~80 output tokens, ~120 input tokens per
entity. For a 5,000-entity store with daily farming, that's
~25,000 input / ~16,667 output tokens per cycle. At hosted
gpt-4o-mini prices (~$0.15/$0.60 per 1M tokens) this is
~$0.014 per cycle, or ~$0.42/month.

**Risk:** lowest of the Tier 1 stages -- the LLM gets structured
input, the output is a single sentence, and the fallback is
identical to today's behaviour.

#### 6.2.2 InsightGenerator (Stage 6) -- narrative meta-insights

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
> unchanged. Total insights: 11 (was 10)."

**Proposed system prompt:**

> You write one-sentence cycle summaries for a knowledge-mining
> pipeline. Given the per-type delta between two cycles, describe
> what shifted in plain language. Mention only types whose count
> changed by 2 or more, OR types that newly appeared, OR types
> that disappeared. Avoid jargon. Do not speculate about causes.

**Proposed user prompt template:**

```
Previous cycle insight counts: {previous_cycle_counts}
Current cycle insight counts: {current_cycle_counts}
Newly appearing types: {new_types}
Disappeared types: {disappeared_types}

Write the one-sentence cycle summary now.
```

**Expected output:**

> "This cycle saw two more relationship insights and two fewer
> trends, with topic insights appearing for the first time."

**Cost:** one call per cycle. Negligible.

#### 6.2.3 CausalMiner (Stage 7) -- pair verifier

**What the stage has** (after the keying fix from
`audit_findings_v0.7.1.md` Section 1.3):

```
event_a       = "raised_concerns"
event_b       = "timeline_extended"
occurrences   = 7
avg_lag_days  = 4.2
reliability   = 0.65    # P(B | A) within window
total_a       = 12
total_b       = 11
```

**Today:** a "causal_candidate" insight is written if
`reliability >= RELIABILITY_THRESHOLD` (0.2). No verification of
whether the pair makes causal sense.

**Proposed system prompt:**

> You evaluate whether a statistically-correlated event pair
> represents plausible cause-and-effect, not mere co-occurrence.
> Given the predicate names of two events and the typical time
> lag, respond with a JSON object:
> `{"causal": true|false, "confidence": 0.0-1.0, "rationale": "<one
> short sentence>"}`. Be conservative: if either predicate is too
> generic to support causation, say `false`. Do not invent context
> not in the input.

**Downstream effect:** insight `confidence` set to LLM's
`confidence` rather than the deterministic `reliability` ratio;
`rationale` appended to the insight title. Pairs with `causal:
false` are filtered out -- addresses the "false-positive flood"
problem the audit identified.

**Cost:** ~50 input / ~80 output tokens per candidate pair. <$0.001
per cycle.

**Risk:** higher than the first two -- the LLM is now a gating
filter, not a narrator. False negatives become a quality
regression. Mitigations: log every rejected pair, expose them in
the dashboard, gate behind a config flag
(`farming.causal_miner.use_llm`).

### 6.3 Tier 2 -- medium-ROI farming stages

- **TopicModeling** (Stage 4) -- topic labels. Today's output is
  `"topic-7"` (cluster id); wired output is `"OAuth2 / SAML
  migration rollout"`. System prompt: assign a 2-4 word title-case
  label given top TF-IDF keywords + sample facts + member entities.
  One call per topic, 5-20 topics per cycle. Trivial cost.
- **Linker** (Stage 13) -- alias adjudicator for non-trivial
  variants ("Alice Johnson" vs "A. Johnson"). System prompt: judge
  whether two names refer to the same person given shared
  co-entity context. Output: `{"same": true|false, "confidence":
  0.0-1.0}`. Gate on `confidence >= 0.85` AND require human
  confirmation through merge UI before applying.
- **Verifier** (Stage 14) -- plausibility checker for
  (subject, predicate, object) triples beyond the schema check.
  Output: `{"plausible": true|false, "confidence": 0.0-1.0, "reason":
  "<one phrase>"}`. Implausible triples get `confidence` halved.

### 6.4 Tier 3 -- lower-ROI / advisory farming stages

- **Completionist** -- generate specific gap-filling questions for
  thin entity profiles. Output queues for next-cycle ingest.
- **TrendDetection** -- explain the likely *driver* of a detected
  trend.
- **GraphAnalysis** -- explain centrality of top PageRank nodes.
- **FeedbackLoop** -- refine extracted gap queries.

These are good batched additions in a "narrative pass" PR once
Tier 1 is in place.

### 6.5 Where LLM doesn't help

EntityAnalytics, Clustering, Pruner, Consolidator (after keying
fix), Calibrator, Rationalizer, Archivist, Defragmenter -- all
deterministic by design. LLM adds cost without quality.

### 6.6 Aggregate cost & latency budget

Daily cycle on a 5,000-entity / 20,000-fact store:

| Stage (wired) | Calls/cycle | Tokens in | Tokens out | Latency (hosted) |
|---|---|---|---|---|
| Distiller | ~5,000 | ~600k | ~400k | ~5 min |
| InsightGenerator | 1 | ~80 | ~30 | <1 s |
| CausalMiner | ~50 | ~2.5k | ~4k | ~10 s |
| TopicModeling | ~15 | ~3k | ~600 | ~3 s |
| Linker | ~30 | ~3k | ~1.5k | ~5 s |
| Verifier | ~2,000 | ~200k | ~80k | ~2 min |
| Tier 3 (4 stages) | ~200 total | ~30k | ~15k | ~30 s |
| **Total** | ~7,300 | ~840k | ~500k | ~8 min |

At gpt-4o-mini hosted prices: **~$0.43 per daily cycle, ~$13/month.**
For an 8B-class local model: ~30-60 ms per call → ~4-8 minutes
wall time. Fits inside an idle window.

Distiller and Verifier dominate (per-entity / per-fact). The
other six stages combined are <1% of cost.

### 6.7 Farming wiring sequence

- **Phase 4.1 -- Distiller alone.** Single stage, single call site,
  no behavioural impact on others, biggest visible user win
  (the "what do you know about X?" answer). Shippable in a day.
- **Phase 4.2 -- TopicModeling + InsightGenerator.** Two small
  narrative additions.
- **Phase 4.3 -- CausalMiner with LLM verification, gated behind
  the keying fix.** First stage where the LLM is a *filter*, not
  a narrator.
- **Phase 4.4 -- Verifier + Linker.** The judgement-style stages.
  Linker requires merge-UI confirmation before applying.
- **Phase 4.5 -- Tier 3 batch.** Narrative polish pass.

---

## 7. Systematic exploration methodology

### 7.1 What we have

- `prompts/` versioned template tree.
- `PromptAssembler` to load any version on demand.
- `cli.py:862` already calls an LLM-as-judge for the
  `ctxmtg evaluate` command.
- Per-role LLM config so we can point different roles at
  different models.

### 7.2 What we need to build -- the eval harness

A small CLI command. Sketch:

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
     retrieval, latency, token cost. No LLM needed.
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

### 7.3 Statistical rigor checklist

- N >= 20 per variant (more for noisy roles like synthesis).
- Each variant evaluated by **at least two judges** when judge
  scores are decisive (mitigates LLM-judge bias).
- Confidence intervals reported alongside means.
- Cost-per-quality-unit (token cost / F1 point) for budgeting.
- Recompute when the model changes (judge or actor).
- Identify "tied" variants (within CI) and report all ties.

### 7.4 Promotion criterion

A variant `vX.Y.Z` becomes the role's default when:

1. It dominates the previous default on at least one primary
   metric and is not worse on any other primary metric (within CI).
2. It has been judged by two distinct judge models.
3. The corpus is large enough (n >= 30) for the CI to be tight.

When dominance is partial, keep both versions in the registry
and let users opt-in via `config/settings.yaml`. Promotion is a
real release event with a CHANGELOG entry.

### 7.5 Versioning

- `vMAJOR.MINOR.PATCH.txt`
- MAJOR -- breaking change to slot syntax.
- MINOR -- new prompt strategy.
- PATCH -- typo / wording fix.
- "Active" version = whichever the assembler is configured to
  load; default = "highest-v1 in the directory"; config can pin
  per role.

---

## 8. Autonomy roadmap (parallel thread)

The same body of work that lets us test prompts also drives the
system toward autonomy. Both threads share infrastructure.

### 8.1 Levels

| Level | Description | This codebase today |
|---|---|---|
| L0 | User-driven only. | Yes (CLI). |
| L1 | Idle-time pipeline. Inbox watcher ingests; scheduler runs farming. | **Partial** -- `farming/scheduler.py` and `cli.py watch` exist. |
| L2 | Self-tuning weights. Calibrator stage measures quality and adjusts farming weights. | **Skeleton** -- CalibratorStage runs but doesn't write to a weights table. |
| L3 | Self-improving prompts. System measures prompt variants, picks winners, rolls out new defaults. | **Not yet** -- needs the eval harness from Section 7. |
| L4 | Agentic farming. LLM is a decision-maker with tool use, not just a narrator. | **Not yet** -- needs L3 plus a tool-use protocol plus an action sandbox. |
| L5 | Multi-agent. Specialised agents (extractor, planner, retriever, synthesiser) coordinate via the knowledge store as shared memory. | **Not yet** -- needs L4 plus inter-agent coordination. |

### 8.2 The dependency chain

```
Eval harness (Section 7)  -->  L2  -->  L3  -->  L4  -->  L5
```

L2 needs deterministic metrics; the eval harness produces them.

L3 = "the system runs the eval harness on itself, proposes new
prompts, ships the winners". Concretely, an LLM-as-author loop:
take the current best prompt + the corpus where it underperforms,
ask the LLM to draft three variants targeting those failure modes,
evaluate, promote.

L4 = "the system uses an LLM to decide what action to take next":
which file to ingest first, which entity to enrich, when to farm,
what to ask the user. Requires L3 (tested prompts) plus a
sandboxed action layer.

L5 = "specialised agents handing off via the knowledge store".
Each existing farming stage is already a narrowly scoped agent
shape -- just without LLM decision-making.

### 8.3 Where to start

The right next move is **the eval harness in Section 7**. It
unlocks both threads:

- For prompt experimentation, it's the missing measurement piece.
- For autonomy, it's the L2 gate, the L3 gate, and the L4
  guardrail.

Without it, every claim about "this prompt is better" is a vibe.
With it, claims are testable.

### 8.4 Scoping autonomy with user trust

A separate axis from the L0-L5 ladder: how much agency the system
has over the user's data.

| Scope | What the system can do without asking |
|---|---|
| Read-only | Ingest, query, farm, write insights to its own tables. (Today.) |
| Suggest | Surface "I think we should merge these entities" but require user click. |
| Apply-with-undo | Auto-apply changes that have a one-click undo. |
| Apply-irrevocably | Delete, archive, prune without asking. |

Today the system is mostly read-only with a few apply-irrevocable
maintenance stages (Pruner, Archivist). A serious autonomy push
should add a Suggest scope first -- the merge UI is exactly that
shape. Apply-with-undo is the long-game default.

---

## 9. Concrete first experiments

Ten experiments, ordered by ROI. Each is one new file in
`prompts/stages/<role>/` plus an entry in the eval harness.

1. **Extraction `v1.1.0-cot.txt`** -- chain-of-thought before JSON.
   Hypothesis: improves F1 on triples by 3-5%.
2. **Synthesis `v1.1.0-structured.txt`** -- JSON output with
   explicit citations, caveats, follow-ups. Hypothesis:
   downstream parsing reliability up >> any quality drop.
3. **Synthesis `v1.1.0-hedged.txt`** -- enforced hedging language.
   Hypothesis: judge faithfulness score up.
4. **Query planning `v1.1.0-ambiguity.txt`** -- surface ambiguity.
   Hypothesis: planning accuracy roughly flat, user-rated
   helpfulness up.
5. **Farming `distiller_v1.0.0.txt`** (after Phase 4.1 wiring) --
   one-sentence natural summary. Hypothesis: query-stack answer
   quality up substantially when user asks "what do you know
   about X?".
6. **Extraction `v1.1.0-fewshot.txt`** -- 3 worked examples.
   Hypothesis: F1 up on edge cases (timestamps, fragments).
7. **Retrieval `v1.1.0-expand.txt`** -- query paraphrase
   expansion. Hypothesis: recall@10 up at modest cost.
8. **Fusion `v1.1.0-coverage.txt`** -- entity-coverage maximisation
   in top-K. Hypothesis: synthesis judge score up.
9. **Farming `causal_miner_v1.0.0.txt`** (after the keying fix
   AND Phase 4.1 wiring) -- LLM verification of mined pairs.
   Hypothesis: false positives cut by half or more.
10. **Layer 1 base `v1.1.0.txt`** -- tightened safety + format
    rules with explicit "if you don't know, say UNKNOWN, never
    invent". Hypothesis: hallucination rate down across all roles.

---

## 10. Recommended sequence

The actual plan, given everything above:

1. **Land Phase 4.1 -- Distiller LLM wiring.** Smallest possible
   first step. One stage, one method, one prompt template. Gives
   us a third "live" farming LLM call site that everything else
   can be measured against. Shippable in a day.
2. **Build the minimal eval harness** (Section 7). One CLI
   command, one fixture corpus, mechanical metrics first, judge
   metrics second. The unlock.
3. **Run baseline eval** on v1.0.0 of every role + the new
   Distiller v1.0.0. Establish the numbers we're trying to beat.
4. **A/B test the top-3 variants** from Section 9 (likely
   experiments 1, 2, 5). Promote the winners as v1.1.0.
5. **Wire Calibrator to a weights table** (L2 autonomy). Eval
   harness produces data; Calibrator writes adjustments; next
   farming cycle reads them.
6. **Add the Suggest scope to the merge UI** (autonomy with user
   trust). System surfaces "merge candidates" backed by evaluated
   prompt confidence; user clicks accept.
7. **Build the prompt-author LLM loop** (L3). LLM proposes new
   variants targeting failure modes; eval harness judges; winners
   promoted. Closed loop.
8. **Design the action sandbox** (L4 prereq). Read-only by
   default, allow-list of write tools, audit log of every action.

Items 1-3: weeks. Items 4-5: weeks-to-months. Items 6-7: months.
Item 8: the start of a longer thread.

---

## 11. Open design questions

- **Where do prompts physically live?** Filesystem (today) is
  versionable in git, easy to diff, easy to load. DB storage
  buys per-user customisation but loses git history. Probably
  keep filesystem; store user-edited Layer 4 separately.
- **Should the eval harness ship in the public release?** Pro:
  users verify their own prompts, trust the system. Con: needs a
  corpus, which adds maintenance. Probably yes, with a small
  public corpus and a hook for user-supplied corpora.
- **Local vs hosted for eval?** Probably support both with the
  recommendation: local actor, hosted judge during development;
  local actor + local judge in production.
- **LLM-as-judge bias.** Self-judging is biased; same-family
  judging less so but still biased. Mitigation: at least two
  judges from different families for promotion decisions.
- **How does Layer 4 (user prefs) get edited?** Web UI? Config
  file? CLI? The 4-layer model needs a clear authoring surface
  for end-users.
- **Self-improving prompts (L3) before user-facing autonomy
  (L4)?** Probably yes -- L3 affects only the system's own
  internal prompts, no user-visible action surface.

---

## 12. What this doc deliberately does NOT do

- Does not pick winners. The whole point is "measure, don't
  speculate".
- Does not write the eval harness. That's the next concrete
  deliverable after Phase 4.1.
- Does not enumerate every prompt variant per role. The 50
  combinations of (role) x (technique) are not all equal; the
  ones in Sections 4 and 9 are the high-information ones.
- Does not commit to L4/L5. Those are downstream of L3 results
  we don't have yet.
- Does not commit to a UI for Layer 4 user prefs. Authoring
  surface is its own design problem.
