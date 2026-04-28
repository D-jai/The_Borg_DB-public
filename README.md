# The_Borg_DB

**Local-first knowledge intelligence system.**

The_Borg_DB ingests your text (meetings, emails, tickets, documents, notes),
extracts structured knowledge (entities, facts, relationships), stores everything
in a dual SQL + vector database, answers hybrid queries, and mines accumulated
data for patterns via an 18-stage farming pipeline. All data stays on your
hardware. LLM features use OpenAI-compatible endpoints you configure — pointed
at a local model server, a hosted API, or both.

## Quick Start

> **Where data lives:** all runtime artifacts (databases, vectors,
> inbox, outbox, auth files, evaluations) live under
> `<project_root>/.runtime/`. Override with `CTXMTG_DATA_ROOT=...`
> if you want them elsewhere. See `src/ctxmtg/paths.py`.

```bash
# Clone and install
git clone https://github.com/D-jai/The_Borg_DB.git
cd The_Borg_DB
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[web]"
python -m spacy download en_core_web_sm

# Ingest some text
ctxmtg ingest "Alice proposed migrating auth to OAuth2. Bob raised concerns."

# Ask a question
ctxmtg query "What did Alice propose?"

# Run farming (mines patterns from accumulated data)
ctxmtg farm run

# Start the web UI (local instance dashboard)
ctxmtg serve
```

## Features

| Feature | Description |
|---------|-------------|
| **10 file formats** | .txt, .md, .csv, .html, .docx, .pdf, .json, .eml, .ics, .vcf |
| **Dual-store architecture** | SQLite (structured facts) + LanceDB (semantic vectors) |
| **4 query modes** | Parallel, Vector→SQL, SQL→Vector, Deep (bidirectional) |
| **18-stage farming** | Entity analytics, trends, clustering, topics, causal mining, rationalizer, maintenance |
| **Garbage entity lifecycle** | Rationalizer marks low-quality entities → Archivist moves them to cold storage |
| **Profile-driven entity filters** | Reject patterns + length rules applied at extraction time |
| **Thinking-token stripping** | `<think>` / `<thinking>` blocks stripped from all LLM responses before storage |
| **Web dashboard** | Record counts, farming controls, entity resolution, LLM usage stats |
| **Per-role LLM config** | Configure API key + base URL + model per pipeline stage (6 roles) |
| **Inbox watcher** | Drop files into `<project>/.runtime/inbox/` for auto-ingestion |
| **HTTP ingest API** | `POST /api/ingest` for programmatic feeding |
| **LLM proxy** | Transparent proxy that captures LLM conversations as knowledge |
| **Hive outbox/inbox sync** | Multi-device knowledge aggregation via file-based manifests |
| **Hive Command Center** | Separate web UI for the hive (link management, pull actions) |
| **OpenAI-compatible API** | Use The_Borg_DB as a "model" in Open WebUI or any chat client |
| **Entity resolution UI** | Find, merge, and delete duplicate entities through the browser |

## Architecture

```
Ingestion → Extraction (spaCy NER + regex + fact triples + filters) → Dual Store (SQLite + LanceDB)
                                                                              ↓
Query → Interpreter → Parallel / V2S / S2V / Deep Retrieval → Fusion → Synthesis → Answer
                                                                              ↓
Farming (18 stages) → Insights, Trends, Clusters, Rationalized Entities → Enriched Queries
                                                                              ↓
Local Outbox (JSON manifests) → Hive Pull → Cross-Stream Discovery → Latent Relationships
```

## LLM Configuration

The_Borg_DB uses LLMs via OpenAI-compatible endpoints for 6 pipeline roles:

| Role | What it does |
|------|--------------|
| **Extraction** | Verifies and enhances NER output |
| **Query Planning** | Interprets natural language queries |
| **Retrieval** | Formulates V2S (Vector→SQL) and S2V (SQL→Vector) bridge queries |
| **Synthesis** | Generates cited answers from retrieval results |
| **Farming** | Generates insight descriptions for farming stages |
| **Fusion** | Reranks fused results by semantic relevance |

Configure them through the web UI at **Local Instance > LLM Roles**, or by
writing to `<project_root>/.runtime/.env` directly. See [.env.example](.env.example)
for the full variable layout.

```bash
export CTXMTG_LLM__EXTRACTION__API_KEY="your-key"
export CTXMTG_LLM__EXTRACTION__BASE_URL="http://localhost:1234/v1"
export CTXMTG_LLM__EXTRACTION__MODEL="your-model-id"
```

All features work without an LLM — the system degrades gracefully to
rule-based extraction, SQL-only queries, and mathematical result fusion.

### Using the LLM features

Once a role is configured, its LLM is used automatically by the matching
pipeline path. No flags, no special commands. The same CLI you ran
without an LLM keeps working; the difference is that answers, insights,
and verifications get richer.

| Command | What changes when LLMs are configured |
|---------|----------------------------------------|
| `ctxmtg ingest …` | Extraction role runs after NER to verify candidates and add domain-specific entities the rule-based extractor missed. |
| `ctxmtg query …` | Query Planning interprets the question, Retrieval formulates V2S/S2V bridges, Fusion re-ranks, Synthesis writes a cited answer. |
| `ctxmtg farm run` | Currently deterministic. Future releases will wire LLMs into Distiller, Causal Mining, and Insight Generation — see `llm_design.md` for the staged plan. |
| `ctxmtg evaluate …` | LLM-as-judge scores answer quality against gold expectations. |

You can mix-and-match per role. A common setup is a small local model
(e.g. an 8B-class model on a localhost endpoint) for Extraction and
Farming, and a larger hosted model for Synthesis. Each role has its
own `BASE_URL` / `MODEL` / `API_KEY` in `.env`.

### How prompts work

The system uses a **4-layer prompt assembler** so prompts can be tuned
without code changes:

1. **Base** (`prompts/base/v1.0.0.txt`) — shared safety + format rules.
2. **Stage** (`prompts/stages/<role>/v1.0.0.txt`) — per-role task instructions.
3. **Domain** — slot-injected from the active `DomainProfile`
   (entity types, terminology, reasoning patterns).
4. **User preferences** — per-user customizations (summary length,
   priority topics, output language).

Each layer is independently versionable. To experiment with a variant
prompt for a role, drop a new file at `prompts/stages/<role>/vX.Y.Z.txt`
— no code change needed. See [llm_design.md](llm_design.md) for the
prompt-experimentation methodology and the planned eval harness.

## CLI Commands

| Command | Description |
|---------|-------------|
| `ctxmtg ingest <file_or_text>` | Ingest a file or raw text |
| `ctxmtg ingest --dir <path>` | Batch ingest a directory |
| `ctxmtg query <question> [--mode parallel\|v2s\|s2v\|deep]` | Ask a question |
| `ctxmtg farm run` | Run an 18-stage farming cycle |
| `ctxmtg farm status` | View farming history |
| `ctxmtg watch` | Auto-ingest from the inbox folder |
| `ctxmtg serve` | Start the local web dashboard |
| `ctxmtg proxy` | Start the LLM proxy |
| `ctxmtg hive push` | Write intelligence to the local outbox |
| `ctxmtg hive serve` | Start the Hive Command Center web UI |
| `ctxmtg suggest` | Query autocomplete |
| `ctxmtg health` | System health check |
| `ctxmtg profile --list` | List domain profiles |

## Documentation

- **[User Guide](docs/user-guide.md)** — Full installation, configuration, and usage guide
- **[Error Codes](docs/error_codes_guidance.md)** — Structured error code reference
- **[Whitepaper](docs/whitepaper.md)** — Architecture and design rationale
- **[Runtime relocation (v0.7.1)](runtimechange.md)** — Why runtime data moved to `<project>/.runtime/`
- **[Autonomous farming design](autonomous_farming.md)** — Working design notes for the 18-stage pipeline
- **[Audit findings (v0.7.1)](audit_findings_v0.7.1.md)** — Keying / LLM wiring / merge-route audit
- **[LLM design spec](llm_design.md)** — All 9 LLM call sites, per-role prompt variants, eval harness sketch, autonomy roadmap. **Section 10 is the recommended next-step sequence.**

## For LLMs and AI Agents

> If you are an AI coding assistant being pointed at this repository to
> work on a task, read this section first. It tells you what to load,
> what to skip, and how the project's invariants are organised.

### Load these on first contact

1. **`README.md`** (this file) -- shape of the system + commands.
2. **`src/ctxmtg/paths.py`** -- 14 helpers, one resolver. Every
   runtime-path question has its answer here.
3. **`src/ctxmtg/farming/__init__.py`** -- the 18-stage roster
   and `create_default_stages()` ordering.
4. **`autonomous_farming.md`** -- the working design doc.
5. **`audit_findings_v0.7.1.md`** -- the live list of known
   issues. Every "why isn't this fixed?" question has its answer
   here, including which items are deliberate non-goals.
6. **`llm_design.md`** -- the LLM strategy working spec. Read
   Section 10 ("Recommended sequence") for the current
   next-step plan. **Phase 4.1 (Distiller LLM wiring) is the
   immediate next concrete deliverable.**
7. **`CHANGELOG.md`** -- what shipped when. The most recent
   section is always the source of truth for current shape.

### Do not pre-read these (load only on demand)

- `docs/whitepaper.md` -- architectural narrative; large; loads
  context budget without changing how you'd write code.
- `src/ctxmtg/web/templates/` -- Jinja2 templates for the web UI;
  load only if you're working on the dashboard.
- `tests/` directory -- does not exist in the public release.
  Don't go looking; do not create new tests speculatively
  unless asked.
- `configs/default.yaml` -- mostly defaults that the resolver
  already handles. Only load if a config-shape question comes up.

### Project invariants (do not break)

- **Runtime data root** = `<project_root>/.runtime/` unless
  `CTXMTG_DATA_ROOT` is set. The resolver in `paths.py` is the
  only place that decides this. Do not introduce a second
  decision point. Do not add new path constants in
  `constants.py`; route through `paths.py`.
- **Per-interaction entity ids** are deliberate. Same logical
  entity in two interactions has two ids. The merge UI is a
  name rename, not an id merge. If you find yourself wanting
  to "fix" this by introducing a `canonical_entity_id` field,
  read `audit_findings_v0.7.1.md` Section 1 first; the
  cross-interaction join question is real but the fix is
  name-keying the two id-keyed stages, not changing the id
  model.
- **`self._llm` is intentionally pre-wired** in 16 of 18 farming
  stages even though none of them call `.generate()` today.
  Adding LLM logic to one stage is a one-method change. Don't
  remove the parameter "to clean up unused state".
- **Schema migrations are unidirectional**. If you add a DDL
  constant, append to `ALL_DDL` and bump `SCHEMA_VERSION`.
  Don't reorder or delete existing constants.
- **The CLI is the contract**. CLI flags and subcommand names
  do not change without a CHANGELOG entry. Adding new commands
  is fine; renaming or removing requires version coordination.

### Common task shapes and how to start them

- **"Add LLM logic to a farming stage"** -- start at the stage
  file, find `self._llm`, add the call site near where the
  stage's deterministic logic produces its summary string.
  Pattern: `if self._llm: <call>` so the stage still runs
  without a configured LLM. See `query/synthesizer.py` for
  reference.
- **"Add a new farming stage"** -- conform to `interfaces/farming.py`,
  register in `farming/__init__.create_default_stages()`,
  decide whether you need a checkpoint row and a progress
  offset row, write a stage docstring that names the stage's
  unit of work and its keying choice (id vs name).
- **"Fix a runtime-path bug"** -- start in `paths.py`, then
  grep for the helper to find the wiring site. Don't add a new
  constant; reuse or extend a helper.
- **"Add an entity-resolution feature"** -- start in
  `web/routes/entities.py` and read the audit's Section 3
  before touching the code; many "obvious bugs" are reframed
  by the rename-not-merge model.

### Things you should ASK about, not assume

- Whether to add tests when the public release ships none.
- Whether a behavioural change (e.g. CausalMiner / Consolidator
  name keying) should land alone or batched with a smoke-test
  suite.
- Whether to migrate `~/.ctxmtg/` data automatically (the
  decision is currently "no" -- see runtimechange.md).
- Whether to add CSRF / rate-limit / request-validation to the
  web routes (currently softened by localhost-only deployment).

## License

Copyright (c) 2024 Aliud Inquisito Inc.

The_Borg_DB is distributed under the **Business Source License 1.1** — see
[LICENSE](LICENSE) and [NOTICE](NOTICE).

In plain English:

- **Free for internal production use** within your own organization. Run it
  on your own data, no license fee and no notification needed.
- **Not free to offer as a competing hosted, embedded, or managed service**
  to third parties. If that's what you want to do, contact the Licensor
  (Dhananjay Raol, on behalf of Aliud Inquisito Inc.) for a commercial
  license.
- **Auto-converts to GPL-3.0-or-later on 2029-01-01** — or on the fourth
  anniversary of the first public release of a given version under the BSL,
  whichever comes first. On that date, that version of The_Borg_DB
  automatically becomes Open Source.

BSL 1.1 is source-available, not Open Source in the OSI sense. Full terms
live in [LICENSE](LICENSE).
