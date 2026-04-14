# The_Borg_DB User Guide

**Version:** 0.7.0
**Last updated:** 2026-04-13

The_Borg_DB is a local-first knowledge intelligence system. It ingests text
(meetings, emails, tickets, documents, notes), extracts structured knowledge
(entities, facts, relationships), stores everything in a dual SQL + vector
database, and answers hybrid queries. All data stays on your hardware. LLM
features use OpenAI-compatible endpoints you configure — no data is sent
anywhere you don't choose.

---

## Table of Contents

1. [What The_Borg_DB Is](#1-what-the_borg_db-is)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Installation](#3-installation)
4. [Quick Start](#4-quick-start)
5. [Data and Security](#5-data-and-security)
6. [Ingestion](#6-ingestion)
7. [Querying](#7-querying)
8. [Autocomplete](#8-autocomplete)
9. [Farming (Deep Dive)](#9-farming-deep-dive)
10. [Hive Outbox/Inbox Sync](#10-hive-outboxinbox-sync)
11. [Domain Profiles](#11-domain-profiles)
12. [Configuration Reference](#12-configuration-reference)
13. [Web Command Center](#13-web-command-center)
14. [LLM Configuration](#14-llm-configuration)
15. [Running Without an LLM](#15-running-without-an-llm)
16. [Upgrading](#16-upgrading)
17. [Troubleshooting](#17-troubleshooting)
18. [Appendix A: Example Queries](#appendix-a-example-queries)

---

## 1. What The_Borg_DB Is

The_Borg_DB sits between your text data and your questions about it. You feed
it content. It extracts who, what, when, and why. You ask questions. It answers
from locally stored facts.

```
Your text (meetings, emails, tickets, notes, documents)
        │
        ▼
  Extraction pipeline
  (spaCy NER + regex + profile-driven filters + fact triples + embeddings)
        │
        ▼
  Dual store
  (SQLite for structured facts │ LanceDB for semantic vectors)
        │
        ▼
  Query engine
  (SQL + semantic search, four modes, fused + reranked results)
        │
        ▼
  Your answer (with citations)
```

Alongside queries, an 18-stage farming pipeline runs in the background: it
mines patterns (co-occurrence, trends, clusters, topics, causal chains),
maintains data quality (rationalizer → archivist lifecycle for garbage
entities, consolidator for duplicates), and pushes intelligence upstream to an
optional Hive node where knowledge from multiple instances is aggregated and
cross-correlated.

The system degrades gracefully: no LLM configured? It uses rule-based
extraction and mathematical result fusion. No embedding model? SQL queries
still work. Every LLM-enhanced feature has a non-LLM fallback.

---

## 2. Hardware Requirements

| Tier | Device | RAM | What works |
|------|--------|-----|-----------|
| **0** | Raspberry Pi 4/5 | 2–4 GB | Ingest, query, lightweight farming, push to hive. No local LLM. |
| **1** | Jetson Orin Nano/NX | 8–16 GB | Same + GPU-accelerated embeddings. |
| **2** | Laptop (M-series Mac, modern x86) | 8–32 GB | Full feature set. Can host the hive. All query modes. |
| **3** | Desktop / DGX-class | 16 GB+ | Same + fast farming, cross-encoder reranking, local large LLMs. |

**Storage:** Budget ~1 GB per 50,000 interactions. SSD recommended for vector
store write performance.

**Python:** 3.10 or newer required.

---

## 3. Installation

### 3.1 From source

```bash
# Clone the repository
git clone https://github.com/D-jai/The_Borg_DB.git
cd The_Borg_DB

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package. Pick the optional extras you need:
#   [web]    - dashboard + OpenAI-compatible API (recommended)
#   [llm]    - local llama.cpp provider
#   [gpu]    - onnxruntime-gpu + faiss-gpu
#   [server] - PostgreSQL driver for Tier 4 deployments
#   [all]    - everything above
pip install -e ".[web]"

# Install the spaCy language model (required for entity extraction)
python -m spacy download en_core_web_sm
```

### 3.2 Verify the installation

```bash
ctxmtg health
```

The health command creates the `~/.ctxmtg/` directory structure on first run
and reports the status of all components (database, embedding model, spaCy,
per-role LLM).

### 3.3 Configure your LLM endpoints

Copy the provided `.env.example` and fill in real values:

```bash
mkdir -p ~/.ctxmtg
cp .env.example ~/.ctxmtg/.env
chmod 600 ~/.ctxmtg/.env
${EDITOR:-nano} ~/.ctxmtg/.env
```

Alternatively, leave the file empty and configure the roles through the web
UI at **Local Instance > LLM Roles** after starting `ctxmtg serve`.

### 3.4 Directory structure created on first run

```
~/.ctxmtg/
├── .env                 # LLM API keys (per-role), instance name — chmod 600
├── knowledge.db         # SQLite: entities, facts, interactions, farming tables
├── archive.db           # SQLite: archived (cold / garbage) entities + facts
├── vectors/             # LanceDB: embedding vectors
├── inbox/               # Drop files here for auto-ingestion
├── processed/           # Successfully ingested files land here
└── outbox/              # Hive sync manifests (JSON) — pulled by the hive
    └── processed/       # Manifests the hive has already ingested
```

The hive (if run) keeps its own `hive.db` and hive-side web config under a
separate `CTXMTG_HOME` directory of your choosing.

---

## 4. Quick Start

Five commands to see The_Borg_DB working end to end:

```bash
source .venv/bin/activate

# 1. Check system health
ctxmtg health

# 2. Ingest some text
ctxmtg ingest "Alice proposed migrating auth to OAuth2. Bob raised timeline concerns."

# 3. Ingest a file
ctxmtg ingest meeting_notes.txt

# 4. Ask a question
ctxmtg query "What did Alice propose?"

# 5. Run a farming cycle
ctxmtg farm run
```

### Zero-friction alternatives

- **Drop files** into `~/.ctxmtg/inbox/` and run `ctxmtg watch` to auto-ingest them.
- **Start the web UI** with `ctxmtg serve` and manage everything through a browser.
- **POST via HTTP** to `http://127.0.0.1:8080/api/ingest` when the web server is running.

---

## 5. Data and Security

### 5.1 Where your data lives

All data is stored in `~/.ctxmtg/` by default. Override with environment
variables `CTXMTG_DB_PATH`, `CTXMTG_VECTOR_PATH`, and `CTXMTG_HOME`.

### 5.2 What The_Borg_DB never does

- Sends data to any external service — unless you configure LLM API keys.
  When configured, only prompts go to the API you chose.
- Logs raw interaction content. Logs contain only counts, durations, and
  error types.
- Opens public network ports. The web server, hive UI, and LLM proxy bind to
  `127.0.0.1` only.

### 5.3 LLM API key security

When you configure LLM API keys (via web UI or `.env.example`), they are
stored in `~/.ctxmtg/.env`:

- Permission 600 (user-only read/write).
- Listed in `.gitignore` (never committed).
- Read at startup by `pydantic-settings` — never passed as command-line args.

### 5.4 Thinking-token hygiene

LLM responses are scrubbed of `<think>`, `<thinking>`, and `<|thinking|>`
blocks before any of the following see them: the extraction pipeline, query
synthesis output, stored facts/insights, and the LLM proxy's captured
transcripts. Model reasoning traces never reach storage or users.

---

## 6. Ingestion

### 6.1 Supported file formats

| Extension | Format | What is extracted |
|-----------|--------|------------------|
| `.txt` | Plain text | Full text → NER, facts, embeddings |
| `.md` | Markdown | Full text (formatting preserved) → NER, facts, embeddings |
| `.csv` | CSV spreadsheet | Rows formatted as `col: value` pairs → NER, facts |
| `.html` / `.htm` | HTML page | Visible text extracted (tags/scripts/styles stripped) |
| `.docx` | Word document | Paragraphs + table text → NER, facts, embeddings |
| `.pdf` | PDF document | Text-based extraction (no OCR for scanned PDFs) |
| `.json` | JSON document | Text fields → same pipeline |
| `.eml` | Email (RFC 2822) | Subject, body, sender, recipients |
| `.ics` | Calendar event (iCal) | Event title, attendees, dates, location |
| `.vcf` | Contact card (vCard) | Name, org, email, phone |

### 6.2 Ingest a single file

```bash
ctxmtg ingest meeting_notes.txt
ctxmtg ingest report.pdf
ctxmtg ingest proposal.docx
```

### 6.3 Ingest raw text

```bash
ctxmtg ingest "Sprint review: Alice reduced build time by 40%."
```

### 6.4 Batch ingest a directory

```bash
ctxmtg ingest --dir ./documents/
```

### 6.5 Inbox watcher (auto-ingest)

Drop files into `~/.ctxmtg/inbox/` and let The_Borg_DB pick them up
automatically.

```bash
# Continuous watching (polls every 30 seconds)
ctxmtg watch

# Custom interval
ctxmtg watch --interval 10

# Process once and exit (suitable for cron)
ctxmtg watch --once
```

On success, files are moved to `~/.ctxmtg/processed/`. On failure, files stay
in `inbox/` and the error is logged. Duplicate filenames get numeric suffixes.

**Systemd user service example:**

```ini
[Unit]
Description=The_Borg_DB inbox watcher

[Service]
Type=simple
ExecStart=%h/.ctxmtg/venv/bin/ctxmtg watch
Restart=on-failure

[Install]
WantedBy=default.target
```

> **Note:** Do not run `ctxmtg watch` and `ctxmtg serve` simultaneously against
> the same database unless you've accepted the SQLite write-lock contention
> trade-off. Ingest batches with the server stopped, or stagger them.

### 6.6 HTTP ingest endpoint

When the web server is running (`ctxmtg serve`):

```bash
curl -X POST http://127.0.0.1:8080/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Alice proposed OAuth2 migration.", "title": "Sprint Review", "source_type": "meeting"}'
```

Can be enabled/disabled from the web UI without restarting the server.

### 6.7 LLM proxy (live chat capture)

The LLM proxy sits between your chat client and your LLM backend, silently
capturing every conversation as searchable knowledge. Thinking tokens are
stripped before capture.

```bash
# Default: proxy on 11435, forwarding to a local Ollama on 11434
ctxmtg proxy

# Forward to any OpenAI-compatible backend (LM Studio, vLLM, a hosted API)
ctxmtg proxy --upstream http://localhost:1234
```

Point your chat client at `http://127.0.0.1:11435` instead of the LLM
backend. All requests are forwarded unchanged. Capture can be toggled from the
web UI.

### 6.8 Intake filtering (Traffic Cop)

Before extraction, every piece of content passes through the Traffic Cop which
applies rules from your active domain profile to accept, defer, or reject
content. Automated notifications and very short texts are rejected by default.

```bash
ctxmtg intake stats    # View intake decisions
```

### 6.9 Profile-driven entity filters

After NER + regex extraction, a second filter stage removes low-quality
entities before they reach storage. Rules live under `ner.entity_filters` in
your active profile YAML and cover:

- `min_name_length` / `max_name_length`
- `reject_patterns` — regex blocklist (timestamps, bare decimals, timezone
  offsets, header labels, etc.)
- `reject_names` — exact-match blocklist

Edit `profiles/<name>.yaml` to customize the filters for your domain.

---

## 7. Querying

### 7.1 Basic query

```bash
ctxmtg query "What did Alice propose?"
```

Results show `[sql]` (structured fact matches) and `[vector]` (semantic
similarity matches), fused and reranked.

### 7.2 Retrieval modes

| Mode | Flag | How it works | Best for |
|------|------|--------------|----------|
| **Parallel** | `--mode parallel` | SQL + vector queried independently, results fused | Most queries (default) |
| **Vector→SQL** | `--mode v2s` | Semantic search first, LLM formulates SQL from findings | Open-ended exploration |
| **SQL→Vector** | `--mode s2v` | Structured query first, LLM finds contextual gaps | Specific facts, timelines |
| **Deep** | `--mode deep` | Both informed paths combined, synthesized together | Complex reasoning, max recall |

Benchmarked on a 101-ticket dataset (1560 entities, 315 facts, 1098 vectors):

| Mode | Avg Results | Avg Latency | Synthesis | Best for |
|------|------------|-------------|-----------|----------|
| parallel | 25 | 1.3s | No | Fast lookups, entity browsing |
| **v2s** | **50** | **22s** | Yes | **Best single path — most results with synthesis** |
| s2v | 31 | 23s | Yes | Gap-filling when the facts table is dense |
| deep | 50–56 | 60s | Yes | Mission-critical queries needing maximum recall |

Without an LLM, `v2s` / `s2v` return unranked results and `deep` falls back to
`parallel`.

```bash
ctxmtg query "Tell me about the OAuth2 migration" --mode v2s
ctxmtg query "When did we agree on the database?" --mode s2v
ctxmtg query "Why are we moving away from old auth?" --mode deep
ctxmtg query "security decisions" --top-k 20
```

---

## 8. Autocomplete

```bash
ctxmtg suggest "What did Al"    # Query completions
ctxmtg suggest --browse         # Browse top entities by distilled relevance
```

Most useful after at least one farming cycle has run (`ctxmtg farm run`). The
distiller stage builds per-entity summaries that power both modes.

---

## 9. Farming (Deep Dive)

Farming is the heart of The_Borg_DB's intelligence. It is a background
pipeline that mines your accumulated knowledge for patterns, trends, and
insights that no single interaction reveals on its own — while also keeping
the store healthy by deduping facts, rationalizing garbage entities, and
archiving cold data.

### 9.1 Why farming matters

Without farming, The_Borg_DB is a search engine — it finds what you put in.
With farming, The_Borg_DB becomes an intelligence system:

- "Alice and Bob always work on the same projects" (co-occurrence pattern)
- "OAuth2 mentions have increased 300% this month" (trend detection)
- "These 12 interactions form a cluster around database migration" (topic clustering)
- "The decision to use Redis was likely caused by the latency incident" (causal mining)

### 9.2 The 18-stage pipeline

Farming runs 18 stages in three groups. Each stage is independent and produces
`FarmingInsight` objects or maintenance actions. Stages progressively scan
through data batch-by-batch — running N cycles covers N × batch_size entities
rather than repeating the same top slice.

#### Intelligence stages (stages 1–7) — Discover new patterns

| # | Stage | What it does | Example output |
|---|-------|--------------|----------------|
| 1 | **Entity Analytics** | Counts entity frequency, builds co-occurrence matrices, identifies the most-mentioned entities and who appears together | "Alice (47) and Bob (31) co-occur in 18 interactions" |
| 2 | **Trend Detection** | Compares entity/topic frequency across time windows to detect rising and falling trends | "OAuth2 mentions rose from 2/week to 11/week" |
| 3 | **Clustering** | Groups related entities using K-Means on their embedding vectors | "Cluster: {Redis, Memcached, caching, latency}" |
| 4 | **Topic Modeling** | Runs LDA across all interaction text to discover recurring themes | "Topic: authentication + migration + security" |
| 5 | **Graph Analysis** | Builds a co-occurrence graph and runs PageRank to find the most connected entities | "Alice has the highest PageRank (0.12)" |
| 6 | **Insight Generator** | Aggregates outputs from stages 1–5 and generates readable insight summaries (LLM-enriched when configured) | "Sprint velocity has been declining since the OAuth2 migration began" |
| 7 | **Causal Miner** | Finds temporal cause-effect patterns in real (non-farming-generated) facts | "Latency incidents tend to follow deployments within 48 hours" |

#### Self-learning stage (stage 8)

| # | Stage | What it does |
|---|-------|--------------|
| 8 | **Feedback Loop** | Analyses the `query_quality_log` to find queries returning zero or poor results. Surfaces extraction gaps |

#### Maintenance stages (stages 9–18) — Keep the store healthy

| # | Stage | What it does |
|---|-------|--------------|
| 9 | **Rationalizer** | Tests each entity name against 8 garbage-detection rules (embedded newlines, URL fragments, markdown link artifacts, bare-decimal phone fragments, truncation markers, pure punctuation, sub-2-char names, excessive whitespace). Matches get `confidence = 0.1`. Entities with important facts (`responsible_for`, `leads`, `reports_to`, `decided`, `committed_to`) are protected. Non-destructive and reversible. |
| 10 | **Consolidator** | Merges duplicate facts (same subject + predicate + object), keeping the highest-confidence version |
| 11 | **Pruner** | Removes facts with very low confidence or expired relevance |
| 12 | **Completionist** | Fills entity-attribute gaps from other interactions where the same entity appears |
| 13 | **Linker** | Connects entities across interactions (e.g. "Alice Chen" in meeting A ↔ "A. Chen" in email B) |
| 14 | **Verifier** | Spot-checks fact consistency and flags contradictions |
| 15 | **Calibrator** | Uses feedback-loop signals to adjust confidence scores and extraction weights |
| 16 | **Distiller** | Builds per-entity summaries with relevance score, top co-entities, top predicates. Powers `ctxmtg suggest` |
| 17 | **Archivist** | Moves old, unreferenced, low-value entities to `archive.db`. Garbage bypass: any entity with `confidence ≤ 0.1` (rationalized) is archived immediately regardless of age |
| 18 | **Defragmenter** | Compacts the SQLite database and rebuilds indexes |

The **rationalizer → archivist** combination implements a non-destructive
garbage lifecycle: bad entities are first marked (confidence downgrade), then
moved to a separate archive DB on the next cycle. A future `ctxmtg maintenance
trim` command will permanently delete archived rows.

### 9.3 Running farming

```bash
# Run a full 18-stage cycle
ctxmtg farm run

# Check recent cycle history
ctxmtg farm status
ctxmtg farm status --limit 10
```

### 9.4 Farming requirements

- **Minimum interactions:** Farming activates after 50 interactions. With
  fewer, cycles complete quickly but produce no insights.
- **LLM optional:** All 18 stages work without an LLM. When configured, the
  `farming` role LLM produces natural-language descriptions for the insight
  generator, completionist, and rationalizer.

### 9.5 Automated scheduling

Set up a systemd timer to run farming during idle periods:

```ini
# ~/.config/systemd/user/ctxmtg-farm.timer
[Unit]
Description=The_Borg_DB farming timer

[Timer]
OnCalendar=*:0/30
Persistent=true

[Install]
WantedBy=timers.target
```

```ini
# ~/.config/systemd/user/ctxmtg-farm.service
[Unit]
Description=The_Borg_DB farming cycle

[Service]
Type=oneshot
ExecStart=%h/.ctxmtg/venv/bin/ctxmtg farm run
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now ctxmtg-farm.timer
```

### 9.6 How farming improves queries

1. **Better autocomplete** — distiller summaries power `ctxmtg suggest`
2. **Fewer duplicates** — consolidator merges duplicate facts
3. **Cleaner entities** — rationalizer + archivist remove garbage
4. **Richer entity profiles** — completionist and linker enrich across interactions
5. **Trend awareness** — trend detection surfaces rising topics
6. **Self-healing** — feedback loop identifies extraction gaps
7. **Faster queries** — defragmenter keeps indexes optimized

### 9.7 Viewing farming insights

Farming insights land in the `meta_insights` table and are surfaced in:

- The **web dashboard** (cycle status and insight counts)
- `ctxmtg suggest --browse` (distiller summaries)
- Query results (farming-enriched entity context improves semantic matching)

---

## 10. Hive Outbox/Inbox Sync

The Hive aggregates knowledge from one or more The_Borg_DB instances. Unlike
direct-database sync, the hive uses a **file-based outbox/inbox** pattern so
locals and the hive do not need to share a database file or even a filesystem.

### 10.1 The architecture

```
Local instance                     Hive instance
───────────────                    ─────────────
ctxmtg hive push                   ctxmtg hive serve
   │                                    │
   ▼                                    ▼
~/.ctxmtg/outbox/                   Hive UI creates a "link" to
  20260413T1500_Local_Tickets_ab12.json  each local's outbox path
  20260413T1530_Local_Tickets_cd34.json  and pulls on demand
          │                                    │
          └──────── ( rsync / scp / S3 / local FS ) ────────┘
```

Each manifest is a self-describing JSON file with a version, instance name,
batch ID, high-water marks, a full local metadata snapshot (DB counts, table
names, farming state, profile, LLM role models, platform), and a payload of
distiller summaries + meta insights. Files are named by ISO timestamp so
lexicographic sort gives FIFO processing. Writes are atomic (`.tmp` →
`os.rename`) so the hive never reads half-written files. Once ingested, the
hive moves the manifest to `outbox/processed/`.

### 10.2 Push from a local

```bash
# Ensure this local has a unique name
grep CTXMTG_HIVE__INSTANCE_NAME ~/.ctxmtg/.env

# Write the next batch of intelligence to ~/.ctxmtg/outbox/
ctxmtg hive push
```

The local tracks per-table high-water marks in its own `outbox_progress`
table, so each push contains only rows that haven't been sent before.

### 10.3 Run the Hive Command Center

```bash
# On the hive machine (can be the same host with a different CTXMTG_HOME)
ctxmtg hive serve
# Defaults to http://127.0.0.1:8082
```

The Hive UI is visually distinguished from the Local UI by an amber/gold
theme. From the dashboard you can:

- Create "links" to each local instance (name + outbox path).
- Click **[Pull]** next to a link to ingest any pending manifests.
- Browse merged entity profiles (cross-stream scored), local insights, and
  hive-native insights.

### 10.4 Cross-stream discovery

Once two or more locals have been pulled, the hive runs its own farming
stages:

- **Entity merging** — identical entity names across locals merge into a
  single profile. Multi-local entities get boosted cross-stream scores
  (~1.2–1.5) vs. single-local (~0.76).
- **Latent discovery** — finds hidden connections between entities that never
  directly co-occur but share common neighbors.
- **Insight correlation** — compares insights across locals to surface
  patterns that only become visible with multiple streams.

### 10.5 Transport options

The outbox directory *is* the protocol boundary. As long as files appear
atomically, the transport can be:

- A local filesystem path (same host, multiple `CTXMTG_HOME` directories).
- An NFS or SMB mount.
- `rsync` / `scp` to a hive host on a schedule.
- An S3 bucket mounted via a FUSE layer.

---

## 11. Domain Profiles

### 11.1 Built-in profiles

| Profile | Best for | Key entity types |
|---------|----------|------------------|
| `general` | Engineering teams, professional meetings | PERSON, ORG, PROJECT, TOOL, DEADLINE, DECISION |
| `legal` | Legal teams, contracts, case notes | PERSON, ORG, STATUTE, RULING, PARTY, JURISDICTION |
| `personal` | Personal notes, journaling | PERSON, PLACE, GOAL, EMOTION, EVENT, RELATIONSHIP |

### 11.2 Choosing a profile

```bash
ctxmtg --profile legal ingest contract.txt
ctxmtg --profile legal query "What are the key obligations?"
```

Or set it permanently in `~/.ctxmtg/config.yaml`:

```yaml
profile_name: legal
```

### 11.3 Viewing profiles

```bash
ctxmtg profile --list      # List all available profiles
ctxmtg profile general     # Show details
```

### 11.4 Creating custom profiles

Copy and edit an existing profile:

```bash
cp profiles/general.yaml profiles/engineering.yaml
```

Key sections to customize:

- `ner.entity_types` — the entity types your domain cares about
- `ner.entity_filters` — rejection rules (length, regex, exact names)
- `intake` — accept/defer/reject rules
- `stages` — LLM parameters per pipeline stage

---

## 12. Configuration Reference

### 12.1 Priority order

1. **Environment variables** (`CTXMTG_*` prefix) — highest priority
2. **`~/.ctxmtg/.env` file** — per-role LLM config + instance name
3. **Default values** — lowest priority

### 12.2 Key settings (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `CTXMTG_HOME` | `~/.ctxmtg` | Root directory for all state |
| `CTXMTG_DB_PATH` | `~/.ctxmtg/knowledge.db` | SQLite database path |
| `CTXMTG_VECTOR_PATH` | `~/.ctxmtg/vectors` | Vector store directory |
| `CTXMTG_PROFILE_NAME` | `general` | Active domain profile |
| `CTXMTG_TOP_K` | `10` | Default query result count |
| `CTXMTG_RETRIEVAL_MODE` | `parallel` | Default query mode |
| `CTXMTG_INBOX_PATH` | `~/.ctxmtg/inbox` | Inbox watcher directory |
| `CTXMTG_PROCESSED_PATH` | `~/.ctxmtg/processed` | Processed files directory |
| `CTXMTG_WATCH_INTERVAL_SECONDS` | `30` | Inbox poll interval |
| `CTXMTG_HTTP_INGEST_ENABLED` | `true` | Enable HTTP ingest endpoint |
| `CTXMTG_PROXY_PORT` | `11435` | LLM proxy port |
| `CTXMTG_PROXY_UPSTREAM` | `http://localhost:11434` | LLM proxy upstream |
| `CTXMTG_HOST` | `127.0.0.1` | Web server bind address |
| `CTXMTG_PORT` | `8080` | Local web server port |
| `CTXMTG_HIVE__INSTANCE_NAME` | _unset_ | Required for hive push — unique per local |
| `CTXMTG_HIVE__OUTBOX_PATH` | `~/.ctxmtg/outbox` | Outbox directory for hive manifests |

### 12.3 Per-role LLM configuration

Each pipeline role can have its own provider. See [Section 14](#14-llm-configuration).

---

## 13. Web Command Center

The_Borg_DB ships with two browser dashboards: one for local instances and one
for the hive. They run as separate processes with separate auth.

### 13.1 Starting the local server

```bash
ctxmtg serve                  # http://127.0.0.1:8080
ctxmtg serve --port 9090      # custom port
```

On first run, you set an admin password. The server binds to localhost only.

### 13.2 Starting the hive server

```bash
ctxmtg hive serve              # http://127.0.0.1:8082
```

Has its own login page and setup flow, distinguished by an amber/gold theme.

### 13.3 Local instance pages

| Page | URL | What it shows |
|------|-----|--------------|
| **Dashboard** | `/` | Record counts, hive status, farming actions, auto-refresh stats |
| **Local Instance** | `/local` | Profile viewer, farming params, LLM role config, service toggles |
| **Entities** | `/entities` | Entity resolution: find, merge, and delete duplicate entities |
| **LLM Usage** | `/usage` | Per-day, per-model, per-role usage stats and token counts |

### 13.4 Hive instance pages

| Page | URL | What it shows |
|------|-----|--------------|
| **Dashboard** | `/` | Merged entity / insight counts, link list, [Pull] buttons |
| **Intelligence** | `/insights` | Merged entity profiles (cross-stream scored) + local/hive insights browser |
| **Profiles** | `/profiles` | Aggregated profile metadata from every linked local |

### 13.5 LLM Role Configuration (via web UI)

Navigate to **Local Instance > LLM Roles** to configure API key, base URL, and
model for each of the 6 pipeline roles. "Save All to .env" persists the config
to `~/.ctxmtg/.env`.

### 13.6 Entity Resolution

Navigate to **Entities** to see:
- **Duplicate candidates** — entities with name variants across interactions
- **Near-duplicate pairs** — entities differing only by case or whitespace
- **Merge / Delete / Dismiss** — merge duplicates, delete garbage outright
  (cascades to related facts, insights, and distiller summaries), or dismiss
  as legitimately distinct

### 13.7 Open WebUI integration

The_Borg_DB exposes an OpenAI-compatible API:

| Setting | Value |
|---------|-------|
| API Base URL | `http://127.0.0.1:8080/v1` |
| API Key | any non-empty string |
| Model | `ctxmtg` (parallel) or `ctxmtg-deep` (bidirectional) |

---

## 14. LLM Configuration

The_Borg_DB uses LLMs via OpenAI-compatible endpoints for 6 pipeline roles.
Each role can use a different provider, model, or API key — and all roles can
point at the same local server if you prefer.

### 14.1 The 6 pipeline roles

| Role | What it does | Recommended size |
|------|--------------|------------------|
| **Extraction** | Verifies and enhances spaCy NER output | Lighter (8B–12B) |
| **Query Planning** | Interprets natural language queries | Lighter (8B–12B) |
| **Farming** | Generates farming insight descriptions | Lighter (8B–12B) |
| **Fusion** | Reranks fused results by semantic relevance | Lighter (8B–12B) |
| **Retrieval** | Formulates V2S and S2V bridge queries | Heavier (20B+) |
| **Synthesis** | Generates cited answers from results | Heavier (20B+) |

The lighter-role / heavier-role split is a recommendation, not a rule. On a
GPU with headroom you can point all six at one large model; on a small
machine you can point all six at one 8B model.

> **Roadmap note:** The `retrieval` role will be split into `retrieval_v2s`
> and `retrieval_s2v` in a future release so you can size them independently
> (V2S benefits from a stronger model for SQL generation; S2V can use a
> lighter one).

### 14.2 Configuring via the web UI (recommended)

1. Start the web server: `ctxmtg serve`
2. Navigate to **Local Instance > LLM Roles**
3. Enter the API key, base URL, and model name for each role
4. Click **Save All to .env**

The config is saved to `~/.ctxmtg/.env` and loaded automatically on next
startup.

### 14.3 Configuring via environment variables

Set variables directly using the `CTXMTG_LLM__ROLE__FIELD` convention:

```bash
# Extraction role — local LM Studio
export CTXMTG_LLM__EXTRACTION__API_KEY="sk-dummy-key"
export CTXMTG_LLM__EXTRACTION__BASE_URL="http://localhost:1234/v1"
export CTXMTG_LLM__EXTRACTION__MODEL="your-light-model-id"

# Synthesis role — a heavier local or hosted model
export CTXMTG_LLM__SYNTHESIS__API_KEY="sk-your-key"
export CTXMTG_LLM__SYNTHESIS__BASE_URL="http://localhost:1234/v1"
export CTXMTG_LLM__SYNTHESIS__MODEL="your-heavy-model-id"
```

See `.env.example` in the repo root for the complete layout.

### 14.4 Supported providers

Any OpenAI-compatible API works:

| Provider | Base URL | Notes |
|----------|----------|-------|
| LM Studio (local) | `http://localhost:1234/v1` | Any model you've loaded; API key can be a dummy string |
| Ollama (local) | `http://localhost:11434/v1` | Any local model; API key can be a dummy string |
| vLLM (local) | `http://localhost:8000/v1` | Self-hosted models |
| OpenAI | `https://api.openai.com/v1` | GPT-4o, GPT-4o-mini |
| Anthropic (via proxy) | `https://api.anthropic.com/v1` | Claude 3.5, Claude 3 |
| Google (via proxy) | `https://generativelanguage.googleapis.com/v1beta` | Gemini |
| Azure OpenAI | `https://<resource>.openai.azure.com/openai/deployments/<model>/` | Azure-hosted models |

### 14.5 Monitoring LLM usage

Navigate to **LLM Usage** (`/usage`) to see:

- **30-day totals** — calls, tokens, and errors
- **Per-model breakdown** — calls, prompt/completion tokens, latency, errors
- **Per-role breakdown** — which pipeline roles consume the most tokens
- **Daily history** — day-by-day usage trends

### 14.6 What improves with an LLM

| Area | Without LLM | With LLM |
|------|-------------|----------|
| Entity extraction | spaCy NER + regex only | + LLM verification and enrichment |
| Fact extraction | Rule-based SVO triples | LLM-corrected triples, fewer false positives |
| Query routing | Regex intent classification | Neural intent + named-entity awareness |
| SQL generation | Template matching | LLM text-to-SQL for complex queries |
| Result synthesis | Raw result list | Natural language answer with citations |
| V2S / S2V modes | Return unranked results | Fully operational |
| Farming insights | Structured output only | Natural language descriptions |

---

## 15. Running Without an LLM

The_Borg_DB is fully functional without any LLM. This is the recommended
starting point when trying out the system.

**Everything works:** Ingestion, extraction (spaCy NER + regex + filters),
embedding, parallel query mode, all 18 farming stages, hive push/pull,
autocomplete, web UI.

**LLM-enhanced features degrade gracefully:** V2S/S2V modes return unranked
results, `deep` mode falls back to `parallel`, extraction uses spaCy output
without LLM verification, farming insights lose their natural-language
descriptions but keep all numerical signals.

---

## 16. Upgrading

### 16.1 Standard upgrade

```bash
cd The_Borg_DB
git pull
source .venv/bin/activate
pip install -e ".[web]"
```

### 16.2 Schema migrations

The_Borg_DB upgrades your database schema automatically on startup. The
current schema version is 5.

---

## 17. Troubleshooting

### Empty query results

1. **Nothing ingested** — run `ctxmtg health` to check record counts
2. **Query too specific** — try broader terms ("OAuth" instead of "OAuth2 migration timeline concerns")
3. **Wrong profile** — use the same profile for ingest and query
4. **Farming hasn't run** — run `ctxmtg farm run` to build intelligence
5. **No embedding_fn** — semantic search requires the ONNX embedder; see next item

### "Warning: Embedding provider unavailable"

```bash
pip install onnxruntime transformers    # CPU
# or
pip install onnxruntime-gpu transformers  # Jetson / DGX
```

Both `onnxruntime` and `transformers` are required — the latter loads the
HuggingFace tokenizer.

### "Warning: Extraction pipeline unavailable"

```bash
python -m spacy download en_core_web_sm
```

### "Database is locked"

Two processes tried to write simultaneously — most commonly `ctxmtg watch` and
`ctxmtg serve` running against the same DB. Stop one, or increase the busy
timeout:

```bash
export CTXMTG_BUSY_TIMEOUT=30000
```

Then check for other processes:

```bash
ps aux | grep ctxmtg
lsof ~/.ctxmtg/knowledge.db
```

### Farming produces no insights

Farming requires at least 50 interactions. Check with `ctxmtg health`. If you
have more than 50 and still see zero insights, check the server logs for
`farm_llm_load_failed` — the farming role may be misconfigured.

### "Event loop is closed" during `--mode deep`

This was a known `aiosqlite` multiple-event-loop issue in earlier versions.
Upgrade to the current release — all CLI async operations now share a single
event loop.

### Hive push writes a file but the hive doesn't see it

1. Confirm the hive's link for this local points at the correct outbox path.
2. Check that no `.tmp` file is stuck in the outbox (indicates a partial write).
3. Inspect `outbox_progress` on the local — if high-water marks are too high,
   the local thinks there is nothing new to send.

---

## Appendix A: Example Queries

These examples show the types of questions you can ask The_Borg_DB and what
kinds of answers to expect. The exact results depend on what content you have
ingested.

### A.1 Finding facts about people

```
ctxmtg query "What did Alice propose?"
ctxmtg query "Who is Bob?"
ctxmtg query "What is Alice's role?"
ctxmtg query "Who worked on the OAuth2 migration?"
ctxmtg query "What has Charlie been involved in this month?"
```

**What happens:** The query engine searches for entities matching "Alice" or
"Bob" in the SQL store and runs a semantic search in the vector store.
Results are fused and ranked by relevance.

### A.2 Finding decisions and outcomes

```
ctxmtg query "What decisions were made about authentication?"
ctxmtg query "Why did we choose Redis over Memcached?"
ctxmtg query "What was the outcome of the security review?"
ctxmtg query "Who approved the budget for Q3?"
ctxmtg query "What are the open action items from last week?"
```

**What happens:** Queries about decisions search for DECISION entity types and
facts with predicates like `decided`, `approved`, `chose`. `--mode deep`
improves these significantly when an LLM is configured.

### A.3 Tracking projects and timelines

```
ctxmtg query "What is the status of the database migration?"
ctxmtg query "When is the OAuth2 deadline?"
ctxmtg query "How has the project timeline changed?"
ctxmtg query "What milestones are coming up this month?"
ctxmtg query "What blocked the release last sprint?"
```

**What happens:** Temporal queries search for DEADLINE entities and
time-annotated facts. `--mode s2v` is effective here: SQL finds the
structured dates, then vector search finds the contextual reasons.

### A.4 Understanding relationships

```
ctxmtg query "Who works with Alice most often?"
ctxmtg query "What teams are involved in the migration?"
ctxmtg query "Which projects are related to security?"
ctxmtg query "What topics does the backend team discuss most?"
ctxmtg query "Who are the key stakeholders for the Q3 release?"
```

**What happens:** Relationship queries benefit heavily from farming. Entity
analytics builds co-occurrence matrices, graph analysis computes relationship
strength, and the distiller produces entity summaries. Run `ctxmtg farm run`
before these queries for the best results.

### A.5 Discovering trends and patterns

```
ctxmtg query "What topics are trending this month?"
ctxmtg query "Has discussion about security increased recently?"
ctxmtg query "What new entities appeared this week?"
ctxmtg query "Are there any recurring issues?"
ctxmtg query "What patterns have emerged across team meetings?"
```

**What happens:** Trend queries pull from farming insights. These work best
after multiple cycles have run over a growing dataset.

### A.6 Cross-referencing across sources

```
ctxmtg query "What did Alice say in emails vs meetings about the migration?"
ctxmtg query "Are there contradictions between the proposal and the meeting notes?"
ctxmtg query "What information from the PDF report relates to sprint discussions?"
ctxmtg query "Combine what we know about Redis from all sources"
```

**What happens:** These queries benefit from `--mode deep`, which runs both
informed retrieval paths and synthesizes a combined answer. If you've linked
multiple locals via the hive, cross-stream discovery broadens the context
further.

### A.7 Exploring your knowledge base

```
ctxmtg suggest --browse                         # Top entities by distilled relevance
ctxmtg suggest "What did"                       # Query completions
ctxmtg query "What do we know about OAuth2?"    # Open-ended exploration
ctxmtg query "Summarize recent activity"        # High-level overview
ctxmtg query "What is the most discussed topic this month?"
```

**What happens:** Browse mode uses distiller summaries from farming.
Open-ended queries work best with `--mode v2s`.

### A.8 Tips for effective queries

| Tip | Example |
|-----|---------|
| Start broad, then narrow | "OAuth" → "OAuth2 migration timeline" |
| Name entities explicitly | "What did Alice say?" (not "What did she say?") |
| Use `--mode v2s` for exploration | "Tell me about the new project" |
| Use `--mode s2v` for specific facts | "When is the deadline for X?" |
| Use `--mode deep` for reasoning | "Why did we make that decision?" |
| Run farming first for trend queries | `ctxmtg farm run` before asking about patterns |
| Check `ctxmtg health` if results are empty | Make sure data is actually ingested |

---

*For bugs and feature requests: https://github.com/D-jai/The_Borg_DB/issues*

---

**Contact**

Aliud Inquisito Inc.
Dhananjay Raol
https://github.com/D-jai/The_Borg_DB

---

*Copyright (c) 2024 Aliud Inquisito Inc. Distributed under the Business Source License 1.1. See [LICENSE](../LICENSE) and [NOTICE](../NOTICE).*
