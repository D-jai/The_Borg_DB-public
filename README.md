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
writing to `~/.ctxmtg/.env` directly. See [.env.example](.env.example) for the
full variable layout.

```bash
export CTXMTG_LLM__EXTRACTION__API_KEY="your-key"
export CTXMTG_LLM__EXTRACTION__BASE_URL="http://localhost:1234/v1"
export CTXMTG_LLM__EXTRACTION__MODEL="your-model-id"
```

All features work without an LLM — the system degrades gracefully to
rule-based extraction, SQL-only queries, and mathematical result fusion.

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
