# The_Borg_DB Error Codes Guidance

**Version:** 0.7.0
**Last updated:** 2026-04-13

**Purpose:** Reference for every structured error code emitted by The_Borg_DB
(internal package name: `ctxmtg`). Each entry lists the code, a plain-English
description, the source location, and troubleshooting hints. Operators and
developers should look here first when an error code appears in a log or
exception traceback.

**Code format:** `CTXMTG-<MODULE>-<NNN>`

| Module prefix | Area |
|---------------|------|
| `STG` | Storage (SQLite, LanceDB) |
| `EXT` | Extraction (NER, facts, summarization) |
| `EMB` | Embedding (ONNX model) |
| `QRY` | Query (planning, execution, fusion, reranking) |
| `FRM` | Farming (pattern mining, scheduling) |
| `PRF` | Profile (domain profile loading/switching) |
| `CFG` | Config (settings, environment variables) |
| `ING` | Ingestion (worker, file I/O, pipeline) |
| `SYN` | Sync (multi-device CRDT/vector delta) |
| `CLI` | CLI (argument validation, server startup) |
| `HLT` | Health (monitor, metrics JSONL) |

---

## How to Read an Entry

```
Code       : CTXMTG-STG-002
Description: SQLite batch write failure
Origin     : src/ctxmtg/storage/sqlite.py — SQLiteStore.store_entities()
             (also raised by store_interaction, store_facts, store_insight)
Hints      : 1. Check available disk space: `df -h ~/.ctxmtg`
             2. Look for a second ctxmtg process holding a write lock: `lsof ~/.ctxmtg/knowledge.db`
             3. Ensure PRAGMA busy_timeout is set (default 5000 ms); if contention is
                high, increase via CTXMTG_BUSY_TIMEOUT env var.
             4. If the database file is corrupt, restore from the most recent backup in
                ~/.ctxmtg/backups/.
```

---

## CTXMTG-STG — Storage Errors

### CTXMTG-STG-001
**Description:** SQLite connection failure — cannot open or create the database file.
**Origin:** `src/ctxmtg/storage/sqlite.py` — `SQLiteStore.__init__()` / `initialize()`
**Hints:**
1. Verify the path in `CTXMTG_DB_PATH` (default `~/.ctxmtg/knowledge.db`) exists and is writable.
2. Check parent directory permissions: `ls -la ~/.ctxmtg/`.
3. On first run, ctxmtg must be able to create the file; ensure the parent directory exists.
4. If running in Docker, confirm the `/data` volume is mounted and writable.

### CTXMTG-STG-002
**Description:** SQLite batch write failure — INSERT or UPDATE transaction failed (disk full, locked, or constraint violation beyond IntegrityError).
**Origin:** `src/ctxmtg/storage/sqlite.py` — `SQLiteStore.store_entities()`, `store_facts()`, `store_interaction()`
**Hints:**
1. Check available disk space: `df -h ~/.ctxmtg`.
2. Look for another ctxmtg process holding a write lock: `lsof ~/.ctxmtg/knowledge.db`.
3. `BEGIN IMMEDIATE` can time out if a long-running read is active; increase `CTXMTG_BUSY_TIMEOUT` (default 5000 ms).
4. Review the full exception chain (`__cause__`) for the underlying sqlite3 error code.

### CTXMTG-STG-003
**Description:** SQLite read/query failure — SELECT or FTS query execution failed.
**Origin:** `src/ctxmtg/storage/sqlite.py` — `SQLiteStore.execute_sql()`, `get_entities()`, `get_facts()`
**Hints:**
1. If the error mentions "no such table", the schema migration has not run; call `initialize()` first.
2. Check for SQL syntax errors in any custom query passed to `execute_sql()`.
3. FTS5 errors often occur after a schema mismatch; re-running `initialize()` rebuilds the FTS index.

### CTXMTG-STG-004
**Description:** SQLite schema migration failure — `PRAGMA user_version` mismatch or DDL error.
**Origin:** `src/ctxmtg/storage/schema.py` — `migrate()`
**Hints:**
1. Check the current schema version: `sqlite3 ~/.ctxmtg/knowledge.db "PRAGMA user_version;"`.
2. Do not manually edit the database schema; always use `ctxmtg` migration commands.
3. If migrating from a very old version, apply migrations sequentially; skipping versions is not supported.
4. Back up the database before running migrations in production.

### CTXMTG-STG-005
**Description:** SQLite transaction lock timeout — `BUSY_TIMEOUT` exceeded (default 5000 ms).
**Origin:** `src/ctxmtg/storage/sqlite.py` — any write method using `BEGIN IMMEDIATE`
**Hints:**
1. Check for stuck ctxmtg processes: `ps aux | grep ctxmtg`.
2. Increase timeout via env var `CTXMTG_BUSY_TIMEOUT=10000` (milliseconds).
3. Farming and ingestion workers should not overlap; check the scheduler configuration.

### CTXMTG-STG-006
**Description:** LanceDB initialization failure — cannot open or create the vector store directory.
**Origin:** `src/ctxmtg/storage/lancedb_store.py` — `LanceDBStore.initialize()`
**Hints:**
1. Verify `CTXMTG_VECTOR_PATH` (default `~/.ctxmtg/vectors`) is writable.
2. LanceDB requires a directory, not a file; delete any stale file at that path if present.
3. Check available disk space; LanceDB pre-allocates index files on first insert.

### CTXMTG-STG-007
**Description:** LanceDB insert failure — vector dimension mismatch or corrupt data.
**Origin:** `src/ctxmtg/storage/lancedb_store.py` — `LanceDBStore.insert()`
**Hints:**
1. All vectors must have the same dimension. Check `EmbeddingProvider.get_dimensions()` matches the table schema.
2. If you changed the embedding model, the existing table has the old dimension; delete the vector store and re-embed.
3. Verify `ids`, `vectors`, and `metadata` lists are the same length.

### CTXMTG-STG-008
**Description:** LanceDB search failure — ANN index error or query vector dimension mismatch.
**Origin:** `src/ctxmtg/storage/lancedb_store.py` — `LanceDBStore.search()`
**Hints:**
1. Confirm the query vector dimension matches the stored vectors.
2. If the index is corrupt, delete `~/.ctxmtg/vectors` and re-embed all content.
3. Check that the table has at least one record before searching; an empty table may raise on some LanceDB versions.

### CTXMTG-STG-009
**Description:** LanceDB delete failure — record ID not found or index corrupt.
**Origin:** `src/ctxmtg/storage/lancedb_store.py` — `LanceDBStore.delete()`
**Hints:**
1. Deleting a non-existent ID is a no-op in most LanceDB versions; if this error fires, the index may be corrupt.
2. Re-initialize the vector store from scratch if the index cannot be repaired.

### CTXMTG-STG-010
**Description:** Embedding metadata write failure — SQL write for the embedding link record failed.
**Origin:** `src/ctxmtg/storage/sqlite.py` — `SQLiteStore.store_embedding_metadata()`
**Hints:**
1. This is a linking table between SQL and vector stores; if it fails, queries may miss results.
2. Follow the same steps as CTXMTG-STG-002 (disk, lock, timeout).
3. Verify the `source_id` in the metadata references a valid record in the source table.

### CTXMTG-STG-011
**Description:** FTS5 index update failure — FTS5 virtual table write failed.
**Origin:** `src/ctxmtg/storage/sqlite.py` — triggered during `store_interaction()`
**Hints:**
1. FTS5 tables update automatically via triggers; check that the trigger DDL in `schema.py` is intact.
2. If the error mentions "fts corrupt", rebuild: `INSERT INTO interactions_fts(interactions_fts) VALUES('rebuild');`.

### CTXMTG-STG-012
**Description:** Foreign key constraint violation — a referenced record does not exist in the parent table.
**Origin:** `src/ctxmtg/storage/sqlite.py` — any INSERT that references another table
**Hints:**
1. Ensure `store_interaction()` is called before `store_entities()` or `store_facts()` for the same interaction.
2. Check that `PRAGMA foreign_keys = ON` is set; without it, violations are silently ignored.

### CTXMTG-STG-013
**Description:** Farming insight storage failure — `FarmingInsight` write to SQL failed.
**Origin:** `src/ctxmtg/storage/sqlite.py` — `SQLiteStore.store_insight()`
**Hints:**
1. Follow the same steps as CTXMTG-STG-002.
2. Farming can continue without storing insights (degraded mode), but insights will be lost.

### CTXMTG-STG-014
**Description:** Interaction storage failure — duplicate interaction ID or constraint violation.
**Origin:** `src/ctxmtg/storage/sqlite.py` — `SQLiteStore.store_interaction()`
**Hints:**
1. The system uses UUIDv5 deterministic IDs; ingesting the same content twice will produce the same ID.
2. Use `INSERT OR IGNORE` semantics to skip duplicates silently, or check for existence first.

### CTXMTG-STG-015
**Description:** Entity batch write failure — entity list INSERT failed.
**Origin:** `src/ctxmtg/storage/sqlite.py` — `SQLiteStore.store_entities()`
**Hints:**
1. Follow the same steps as CTXMTG-STG-002.
2. Check that `entity_type` values match the CHECK constraint in the schema (must be one of the `EntityType` enum values).

---

## CTXMTG-EXT — Extraction Errors

### CTXMTG-EXT-001
**Description:** spaCy model not found — `en_core_web_sm` (or md/lg) is not downloaded.
**Origin:** `src/ctxmtg/extraction/spacy_ner.py` — `SpacyNERProvider.__init__()`
**Hints:**
1. Run: `python -m spacy download en_core_web_sm`
2. In Docker builds, add the download step to the Dockerfile (see `docker/Dockerfile.edge`).
3. If using `en_core_web_md` or `lg`, ensure the correct model name is set in config.
4. Use `scripts/download_models.py` to download all required models in one step.

### CTXMTG-EXT-002
**Description:** spaCy inference failure — unexpected input format or internal spaCy error.
**Origin:** `src/ctxmtg/extraction/spacy_ner.py` — `SpacyNERProvider.extract_entities()`
**Hints:**
1. Check that the input text is a non-empty Python `str`, not bytes.
2. Very long texts (>1 MB) can exhaust spaCy's internal buffers; chunk the text first.
3. Enable spaCy debug logging: `import spacy; spacy.prefer_gpu()` and inspect the pipeline.

### CTXMTG-EXT-003
**Description:** Regex pattern compilation failure — invalid regex in domain profile `custom_patterns`.
**Origin:** `src/ctxmtg/extraction/regex_extractor.py` — `RegexExtractor.__init__()`
**Hints:**
1. Validate every pattern in your `.yaml` profile with `re.compile(pattern)` before deploying.
2. Common mistakes: unescaped backslashes in YAML (use `\\d` not `\d`), unclosed groups.
3. The error message includes the offending pattern string; fix it in the profile YAML.

### CTXMTG-EXT-004
**Description:** Dependency parse failure — spaCy parse tree is incomplete or malformed.
**Origin:** `src/ctxmtg/extraction/fact_extractor.py` — `SimpleFactExtractor.extract_facts()`
**Hints:**
1. This is non-fatal; the fact extractor will return an empty list rather than crashing.
2. Ensure `en_core_web_sm` (or a larger model) is loaded — the `tagger` and `parser` components must be enabled.
3. Single-word or very short inputs produce no parse tree; this is expected.

### CTXMTG-EXT-005
**Description:** Summarization failure — text is too short, empty, or is a single sentence.
**Origin:** `src/ctxmtg/extraction/summarizer.py` — `TextRankSummarizer.summarize()`
**Hints:**
1. TextRank requires at least 3 sentences to produce a meaningful summary.
2. For short inputs, fall back to returning the full text as the summary.
3. Check that `spacy` sentence segmentation is working: `[s.text for s in doc.sents]`.

### CTXMTG-EXT-006
**Description:** Entity deduplication failure — ID generation collision or internal logic error.
**Origin:** `src/ctxmtg/extraction/pipeline.py` — `BasicExtractionPipeline._deduplicate_entities()`
**Hints:**
1. This should be extremely rare; UUIDv5 collisions are astronomically unlikely.
2. If it fires consistently, check the `generate_id()` function for a logic bug (`id_gen.py`).

### CTXMTG-EXT-007
**Description:** Chunking failure — input text is empty or `chunk_size` is zero or negative.
**Origin:** `src/ctxmtg/embedding/chunker.py` — `TextChunker.chunk()`
**Hints:**
1. Validate input before calling the chunker; reject empty strings at the ingestion boundary.
2. Ensure `embedding.chunk_size` in the domain profile is a positive integer (default 256).

### CTXMTG-EXT-008
**Description:** Extraction pipeline stage failure — one stage failed and the pipeline is stopping.
**Origin:** `src/ctxmtg/extraction/pipeline.py` — `BasicExtractionPipeline.process()`
**Hints:**
1. Check the `__cause__` on this exception; it will be one of EXT-001 through EXT-007.
2. The pipeline is designed to be partially fault-tolerant; NER failure does not prevent chunking.
3. If a specific stage consistently fails, disable it in the domain profile.

### CTXMTG-EXT-009
**Description:** Fact extraction produced no usable subject-predicate-object triples.
**Origin:** `src/ctxmtg/extraction/fact_extractor.py` — `SimpleFactExtractor.extract_facts()`
**Hints:**
1. This is a warning, not a hard error; the pipeline continues with an empty facts list.
2. Template-based fact extraction requires sentences with at least 2 detected entities.
3. Upgrade to LLM-assisted extraction (Phase 2) for better fact coverage.

### CTXMTG-EXT-010
**Description:** Entity type mapping failure — a spaCy label is not in the `EntityType` enum.
**Origin:** `src/ctxmtg/extraction/spacy_ner.py` — `SpacyNERProvider._map_label()`
**Hints:**
1. Unmapped labels are silently assigned `EntityType.OTHER`; this error fires only if the mapping dict is broken.
2. Check the label-to-EntityType mapping dictionary in `spacy_ner.py` for the offending label.

---

## CTXMTG-EMB — Embedding Errors

### CTXMTG-EMB-001
**Description:** ONNX model file not found — the `.onnx` file is missing from the local cache.
**Origin:** `src/ctxmtg/embedding/onnx_embedder.py` — `ONNXEmbeddingProvider.__init__()`
**Hints:**
1. Run `python scripts/download_models.py` to download the default model (`all-MiniLM-L6-v2`).
2. Check the cache directory: `~/.ctxmtg/models/` (or `CTXMTG_MODEL_PATH`).
3. If using a custom model, confirm the `.onnx` path in the domain profile is correct.

### CTXMTG-EMB-002
**Description:** ONNX session creation failure — model file is corrupt or incompatible with the installed ONNX Runtime version.
**Origin:** `src/ctxmtg/embedding/onnx_embedder.py` — `ONNXEmbeddingProvider.__init__()`
**Hints:**
1. Delete the cached model and re-download: `rm -rf ~/.ctxmtg/models/ && python scripts/download_models.py`.
2. Check `onnxruntime` version compatibility with the model's opset: `python -c "import onnxruntime; print(onnxruntime.__version__)"`.
3. INT8 quantized models require ONNX Runtime >= 1.16 with the appropriate execution provider.

### CTXMTG-EMB-003
**Description:** ONNX inference failure — input shape mismatch or internal runtime error.
**Origin:** `src/ctxmtg/embedding/onnx_embedder.py` — `ONNXEmbeddingProvider.embed()`
**Hints:**
1. Check that input token sequences do not exceed `max_seq_len` (typically 512 for MiniLM).
2. Reduce `embedding.chunk_size` in the domain profile if texts are being truncated unexpectedly.
3. Inspect the ONNX session's expected input names: `session.get_inputs()`.

### CTXMTG-EMB-004
**Description:** Tokenization failure — text exceeds the model's maximum sequence length.
**Origin:** `src/ctxmtg/embedding/onnx_embedder.py` — `ONNXEmbeddingProvider._tokenize()`
**Hints:**
1. The chunker should prevent this; check that `embedding.chunk_size` in the profile is ≤ 400 tokens.
2. For unusually long words or no-whitespace text (e.g., Base64 blobs), the tokenizer may produce more tokens than characters.
3. Enable truncation in the tokenizer as a safety fallback.

### CTXMTG-EMB-005
**Description:** Batch embedding failure — one or more texts in the batch caused an inference error.
**Origin:** `src/ctxmtg/embedding/onnx_embedder.py` — `ONNXEmbeddingProvider.embed()`
**Hints:**
1. Try reducing batch size (`CTXMTG_EMBEDDING_BATCH_SIZE`, default 32) to isolate the failing text.
2. Check for empty strings in the batch; filter them before calling `embed()`.
3. On memory-constrained devices, a batch of 32 may exhaust RAM; reduce to 8 or 16.

### CTXMTG-EMB-006
**Description:** Model download failure — network error or HuggingFace Hub is unavailable.
**Origin:** `scripts/download_models.py` — `download_model()`
**Hints:**
1. Check internet connectivity: `curl -I https://huggingface.co`.
2. Set `HF_HUB_OFFLINE=1` to use only cached models.
3. Download models manually and place them in `~/.ctxmtg/models/<model_name>/`.
4. If behind a proxy, set `HTTPS_PROXY` before running the download script.

### CTXMTG-EMB-007
**Description:** Dimension mismatch — the model's output dimension does not match the value stored in the vector table schema.
**Origin:** `src/ctxmtg/embedding/onnx_embedder.py` — `ONNXEmbeddingProvider.get_dimensions()` check in `LanceDBStore.insert()`
**Hints:**
1. This occurs when the embedding model is changed after vectors were already stored.
2. Delete `~/.ctxmtg/vectors` and re-run ingestion with the new model.
3. Always store the model name and version in `EmbeddingMetadata` so the mismatch can be detected early.

### CTXMTG-EMB-008
**Description:** Requested execution provider is unavailable (CUDA, CoreML, etc.).
**Origin:** `src/ctxmtg/embedding/onnx_embedder.py` — `ONNXEmbeddingProvider._get_provider()`
**Hints:**
1. The system falls back to CPU automatically; this is a warning, not a fatal error.
2. To use CUDA: install `onnxruntime-gpu` and ensure CUDA drivers are present.
3. On Apple Silicon, CoreML provider is available via `onnxruntime-silicon`.

---

## CTXMTG-QRY — Query Errors

### CTXMTG-QRY-001
**Description:** Intent classification produced no confident match; defaulting to `SEMANTIC`.
**Origin:** `src/ctxmtg/query/intent.py` — `RuleBasedIntentClassifier.plan()`
**Hints:**
1. This is a soft warning; the query still executes against the vector store.
2. Add domain-specific regex patterns to the intent classifier for common query forms in your domain.
3. Queries that are very short (1–2 words) often trigger this; encourage users to ask full questions.

### CTXMTG-QRY-002
**Description:** SQL template not found — no SQL template matches the detected intent.
**Origin:** `src/ctxmtg/query/planner.py` — `TemplateQueryPlanner.plan()`
**Hints:**
1. Check that the `intent` value in the `QueryPlan` is a member of the `QueryIntent` enum.
2. Extend `planner.py` with a new template for any intent type that consistently triggers this.
3. The planner falls back to FTS search when no template matches; results may be less precise.

### CTXMTG-QRY-003
**Description:** SQL query execution failure — syntax error, missing table, or query timeout.
**Origin:** `src/ctxmtg/query/executor.py` — `QueryExecutor._execute_sql()`
**Hints:**
1. Log and inspect the generated SQL in the `QueryPlan.sql_query` field before execution.
2. If the error is "no such table", call `SQLiteStore.initialize()` to create missing tables.
3. Parameterized queries are used; if this fires with a syntax error, check the template in `planner.py`.
4. The executor falls back to vector-only results when SQL fails (see CTXMTG-QRY-008 if both fail).

### CTXMTG-QRY-004
**Description:** Vector query failure — embedding the query text failed, or the ANN search failed.
**Origin:** `src/ctxmtg/query/executor.py` — `QueryExecutor._execute_vector()`
**Hints:**
1. Check the embedding error first (CTXMTG-EMB-*) via the `__cause__` chain.
2. If the vector store is empty, search returns an empty list, not an error; check `VectorStore.count()`.
3. Dimension mismatch between query vector and stored vectors raises here; see CTXMTG-EMB-007.

### CTXMTG-QRY-005
**Description:** RRF fusion failure — both result sets are empty or structurally incompatible.
**Origin:** `src/ctxmtg/query/fusion.py` — `RRFFuser.fuse()`
**Hints:**
1. Empty result sets from both stores are valid; the fuser returns an empty list, not an error.
2. This error fires only if the input lists have an unexpected type; check that `SearchResult` objects are passed.
3. Review the executor for upstream errors (QRY-003, QRY-004) that may have prevented results from being fetched.

### CTXMTG-QRY-006
**Description:** TF-IDF reranking failure — vectorizer error or empty document set.
**Origin:** `src/ctxmtg/query/reranker.py` — `TFIDFReranker.rerank()`
**Hints:**
1. The reranker is optional; if it fails, the system returns fused results without reranking.
2. An empty results list passed to the reranker is a no-op; check that fusion (QRY-005) returned results.
3. `scikit-learn` is an optional dependency; if unavailable, the fallback manual TF-IDF is used.

### CTXMTG-QRY-007
**Description:** Query log write failure — feedback loop JSONL write failed.
**Origin:** `src/ctxmtg/query/executor.py` — query logging step
**Hints:**
1. The query still returns results; this is a non-fatal background write.
2. Check write permissions on `~/.ctxmtg/query_log.jsonl`.
3. Disk full is the most common cause; see CTXMTG-STG-002 hints.

### CTXMTG-QRY-008
**Description:** Total query failure — both SQL and vector stores failed; no results returned.
**Origin:** `src/ctxmtg/query/executor.py` — `QueryExecutor.execute()`
**Hints:**
1. Check upstream storage errors (CTXMTG-STG-* and CTXMTG-EMB-*) in the log for the same request.
2. Verify that `SQLiteStore.initialize()` and `LanceDBStore.initialize()` completed successfully at startup.
3. This is the most severe query error; investigate storage health with `ctxmtg health`.

### CTXMTG-QRY-009
**Description:** Query parameter validation failure — invalid `limit`, `offset`, or filter values.
**Origin:** `src/ctxmtg/query/executor.py` — parameter validation step
**Hints:**
1. `limit` must be a positive integer ≤ 1000 (configurable).
2. `offset` must be ≥ 0.
3. Filter keys must match column names in the schema; check `research/round-2/03-unified-schema-design.md`.

---

## CTXMTG-FRM — Farming Errors

### CTXMTG-FRM-001
**Description:** General farming stage failure — a stage raised an unhandled exception.
**Origin:** `src/ctxmtg/farming/pipeline.py` — `FarmingPipeline.run()`
**Hints:**
1. Check `__cause__` for one of FRM-002 through FRM-010.
2. Farming is designed to be fault-tolerant; a failed stage is logged and skipped, not re-run.
3. Check the checkpoint file in `~/.ctxmtg/farming_checkpoint.json` to see which stage failed.

### CTXMTG-FRM-002
**Description:** Entity analytics failure — frequency or co-occurrence computation error.
**Origin:** `src/ctxmtg/farming/entity_analytics.py` — `EntityAnalyticsStage.run()`
**Hints:**
1. Usually caused by an empty entities table; ensure at least some interactions have been ingested.
2. The co-occurrence matrix requires at least 2 entities per interaction; check entity extraction quality.

### CTXMTG-FRM-003
**Description:** Trend detection failure — insufficient data points (below `min_samples`).
**Origin:** `src/ctxmtg/farming/trend_detection.py` — `TrendDetectionStage.run()`
**Hints:**
1. Trend detection requires interactions spanning at least 7 days (default `lookback_window_days`).
2. Increase the lookback window in the domain profile if you have older data.
3. This is a soft failure; no trends are reported but farming continues.

### CTXMTG-FRM-004
**Description:** Clustering failure — K-Means or HDBSCAN convergence error.
**Origin:** `src/ctxmtg/farming/clustering.py` — `ClusteringStage.run()`
**Hints:**
1. K-Means requires at least `k` vectors; if the vector store has fewer than `cluster_min_size`, skip.
2. HDBSCAN requires `min_cluster_size ≥ 2`; check the profile's `cluster_min_size`.
3. Reduce the number of clusters if the dataset is small.

### CTXMTG-FRM-005
**Description:** Topic modeling failure — LDA training error or empty corpus.
**Origin:** `src/ctxmtg/farming/topic_modeling.py` — `TopicModelingStage.run()`
**Hints:**
1. LDA requires a non-empty corpus of tokenized documents.
2. If interactions are very short (< 20 words), LDA produces poor topics; this is expected.
3. Consider switching to BERTopic (requires more RAM) for short-text corpora.

### CTXMTG-FRM-006
**Description:** Graph analysis failure — PageRank computation error or disconnected graph.
**Origin:** `src/ctxmtg/farming/graph_analysis.py` — `GraphAnalysisStage.run()`
**Hints:**
1. PageRank requires a connected graph; if entity co-occurrence is sparse, the graph may be empty.
2. Lower `cluster_min_size` in the domain profile to include more edges.
3. On very small datasets (< 10 interactions), graph analysis may produce trivial results.

### CTXMTG-FRM-007
**Description:** Farming insight storage failure — writing a `FarmingInsight` to SQL failed.
**Origin:** `src/ctxmtg/farming/insight_generator.py` — `InsightGeneratorStage.run()`
**Hints:**
1. See CTXMTG-STG-013 for storage-level debugging.
2. Farming can still run; insights for this cycle will be lost but the pipeline continues.

### CTXMTG-FRM-008
**Description:** Farming scheduler failure — idle detection or timer initialization error.
**Origin:** `src/ctxmtg/farming/scheduler.py` — `FarmingScheduler.start()`
**Hints:**
1. Check that the process has permission to read `/proc/stat` (Linux) or `psutil` equivalents.
2. The scheduler defaults to a time-based fallback if idle detection fails.
3. Force a farming run manually with `ctxmtg farm --force` to bypass the scheduler.

### CTXMTG-FRM-009
**Description:** Feedback loop read failure — cannot read query quality signals from the log.
**Origin:** `src/ctxmtg/farming/feedback_loop.py` — `FeedbackLoop.read_signals()`
**Hints:**
1. Check that `~/.ctxmtg/query_log.jsonl` exists and is readable.
2. If the file is corrupt (truncated JSON lines), delete it and allow it to be rebuilt from the next queries.
3. The feedback loop is non-essential; farming still runs without quality signals.

### CTXMTG-FRM-010
**Description:** Checkpoint write failure — cannot persist farming stage progress.
**Origin:** `src/ctxmtg/farming/pipeline.py` — `FarmingPipeline._write_checkpoint()`
**Hints:**
1. Checkpoints are written to `~/.ctxmtg/farming_checkpoint.json`.
2. Without checkpoints, a farming run that crashes mid-pipeline will restart from Stage 1 next time.
3. Check write permissions on `~/.ctxmtg/`.

---

## CTXMTG-PRF — Profile Errors

### CTXMTG-PRF-001
**Description:** Profile file not found — the `.yaml` file is missing from the profiles directory.
**Origin:** `src/ctxmtg/profile/loader.py` — `ProfileLoader.load()`
**Hints:**
1. Default bundled profiles are `general`, `legal`, `personal`, `engineering`, `medical`.
2. Custom profiles must be placed in `~/.ctxmtg/profiles/<name>.yaml` or `profiles/<name>.yaml`.
3. List available profiles with `ctxmtg profile --list`.

### CTXMTG-PRF-002
**Description:** Profile YAML parse failure — malformed YAML syntax in the profile file.
**Origin:** `src/ctxmtg/profile/loader.py` — `ProfileLoader.load()`
**Hints:**
1. Validate the YAML online at https://yaml-online-parser.appspot.com/ or with `python -c "import yaml; yaml.safe_load(open('profile.yaml'))"`.
2. Common YAML mistakes: tabs instead of spaces, unquoted colons, incorrect indentation.
3. The error message includes the line number of the syntax error.

### CTXMTG-PRF-003
**Description:** Profile Pydantic validation failure — a field value in the YAML is the wrong type or out of range.
**Origin:** `src/ctxmtg/profile/loader.py` — `ProfileLoader.load()` — Pydantic parsing step
**Hints:**
1. `temperature` must be a float between 0.0 and 2.0.
2. `max_tokens` must be a positive integer.
3. Read the Pydantic validation error message; it names the offending field and the expected type.

### CTXMTG-PRF-004
**Description:** Profile version incompatible — the profile's `version` field is not supported by this ctxmtg version.
**Origin:** `src/ctxmtg/profile/loader.py` — `ProfileLoader._check_version()`
**Hints:**
1. Check `ctxmtg --version` and compare with the `version` field in the profile YAML.
2. Update the profile to the current schema, or upgrade ctxmtg.

### CTXMTG-PRF-005
**Description:** Unknown entity type in profile — `entity_types` list contains an unrecognized value.
**Origin:** `src/ctxmtg/profile/loader.py` — `ProfileLoader._validate_entity_types()`
**Hints:**
1. Valid types are defined in `EntityType` enum: `person`, `org`, `project`, `topic`, `tool`, `location`, `event`, `other`.
2. Custom entity types must be added to the `EntityType` enum before being used in a profile.

### CTXMTG-PRF-006
**Description:** Profile switch failure — the target profile cannot be found or loaded.
**Origin:** `src/ctxmtg/profile/switcher.py` — `ProfileSwitcher.switch()`
**Hints:**
1. Verify the profile name with `ctxmtg profile --list`.
2. The system reverts to the previously active profile if switching fails.

---

## CTXMTG-CFG — Config Errors

### CTXMTG-CFG-001
**Description:** Config file not found — the YAML config file at the expected path does not exist.
**Origin:** `src/ctxmtg/config/settings.py` — `CtxMtgSettings` initialization
**Hints:**
1. On first run, ctxmtg creates a default config at `~/.ctxmtg/config.yaml`.
2. If a custom path is set via `CTXMTG_CONFIG_PATH`, verify it exists.
3. Copy `configs/default.yaml` to `~/.ctxmtg/config.yaml` as a starting point.

### CTXMTG-CFG-002
**Description:** Config YAML parse failure — malformed YAML in the config file.
**Origin:** `src/ctxmtg/config/settings.py` — YAML load step
**Hints:**
1. Validate with `python -c "import yaml; yaml.safe_load(open('~/.ctxmtg/config.yaml'))"`.
2. Restore from `configs/default.yaml` if the config is corrupt.

### CTXMTG-CFG-003
**Description:** Config validation failure — Pydantic `BaseSettings` validation error on a field.
**Origin:** `src/ctxmtg/config/settings.py` — `CtxMtgSettings` model validation
**Hints:**
1. The error message names the offending field and expected type.
2. Environment variables override YAML; check for conflicting `CTXMTG_*` env vars.
3. `db_path` and `vector_path` must be valid filesystem paths (tilde expansion is supported).

### CTXMTG-CFG-004
**Description:** Environment variable override failure — a `CTXMTG_*` env var has an invalid value.
**Origin:** `src/ctxmtg/config/settings.py` — Pydantic env parsing
**Hints:**
1. Boolean fields accept `true`/`false` (case-insensitive), `1`/`0`.
2. Integer fields must be pure digits; do not include units (e.g., use `5000` not `5000ms`).
3. List fields use comma-separated values: `CTXMTG_ENTITY_TYPES=person,org,topic`.

### CTXMTG-CFG-005
**Description:** Required configuration is missing — a required setting has no default value and no env var.
**Origin:** `src/ctxmtg/config/settings.py` — `CtxMtgSettings` initialization
**Hints:**
1. Check `configs/default.yaml` for all required fields and their defaults.
2. This typically means a mandatory field was added to the schema but not given a default.

---

## CTXMTG-ING — Ingestion Errors

### CTXMTG-ING-001
**Description:** Ingestion worker initialization failure — stores or extraction pipeline not ready.
**Origin:** `src/ctxmtg/ingestion/worker.py` — `IngestionWorker.__init__()`
**Hints:**
1. Ensure `SQLiteStore.initialize()` and `LanceDBStore.initialize()` are called before starting the worker.
2. Check that the extraction pipeline loaded the spaCy model (CTXMTG-EXT-001).
3. Run `ctxmtg health` to verify all subsystems are ready.

### CTXMTG-ING-002
**Description:** Input file not found or unreadable — the file path does not exist or permission is denied.
**Origin:** `src/ctxmtg/ingestion/worker.py` — `IngestionWorker.ingest_file()`
**Hints:**
1. Verify the file exists: `ls -la <path>`.
2. Check read permissions: `stat <path>`.
3. When piping text, use `ctxmtg ingest -` to read from stdin instead.

### CTXMTG-ING-003
**Description:** Input text is too short to process meaningfully (below minimum character threshold).
**Origin:** `src/ctxmtg/ingestion/worker.py` — `IngestionWorker._validate_input()`
**Hints:**
1. Default minimum is 10 characters; configurable via `CTXMTG_MIN_INGEST_CHARS`.
2. Empty strings and whitespace-only inputs are rejected here.
3. For very short inputs (action items, one-liners), use the `note` source type which has relaxed thresholds.

### CTXMTG-ING-004
**Description:** Ingestion pipeline stage failure — extraction, embedding, or storage step failed.
**Origin:** `src/ctxmtg/ingestion/worker.py` — `IngestionWorker.process()`
**Hints:**
1. Check `__cause__` for the underlying error (CTXMTG-EXT-*, CTXMTG-EMB-*, or CTXMTG-STG-*).
2. Partial failures are possible: entities may be stored but embeddings may not.
3. Re-ingest the same content after fixing the underlying error; duplicate detection prevents double-storage.

### CTXMTG-ING-005
**Description:** Duplicate interaction detected — an interaction with the same deterministic ID has already been ingested.
**Origin:** `src/ctxmtg/ingestion/worker.py` — `IngestionWorker._check_duplicate()`
**Hints:**
1. This is a warning, not a hard error; the worker skips the duplicate and logs this code.
2. If you need to re-ingest updated content, change the `source_id` or `updated_at` field to produce a new ID.
3. Use `--force-reingest` flag (Phase 2+) to override duplicate detection.

---

## CTXMTG-SYN — Sync Errors

### CTXMTG-SYN-001
**Description:** Sync connection failure — cannot reach the remote sync endpoint.
**Origin:** `src/ctxmtg/sync/crdt_sync.py` — `CRDTSync.push()` / `pull()`
**Hints:**
1. Verify network connectivity to the sync server.
2. Check `CTXMTG_SYNC_ENDPOINT` is set correctly.
3. Sync is optional and non-destructive; local data is never modified by a failed pull.

### CTXMTG-SYN-002
**Description:** CRDT conflict resolution failure — two conflicting changes cannot be automatically merged.
**Origin:** `src/ctxmtg/sync/crdt_sync.py` — `CRDTSync._resolve_conflict()`
**Hints:**
1. cr-sqlite handles most conflicts automatically via Last-Write-Wins; this fires only for schema-level conflicts.
2. Ensure all syncing devices are on the same ctxmtg version.
3. Manual resolution: pull the conflicting record from both devices and decide which version to keep.

### CTXMTG-SYN-003
**Description:** Push failure — the remote rejected the changes (auth error, quota exceeded, or version mismatch).
**Origin:** `src/ctxmtg/sync/crdt_sync.py` — `CRDTSync.push()`
**Hints:**
1. Check sync server authentication credentials.
2. If quota exceeded, archive old interactions to free space on the remote.
3. Version mismatch: upgrade both client and server to the same ctxmtg version.

### CTXMTG-SYN-004
**Description:** Pull failure — remote is unreachable or returned invalid data.
**Origin:** `src/ctxmtg/sync/crdt_sync.py` — `CRDTSync.pull()`
**Hints:**
1. Sync will retry automatically on the next scheduled interval.
2. Check sync server health independently.
3. Local data is unaffected by a failed pull.

### CTXMTG-SYN-005
**Description:** Vector delta replay failure — the delta log is corrupt or events are out of sequence.
**Origin:** `src/ctxmtg/sync/vector_delta.py` — `VectorDelta.replay()`
**Hints:**
1. Delete `~/.ctxmtg/vector_delta.log` and perform a full vector re-sync.
2. Out-of-sequence events indicate clock skew between devices; synchronize system clocks (NTP).

---

## CTXMTG-CLI — CLI Errors

### CTXMTG-CLI-001
**Description:** CLI argument validation failure — an invalid or missing required argument was passed.
**Origin:** `src/ctxmtg/cli.py` — Click argument parsing
**Hints:**
1. Run `ctxmtg --help` or `ctxmtg <command> --help` to see valid arguments.
2. Source type must be one of: `meeting`, `email`, `document`, `note`, `chat`, `other`.
3. Query mode must be `parallel` or `deep`.

### CTXMTG-CLI-002
**Description:** Server startup failure — the port is already in use or permission is denied.
**Origin:** `src/ctxmtg/cli.py` — `start` command
**Hints:**
1. Check if another ctxmtg instance is already running: `ps aux | grep ctxmtg`.
2. Change the port: `ctxmtg start --port 8081`.
3. Ports below 1024 require root on Linux; use a port ≥ 1024 for unprivileged operation.

### CTXMTG-CLI-003
**Description:** Query server connection failure — the server is not running or is unreachable.
**Origin:** `src/ctxmtg/cli.py` — `query` command
**Hints:**
1. Start the server first: `ctxmtg start`.
2. Check the configured host and port match: `CTXMTG_HOST` and `CTXMTG_PORT`.
3. The query command can also run in standalone mode (no server) for single queries.

### CTXMTG-CLI-004
**Description:** Graceful shutdown failure — background workers did not stop within the timeout.
**Origin:** `src/ctxmtg/cli.py` — signal handler / shutdown sequence
**Hints:**
1. The default shutdown timeout is 10 seconds.
2. If a farming run is in progress, it completes its current stage before shutting down.
3. Use `SIGKILL` (`kill -9 <pid>`) only as a last resort; the database will still be consistent due to WAL mode.

---

## CTXMTG-HLT — Health Errors

### CTXMTG-HLT-001
**Description:** Health monitor initialization failure — cannot start metrics collection.
**Origin:** `src/ctxmtg/health/monitor.py` — `HealthMonitor.__init__()`
**Hints:**
1. Check that `psutil` is installed: `python -c "import psutil"`.
2. The health monitor is non-essential; ctxmtg runs without it in degraded mode.
3. Verify write permissions on `~/.ctxmtg/metrics.jsonl`.

### CTXMTG-HLT-002
**Description:** Metrics write failure — the JSONL log file write failed.
**Origin:** `src/ctxmtg/health/monitor.py` — `HealthMonitor.record()`
**Hints:**
1. Check disk space and write permissions on `~/.ctxmtg/metrics.jsonl`.
2. This is non-fatal; the system continues operating without metrics.
3. Rotate or truncate the metrics file if it has grown very large.

### CTXMTG-HLT-003
**Description:** Resource check failure — a `psutil` or OS resource query failed.
**Origin:** `src/ctxmtg/health/monitor.py` — `HealthMonitor.get_status()`
**Hints:**
1. On some containerized environments, `/proc` access is restricted; this is expected.
2. The health endpoint returns partial data with a warning when one metric cannot be collected.
3. Check container capabilities if running in Docker with `--cap-drop=ALL`.
