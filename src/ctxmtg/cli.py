# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Click CLI Entry Point
=====================

This module provides the command-line interface for ctxmtg. It uses
the Click library for argument parsing, subcommand routing, and help
text generation.

Available commands:
    ctxmtg ingest <file_or_text>   -- Ingest a file or raw text
    ctxmtg ingest --dir <path>     -- Batch ingest all supported files
    ctxmtg query <question>        -- Ask a question
    ctxmtg profile --list          -- List available profiles
    ctxmtg profile <name>          -- Show a profile's configuration
    ctxmtg intake stats            -- Show Traffic Cop statistics
    ctxmtg health                  -- Show system health metrics

The CLI wires up all pipeline components (storage, extraction,
embedding, intake) and delegates to the appropriate subsystem.

Depends on:
    - click (CLI framework)
    - ctxmtg.config.settings (CtxMtgSettings)
    - ctxmtg.ingestion.worker (IngestionWorker)
    - ctxmtg.intake.rules (RuleBasedIntakeGateway)
    - ctxmtg.profile.loader (ProfileLoader)
    - ctxmtg.health.monitor (HealthMonitor)
    - ctxmtg.storage.sqlite (SQLiteStore)
    - ctxmtg.storage.lancedb_store (LanceDBStore)

Used by:
    - pyproject.toml [project.scripts] entry point
"""

from __future__ import annotations

from pathlib import Path

import click
import structlog

# ---------------------------------------------------------------
# Module-level logger for CLI operations.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.cli")


# =====================================================================
# Main CLI group
# =====================================================================
@click.group()
@click.version_option(package_name="ctxmtg")
@click.option(
    "--profile",
    default="general",
    envvar="CTXMTG_PROFILE_NAME",
    help="Domain profile to use (e.g., general, legal, personal).",
)
@click.option(
    "--db-path",
    default=None,
    envvar="CTXMTG_DB_PATH",
    help="Path to the SQLite database file.",
)
@click.option(
    "--vector-path",
    default=None,
    envvar="CTXMTG_VECTOR_PATH",
    help="Path to the vector store directory.",
)
@click.pass_context
def main(ctx: click.Context, profile: str, db_path: str | None, vector_path: str | None) -> None:
    """
    ctxmtg: Local-first knowledge system.

    A multi-agent system that extracts intelligence from your
    interactions, stores it locally, and answers hybrid queries.
    Controlled by domain profiles for any vertical (legal, medical,
    engineering, personal, etc.).
    """
    # Store options in context for subcommands to access
    ctx.ensure_object(dict)
    ctx.obj["profile_name"] = profile
    ctx.obj["db_path"] = db_path
    ctx.obj["vector_path"] = vector_path


# =====================================================================
# Ingest command
# =====================================================================
@main.command()
@click.argument("file_or_text", required=False)
@click.option(
    "--dir",
    "directory",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Batch ingest all supported files in a directory.",
)
@click.option(
    "--source-type",
    default=None,
    help="Source type override (meeting, email, document, etc.).",
)
@click.pass_context
def ingest(
    ctx: click.Context,
    file_or_text: str | None,
    directory: str | None,
    source_type: str | None,
) -> None:
    """
    Ingest a file, text, or directory into the knowledge system.

    Auto-detects file format by extension (.txt, .json, .eml, .ics, .vcf).
    Raw text strings (no file extension) are ingested as plain text.

    Examples:
        ctxmtg ingest meeting_notes.txt
        ctxmtg ingest email.eml
        ctxmtg ingest "Alice proposed migrating to OAuth2."
        ctxmtg ingest --dir ./documents/
    """
    # Must provide either a file/text or a directory
    if not file_or_text and not directory:
        click.echo("Error: Provide a file path, text string, or use --dir.", err=True)
        ctx.exit(1)
        return

    # Validate --source-type early with a helpful error message.
    if source_type is not None:
        from ctxmtg.models.interaction import SourceType

        try:
            SourceType(source_type)
        except ValueError:
            valid = ", ".join(st.value for st in SourceType)
            click.echo(
                f"Error: Unknown source_type '{source_type}'. "
                f"Valid values: {valid}",
                err=True,
            )
            ctx.exit(1)
            return

    try:
        # Initialize pipeline components
        worker = _create_worker(ctx)

        if directory:
            # Batch ingest directory
            dir_path = Path(directory)
            click.echo(f"Batch ingesting from: {dir_path}")
            all_stats = worker.ingest_directory(dir_path)

            # Print summary
            total_files = len(all_stats)
            total_ok = sum(1 for s in all_stats if s.get("status") != "error")
            click.echo(f"\nProcessed {total_files} files ({total_ok} successful)")

            for stat in all_stats:
                status = stat.get("status", "ok")
                f = stat.get("file", "unknown")
                if status == "error":
                    click.echo(f"  ✗ {f}: {stat.get('error', 'unknown error')}")
                else:
                    entities = stat.get("entities_stored", 0)
                    facts = stat.get("facts_stored", 0)
                    click.echo(f"  ✓ {f}: {entities} entities, {facts} facts")

        elif file_or_text:
            file_path = Path(file_or_text)

            if file_path.exists() and file_path.is_file():
                # It's a file -- ingest via loader
                click.echo(f"Ingesting file: {file_path}")
                stats = worker.ingest_file(file_path)
            else:
                # It's raw text -- ingest as text
                click.echo("Ingesting text input...")
                stats = worker.ingest_text(file_or_text)

            # Print result
            entities = stats.get("entities_stored", 0)
            facts = stats.get("facts_stored", 0)
            embeddings = stats.get("embeddings_stored", 0)
            duration = stats.get("duration_ms", 0)
            click.echo(
                f"Done: {entities} entities, {facts} facts, "
                f"{embeddings} embeddings ({duration}ms)"
            )

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


# =====================================================================
# Query command
# =====================================================================
@main.command()
@click.argument("question")
@click.option(
    "--mode",
    default="parallel",
    type=click.Choice(["parallel", "v2s", "s2v", "deep"]),
    help=(
        "Retrieval mode: parallel (both stores independently), "
        "v2s (vector→SQL informed), s2v (SQL→vector informed), "
        "deep (bidirectional, both informed paths)."
    ),
)
@click.option("--top-k", default=10, help="Number of results to return.")
@click.pass_context
def query(ctx: click.Context, question: str, mode: str, top_k: int) -> None:
    """
    Ask a question and get results from the knowledge system.

    Uses hybrid query (SQL + vector search) to find relevant information.
    Retrieval modes:
      parallel  -- Both stores queried independently (default, no LLM needed)
      v2s       -- Vector→SQL: semantic discovery first, then precise SQL
      s2v       -- SQL→Vector: structured facts first, then contextual depth
      deep      -- Bidirectional: both informed paths, maximum thoroughness

    Examples:
        ctxmtg query "What did Alice propose?"
        ctxmtg query "How many meetings this week?" --mode deep
        ctxmtg query "Tell me about OAuth2" --mode v2s
        ctxmtg query "Why did we choose Redis?" --mode s2v
    """
    try:
        import asyncio

        from ctxmtg.config.settings import CtxMtgSettings
        from ctxmtg.models.query import RetrievalMode
        from ctxmtg.profile.loader import ProfileLoader
        from ctxmtg.query.executor import QueryExecutor
        from ctxmtg.query.fusion import RRFFuser
        from ctxmtg.query.interpreter import RuleBasedQueryInterpreter
        from ctxmtg.query.planner import TemplateQueryPlanner
        from ctxmtg.query.quality_logger import QueryQualityLogger
        from ctxmtg.query.reranker import TFIDFReranker

        settings = CtxMtgSettings()

        # Override settings from CLI options
        db_path = ctx.obj.get("db_path") or settings.db_path
        vector_path = ctx.obj.get("vector_path") or settings.vector_path
        profile_name = ctx.obj.get("profile_name", "general")

        # Load profile
        profile = ProfileLoader.load(profile_name)

        # Initialize stores
        sql_store, vector_store = _init_stores(db_path, vector_path)

        # Map CLI mode strings to RetrievalMode enum values.
        mode_map = {
            "parallel": RetrievalMode.PARALLEL,
            "v2s": RetrievalMode.VECTOR_TO_SQL,
            "s2v": RetrievalMode.SQL_TO_VECTOR,
            "deep": RetrievalMode.BIDIRECTIONAL,
        }
        retrieval_mode = mode_map.get(mode, RetrievalMode.PARALLEL)

        # Initialize query components
        interpreter = RuleBasedQueryInterpreter(sql_store)
        planner = TemplateQueryPlanner()
        fuser = RRFFuser()
        reranker = TFIDFReranker()

        # Initialize LLM for informed retrieval modes.
        # Uses the per-role API config from ~/.ctxmtg/.env.
        # If no API key is configured, the executor falls back to Parallel.
        llm = None
        prompt_assembler = None
        if retrieval_mode != RetrievalMode.PARALLEL:
            try:
                from ctxmtg.llm.factory import get_best_provider
                from ctxmtg.llm.prompt_assembler import PromptAssembler

                prompt_assembler = PromptAssembler()
                llm = get_best_provider(
                    "synthesis", "retrieval", "query_planning",
                    db_path=db_path,
                )
                if llm:
                    click.echo(f"LLM: {llm.get_model_name()}")
            except Exception:
                pass

        # Wire quality logger for the self-learning feedback loop.
        # Logs every query to query_quality_log so FeedbackLoopStage
        # can detect zero-result and refinement patterns.
        quality_logger = QueryQualityLogger(sql_store)

        # Load embedding function for vector search.
        # ORIGINAL CODE (disabled 2026-04-07): embedding_fn was not passed
        # to QueryExecutor, so vector search was always skipped. All queries
        # ran SQL-only, ignoring the 1000+ embeddings in LanceDB.
        # executor = QueryExecutor(...) # no embedding_fn
        embedding_fn = None
        try:
            from ctxmtg.embedding.onnx_embedder import ONNXEmbeddingProvider
            embedder = ONNXEmbeddingProvider()
            embedding_fn = lambda text: embedder.embed([text])[0]
        except Exception:
            pass

        executor = QueryExecutor(
            sql_store=sql_store,
            vector_store=vector_store,
            interpreter=interpreter,
            planner=planner,
            fuser=fuser,
            reranker=reranker,
            embedding_fn=embedding_fn,
            llm=llm,
            prompt_assembler=prompt_assembler,
            profile=profile,
            quality_logger=quality_logger,
        )

        # Run query, hive query, and cleanup in a single event loop
        # to avoid aiosqlite "Event loop is closed" errors from
        # multiple asyncio.run() calls (aiosqlite keeps background
        # threads tied to the first event loop).
        async def _run_query_and_cleanup():
            result = await executor.execute(
                question, profile, mode=retrieval_mode, top_k=top_k,
            )

            # Log and run hive query while the event loop is still alive
            eval_folder = None
            hive_result = None
            try:
                from ctxmtg.query.evaluation import log_query_answer

                eval_path = log_query_answer(result)
                eval_folder = eval_path.parent

                try:
                    from ctxmtg.query.hive_runner import run_hive_query

                    hive_db_path = str(
                        Path(settings.hive.local_db_path).expanduser()
                    )
                    if Path(hive_db_path).exists():
                        hive_result = await run_hive_query(
                            query=question,
                            mode=retrieval_mode,
                            top_k=top_k,
                            hive_db_path=hive_db_path,
                            eval_folder=eval_folder,
                        )
                except Exception:
                    pass  # Hive query is best-effort
            except Exception:
                pass  # Logging is best-effort

            # Close stores while the event loop is still running
            await sql_store.close()
            await vector_store.close()

            return result, eval_folder, hive_result

        result, eval_folder, hive_result = asyncio.run(
            _run_query_and_cleanup()
        )

        # Print results (sync, after event loop is done)
        click.echo(f"\nQuery: {question}")
        click.echo(
            f"Mode: {result.mode.value} | Results: {result.total_results} "
            f"| Latency: {result.latency_ms}ms"
        )

        if result.fallback_reason:
            click.echo(
                f"Note: requested mode {mode} fell back to "
                f"{result.mode.value} ({result.fallback_reason})"
            )

        click.echo("-" * 60)

        if result.synthesis:
            click.echo(f"\n{result.synthesis}")
            click.echo("-" * 60)

        if not result.results:
            click.echo("No results found.")
        else:
            for i, r in enumerate(result.results[:top_k], 1):
                content_preview = r.content[:150].replace("\n", " ")
                click.echo(f"\n{i}. [{r.source_store}] (score: {r.score:.3f})")
                click.echo(f"   {content_preview}")

        if eval_folder:
            click.echo(f"\n(Logged to {eval_folder.name})")
        if hive_result:
            click.echo(
                f"(Hive: {hive_result.get('results', 0)} results, "
                f"{hive_result.get('latency_ms', 0)}ms)"
            )

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


# =====================================================================
# Profile command
# =====================================================================
@main.command()
@click.argument("name", required=False)
@click.option("--list", "list_profiles", is_flag=True, help="List available profiles.")
@click.pass_context
def profile(ctx: click.Context, name: str | None, list_profiles: bool) -> None:
    """
    List or show domain profiles.

    Examples:
        ctxmtg profile --list
        ctxmtg profile general
    """
    from ctxmtg.profile.loader import ProfileLoader

    if list_profiles or not name:
        # List available profiles
        profiles = ProfileLoader.list_profiles()
        click.echo("Available profiles:")
        for p in profiles:
            click.echo(f"  - {p}")
        return

    # Show a specific profile's configuration
    try:
        prof = ProfileLoader.load(name)
        click.echo(f"Profile: {prof.name} (v{prof.version})")
        click.echo(f"Description: {prof.description}")
        click.echo(f"Entity types: {', '.join(prof.ner.entity_types)}")
        click.echo(f"Embedding model: {prof.embedding.preferred_model}")
        click.echo(f"Intake mode: {prof.intake.mode}")
        click.echo(f"Reject rules: {len(prof.intake.reject)}")
        click.echo(f"Defer rules: {len(prof.intake.defer)}")
        click.echo(f"Accept rules: {len(prof.intake.accept)}")
    except Exception as exc:
        click.echo(f"Error loading profile '{name}': {exc}", err=True)
        ctx.exit(1)


# =====================================================================
# Intake stats command
# =====================================================================
@main.group()
def intake() -> None:
    """Intake (Traffic Cop) management commands."""
    pass


@intake.command("stats")
def intake_stats() -> None:
    """Show Traffic Cop accept/defer/reject statistics."""
    # In a real deployment, these stats would be persisted.
    # For now, show that the command exists and works.
    click.echo("Intake Statistics")
    click.echo("-" * 30)
    click.echo("Note: Stats are tracked per-session.")
    click.echo("Use 'ctxmtg ingest' to process files,")
    click.echo("then check stats via the worker API.")


# Register the intake group under main
main.add_command(intake)


# =====================================================================
# Hive commands (intelligence aggregator)
# =====================================================================
@main.group()
def hive() -> None:
    """Hive intelligence management commands."""
    pass


@hive.command("status")
@click.pass_context
def hive_status(ctx: click.Context) -> None:
    """
    Show hive intelligence status: profiles, insights, source locals.

    Examples:
        ctxmtg hive status
    """
    import asyncio

    from ctxmtg.config.settings import CtxMtgSettings
    from ctxmtg.sync.hive_db import HiveDatabase

    settings = CtxMtgSettings()

    try:
        hive_db = HiveDatabase(
            mode=settings.hive.mode,
            local_db_path=str(Path(settings.hive.local_db_path).expanduser()),
        )
        asyncio.run(hive_db.initialize())

        status = asyncio.run(hive_db.get_status())
        counts = status.get("record_counts", {})

        click.echo("Hive Intelligence Status")
        click.echo("=" * 40)
        click.echo(f"Mode:             {status.get('mode', 'local')}")
        click.echo(f"Entity profiles:  {counts.get('hive_entity_profiles', 0)}")
        click.echo(f"Local insights:   {counts.get('hive_insights', 0)}")
        click.echo(f"Native insights:  {counts.get('hive_native_insights', 0)}")
        sources = status.get("source_instances", [])
        click.echo(f"Source locals:    {', '.join(sources) if sources else 'none'}")

        asyncio.run(hive_db.close())

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


@hive.command("push")
@click.pass_context
def hive_push(ctx: click.Context) -> None:
    """
    Write distilled intelligence to the local outbox.

    Writes new distiller_summaries and meta_insights as a JSON
    manifest to the outbox directory.  The hive pulls from this
    outbox via configured links.

    Examples:
        ctxmtg hive push
    """
    import asyncio

    from ctxmtg.config.settings import CtxMtgSettings
    from ctxmtg.sync.hive_sync import HiveSyncWorker
    from ctxmtg.sync.outbox_writer import OutboxWriter

    settings = CtxMtgSettings()
    db_path = ctx.obj.get("db_path") or settings.db_path

    try:
        sql_store = _init_sql_store(db_path)

        # 2026-04-08: Outbox pattern -- write to local outbox, not hive DB
        outbox_path = Path(settings.hive.outbox_path).expanduser()
        instance_name = settings.hive.instance_name

        writer = OutboxWriter(
            outbox_path=outbox_path,
            instance_name=instance_name,
        )
        worker = HiveSyncWorker(
            local_store=sql_store,
            outbox_writer=writer,
        )
        click.echo(f"Writing intelligence to outbox ({outbox_path})...")
        counts = asyncio.run(worker.sync())

        click.echo(f"Written to outbox:")
        click.echo(f"  Distiller summaries: {counts.get('summaries', 0)}")
        click.echo(f"  Meta insights:       {counts.get('insights', 0)}")
        if counts.get("manifest"):
            click.echo(f"  Manifest: {counts['manifest']}")

        asyncio.run(sql_store.close())

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)

    # ORIGINAL hive_push (disabled 2026-04-08): direct push to hive DB
    # from ctxmtg.sync.hive_db import HiveDatabase
    # hive_settings = settings.hive
    # hive_db_path = str(Path(hive_settings.local_db_path).expanduser())
    # Path(hive_db_path).parent.mkdir(parents=True, exist_ok=True)
    # hive_db = HiveDatabase(
    #     mode=hive_settings.mode,
    #     local_db_path=hive_db_path,
    # )
    # asyncio.run(hive_db.initialize())
    # worker = HiveSyncWorker(
    #     local_store=sql_store,
    #     hive_db=hive_db,
    # )
    # click.echo("Pushing intelligence to hive...")
    # counts = asyncio.run(worker.sync())
    # click.echo(f"Pushed:")
    # click.echo(f"  Entity profiles: {counts.get('profiles', 0)}")
    # click.echo(f"  Insights:        {counts.get('insights', 0)}")
    # asyncio.run(sql_store.close())
    # asyncio.run(hive_db.close())


@hive.command("sync")
@click.pass_context
def hive_sync(ctx: click.Context) -> None:
    """
    Run a hive sync cycle (alias for 'hive push').

    Examples:
        ctxmtg hive sync
    """
    ctx.invoke(hive_push)


@hive.command("farm")
@click.pass_context
def hive_farm(ctx: click.Context) -> None:
    """
    Run the 3-stage hive farming pipeline.

    Analyzes the merged collective intelligence for cross-stream
    patterns, latent relationships, and insight correlations.

    Examples:
        ctxmtg hive farm
    """
    import asyncio

    from ctxmtg.config.settings import CtxMtgSettings
    from ctxmtg.sync.hive_db import HiveDatabase
    from ctxmtg.sync.hive_farming import HiveFarmingPipeline

    settings = CtxMtgSettings()

    try:
        hive_db = HiveDatabase(
            mode=settings.hive.mode,
            local_db_path=str(Path(settings.hive.local_db_path).expanduser()),
        )
        asyncio.run(hive_db.initialize())

        pipeline = HiveFarmingPipeline(hive_db)
        click.echo("Running hive farming (3 stages)...")
        result = asyncio.run(pipeline.run())

        click.echo(f"\nHive Farming Complete")
        click.echo("=" * 40)
        click.echo(f"Status:     {result.get('status', 'unknown')}")
        click.echo(f"Stages run: {result.get('stages_run', 0)}")
        click.echo(f"Succeeded:  {result.get('stages_succeeded', 0)}")
        click.echo(f"Failed:     {result.get('stages_failed', 0)}")

        for stage_name, stage_result in result.get("stage_results", {}).items():
            click.echo(f"\n  {stage_name}: {stage_result}")

        asyncio.run(hive_db.close())

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


@hive.command("serve")
@click.option("--port", default=8081, help="Port for hive web UI")
@click.option("--host", default="127.0.0.1", help="Bind address")
def hive_serve(port: int, host: str) -> None:
    """
    Start the Hive Command Center web UI.

    Runs on a separate port from the local Command Center.
    Manage links to locals, pull intelligence, browse profiles.

    Examples:
        ctxmtg hive serve
        ctxmtg hive serve --port 8082
    """
    from ctxmtg.web.hive_app import run_hive_server

    click.echo(f"Starting Hive Command Center on http://{host}:{port}")
    run_hive_server(host=host, port=port)


# Register hive group under main
main.add_command(hive)


# =====================================================================
# Intelligence commands
# =====================================================================
@main.group()
def intelligence() -> None:
    """Intelligence management commands."""
    pass


@intelligence.command("pull")
@click.pass_context
def intelligence_pull(ctx: click.Context) -> None:
    """
    Pull hive intelligence to local cache.

    Fetches merged entity profiles from the hive and caches
    them locally for use by the extraction pipeline.

    Examples:
        ctxmtg intelligence pull
    """
    import asyncio

    from ctxmtg.config.settings import CtxMtgSettings
    from ctxmtg.sync.hive_db import HiveDatabase
    from ctxmtg.sync.intelligence_pull import IntelligencePullWorker

    settings = CtxMtgSettings()
    db_path = ctx.obj.get("db_path") or settings.db_path

    try:
        sql_store = _init_sql_store(db_path)

        hive_db = HiveDatabase(
            mode=settings.hive.mode,
            local_db_path=str(Path(settings.hive.local_db_path).expanduser()),
        )
        asyncio.run(hive_db.initialize())

        worker = IntelligencePullWorker(
            local_store=sql_store,
            hive_db=hive_db,
        )
        click.echo("Pulling hive intelligence...")
        cached = asyncio.run(worker.pull())
        click.echo(f"Cached {cached} entity profiles from hive.")

        asyncio.run(sql_store.close())
        asyncio.run(hive_db.close())

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


main.add_command(intelligence)


# =====================================================================
# Evaluate commands (query evaluation pipeline)
# =====================================================================
@main.group()
def evaluate() -> None:
    """Query evaluation pipeline commands."""
    pass


@evaluate.command("list")
@click.option("--limit", default=20, help="Number of recent evaluations to show.")
def evaluate_list(limit: int) -> None:
    """
    List recent query evaluations.

    Examples:
        ctxmtg evaluate list
        ctxmtg evaluate list --limit 5
    """
    from ctxmtg.query.evaluation import list_evaluations

    evals = list_evaluations(limit=limit)

    if not evals:
        click.echo("No evaluations found. Run 'ctxmtg query' first.")
        return

    click.echo("Recent Query Evaluations")
    click.echo("=" * 60)

    for e in evals:
        has_hive = "H" if e.get("has_hive") else "-"
        has_eval = "E" if e.get("has_evaluation") else "-"
        query = e.get("query", "?")
        if len(query) > 50:
            query = query[:47] + "..."
        click.echo(
            f"  {e['folder']}  [{has_hive}{has_eval}]  "
            f"{e.get('total_results', 0)} results  {query}"
        )

    click.echo(f"\n[H]=has hive answer  [E]=has evaluation")


@evaluate.command("run")
@click.argument("folder_name")
@click.pass_context
def evaluate_run(ctx: click.Context, folder_name: str) -> None:
    """
    Run an LLM evaluation on a logged query.

    Compares local vs hive answers using an LLM and produces a
    comparison report. Requires a configured LLM provider.

    Examples:
        ctxmtg evaluate run 20260324_120000_abc12345
    """
    import asyncio

    from ctxmtg.query.evaluation import (
        build_evaluation_prompt,
        get_eval_dir,
        load_evaluation_inputs,
        save_evaluation_result,
    )

    eval_dir = get_eval_dir()
    eval_folder = eval_dir / folder_name

    if not eval_folder.exists():
        click.echo(f"Error: Evaluation folder not found: {folder_name}", err=True)
        ctx.exit(1)
        return

    inputs = load_evaluation_inputs(eval_folder)
    local_answer = inputs.get("local_answer")

    if not local_answer:
        click.echo("Error: No local_answer.json in evaluation folder.", err=True)
        ctx.exit(1)
        return

    hive_answer = inputs.get("hive_answer")

    # Load meta insights from local store
    meta_insights = []
    try:
        from ctxmtg.config.settings import CtxMtgSettings

        settings = CtxMtgSettings()
        db_path = ctx.obj.get("db_path") or settings.db_path
        sql_store = _init_sql_store(db_path)

        rows = asyncio.run(
            sql_store.execute_sql(
                "SELECT insight_type, title, description, confidence "
                "FROM meta_insights ORDER BY created_at DESC LIMIT 10",
                {},
            )
        )
        meta_insights = rows
        asyncio.run(sql_store.close())
    except Exception:
        pass

    # Build prompt
    prompt = build_evaluation_prompt(local_answer, hive_answer, meta_insights)

    # Try to get an LLM evaluation
    try:
        from ctxmtg.llm.factory import get_best_provider

        llm = get_best_provider("synthesis", "extraction")
        if llm and llm.is_available():
            evaluation_text = llm.generate(
                prompt=prompt,
                system_prompt=(
                    "You are a knowledge retrieval system evaluator. "
                    "Compare local and hive answers objectively."
                ),
                temperature=0.3,
                max_tokens=512,
            )
        else:
            evaluation_text = "(No LLM available. Showing prompt only.)"
    except Exception:
        evaluation_text = "(LLM evaluation failed. Showing prompt only.)"

    # Display
    click.echo(f"\nEvaluation: {folder_name}")
    click.echo("=" * 60)
    click.echo(f"Query: {local_answer.get('query', '?')}")
    click.echo(f"Local results: {local_answer.get('total_results', 0)}")
    if hive_answer:
        click.echo(f"Hive results: {hive_answer.get('total_results', 0)}")
    else:
        click.echo("Hive results: (none)")
    click.echo(f"Meta insights: {len(meta_insights)}")
    click.echo("-" * 60)
    click.echo(evaluation_text)

    # Save
    save_evaluation_result(eval_folder, {
        "query": local_answer.get("query", "?"),
        "local_results": local_answer.get("total_results", 0),
        "hive_results": hive_answer.get("total_results", 0) if hive_answer else 0,
        "meta_insights_count": len(meta_insights),
        "evaluation": evaluation_text,
    })
    click.echo(f"\n(Saved to {eval_folder / 'evaluation.json'})")


main.add_command(evaluate)


# =====================================================================
# Farm commands (farming pipeline management)
# =====================================================================
@main.group()
def farm() -> None:
    """Farming pipeline management commands."""
    pass


@farm.command("run")
@click.pass_context
def farm_run(ctx: click.Context) -> None:
    """
    Run a full farming cycle (all 16 stages).

    Executes intelligence stages, self-learning feedback loop,
    and maintenance agents in order. Hive pull is suspended
    during the farming window.

    Examples:
        ctxmtg farm run
    """
    import asyncio

    from ctxmtg.config.settings import CtxMtgSettings
    from ctxmtg.farming import FarmingPipeline, create_default_stages

    settings = CtxMtgSettings()
    db_path = ctx.obj.get("db_path") or settings.db_path
    vector_path = ctx.obj.get("vector_path") or settings.vector_path

    try:
        sql_store, vector_store = _init_stores(db_path, vector_path)

        # Try to load a per-role LLM for farming stages.
        llm = None
        try:
            from ctxmtg.llm.factory import get_best_provider
            llm = get_best_provider("farming", "extraction", db_path=db_path)
            if llm:
                click.echo(f"LLM: {llm.get_model_name()}")
        except Exception:
            pass

        # Build pipeline with all default stages
        pipeline = FarmingPipeline(sql_store, vector_store)
        for stage in create_default_stages(llm=llm):
            pipeline.register_stage(stage)

        click.echo("Running farming cycle (16 stages)...")
        result = asyncio.run(pipeline.run_cycle(trigger="manual"))

        # Display results
        click.echo(f"\nFarming Cycle #{result['cycle_id']}")
        click.echo("=" * 40)
        click.echo(f"Status:           {result['status']}")
        click.echo(f"Stages run:       {result['stages_run']}")
        click.echo(f"Stages succeeded: {result['stages_succeeded']}")
        click.echo(f"Stages failed:    {result['stages_failed']}")
        click.echo(f"Insights produced:{result['insights_produced']}")
        click.echo(f"Duration:         {result['duration_ms']:.0f}ms")

        # Cleanup
        asyncio.run(sql_store.close())
        asyncio.run(vector_store.close())

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


@farm.command("status")
@click.option("--limit", default=5, help="Number of recent cycles to show.")
@click.pass_context
def farm_status(ctx: click.Context, limit: int) -> None:
    """
    Show recent farming cycle status.

    Displays the most recent farming cycles with their status,
    stage counts, and duration.

    Examples:
        ctxmtg farm status
        ctxmtg farm status --limit 10
    """
    import asyncio

    from ctxmtg.config.settings import CtxMtgSettings

    settings = CtxMtgSettings()
    db_path = ctx.obj.get("db_path") or settings.db_path

    try:
        sql_store = _init_sql_store(db_path)

        rows = asyncio.run(
            sql_store.execute_sql(
                "SELECT cycle_id, status, trigger, stages_done, "
                "started_at, completed_at "
                "FROM farming_cycles "
                "ORDER BY cycle_id DESC LIMIT :limit",
                {"limit": limit},
            )
        )

        if not rows:
            click.echo("No farming cycles found.")
            asyncio.run(sql_store.close())
            return

        click.echo("Recent Farming Cycles")
        click.echo("=" * 60)
        for row in rows:
            click.echo(
                f"  Cycle #{row['cycle_id']}  "
                f"status={row['status']}  "
                f"trigger={row.get('trigger', '?')}  "
                f"stages={row.get('stages_done', '?')}  "
                f"started={row.get('started_at', '?')}"
            )

        # Also show insight counts
        insight_rows = asyncio.run(
            sql_store.execute_sql(
                "SELECT COUNT(*) as cnt FROM meta_insights", {}
            )
        )
        if insight_rows:
            click.echo(f"\nTotal farming insights: {insight_rows[0]['cnt']}")

        asyncio.run(sql_store.close())

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


# Register farm group under main
main.add_command(farm)


def _init_sql_store(db_path: str):
    """
    Initialize a SQLiteStore from a database path.

    This is a lightweight helper for CLI commands that only need
    the SQL store (not the vector store).

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        An initialised SQLiteStore instance.
    """
    import asyncio

    from ctxmtg.storage.sqlite import SQLiteStore

    resolved = Path(db_path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    store = SQLiteStore(db_path=str(resolved))
    asyncio.run(store.initialize())
    return store


# =====================================================================
# Health command
# =====================================================================
@main.command()
@click.pass_context
def health(ctx: click.Context) -> None:
    """
    Show system health: RAM, DB size, record counts, vector count.
    """
    import asyncio

    from ctxmtg.config.settings import CtxMtgSettings
    from ctxmtg.health.monitor import HealthMonitor

    settings = CtxMtgSettings()
    db_path = Path(ctx.obj.get("db_path") or settings.db_path).expanduser()
    vector_path = ctx.obj.get("vector_path") or settings.vector_path

    # Try to connect to stores for record counts
    sql_store = None
    vector_store = None

    import contextlib

    with contextlib.suppress(Exception):
        sql_store, vector_store = _init_stores(str(db_path), vector_path)

    monitor = HealthMonitor(
        sql_store=sql_store,
        vector_store=vector_store,
        db_path=db_path,
    )

    health_data = monitor.get_health()

    click.echo("ctxmtg Health Report")
    click.echo("=" * 40)
    click.echo(f"Status:     {health_data.get('status', 'unknown')}")
    click.echo(f"RAM Usage:  {health_data.get('ram_mb', 0)} MB")
    click.echo(f"DB Size:    {health_data.get('db_size_mb', 0)} MB")

    counts = health_data.get("record_counts", {})
    if counts:
        click.echo(f"Interactions: {counts.get('interactions', 0)}")
        click.echo(f"Entities:     {counts.get('entities', 0)}")
        click.echo(f"Facts:        {counts.get('facts', 0)}")

    vec_count = health_data.get("vector_count", 0)
    click.echo(f"Vectors:    {vec_count}")

    intake_st = health_data.get("intake_stats", {})
    if intake_st:
        click.echo("\nIntake Stats:")
        for action, count in intake_st.items():
            click.echo(f"  {action}: {count}")

    # Cleanup
    if sql_store:
        with contextlib.suppress(Exception):
            asyncio.run(sql_store.close())
    if vector_store:
        with contextlib.suppress(Exception):
            asyncio.run(vector_store.close())


# =====================================================================
# Suggest command (query autocomplete from distiller intelligence)
# =====================================================================
@main.command()
@click.argument("partial", required=False, default="")
@click.option("--browse", is_flag=True, help="Show top entities for browsing.")
@click.pass_context
def suggest(ctx: click.Context, partial: str, browse: bool) -> None:
    """
    Suggest query completions from accumulated knowledge.

    Uses distilled entity intelligence (from farming or hive sync) to
    generate query suggestions.  If no intelligence data exists yet
    (fresh install, no farming done), shows an empty result.

    Examples:
        ctxmtg suggest "What did Al"
        ctxmtg suggest --browse
    """
    import asyncio

    from ctxmtg.config.settings import CtxMtgSettings
    from ctxmtg.query.autocomplete import AutocompleteEngine

    settings = CtxMtgSettings()
    db_path = ctx.obj.get("db_path") or settings.db_path

    try:
        # Initialize the SQL store (vector store not needed for suggest).
        sql_store = _init_sql_store(db_path)

        # Create the autocomplete engine backed by the local store.
        engine = AutocompleteEngine(sql_store)

        if browse or not partial.strip():
            # Browse mode: show the top entities by relevance.
            entities = asyncio.run(engine.get_top_entities(limit=10))

            if not entities:
                click.echo("No intelligence data yet. Run 'ctxmtg farm run' first.")
            else:
                click.echo("Top Entities")
                click.echo("=" * 50)
                for i, ent in enumerate(entities, 1):
                    score = ent.get("relevance_score", 0.0)
                    etype = ent.get("entity_type", "?")
                    name = ent.get("entity_name", "?")
                    summary = ent.get("summary", "")
                    # Truncate long summaries for display.
                    if len(summary) > 80:
                        summary = summary[:77] + "..."
                    click.echo(f"  {i}. [{etype}] {name} (score: {score:.2f})")
                    if summary:
                        click.echo(f"     {summary}")
        else:
            # Suggestion mode: generate query completions.
            suggestions = asyncio.run(engine.suggest(partial, max_suggestions=10))

            if not suggestions:
                click.echo("No matching suggestions found.")
            else:
                click.echo("Suggestions:")
                for i, s in enumerate(suggestions, 1):
                    click.echo(f"  {i}. {s}")

        # Cleanup
        asyncio.run(sql_store.close())

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


# =====================================================================
# Watch command (inbox watcher daemon)
# =====================================================================
@main.command()
@click.option("--once", is_flag=True, help="Process inbox once and exit (for cron).")
@click.option("--interval", default=30, help="Seconds between polls (default 30).")
@click.pass_context
def watch(ctx: click.Context, once: bool, interval: int) -> None:
    """
    Watch the inbox directory and auto-ingest new files.

    Files dropped into ~/.ctxmtg/inbox/ are ingested and moved to
    ~/.ctxmtg/processed/ on success.

    Examples:
        ctxmtg watch
        ctxmtg watch --once
        ctxmtg watch --interval 10
    """
    from ctxmtg.config.settings import CtxMtgSettings
    from ctxmtg.ingestion.watcher import InboxWatcher

    settings = CtxMtgSettings()
    inbox = Path(settings.inbox_path).expanduser()
    processed = Path(settings.processed_path).expanduser()
    inbox.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    try:
        worker = _create_worker(ctx)
    except Exception as exc:
        click.echo(f"Error initialising pipeline: {exc}", err=True)
        ctx.exit(1)
        return

    watcher = InboxWatcher(
        inbox_path=inbox,
        processed_path=processed,
        worker=worker,
        interval_seconds=interval,
    )

    if once:
        results = watcher.scan_once()
        ok = sum(1 for r in results if r["status"] == "ok")
        failed = sum(1 for r in results if r["status"] == "error")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        click.echo(f"Processed: {ok} ok, {failed} failed, {skipped} skipped")
        for r in results:
            if r["status"] == "ok":
                click.echo(f"  + {r['file']}: {r.get('entities', 0)} entities, {r.get('facts', 0)} facts")
            elif r["status"] == "error":
                click.echo(f"  x {r['file']}: {r.get('error', 'unknown')}")
        if not results:
            click.echo("Inbox is empty.")
    else:
        click.echo(f"Watching {inbox} (every {interval}s). Ctrl+C to stop.")
        watcher.run()


# =====================================================================
# Proxy command (LLM proxy for live chat capture)
# =====================================================================
@main.command()
@click.option("--port", default=11435, help="Port to listen on (default 11435).")
@click.option(
    "--upstream",
    default="http://localhost:11434",
    help="Upstream LLM backend URL (default: Ollama on 11434).",
)
@click.pass_context
def proxy(ctx: click.Context, port: int, upstream: str) -> None:
    """
    Start an LLM proxy that captures conversations.

    Forwards all requests to the upstream LLM and silently captures
    user+assistant exchanges as chat interactions.

    Point your chat client at http://127.0.0.1:<port> instead of
    the upstream URL.

    Examples:
        ctxmtg proxy
        ctxmtg proxy --port 11435 --upstream http://localhost:11434
        ctxmtg proxy --upstream https://api.openai.com
    """
    try:
        from ctxmtg.proxy import run_proxy

        click.echo(f"LLM proxy on http://127.0.0.1:{port} -> {upstream}")
        click.echo("Chat conversations will be captured automatically.")
        run_proxy(port=port, upstream=upstream)
    except ImportError as exc:
        click.echo(
            f"Error: Missing dependency. Run: pip install httpx\n{exc}",
            err=True,
        )
        ctx.exit(1)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


# =====================================================================
# Serve command (web command center)
# =====================================================================
@main.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Bind address (forced to 127.0.0.1 for security).",
)
@click.option("--port", default=8080, help="Port to listen on.")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """
    Start the web command center (dashboard + API).

    Binds to localhost only. On first run you will be prompted to
    set an admin password.

    Examples:
        ctxmtg serve
        ctxmtg serve --port 9090
    """
    try:
        from ctxmtg.web.app import run_server

        click.echo(f"Starting command center on http://127.0.0.1:{port}")
        run_server(host=host, port=port)
    except ImportError as exc:
        click.echo(
            f"Error: Web dependencies not installed. "
            f"Run: pip install 'ctxmtg[web]'\n{exc}",
            err=True,
        )
        ctx.exit(1)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


# =====================================================================
# Helper functions for wiring up pipeline components
# =====================================================================


def _init_stores(db_path: str, vector_path: str) -> tuple:
    """
    Initialize SQL and vector stores.

    Args:
        db_path: Path to the SQLite database.
        vector_path: Path to the vector store directory.

    Returns:
        A tuple of (SQLiteStore, LanceDBStore).
    """
    import asyncio

    from ctxmtg.storage.lancedb_store import LanceDBStore
    from ctxmtg.storage.sqlite import SQLiteStore

    # Ensure parent directories exist
    resolved_db = Path(db_path).expanduser()
    resolved_db.parent.mkdir(parents=True, exist_ok=True)

    resolved_vec = Path(vector_path).expanduser()
    resolved_vec.mkdir(parents=True, exist_ok=True)

    sql_store = SQLiteStore(db_path=str(resolved_db))
    vector_store = LanceDBStore(db_path=str(resolved_vec))

    asyncio.run(sql_store.initialize())
    asyncio.run(vector_store.initialize())

    return sql_store, vector_store


def _create_worker(ctx: click.Context):
    """
    Create an IngestionWorker with all pipeline components wired up.

    Reads configuration from the Click context and settings.

    Args:
        ctx: The Click context with CLI options.

    Returns:
        A configured IngestionWorker instance.
    """
    from ctxmtg.config.settings import CtxMtgSettings
    from ctxmtg.ingestion.worker import IngestionWorker
    from ctxmtg.intake.rules import RuleBasedIntakeGateway
    from ctxmtg.profile.loader import ProfileLoader

    settings = CtxMtgSettings()

    # Override settings from CLI options
    db_path = ctx.obj.get("db_path") or settings.db_path
    vector_path = ctx.obj.get("vector_path") or settings.vector_path
    profile_name = ctx.obj.get("profile_name", "general")

    # Load profile
    profile = ProfileLoader.load(profile_name)

    # Initialize stores
    sql_store, vector_store = _init_stores(db_path, vector_path)

    # Initialize Traffic Cop
    gateway = RuleBasedIntakeGateway(profile.intake)

    # Try to initialize extraction pipeline (may fail if spaCy model missing)
    extraction = None
    try:
        from ctxmtg.extraction.pipeline import BasicExtractionPipeline

        # Wire LLM verifier if extraction API key is configured.
        llm_verifier = None
        try:
            from ctxmtg.llm.factory import create_provider
            extraction_llm = create_provider("extraction", db_path=db_path)
            if extraction_llm:
                from ctxmtg.llm.prompt_assembler import PromptAssembler
                from ctxmtg.extraction.llm_verifier import LLMExtractionVerifier
                assembler = PromptAssembler()
                llm_verifier = LLMExtractionVerifier(
                    llm=extraction_llm,
                    prompt_assembler=assembler,
                    profile=profile,
                )
                click.echo(f"Extraction LLM: {extraction_llm.get_model_name()}")
        except Exception:
            pass

        extraction = BasicExtractionPipeline(profile, llm_verifier=llm_verifier)
    except Exception as exc:
        click.echo(f"Warning: Extraction pipeline unavailable: {exc}", err=True)

    # Try to initialize embedding provider (may fail if ONNX model missing)
    embedder = None
    try:
        from ctxmtg.embedding.onnx_embedder import ONNXEmbeddingProvider

        embedder = ONNXEmbeddingProvider()
    except Exception as exc:
        click.echo(f"Warning: Embedding provider unavailable: {exc}", err=True)

    return IngestionWorker(
        sql_store=sql_store,
        vector_store=vector_store,
        extraction_pipeline=extraction,
        embedding_provider=embedder,
        intake_gateway=gateway,
    )
