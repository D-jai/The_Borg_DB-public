# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Query Evaluation Pipeline
==========================

Logs query answers and provides an LLM-powered evaluation command
that compares local vs hive answers.

P5-10: Every ``ctxmtg query`` logs the question, results, synthesis,
mode, and latency to ``~/.ctxmtg/evaluations/{timestamp}_{hash}/
local_answer.json``.

P5-12: ``ctxmtg evaluate`` reads a logged evaluation folder, feeds
local answer + hive answer + meta insights to an LLM for comparison.

Depends on:
    - json (serialization)
    - hashlib (query hash for folder naming)
    - pathlib (evaluation folder management)
    - ctxmtg.models.query (QueryResult, SearchResult)

Used by:
    - ctxmtg.cli (query command logging, evaluate command)
    - ctxmtg.query.hive_runner (writes hive_answer.json)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from ctxmtg.models.query import QueryResult

logger = structlog.get_logger("ctxmtg.query.evaluation")

DEFAULT_EVAL_DIR = "~/.ctxmtg/evaluations"


def get_eval_dir(base_dir: str | None = None) -> Path:
    """Return the evaluations directory, creating it if needed."""
    path = Path(base_dir or DEFAULT_EVAL_DIR).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _query_hash(query: str) -> str:
    """Short hash of the query text for folder naming."""
    return hashlib.sha256(query.encode()).hexdigest()[:8]


def _result_to_dict(result: QueryResult) -> dict[str, Any]:
    """Serialize a QueryResult to a JSON-friendly dict."""
    return {
        "query": result.query,
        "mode": result.mode.value,
        "total_results": result.total_results,
        "sql_results_count": result.sql_results_count,
        "vector_results_count": result.vector_results_count,
        "synthesis": result.synthesis,
        "fallback_reason": result.fallback_reason,
        "latency_ms": round(result.latency_ms, 2),
        "results": [
            {
                "id": r.id,
                "source_store": r.source_store,
                "content": r.content[:500],
                "score": round(r.score, 4),
                "metadata": r.metadata,
            }
            for r in result.results[:20]
        ],
    }


def log_query_answer(
    result: QueryResult,
    eval_dir: str | None = None,
    answer_type: str = "local_answer",
) -> Path:
    """
    Log a query result to the evaluations directory.

    Creates a folder named ``{timestamp}_{hash}`` and writes the
    result as ``{answer_type}.json``.

    Args:
        result: The QueryResult to log.
        eval_dir: Override for the evaluations directory.
        answer_type: File name prefix ("local_answer" or "hive_answer").

    Returns:
        Path to the created JSON file.
    """
    import uuid

    base = get_eval_dir(eval_dir)
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    qhash = _query_hash(result.query)
    uid = uuid.uuid4().hex[:4]
    folder_name = f"{ts}_{qhash}_{uid}"
    folder = base / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    data = _result_to_dict(result)
    data["logged_at"] = now.isoformat()

    file_path = folder / f"{answer_type}.json"
    file_path.write_text(json.dumps(data, indent=2, default=str))

    logger.info(
        "query_answer_logged",
        folder=str(folder),
        answer_type=answer_type,
        query=result.query[:80],
    )

    return file_path


def log_hive_answer(
    result: QueryResult,
    eval_folder: Path,
) -> Path:
    """
    Log a hive query result to an existing evaluation folder.

    Args:
        result: The hive QueryResult to log.
        eval_folder: The evaluation folder (created by log_query_answer).

    Returns:
        Path to the created hive_answer.json file.
    """
    data = _result_to_dict(result)
    data["logged_at"] = datetime.now(timezone.utc).isoformat()

    file_path = eval_folder / "hive_answer.json"
    file_path.write_text(json.dumps(data, indent=2, default=str))

    logger.info(
        "hive_answer_logged",
        folder=str(eval_folder),
        query=result.query[:80],
    )

    return file_path


def list_evaluations(
    eval_dir: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    List recent evaluation folders with summary info.

    Returns:
        List of dicts with folder name, query, has_hive, timestamp.
    """
    base = get_eval_dir(eval_dir)
    folders = sorted(base.iterdir(), reverse=True) if base.exists() else []

    results = []
    for folder in folders[:limit]:
        if not folder.is_dir():
            continue

        local_file = folder / "local_answer.json"
        hive_file = folder / "hive_answer.json"

        entry: dict[str, Any] = {
            "folder": folder.name,
            "path": str(folder),
            "has_local": local_file.exists(),
            "has_hive": hive_file.exists(),
            "has_evaluation": (folder / "evaluation.json").exists(),
        }

        if local_file.exists():
            try:
                data = json.loads(local_file.read_text())
                entry["query"] = data.get("query", "?")
                entry["mode"] = data.get("mode", "?")
                entry["logged_at"] = data.get("logged_at", "?")
                entry["total_results"] = data.get("total_results", 0)
            except (json.JSONDecodeError, OSError):
                entry["query"] = "(unreadable)"

        results.append(entry)

    return results


def load_evaluation_inputs(
    eval_folder: Path,
) -> dict[str, Any]:
    """
    Load local_answer.json and hive_answer.json from an evaluation folder.

    Returns:
        Dict with keys: local_answer, hive_answer (both may be None).
    """
    result: dict[str, Any] = {
        "local_answer": None,
        "hive_answer": None,
    }

    local_file = eval_folder / "local_answer.json"
    if local_file.exists():
        try:
            result["local_answer"] = json.loads(local_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    hive_file = eval_folder / "hive_answer.json"
    if hive_file.exists():
        try:
            result["hive_answer"] = json.loads(hive_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    return result


def build_evaluation_prompt(
    local_answer: dict[str, Any],
    hive_answer: dict[str, Any] | None,
    meta_insights: list[dict[str, Any]] | None = None,
) -> str:
    """
    Build an LLM prompt for evaluating local vs hive query answers.

    Args:
        local_answer: The local query result dict.
        hive_answer: The hive query result dict (may be None).
        meta_insights: Optional farming meta insights for context.

    Returns:
        A prompt string for the LLM evaluation.
    """
    query = local_answer.get("query", "Unknown query")
    local_results = local_answer.get("results", [])
    local_synthesis = local_answer.get("synthesis")

    prompt_parts = [
        "You are evaluating a knowledge retrieval system that has two data sources:",
        "a local store (single device) and a hive (aggregated intelligence from multiple devices).",
        "",
        f"QUERY: {query}",
        "",
        "=== LOCAL ANSWER ===",
        f"Mode: {local_answer.get('mode', '?')}",
        f"Results: {local_answer.get('total_results', 0)}",
        f"Latency: {local_answer.get('latency_ms', 0)}ms",
    ]

    if local_synthesis:
        prompt_parts.append(f"Synthesis: {local_synthesis}")

    for i, r in enumerate(local_results[:5], 1):
        prompt_parts.append(f"  {i}. [{r.get('source_store', '?')}] {r.get('content', '')[:200]}")

    if hive_answer:
        hive_results = hive_answer.get("results", [])
        hive_synthesis = hive_answer.get("synthesis")

        prompt_parts.extend([
            "",
            "=== HIVE ANSWER ===",
            f"Mode: {hive_answer.get('mode', '?')}",
            f"Results: {hive_answer.get('total_results', 0)}",
            f"Latency: {hive_answer.get('latency_ms', 0)}ms",
        ])

        if hive_synthesis:
            prompt_parts.append(f"Synthesis: {hive_synthesis}")

        for i, r in enumerate(hive_results[:5], 1):
            prompt_parts.append(f"  {i}. [{r.get('source_store', '?')}] {r.get('content', '')[:200]}")
    else:
        prompt_parts.extend(["", "=== HIVE ANSWER ===", "(No hive answer available)"])

    if meta_insights:
        prompt_parts.extend(["", "=== META INSIGHTS (from farming) ==="])
        for ins in meta_insights[:5]:
            prompt_parts.append(
                f"  - [{ins.get('insight_type', '?')}] "
                f"{ins.get('title', '?')}: {ins.get('description', '')[:200]}"
            )

    prompt_parts.extend([
        "",
        "=== EVALUATION INSTRUCTIONS ===",
        "Compare the local and hive answers. Address:",
        "1. Did the hive surface information the local missed?",
        "2. Do the meta insights provide context that changes interpretation?",
        "3. Is there emergent wisdom from combining all three sources?",
        "4. Overall verdict: does the hive add value for this query?",
        "",
        "Provide a concise evaluation (3-5 sentences).",
    ])

    return "\n".join(prompt_parts)


def save_evaluation_result(
    eval_folder: Path,
    evaluation: dict[str, Any],
) -> Path:
    """Save an LLM evaluation result to the evaluation folder."""
    file_path = eval_folder / "evaluation.json"
    evaluation["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    file_path.write_text(json.dumps(evaluation, indent=2, default=str))

    logger.info(
        "evaluation_saved",
        folder=str(eval_folder),
    )

    return file_path
