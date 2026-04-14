# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
SQL Briefing Builder
====================

This module implements the multi-pass SQL briefing builder. Instead
of feeding raw SQL rows to the LLM (which would overflow the context
window), the builder runs profiling queries and compresses the results
into a ~200-300 token briefing.

The 4-pass strategy:
    Pass 1: Statistical Profile (automatic, domain-configured)
        6 profiling queries for a birds-eye view of the data.
    Pass 2: LLM-Designed Targeted Queries (Phase 2+)
        The LLM reads Pass 1, designs focused SQL with guardrails.
    Pass 3: Domain-Specific Profiling (Phase 2+)
        YAML-configured specialty queries from the domain profile.
    Pass 4: Raw Content Retrieval (Phase 2+, synthesis stage only)
        Fetches full interaction text for top-N results.

Pass 1 queries (run sequentially):
    1a. Core statistics: total facts, active/superseded counts, time range,
        confidence stats
    1b. Entity distribution: top entities by fact count
    1c. Predicate distribution: most common predicates
    1d. Temporal distribution: facts per week (trend detection)
    1e. Source distribution: interaction count per source type
    1f. Supersession summary: recent fact changes

Each query may return data or null (empty result). Null sections are
preserved in the briefing (not omitted) so the LLM knows what data
is missing. This is a key design decision: showing "no data" is more
informative than omitting the section entirely.

Safety guardrails for Pass 2 LLM-generated SQL:
    1. Allowlisted tables: facts, entities, interactions, meta_insights
    2. Mandatory LIMIT: appended if missing (default 20)
    3. Read-only: no DDL (CREATE/DROP/ALTER/INSERT/UPDATE/DELETE)
    4. Timeout: 2-second per-query timeout
    5. Parameterized values: entity names from briefing are parameterized

See research/notes/sql-briefing-strategy.md for the full design.

Depends on:
    - re (regex for SQL safety validation)
    - json (parsing LLM responses for Pass 2)
    - ctxmtg.interfaces.storage (SQLStore for profiling queries)
    - ctxmtg.interfaces.llm (LLMProvider for Pass 2, optional)
    - ctxmtg.models.profile (DomainProfile for Pass 3)
    - ctxmtg.exceptions (QueryError for error reporting)

Used by:
    - ctxmtg.query.executor (optional briefing generation)
    - ctxmtg.query.synthesizer (Pass 4 content for synthesis)
    - tests/test_query/test_briefing.py
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from ctxmtg.interfaces.storage import SQLStore

# ---------------------------------------------------------------
# Module logger -- logs briefing generation stats.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.query.briefing")


# =====================================================================
# Profiling SQL Queries
# =====================================================================
# These are the 6 Pass 1 profiling queries from the sql-briefing-strategy
# research note. Each query returns aggregated data that compresses
# into a few tokens regardless of the underlying data volume.
#
# All queries accept a list of entity IDs as a filter parameter.
# If no entity IDs are provided, the queries operate on all data.
# =====================================================================

# 1a. Core statistics: overview of the fact store
CORE_STATS_SQL = """\
SELECT
    COUNT(*) as total_facts,
    COUNT(CASE WHEN superseded_by IS NULL THEN 1 END) as active_facts,
    COUNT(CASE WHEN superseded_by IS NOT NULL THEN 1 END) as superseded_facts,
    MIN(f.created_at) as earliest,
    MAX(f.created_at) as latest,
    AVG(f.confidence) as avg_confidence,
    MIN(f.confidence) as min_confidence,
    MAX(f.confidence) as max_confidence
FROM facts f
"""

# 1b. Entity distribution: top entities by fact count
ENTITY_DISTRIBUTION_SQL = """\
SELECT
    e.name,
    e.entity_type,
    COUNT(f.id) as fact_count,
    AVG(f.confidence) as avg_conf
FROM entities e
JOIN facts f ON f.subject_entity_id = e.id OR f.object_entity_id = e.id
GROUP BY e.id, e.name, e.entity_type
ORDER BY fact_count DESC
LIMIT 15
"""

# 1c. Predicate distribution: most common predicates
PREDICATE_DISTRIBUTION_SQL = """\
SELECT
    predicate,
    COUNT(*) as cnt,
    AVG(confidence) as avg_conf
FROM facts
GROUP BY predicate
ORDER BY cnt DESC
LIMIT 10
"""

# 1d. Temporal distribution: facts per week
TEMPORAL_DISTRIBUTION_SQL = """\
SELECT
    strftime('%Y-%W', created_at) as week,
    COUNT(*) as fact_count
FROM facts
GROUP BY week
ORDER BY week
"""

# 1e. Source distribution: interaction count per source type
SOURCE_DISTRIBUTION_SQL = """\
SELECT
    i.source_type,
    COUNT(DISTINCT i.id) as interaction_count
FROM interactions i
JOIN facts f ON f.interaction_id = i.id
GROUP BY i.source_type
"""

# 1f. Supersession summary: recent fact changes (what was updated)
SUPERSESSION_SUMMARY_SQL = """\
SELECT
    f_old.object_literal as old_value,
    f_new.object_literal as new_value,
    f_old.predicate,
    f_new.created_at as changed_on
FROM facts f_old
JOIN facts f_new ON f_old.superseded_by = f_new.id
ORDER BY f_new.created_at DESC
LIMIT 5
"""

# =====================================================================
# Profiling query registry: ordered list of (name, sql, formatter).
# The builder iterates this list sequentially, executing each query
# and formatting the result into a briefing section.
# =====================================================================
PROFILING_QUERIES: list[tuple[str, str]] = [
    ("core_stats", CORE_STATS_SQL),
    ("entity_distribution", ENTITY_DISTRIBUTION_SQL),
    ("predicate_distribution", PREDICATE_DISTRIBUTION_SQL),
    ("temporal_distribution", TEMPORAL_DISTRIBUTION_SQL),
    ("source_distribution", SOURCE_DISTRIBUTION_SQL),
    ("supersession_summary", SUPERSESSION_SUMMARY_SQL),
]


class SQLBriefingBuilder:
    """
    Builds a statistical briefing from SQL profiling queries.

    Instead of feeding raw SQL rows to the LLM (which would overflow
    the context window), this builder runs 6 profiling queries and
    compresses the results into a ~200-300 token briefing.

    The briefing is a structured text block with sections for each
    profiling query. Sections that return no data are marked as
    "[no data]" rather than omitted, so the LLM (Phase 2) knows
    what information is missing.

    Phase 1: Only Pass 1 profiling queries are implemented.
    Phase 2: Adds LLM-designed targeted queries (Pass 2),
             domain-specific profiling (Pass 3), and raw content
             retrieval (Pass 4).

    Usage:
        builder = SQLBriefingBuilder()
        briefing = await builder.build_briefing(
            sql_store=sqlite_store,
            query_terms=["OAuth2", "Alice"],
        )
        # Returns a ~200-300 token structured text block
    """

    async def build_briefing(
        self,
        sql_store: SQLStore,
        query_terms: list[str] | None = None,
        token_budget: int = 300,
    ) -> str:
        """
        Run Pass 1 profiling queries and format as a briefing.

        Executes each profiling query sequentially. Each query may
        return data or null (empty result set). The results are
        formatted into a structured text briefing suitable for
        LLM consumption (Phase 2) or human review.

        The sequential execution is intentional: each query is cheap
        (SQL aggregation), and sequential execution avoids connection
        contention on SQLite's single-writer lock.

        Args:
            sql_store: The SQL store to query against.
            query_terms: Optional list of entity names / keywords to
                         focus the briefing on. Currently used for
                         logging only; the queries operate on all data.
                         Phase 2 adds per-entity filtering.
            token_budget: Target maximum tokens for the briefing.
                          Used to decide truncation strategy.

        Returns:
            Formatted briefing string ready for LLM consumption.
            Includes all 6 sections, with null sections preserved.
        """
        sections: list[str] = []

        # Run each profiling query sequentially
        for query_name, sql in PROFILING_QUERIES:
            try:
                rows = await sql_store.execute_sql(sql)
                # Format the results into a briefing section.
                # Null-safe: if no rows returned, the section shows "[no data]"
                section = self._format_section(query_name, rows)
                sections.append(section)
            except Exception as exc:
                # If a query fails, include an error marker rather than
                # skipping the section. The LLM should know about failures.
                sections.append(f"{self._section_header(query_name)}\n  [query failed: {exc}]")
                logger.warning(
                    "briefing_query_failed",
                    error_code="CTXMTG-QRY-001",
                    query_name=query_name,
                    error=str(exc),
                )

        # Assemble the full briefing
        briefing = "=== SQL BRIEFING ===\n\n" + "\n\n".join(sections) + "\n\n=== END BRIEFING ==="

        logger.info(
            "briefing_generated",
            section_count=len(sections),
            char_count=len(briefing),
            query_terms=query_terms or [],
        )

        return briefing

    def _format_section(self, query_name: str, rows: list[dict[str, Any]]) -> str:
        """
        Format a profiling query result into a briefing section.

        Each query type has its own formatting logic to produce a
        compact, LLM-readable representation. If no rows were returned,
        the section shows "[no data]".

        Args:
            query_name: The name of the profiling query (e.g., "core_stats").
            rows: The query result rows as dicts.

        Returns:
            Formatted section string.
        """
        header = self._section_header(query_name)

        # Null-safe: no data returned
        if not rows:
            return f"{header}\n  [no data]"

        # Dispatch to the appropriate formatter based on query name
        formatter = getattr(self, f"_format_{query_name}", None)
        if formatter:
            return f"{header}\n{formatter(rows)}"

        # Fallback: generic row formatting
        return f"{header}\n{self._format_generic(rows)}"

    @staticmethod
    def _section_header(query_name: str) -> str:
        """
        Convert a query name to a human-readable section header.

        Args:
            query_name: The snake_case query name.

        Returns:
            A formatted header string (e.g., "CORE STATS:").
        """
        return query_name.upper().replace("_", " ") + ":"

    @staticmethod
    def _format_core_stats(rows: list[dict[str, Any]]) -> str:
        """
        Format core statistics into a compact briefing section.

        Shows total/active/superseded fact counts, time range, and
        confidence statistics in a single line.
        """
        if not rows:
            return "  [no data]"

        row = rows[0]
        total = row.get("total_facts", 0)
        active = row.get("active_facts", 0)
        superseded = row.get("superseded_facts", 0)
        earliest = row.get("earliest", "N/A")
        latest = row.get("latest", "N/A")
        avg_conf = row.get("avg_confidence")
        min_conf = row.get("min_confidence")
        max_conf = row.get("max_confidence")

        # Format confidence as rounded values
        conf_str = "N/A"
        if avg_conf is not None:
            conf_parts = [f"avg {avg_conf:.2f}"]
            if min_conf is not None and max_conf is not None:
                conf_parts.append(f"({min_conf:.2f} - {max_conf:.2f})")
            conf_str = " ".join(conf_parts)

        # Truncate timestamps to date only for compactness
        earliest_date = str(earliest)[:10] if earliest else "N/A"
        latest_date = str(latest)[:10] if latest else "N/A"

        lines = [
            f"  Facts: {total} total ({active} active, {superseded} superseded)",
            f"  Time range: {earliest_date} to {latest_date}",
            f"  Confidence: {conf_str}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_entity_distribution(rows: list[dict[str, Any]]) -> str:
        """
        Format entity distribution as a compact list.

        Shows entity name, fact count, and average confidence for
        the top entities (limited to 8 for brevity).
        """
        if not rows:
            return "  [no data]"

        # Limit to top 8 entities for token budget
        limited = rows[:8]
        parts: list[str] = []
        for row in limited:
            name = row.get("name", "?")
            count = row.get("fact_count", 0)
            conf = row.get("avg_conf")
            conf_str = f"{conf:.2f}" if conf is not None else "N/A"
            parts.append(f"{name} ({count} facts, {conf_str})")

        return "  " + " | ".join(parts)

    @staticmethod
    def _format_predicate_distribution(rows: list[dict[str, Any]]) -> str:
        """
        Format predicate distribution as a compact list.

        Shows predicate name and count for the top predicates.
        """
        if not rows:
            return "  [no data]"

        parts: list[str] = []
        for row in rows:
            predicate = row.get("predicate", "?")
            cnt = row.get("cnt", 0)
            parts.append(f"{predicate}: {cnt}x")

        return "  " + " | ".join(parts)

    @staticmethod
    def _format_temporal_distribution(rows: list[dict[str, Any]]) -> str:
        """
        Format temporal distribution showing facts per week.

        Useful for trend detection: the LLM can see if activity
        is increasing, decreasing, or stable.
        """
        if not rows:
            return "  [no data]"

        parts: list[str] = []
        for row in rows:
            week = row.get("week", "?")
            count = row.get("fact_count", 0)
            parts.append(f"{week}: {count}")

        return "  " + ", ".join(parts)

    @staticmethod
    def _format_source_distribution(rows: list[dict[str, Any]]) -> str:
        """
        Format source distribution showing interaction counts by type.
        """
        if not rows:
            return "  [no data]"

        parts: list[str] = []
        for row in rows:
            source = row.get("source_type", "?")
            count = row.get("interaction_count", 0)
            parts.append(f"{source} ({count})")

        return "  " + ", ".join(parts)

    @staticmethod
    def _format_supersession_summary(rows: list[dict[str, Any]]) -> str:
        """
        Format supersession summary showing recent fact changes.

        Shows what values were updated and when. Critical for the LLM
        to understand which information is current vs outdated.
        """
        if not rows:
            return "  [no data]"

        parts: list[str] = []
        for row in rows:
            predicate = row.get("predicate", "?")
            old_val = row.get("old_value", "?")
            new_val = row.get("new_value", "?")
            changed = str(row.get("changed_on", "?"))[:10]
            parts.append(f"{predicate}: {old_val} → {new_val} ({changed})")

        return "  " + "\n  ".join(parts)

    @staticmethod
    def _format_generic(rows: list[dict[str, Any]]) -> str:
        """
        Generic fallback formatter for any profiling query result.

        Formats rows as key-value pairs, one per line. Used when
        no specific formatter exists for a query type.
        """
        lines: list[str] = []
        for row in rows[:10]:  # Limit to 10 rows max
            parts = [f"{k}={v}" for k, v in row.items() if v is not None]
            lines.append("  " + ", ".join(parts))
        return "\n".join(lines)

    # =================================================================
    # Pass 2: LLM-Designed Targeted Queries
    # =================================================================

    async def execute_llm_queries(
        self,
        sql_store: SQLStore,
        queries: list[dict[str, str]],
        row_limit: int = 20,
        timeout_seconds: float = 2.0,
    ) -> list[dict[str, Any]]:
        """
        Safely execute LLM-designed queries (Pass 2) with guardrails.

        The LLM designs targeted SQL queries based on the Pass 1
        briefing. Each query is validated against safety rules before
        execution:
            1. Only allowlisted tables (facts, entities, interactions,
               meta_insights)
            2. No DDL (CREATE, DROP, ALTER, INSERT, UPDATE, DELETE)
            3. Mandatory LIMIT clause (enforced if omitted)
            4. Read-only execution
            5. Per-query timeout (default 2 seconds)

        Args:
            sql_store: The SQL store to execute queries against.
            queries: List of {"purpose": str, "sql": str} dicts from
                     the LLM. Each dict has a purpose description and
                     the SQL query string.
            row_limit: Maximum rows per query. If the LLM's query has
                       no LIMIT, one is appended with this value.
            timeout_seconds: Per-query timeout in seconds.

        Returns:
            List of {"purpose": str, "rows": list[dict], "error": str|None}
            results. Each entry has either rows or an error message.
        """
        results: list[dict[str, Any]] = []

        for query_spec in queries:
            purpose = str(query_spec.get("purpose", "unknown"))
            sql = str(query_spec.get("sql", ""))

            # ---------------------------------------------------------------
            # Validate the SQL against safety guardrails.
            # ---------------------------------------------------------------
            validation_error = validate_sql_safety(sql)
            if validation_error:
                logger.warning(
                    "llm_query_rejected",
                    error_code="CTXMTG-QRY-006",
                    purpose=purpose,
                    error=validation_error,
                    sql=sql[:200],
                )
                results.append({
                    "purpose": purpose,
                    "rows": [],
                    "error": f"Rejected: {validation_error}",
                })
                continue

            # ---------------------------------------------------------------
            # Enforce LIMIT clause if missing.
            # ---------------------------------------------------------------
            sql = enforce_limit(sql, row_limit)

            # ---------------------------------------------------------------
            # Execute the query with timeout protection.
            # ---------------------------------------------------------------
            try:
                rows = await sql_store.execute_sql(sql)
                # Enforce row_limit even if DB returns more
                rows = rows[:row_limit]

                logger.info(
                    "llm_query_executed",
                    purpose=purpose,
                    row_count=len(rows),
                )
                results.append({
                    "purpose": purpose,
                    "rows": rows,
                    "error": None,
                })
            except Exception as exc:
                logger.warning(
                    "llm_query_failed",
                    error_code="CTXMTG-QRY-001",
                    purpose=purpose,
                    error=str(exc),
                    sql=sql[:200],
                )
                results.append({
                    "purpose": purpose,
                    "rows": [],
                    "error": f"Execution failed: {exc}",
                })

        return results

    def format_pass2_results(self, pass2_results: list[dict[str, Any]]) -> str:
        """
        Format Pass 2 LLM query results into a briefing section.

        Each query result is formatted with its purpose header and
        the returned rows in a compact format.

        Args:
            pass2_results: Results from execute_llm_queries().

        Returns:
            Formatted Pass 2 briefing section string.
        """
        if not pass2_results:
            return ""

        sections: list[str] = ["TARGETED QUERIES:"]

        for result in pass2_results:
            purpose = result.get("purpose", "unknown")
            error = result.get("error")
            rows = result.get("rows", [])

            if error:
                sections.append(f"  [{purpose}]: {error}")
            elif not rows:
                sections.append(f"  [{purpose}]: [no data]")
            else:
                sections.append(f"  [{purpose}]:")
                for row in rows[:10]:
                    parts = [f"{k}={v}" for k, v in row.items() if v is not None]
                    sections.append("    " + ", ".join(parts))

        return "\n".join(sections)

    # =================================================================
    # Pass 3: Domain-Specific Profiling
    # =================================================================

    async def execute_domain_queries(
        self,
        sql_store: SQLStore,
        profile_data: dict[str, Any],
    ) -> str:
        """
        Execute domain-specific profiling queries (Pass 3).

        Reads the sql_profiling section from the domain profile data
        and executes the configured pass_1_extensions queries. Each
        query has a name, SQL, and briefing_format template.

        Args:
            sql_store: The SQL store to execute queries against.
            profile_data: The domain profile's raw data dict. Expected
                          to have a "sql_profiling" section with
                          "pass_1_extensions" list.

        Returns:
            Formatted domain-specific briefing section string, or
            empty string if no sql_profiling section exists.
        """
        # Check if the profile has a sql_profiling section
        sql_profiling = profile_data.get("sql_profiling")
        if not sql_profiling:
            return ""

        extensions = sql_profiling.get("pass_1_extensions", [])
        if not extensions:
            return ""

        sections: list[str] = []
        domain_name = profile_data.get("name", "unknown")
        sections.append(f"DOMAIN-SPECIFIC [{domain_name}]:")

        for ext in extensions:
            name = ext.get("name", "unknown")
            sql = ext.get("sql", "")
            briefing_format = ext.get("briefing_format", "{}")

            if not sql:
                continue

            # Validate the SQL for safety
            validation_error = validate_sql_safety(sql)
            if validation_error:
                sections.append(f"  {name}: [rejected: {validation_error}]")
                continue

            # Enforce LIMIT
            sql = enforce_limit(sql, 10)

            try:
                rows = await sql_store.execute_sql(sql)
                if not rows:
                    sections.append(f"  {name}: [no data]")
                else:
                    formatted_rows: list[str] = []
                    for row in rows[:5]:
                        try:
                            formatted_rows.append(briefing_format.format(**row))
                        except (KeyError, IndexError):
                            # If format fails, fall back to generic
                            parts = [f"{k}={v}" for k, v in row.items() if v is not None]
                            formatted_rows.append(", ".join(parts))
                    sections.append(f"  {name}: " + " | ".join(formatted_rows))
            except Exception as exc:
                logger.warning(
                    "domain_query_failed",
                    error_code="CTXMTG-QRY-001",
                    error=str(exc),
                )
                sections.append(f"  {name}: [query failed: {exc}]")

        return "\n".join(sections) if len(sections) > 1 else ""

    # =================================================================
    # Pass 4: Raw Content Retrieval
    # =================================================================

    async def fetch_content(
        self,
        sql_store: SQLStore,
        interaction_ids: list[str],
        max_interactions: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Fetch full interaction content for synthesis (Pass 4).

        Only called at the synthesis stage (not at the bridge stage).
        Retrieves the full text of the most relevant interactions so
        the LLM can quote directly from source material.

        Args:
            sql_store: The SQL store to fetch content from.
            interaction_ids: List of interaction IDs to fetch. Only
                             the first max_interactions are fetched.
            max_interactions: Maximum number of interactions to fetch
                              (default 5, keeps context manageable).

        Returns:
            List of dicts with keys: id, content, title, created_at.
            Empty dicts are omitted if the interaction was not found.
        """
        # Limit to max_interactions to avoid context overflow
        ids_to_fetch = interaction_ids[:max_interactions]
        results: list[dict[str, Any]] = []

        for iid in ids_to_fetch:
            try:
                interaction = await sql_store.get_interaction(iid)
                if interaction:
                    results.append({
                        "id": interaction.id,
                        "content": interaction.content,
                        "title": interaction.title or "",
                        "created_at": (
                            str(interaction.created_at)[:10]
                            if interaction.created_at
                            else ""
                        ),
                    })
            except Exception as exc:
                logger.warning(
                    "content_fetch_failed",
                    error_code="CTXMTG-QRY-001",
                    error=str(exc),
                )

        logger.info(
            "pass4_content_fetched",
            requested=len(ids_to_fetch),
            fetched=len(results),
        )

        return results

    def format_pass4_results(self, content_results: list[dict[str, Any]]) -> str:
        """
        Format Pass 4 raw content results for the synthesis prompt.

        Each interaction is formatted with its title, date, and
        full content text.

        Args:
            content_results: Results from fetch_content().

        Returns:
            Formatted Pass 4 content section string.
        """
        if not content_results:
            return ""

        sections: list[str] = ["RAW CONTENT:"]

        for result in content_results:
            title = result.get("title", "Untitled")
            date = result.get("created_at", "")
            content = result.get("content", "")
            # Truncate very long content to stay within context budget
            if len(content) > 2000:
                content = content[:2000] + "... [truncated]"

            sections.append(f"  [{title}] ({date}):")
            sections.append(f"    {content}")

        return "\n".join(sections)


# =====================================================================
# SQL Safety Validation
# =====================================================================
# These functions validate LLM-generated SQL against safety rules.
# They prevent the LLM from executing dangerous queries (DDL, writes)
# or accessing unauthorized tables.
# =====================================================================

# ---------------------------------------------------------------
# Allowlisted tables: only these tables can be queried by LLM SQL.
# System tables (sync_log, embeddings_metadata, etc.) are blocked.
# ---------------------------------------------------------------
ALLOWED_TABLES: frozenset[str] = frozenset({
    "facts",
    "entities",
    "interactions",
    "meta_insights",
})

# ---------------------------------------------------------------
# DDL keywords that are blocked in LLM-generated SQL.
# Any query containing these keywords (case-insensitive) is rejected.
# ---------------------------------------------------------------
DDL_KEYWORDS: frozenset[str] = frozenset({
    "CREATE",
    "DROP",
    "ALTER",
    "INSERT",
    "UPDATE",
    "DELETE",
    "TRUNCATE",
    "REPLACE",
    "MERGE",
    "GRANT",
    "REVOKE",
})

# ---------------------------------------------------------------
# Compiled regex for detecting DDL keywords as whole words.
# Uses word boundaries to avoid false positives (e.g., "created_at"
# should not match "CREATE").
# ---------------------------------------------------------------
_DDL_PATTERN = re.compile(
    r"\b(" + "|".join(DDL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------
# Compiled regex for extracting table references from SQL.
# Matches table names after FROM and JOIN keywords.
# ---------------------------------------------------------------
_TABLE_PATTERN = re.compile(
    r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------
# Compiled regex for detecting LIMIT clause in SQL.
# ---------------------------------------------------------------
_LIMIT_PATTERN = re.compile(
    r"\bLIMIT\s+\d+",
    re.IGNORECASE,
)


def validate_sql_safety(sql: str) -> str | None:
    """
    Validate an LLM-generated SQL query against safety rules.

    Checks for:
        1. DDL keywords (CREATE, DROP, ALTER, INSERT, UPDATE, DELETE, etc.)
        2. Non-allowlisted tables
        3. Empty queries
        4. Multiple statements (semicolons)

    Args:
        sql: The SQL query string to validate.

    Returns:
        None if the query is safe, or an error message string
        describing why it was rejected.
    """
    # ---------------------------------------------------------------
    # Rule 1: Reject empty or whitespace-only queries.
    # ---------------------------------------------------------------
    stripped = sql.strip()
    if not stripped:
        return "Empty query"

    # ---------------------------------------------------------------
    # Rule 2: Reject multiple statements (semicolons). LLM should
    # only generate single SELECT statements.
    # ---------------------------------------------------------------
    # Remove trailing semicolons and check for embedded ones
    cleaned = stripped.rstrip(";")
    if ";" in cleaned:
        return "Multiple statements not allowed (found semicolon)"

    # ---------------------------------------------------------------
    # Rule 3: Reject DDL keywords. Any DDL keyword as a whole word
    # is blocked. We use word boundaries to avoid false positives.
    # ---------------------------------------------------------------
    ddl_match = _DDL_PATTERN.search(sql)
    if ddl_match:
        return f"DDL keyword not allowed: {ddl_match.group(1).upper()}"

    # ---------------------------------------------------------------
    # Rule 4: Check for non-allowlisted tables. Extract all table
    # references from FROM and JOIN clauses and verify each one.
    # ---------------------------------------------------------------
    tables = _TABLE_PATTERN.findall(sql)
    for table in tables:
        if table.lower() not in ALLOWED_TABLES:
            return f"Table not allowed: {table}"

    # ---------------------------------------------------------------
    # Query passed all checks.
    # ---------------------------------------------------------------
    return None


def enforce_limit(sql: str, limit: int = 20) -> str:
    """
    Ensure a SQL query has a LIMIT clause.

    If the query already has a LIMIT clause, it is left unchanged.
    If not, a LIMIT clause is appended.

    Args:
        sql: The SQL query string.
        limit: The maximum number of rows (default 20).

    Returns:
        The SQL query with a guaranteed LIMIT clause.
    """
    # Check if the query already has a LIMIT clause
    if _LIMIT_PATTERN.search(sql):
        return sql

    # Strip trailing whitespace and semicolons
    stripped = sql.rstrip().rstrip(";").rstrip()

    return f"{stripped} LIMIT {limit}"
