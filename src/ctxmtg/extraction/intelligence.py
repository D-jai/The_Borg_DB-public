# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Intelligence Context for Extraction Pipeline
==============================================

Provides cross-instance intelligence context to the extraction pipeline.
When the hive has distilled entity summaries (from the DistillerStage),
this module formats them into human-readable prompt context that the
LLM verifier can use to make better extraction decisions.

The IntelligenceContext class holds:
    - ``hive_hints``: pre-fetched intelligence hints (entity_name → dict)
      pulled from the local_intelligence_cache by IntelligencePullWorker.
    - ``batch_entities``: rolling context of entities discovered during
      the current extraction session (lightweight: name + type + count).

When ``build_prompt_context()`` is called, it produces a block of text
like:

    Known intelligence (from prior analysis):
    - Alice: person. Predicates: proposed, decided, approved.
      Co-occurs with: Bob, OAuth2. Active in 5 interactions.
    - Bob: person. Predicates: raised_concern, suggested.
      Co-occurs with: Alice, Redis.

This context is injected into the extraction prompt template via the
``{{intelligence_context}}`` slot.

Graceful degradation:
    - If no hive hints exist, ``build_prompt_context()`` returns an
      empty string and the extraction pipeline works identically to
      Phase 3 (no intelligence enrichment).
    - The ``max_prompt_chars`` cap prevents the intelligence context
      from consuming too much of the LLM's context window.

Depends on:
    - ctxmtg.models.interaction (Entity model for batch entity tracking)

Used by:
    - ctxmtg.extraction.pipeline (builds and injects intelligence context)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from ctxmtg.models.interaction import Entity

# ---------------------------------------------------------------
# Module-level logger -- structured JSON output, no PII in logs.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.extraction.intelligence")


class IntelligenceContext:
    """
    Holds and formats cross-instance intelligence for extraction.

    Constructed with hive hints (a dict of entity_name → hint dict)
    and a character budget for prompt injection.  Provides methods to:

    - Build a human-readable intelligence prompt section.
    - Track entities discovered during the current batch (rolling
      context across chunks).
    - Look up enrichment data for individual entities.

    Usage:
        ctx = IntelligenceContext(hive_hints=hints, max_prompt_chars=2000)
        prompt_section = ctx.build_prompt_context(["Alice", "Bob"])
        ctx.add_batch_entities(merged_entities)
        enrichment = ctx.get_entity_context_enrichment("Alice")
    """

    def __init__(
        self,
        hive_hints: dict[str, dict],
        max_prompt_chars: int = 2000,
    ) -> None:
        """
        Initialise the intelligence context.

        Args:
            hive_hints:      Dict of entity_name → hint dict pulled from
                             the local_intelligence_cache.  Each hint has:
                             summary, top_co_entities, top_predicates,
                             relevance_score.
            max_prompt_chars: Maximum character budget for the intelligence
                             context section in the LLM prompt.  Longer
                             contexts are truncated, prioritising by
                             relevance_score.  Default 2000.
        """
        # Store the hive hints dict for quick lookups.
        self._hive_hints: dict[str, dict] = hive_hints or {}

        # Character budget for prompt injection.  Prevents the
        # intelligence context from consuming too much LLM context.
        self._max_prompt_chars = max_prompt_chars

        # Rolling batch entities: tracks entities discovered during
        # the current extraction session.  Lightweight representation:
        # name → {type, mention_count}.  Updated by add_batch_entities().
        self._batch_entities: dict[str, dict[str, Any]] = {}

    # =================================================================
    # build_prompt_context: format intelligence for LLM injection
    # =================================================================

    def build_prompt_context(self, current_entity_names: list[str]) -> str:
        """
        Build a human-readable intelligence context string.

        Collects hints for the given entity names (from hive hints)
        and rolling batch entities.  Formats each hint as a concise
        one-liner with type, predicates, co-entities, and relevance.

        The output is truncated to ``max_prompt_chars``, prioritising
        entities with higher relevance scores.

        Args:
            current_entity_names: Entity names found in the current
                                  chunk or interaction.  Used to select
                                  relevant hints from the hive cache.

        Returns:
            A formatted string for prompt injection, or empty string
            if no intelligence is available.
        """
        # ----------------------------------------------------------
        # Step 1: Collect relevant hints.
        # Merge hive hints for current entities + batch entities.
        # ----------------------------------------------------------
        hint_entries: list[tuple[str, dict, float]] = []

        # Gather hive hints for explicitly requested entity names.
        all_names = set(current_entity_names)

        # Also include batch entities (from earlier chunks in session).
        all_names.update(self._batch_entities.keys())

        for name in all_names:
            # Check hive hints first (richer data).
            if name in self._hive_hints:
                hint = self._hive_hints[name]
                relevance = hint.get("relevance_score", 0.0)
                hint_entries.append((name, hint, relevance))
            elif name in self._batch_entities:
                # Batch entities are lightweight -- build a minimal hint.
                batch = self._batch_entities[name]
                lightweight_hint = {
                    "summary": "",
                    "top_co_entities": [],
                    "top_predicates": [],
                    "relevance_score": 0.01,
                    "entity_type": batch.get("entity_type", "unknown"),
                    "mention_count": batch.get("mention_count", 1),
                }
                hint_entries.append((name, lightweight_hint, 0.01))

        # No intelligence available -- return empty string.
        if not hint_entries:
            return ""

        # ----------------------------------------------------------
        # Step 2: Sort by relevance score descending (most important
        # entities first) to prioritise when truncating.
        # ----------------------------------------------------------
        hint_entries.sort(key=lambda x: x[2], reverse=True)

        # ----------------------------------------------------------
        # Step 3: Build the formatted text lines.
        # Each entity gets a concise one-liner with its key attributes.
        # ----------------------------------------------------------
        lines: list[str] = ["Known intelligence (from prior analysis):"]
        total_chars = len(lines[0])

        for name, hint, _rel in hint_entries:
            line = _format_hint_line(name, hint)

            # Check if adding this line would exceed the budget.
            # +1 for the newline character between lines.
            if total_chars + len(line) + 1 > self._max_prompt_chars:
                break

            lines.append(line)
            total_chars += len(line) + 1

        # If only the header made it (no entity lines), return empty.
        if len(lines) <= 1:
            return ""

        return "\n".join(lines)

    # =================================================================
    # add_batch_entities: rolling context from current session
    # =================================================================

    def add_batch_entities(self, entities: list[Entity]) -> None:
        """
        Add entities from the current extraction batch to rolling context.

        These lightweight entries (name, type, mention_count) let later
        chunks in the same session know about entities discovered in
        earlier chunks -- even without full hive intelligence.

        Modifies internal state.  Accumulates across calls (e.g. one
        call per chunk in a multi-chunk interaction).

        Args:
            entities: List of Entity objects from the current batch.
                      Each contributes name + type to the rolling context.
        """
        for entity in entities:
            name = entity.name
            if name in self._batch_entities:
                # Increment mention count for repeated entities.
                self._batch_entities[name]["mention_count"] += 1
            else:
                # New entity -- record type and initial count.
                self._batch_entities[name] = {
                    "entity_type": entity.entity_type.value
                    if hasattr(entity.entity_type, "value")
                    else str(entity.entity_type),
                    "mention_count": 1,
                }

    # =================================================================
    # get_entity_context_enrichment: per-entity hint lookup
    # =================================================================

    def get_entity_context_enrichment(self, entity_name: str) -> dict | None:
        """
        Look up a hive intelligence hint for a specific entity.

        Used by ``_build_entity_context`` in the extraction pipeline
        to prepend hive knowledge to an entity's context dict.

        Args:
            entity_name: The entity name to look up.

        Returns:
            The hint dict (summary, top_co_entities, top_predicates,
            relevance_score) if a hint exists, or None if not found.
        """
        return self._hive_hints.get(entity_name, None)


# =====================================================================
# Private helpers
# =====================================================================


def _format_hint_line(name: str, hint: dict) -> str:
    """
    Format a single entity hint as a concise one-liner for prompt injection.

    The output looks like:
        - Alice: person. Predicates: proposed, decided. Co-occurs with: Bob.
    or for lightweight batch entities:
        - Alice: person. Seen in current session (1 mention).

    Args:
        name: The entity name.
        hint: The hint dict with summary, predicates, co-entities, etc.

    Returns:
        A formatted string line starting with "- ".
    """
    # Determine the entity type from the hint.
    entity_type = hint.get("entity_type", "unknown")

    # Start with the entity name and type.
    parts = [f"- {name}: {entity_type}."]

    # Add predicates if available.
    predicates = hint.get("top_predicates", [])
    if predicates:
        pred_str = ", ".join(predicates[:5])
        parts.append(f"Predicates: {pred_str}.")

    # Add co-entities if available.
    co_entities = hint.get("top_co_entities", [])
    if co_entities:
        co_str = ", ".join(co_entities[:5])
        parts.append(f"Co-occurs with: {co_str}.")

    # Add mention count if this is a batch entity (no predicates/co-entities).
    mention_count = hint.get("mention_count")
    if mention_count and not predicates and not co_entities:
        parts.append(f"Seen in current session ({mention_count} mention(s)).")

    # If there's a non-empty summary, add a snippet (truncated to 80 chars).
    summary = hint.get("summary", "")
    if summary and not mention_count:
        snippet = summary[:80] + "..." if len(summary) > 80 else summary
        parts.append(f"Summary: {snippet}")

    return " ".join(parts)
