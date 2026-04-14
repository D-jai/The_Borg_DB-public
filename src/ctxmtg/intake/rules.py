# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Rule-Based Intake Gateway (Traffic Cop)
========================================

This module implements the IntakeGateway interface using rule-based
pattern matching from the domain profile's intake section. It is the
Phase 1 Traffic Cop -- simple YAML rules, no LLM.

The Traffic Cop does two jobs:
    1. **Classify:** Evaluate rules in order (REJECT → DEFER → ACCEPT)
       to decide whether the interaction should be processed, queued,
       or discarded. First match wins. Default is ACCEPT.
    2. **Transform:** Replace heavy content (email attachments, code
       blocks, large inline content) with lightweight metadata stubs
       BEFORE the extraction pipeline sees the interaction.

Rule evaluation order:
    REJECT rules → DEFER rules → ACCEPT rules → default ACCEPT.

Content transformation:
    - Email attachments → [ATTACHMENT: name, type, size]
    - Code blocks (``` or indented) → [CODE_BLOCK: language, line_count]
    - Inline content > threshold → [TRUNCATED: original_size]

This runs BEFORE extraction, so the NLP pipeline never wastes time
processing binary attachments, large code dumps, or spam.

Depends on:
    - fnmatch (glob-style pattern matching for sender/subject)
    - re (regex for code block detection)
    - ctxmtg.interfaces.intake (IntakeGateway ABC)
    - ctxmtg.models.interaction (Interaction, IntakeAction)
    - ctxmtg.models.profile (IntakeConfig, IntakeRule)

Used by:
    - ctxmtg.ingestion.worker (calls before extraction pipeline)
    - ctxmtg.cli (runs Traffic Cop on each ingested interaction)
"""

from __future__ import annotations

import fnmatch
import re
from copy import deepcopy

import structlog

from ctxmtg.interfaces.intake import IntakeGateway
from ctxmtg.models.interaction import IntakeAction, Interaction
from ctxmtg.models.profile import IntakeConfig, IntakeRule

# ---------------------------------------------------------------
# Module-level logger -- logs classification decisions, not content.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.intake.rules")

# ---------------------------------------------------------------
# Regex for detecting fenced code blocks (```language ... ```)
# and indented code blocks (4+ spaces or tab at line start).
# ---------------------------------------------------------------
_FENCED_CODE_RE = re.compile(
    r"```(\w*)\n(.*?)```",
    re.DOTALL,
)

# Indented code blocks: lines starting with 4+ spaces or a tab
_INDENTED_CODE_RE = re.compile(
    r"(?:^(?:    |\t).+\n?){2,}",
    re.MULTILINE,
)


class RuleBasedIntakeGateway(IntakeGateway):
    """
    Rule-based Traffic Cop implementation.

    Evaluates YAML-defined rules against incoming interactions to
    classify them (ACCEPT/DEFER/REJECT) and transform their content
    by replacing heavy elements with metadata stubs.

    The intake configuration comes from the domain profile's intake
    section, which defines reject, defer, and accept rules, plus
    content transformation limits.

    Usage:
        from ctxmtg.models.profile import IntakeConfig

        config = IntakeConfig(
            reject=[IntakeRule(sender_pattern="*@noreply.*")],
            defer=[IntakeRule(cc_only=True)],
            accept=[IntakeRule(contains_keywords=["deadline"])],
        )
        gateway = RuleBasedIntakeGateway(config)
        action, modified = gateway.process(interaction)
    """

    def __init__(self, config: IntakeConfig) -> None:
        """
        Initialize the gateway with intake configuration.

        Args:
            config: The IntakeConfig from a domain profile, containing
                    reject/defer/accept rules and content limits.
        """
        self._config = config

        # Track intake statistics for reporting
        self._stats: dict[str, int] = {
            "accept": 0,
            "defer": 0,
            "reject": 0,
            "route": 0,
        }

        logger.info(
            "intake_gateway_initialized",
            mode=config.mode,
            reject_rules=len(config.reject),
            defer_rules=len(config.defer),
            accept_rules=len(config.accept),
        )

    def process(
        self, interaction: Interaction
    ) -> tuple[IntakeAction, Interaction]:
        """
        Classify and transform an inbound interaction.

        Steps:
        1. If mode is "passthrough", accept everything with no transform.
        2. Apply content transformations (attachment stubs, code stubs, truncation).
        3. Evaluate rules: REJECT → DEFER → ACCEPT → default ACCEPT.
        4. Update the interaction's intake_action field.
        5. Update intake statistics.

        Args:
            interaction: The inbound Interaction to classify.

        Returns:
            A tuple of (IntakeAction, modified Interaction).
        """
        # Passthrough mode: accept everything, no transformation
        if self._config.mode == "passthrough":
            self._stats["accept"] += 1
            return (IntakeAction.ACCEPT, interaction)

        # ---------------------------------------------------------------
        # Step 1: Apply content transformations.
        # Create a deep copy so we don't modify the original interaction.
        # ---------------------------------------------------------------
        modified = deepcopy(interaction)
        modified = self._transform_content(modified)

        # ---------------------------------------------------------------
        # Step 2: Evaluate classification rules in priority order.
        # First match wins. Order: REJECT → DEFER → ACCEPT → default.
        # ---------------------------------------------------------------
        action = self._classify(modified)

        # ---------------------------------------------------------------
        # Step 3: Update the interaction's intake_action field.
        # ---------------------------------------------------------------
        modified.intake_action = action

        # ---------------------------------------------------------------
        # Step 4: Update statistics.
        # ---------------------------------------------------------------
        self._stats[action.value] += 1

        logger.info(
            "intake_classified",
            interaction_id=interaction.id,
            action=action.value,
            source_type=interaction.source_type.value,
        )

        return (action, modified)

    @property
    def stats(self) -> dict[str, int]:
        """Return a copy of the intake classification statistics."""
        return dict(self._stats)

    def _classify(self, interaction: Interaction) -> IntakeAction:
        """
        Evaluate rules against an interaction and return the action.

        Rules are evaluated in strict priority order:
        1. REJECT rules -- if any match, interaction is rejected.
        2. DEFER rules -- if any match, interaction is deferred.
        3. ACCEPT rules -- if any match, interaction is explicitly accepted.
        4. Default: ACCEPT (permissive by default in Phase 1).

        First match wins within each priority level.

        Args:
            interaction: The interaction to classify.

        Returns:
            The IntakeAction classification.
        """
        # Check REJECT rules first (highest priority)
        for rule in self._config.reject:
            if self._rule_matches(rule, interaction):
                logger.debug(
                    "reject_rule_matched",
                    interaction_id=interaction.id,
                    rule=str(rule),
                )
                return IntakeAction.REJECT

        # Check DEFER rules second
        for rule in self._config.defer:
            if self._rule_matches(rule, interaction):
                logger.debug(
                    "defer_rule_matched",
                    interaction_id=interaction.id,
                    rule=str(rule),
                )
                return IntakeAction.DEFER

        # Check ACCEPT rules third (explicit accept)
        for rule in self._config.accept:
            if self._rule_matches(rule, interaction):
                return IntakeAction.ACCEPT

        # Default: ACCEPT (permissive by default)
        return IntakeAction.ACCEPT

    def _rule_matches(self, rule: IntakeRule, interaction: Interaction) -> bool:
        """
        Check if a single rule matches an interaction.

        A rule matches if ALL of its non-None conditions are satisfied.
        Conditions that are None (not set) are ignored.

        Args:
            rule: The IntakeRule to evaluate.
            interaction: The interaction to check against.

        Returns:
            True if all set conditions in the rule match.
        """
        # Track whether any condition was actually checked.
        # A rule with all None fields should not match anything.
        any_condition_set = False

        # Check sender_pattern against participants or metadata
        if rule.sender_pattern is not None:
            any_condition_set = True
            sender = interaction.metadata.get("from", "")
            if isinstance(sender, str) and not fnmatch.fnmatch(sender, rule.sender_pattern):
                return False

        # Check subject_pattern against title
        if rule.subject_pattern is not None:
            any_condition_set = True
            title = interaction.title or ""
            if not fnmatch.fnmatch(title, rule.subject_pattern):
                return False

        # Check source_type match
        if rule.source_type is not None:
            any_condition_set = True
            if interaction.source_type.value != rule.source_type:
                return False

        # Check cc_only: true means the interaction is a CC'd email
        if rule.cc_only is not None and rule.cc_only:
            any_condition_set = True
            # CC-only is determined by metadata (set by eml_loader)
            is_cc = interaction.metadata.get("cc_only", False)
            if not is_cc:
                return False

        # Check thread_depth_gt: match if thread depth exceeds threshold
        if rule.thread_depth_gt is not None:
            any_condition_set = True
            thread_depth = interaction.metadata.get("thread_depth", 0)
            if not isinstance(thread_depth, int) or thread_depth <= rule.thread_depth_gt:
                return False

        # Check contains_keywords: match if content contains ANY keyword
        if rule.contains_keywords is not None:
            any_condition_set = True
            content_lower = interaction.content.lower()
            has_keyword = any(kw.lower() in content_lower for kw in rule.contains_keywords)
            if not has_keyword:
                return False

        # sender_in_entities is deferred to Phase 2 (needs DB access).
        # For now, skip this condition.
        if rule.sender_in_entities is not None:
            any_condition_set = True
            # Phase 1: always passes (we can't check entities from here)
            # Phase 2: will query the entity store to check
            pass

        # A rule with no conditions set should not match
        return any_condition_set

    def _transform_content(self, interaction: Interaction) -> Interaction:
        """
        Transform interaction content by replacing heavy elements.

        Applies three transformations:
        1. Replace fenced and indented code blocks with stubs.
        2. Truncate content beyond max_inline_content_chars.

        Note: email attachment stubs are already handled by the
        eml_loader -- it produces [ATTACHMENT: ...] stubs at load time.

        Args:
            interaction: The interaction to transform (modified in-place).

        Returns:
            The transformed interaction.
        """
        content = interaction.content

        # ---------------------------------------------------------------
        # Transform 1: Replace code blocks with stubs.
        # If max_code_block_lines is 0, replace ALL code blocks.
        # ---------------------------------------------------------------
        max_code_lines = self._config.max_code_block_lines

        # Replace fenced code blocks (```lang ... ```)
        def _replace_fenced(match: re.Match) -> str:
            language = match.group(1) or "unknown"
            code_text = match.group(2)
            line_count = code_text.count("\n") + 1
            if max_code_lines == 0 or line_count > max_code_lines:
                return f"[CODE_BLOCK: {language}, {line_count} lines]"
            return match.group(0)

        content = _FENCED_CODE_RE.sub(_replace_fenced, content)

        # Replace indented code blocks (4+ spaces or tab)
        def _replace_indented(match: re.Match) -> str:
            code_text = match.group(0)
            line_count = code_text.count("\n") + 1
            if max_code_lines == 0 or line_count > max_code_lines:
                return f"[CODE_BLOCK: indented, {line_count} lines]"
            return match.group(0)

        content = _INDENTED_CODE_RE.sub(_replace_indented, content)

        # ---------------------------------------------------------------
        # Transform 2: Truncate content exceeding max_inline_content_chars.
        # ---------------------------------------------------------------
        max_chars = self._config.max_inline_content_chars
        if max_chars > 0 and len(content) > max_chars:
            original_size = len(content)
            content = content[:max_chars] + f"\n[TRUNCATED: {original_size} chars original]"

        interaction.content = content
        return interaction
