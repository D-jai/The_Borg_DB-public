# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Intake Gateway Interface ABC
=============================

This module defines the abstract base class for the intake gateway --
the "Traffic Cop" that classifies AND transforms inbound data before
it reaches the extraction pipeline.

The intake gateway serves two critical functions:
1. Classify: decide whether to ACCEPT, DEFER, REJECT, or ROUTE each
   incoming interaction. Only ACCEPT interactions proceed to extraction.
2. Transform: strip heavy content (attachments, code blocks, binary data)
   and replace them with metadata stubs. The original data stays at the
   source -- we store only lightweight references.

This is the first component that touches inbound data. It runs BEFORE
the extraction pipeline and reduces noise, storage, and processing cost
by filtering and slimming incoming content.

Phase 1: RuleBasedGateway using YAML pattern matching from the domain
profile's intake section. Simple but effective for common filtering.
Future: LLMGateway (Tier 2+) for semantic classification -- understanding
context, not just pattern matching.

Depends on:
    - abc (Python's Abstract Base Class machinery)
    - ctxmtg.models.interaction (Interaction, IntakeAction)

Used by:
    - ctxmtg.intake.rules (implements IntakeGateway with YAML rules)
    - ctxmtg.ingestion.worker (calls IntakeGateway before extraction)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# ---------------------------------------------------------------
# Import the data models for the intake gateway.
# The gateway receives an Interaction and returns an (action, interaction)
# tuple. The IntakeAction enum classifies the decision.
# ---------------------------------------------------------------
from ctxmtg.models.interaction import IntakeAction, Interaction

# =====================================================================
# IntakeGateway ABC -- Traffic Cop Interface
# =====================================================================


class IntakeGateway(ABC):
    """
    Traffic Cop: classifies AND transforms inbound data before extraction.

    The gateway does two jobs:
    1. Classify: ACCEPT, DEFER, REJECT, or ROUTE.
    2. Transform: strip heavy content (attachments, code blocks, binary),
       replacing them with metadata stubs. The original data stays at
       the source -- we store only the reference.

    Phase 1: RuleBasedGateway using YAML patterns.
    Future: LLMGateway (Tier 2+) for semantic classification.

    Classification rules are evaluated in priority order:
    - REJECT rules first (sender patterns, source types to exclude)
    - DEFER rules second (CC-only emails, deep threads)
    - ACCEPT rules third (known entities, priority keywords)
    - Default: ACCEPT (permissive by default in Phase 1)

    Content transformation:
    - Email attachments → [ATTACHMENT: name, type, size] stubs
    - Code blocks → [CODE_BLOCK: language, line_count] stubs
    - Large inline content → [TRUNCATED: original_size] markers

    The returned Interaction may differ from the input (heavy content
    replaced with stubs). The intake_action field is set on the
    returned Interaction to record the classification decision.

    Usage:
        gateway = RuleBasedIntakeGateway(profile.intake)
        action, modified_interaction = gateway.process(interaction)
        if action == IntakeAction.ACCEPT:
            result = extraction_pipeline.process(modified_interaction)
        elif action == IntakeAction.DEFER:
            queue.add(modified_interaction)
        elif action == IntakeAction.REJECT:
            logger.info("interaction_rejected", id=interaction.id)
    """

    @abstractmethod
    def process(
        self, interaction: Interaction
    ) -> tuple[IntakeAction, Interaction]:
        """
        Classify and transform an inbound interaction.

        Evaluates the interaction against the configured rules
        (reject, defer, accept) and transforms its content by
        replacing heavy elements with lightweight metadata stubs.

        The classification decision is also set on the returned
        Interaction's intake_action field for audit/logging purposes.

        Args:
            interaction: The inbound Interaction to classify and
                         transform. The original object is not modified;
                         a new Interaction is returned with any
                         transformations applied.

        Returns:
            A tuple of:
            - IntakeAction: the classification decision
              (ACCEPT/DEFER/REJECT/ROUTE)
            - Interaction: a potentially modified copy of the input
              interaction. Attachments are replaced with metadata stubs,
              code blocks are summarized, and heavy content is stripped.
              The intake_action field is set to match the classification.
        """
        ...
