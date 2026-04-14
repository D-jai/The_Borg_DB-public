# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
LLM Extraction Verifier
========================

This module provides the LLMExtractionVerifier class, which runs after
the Phase 1 extraction pipeline (spaCy + regex) to verify and enhance
extraction results using a local LLM.

The LLM reviews the NER/regex output and:
    1. Verifies entities (removes false positives by setting confidence=0.0)
    2. Adds missed entities (especially domain-specific ones)
    3. Verifies facts (corrects bad triples)
    4. Adds missed facts

Graceful degradation:
    If the LLM is unavailable, returns invalid JSON, or errors out,
    the verifier returns the original ExtractionResult unchanged.
    This means Phase 1 behavior is always the fallback -- the LLM
    only improves results, never breaks them.

Provenance tracking:
    New entities added by the LLM get provenance "llm:{model_name}",
    e.g., "llm:mock" in tests, "llm:llama-3.2-3b-instruct.Q4_K_M"
    in production. This lets downstream consumers know which entities
    came from the NER pipeline vs. the LLM.

    Rejected entities keep their original provenance but get their
    confidence set to 0.0 (NOT deleted). This preserves the audit
    trail -- you can always see what the NER found and what the LLM
    disagreed with.

Depends on:
    - ctxmtg.interfaces.llm (LLMProvider ABC)
    - ctxmtg.llm.prompt_assembler (PromptAssembler for 4-layer prompts)
    - ctxmtg.models.interaction (Interaction, Entity, Fact, ExtractionResult)
    - ctxmtg.models.profile (DomainProfile)
    - ctxmtg.storage.id_gen (generate_entity_id, generate_fact_id)

Used by:
    - ctxmtg.extraction.pipeline (calls verify_and_enhance after Phase 1)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import structlog

from ctxmtg.interfaces.llm import LLMProvider
from ctxmtg.llm.prompt_assembler import PromptAssembler
from ctxmtg.models.interaction import (
    Entity,
    EntityType,
    ExtractionResult,
    Fact,
    Interaction,
)
from ctxmtg.models.profile import DomainProfile
from ctxmtg.storage.id_gen import generate_entity_id, generate_fact_id

# ---------------------------------------------------------------
# Module-level logger: logs verification actions but never logs
# content (PII concern). Tags all entries with this module name.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.extraction.llm_verifier")

# ---------------------------------------------------------------
# Valid entity type values for mapping LLM output to EntityType enum.
# LLM may return type strings that need normalization.
# ---------------------------------------------------------------
_ENTITY_TYPE_MAP: dict[str, EntityType] = {
    "person": EntityType.PERSON,
    "org": EntityType.ORG,
    "project": EntityType.PROJECT,
    "topic": EntityType.TOPIC,
    "tool": EntityType.TOOL,
    "location": EntityType.LOCATION,
    "event": EntityType.EVENT,
    "other": EntityType.OTHER,
}


# =====================================================================
# LLMExtractionVerifier -- Post-extraction LLM verification
# =====================================================================


class LLMExtractionVerifier:
    """
    Post-extraction LLM verification and enhancement.

    Runs after the Phase 1 extraction pipeline. The LLM reviews
    spaCy+regex output and:
    1. Verifies entities (removes false positives)
    2. Adds missed entities (especially domain-specific)
    3. Verifies facts (corrects bad triples)
    4. Adds missed facts

    Gracefully degrades: if LLM is unavailable, returns the original
    ExtractionResult unchanged.

    Usage:
        verifier = LLMExtractionVerifier(llm, prompt_assembler, profile)
        enhanced_result = verifier.verify_and_enhance(interaction, result)
    """

    def __init__(
        self,
        llm: LLMProvider,
        prompt_assembler: PromptAssembler,
        profile: DomainProfile,
    ) -> None:
        """
        Initialize the LLM extraction verifier.

        Args:
            llm: The LLM provider to use for verification. Can be a
                 real LlamaCppProvider or MockLLMProvider for tests.
            prompt_assembler: The 4-layer prompt assembler for building
                              system prompts from the extraction template.
            profile: The active domain profile, used for assembling
                     domain-specific prompts and stage parameters.
        """
        self._llm = llm
        self._prompt_assembler = prompt_assembler
        self._profile = profile

    def verify_and_enhance(
        self,
        interaction: Interaction,
        extraction_result: ExtractionResult,
        intelligence_context: str | None = None,
    ) -> ExtractionResult:
        """
        Verify and enhance an extraction result using the LLM.

        Sends the original text and Phase 1 extraction output to the
        LLM for review. The LLM returns a structured JSON response
        indicating which entities/facts are verified, which are new,
        and which should be rejected.

        Returns a new ExtractionResult with verified/added entities
        and facts. Original entities marked as rejected get their
        confidence set to 0.0 (not deleted -- provenance preserved).

        If the LLM is unavailable or returns invalid JSON, returns
        the original extraction_result unchanged.

        Args:
            interaction: The original Interaction being processed.
            extraction_result: The Phase 1 extraction output to verify.
            intelligence_context: Optional intelligence context string
                                  from the hive (Phase 4+).  Injected
                                  into the ``{{intelligence_context}}``
                                  slot of the extraction prompt.  When
                                  None, the slot is cleared to empty.

        Returns:
            An enhanced ExtractionResult, or the original if LLM
            is unavailable or errors out.
        """
        # ---------------------------------------------------------------
        # Guard: if the LLM is not available, return the original result.
        # This preserves Phase 1 behavior when no LLM is configured.
        # ---------------------------------------------------------------
        if not self._llm.is_available():
            logger.info(
                "llm_verifier_skipped",
                reason="llm_not_available",
                interaction_id=interaction.id,
            )
            return extraction_result

        # ---------------------------------------------------------------
        # Step 1: Build the system prompt using the 4-layer assembler.
        # This loads the extraction stage template and fills domain slots.
        # ---------------------------------------------------------------
        system_prompt = self._prompt_assembler.assemble("extraction", self._profile)

        # ---------------------------------------------------------------
        # Step 1b: Inject the intelligence context into the prompt.
        # The extraction template has a {{intelligence_context}} slot
        # that the assembler leaves unfilled (not in slot_values).
        # Replace it with the actual intelligence string, or remove
        # the placeholder if no intelligence is available.
        # ---------------------------------------------------------------
        intel_replacement = intelligence_context if intelligence_context else ""
        system_prompt = system_prompt.replace(
            "{{intelligence_context}}", intel_replacement
        )

        # ---------------------------------------------------------------
        # Step 2: Build the user prompt with original text + Phase 1 output.
        # The extraction template expects JSON input with text, entities,
        # and facts.
        # ---------------------------------------------------------------
        user_prompt = self._build_user_prompt(interaction, extraction_result)

        # ---------------------------------------------------------------
        # Step 3: Get LLM parameters for the extraction stage.
        # ---------------------------------------------------------------
        stage_params = self._prompt_assembler.get_stage_params("extraction", self._profile)

        # ---------------------------------------------------------------
        # Step 4: Call the LLM with json_mode=True for structured output.
        # ---------------------------------------------------------------
        try:
            llm_response = self._llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=stage_params.temperature,
                max_tokens=stage_params.max_tokens,
                top_p=stage_params.top_p,
                frequency_penalty=stage_params.frequency_penalty,
                presence_penalty=stage_params.presence_penalty,
                json_mode=True,
            )
        except Exception:
            logger.warning(
                "llm_verifier_generate_failed",
                error_code="CTXMTG-EXT-008",
                interaction_id=interaction.id,
            )
            return extraction_result

        # ---------------------------------------------------------------
        # Step 5: Parse the LLM response as JSON.
        # If parsing fails, log a warning and return the original result.
        # ---------------------------------------------------------------
        parsed = self._parse_llm_response(llm_response)
        if parsed is None:
            return extraction_result

        # ---------------------------------------------------------------
        # Step 6: Merge the LLM's review back into the extraction result.
        # This handles new entities, rejected entities, new facts, etc.
        # ---------------------------------------------------------------
        return self._merge_results(interaction, extraction_result, parsed)

    # =================================================================
    # Private helper methods
    # =================================================================

    def _build_user_prompt(
        self,
        interaction: Interaction,
        extraction_result: ExtractionResult,
    ) -> str:
        """
        Build the user prompt containing original text + Phase 1 output.

        Formats the extraction data as JSON matching the input format
        expected by the extraction prompt template:
            {"text": "...", "entities": [...], "facts": [...]}

        Args:
            interaction: The original Interaction with the source text.
            extraction_result: The Phase 1 extraction output.

        Returns:
            A JSON string containing the input data for the LLM.
        """
        # Build entity list as simple dicts for the LLM
        entities_json = []
        for entity in extraction_result.entities:
            entities_json.append({
                "name": entity.name,
                "type": entity.entity_type.value,
                "confidence": entity.confidence,
            })

        # Build fact list as simple dicts for the LLM
        facts_json = []
        for fact in extraction_result.facts:
            fact_dict: dict[str, object] = {
                "subject": fact.subject_entity_id,
                "predicate": fact.predicate,
                "confidence": fact.confidence,
            }
            if fact.object_entity_id:
                fact_dict["object"] = fact.object_entity_id
            elif fact.object_literal:
                fact_dict["object"] = fact.object_literal
            facts_json.append(fact_dict)

        input_data = {
            "text": interaction.content,
            "entities": entities_json,
            "facts": facts_json,
        }

        return json.dumps(input_data, ensure_ascii=False)

    def _parse_llm_response(self, response: str) -> dict | None:
        """
        Parse the LLM's JSON response.

        Validates that the response is valid JSON and contains the
        expected structure. If parsing fails, logs a warning and
        returns None (caller should return the original result).

        Args:
            response: The raw string response from the LLM.

        Returns:
            A parsed dict with the verification data, or None if
            the response is invalid.
        """
        if not response or not response.strip():
            logger.warning(
                "llm_verifier_empty_response",
                error_code="CTXMTG-EXT-008",
            )
            return None

        try:
            parsed = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "llm_verifier_invalid_json",
                error_code="CTXMTG-EXT-008",
                response_length=len(response),
            )
            return None

        if not isinstance(parsed, dict):
            logger.warning(
                "llm_verifier_unexpected_type",
                error_code="CTXMTG-EXT-008",
                response_type=type(parsed).__name__,
            )
            return None

        return parsed

    def _merge_results(
        self,
        interaction: Interaction,
        original: ExtractionResult,
        llm_review: dict,
    ) -> ExtractionResult:
        """
        Merge the LLM's review into the original extraction result.

        Handles:
        - rejected: entities whose names match get confidence=0.0
        - new_entities: added with provenance "llm:{model_name}"
        - new_facts: added with provenance "llm:{model_name}"
        - verified_entities and verified_facts: acknowledged (no changes)

        Args:
            interaction: The original Interaction.
            original: The Phase 1 ExtractionResult to enhance.
            llm_review: The parsed JSON from the LLM.

        Returns:
            A new ExtractionResult with LLM enhancements applied.
        """
        model_name = self._llm.get_model_name()
        provenance = f"llm:{model_name}"
        now = datetime.now(timezone.utc)

        # ---------------------------------------------------------------
        # Step 1: Copy the original entities list (we modify in place).
        # We create new Entity objects to avoid mutating the originals.
        # ---------------------------------------------------------------
        entities = [entity.model_copy() for entity in original.entities]

        # ---------------------------------------------------------------
        # Step 2: Process rejected entities -- set confidence to 0.0.
        # Match by name (case-insensitive). Don't delete -- audit trail.
        # ---------------------------------------------------------------
        rejected_list = llm_review.get("rejected", [])
        if isinstance(rejected_list, list):
            rejected_names = set()
            for item in rejected_list:
                if isinstance(item, dict) and "name" in item:
                    rejected_names.add(item["name"].lower().strip())
                elif isinstance(item, str):
                    rejected_names.add(item.lower().strip())

            for entity in entities:
                if entity.name.lower().strip() in rejected_names:
                    entity.confidence = 0.0

        # ---------------------------------------------------------------
        # Step 3: Process new entities from the LLM.
        # Create Entity objects with LLM provenance and generate IDs.
        # ---------------------------------------------------------------
        new_entities_list = llm_review.get("new_entities", [])
        if isinstance(new_entities_list, list):
            # Build a set of existing entity names (lowercase) to avoid duplicates
            existing_names = {e.name.lower().strip() for e in entities}

            for item in new_entities_list:
                if not isinstance(item, dict) or "name" not in item:
                    continue

                name = str(item["name"]).strip()
                if not name:
                    continue

                # Skip if this entity already exists (case-insensitive)
                if name.lower() in existing_names:
                    continue

                # Determine entity type from LLM output
                raw_type = str(item.get("type", "other")).lower().strip()
                entity_type = _ENTITY_TYPE_MAP.get(raw_type, EntityType.OTHER)

                # Determine confidence (default 0.8 for LLM-sourced entities)
                confidence = item.get("confidence", 0.8)
                if not isinstance(confidence, (int, float)):
                    confidence = 0.8
                confidence = max(0.0, min(1.0, float(confidence)))

                # Generate deterministic entity ID
                entity_id = generate_entity_id(
                    interaction_id=interaction.id,
                    name=name,
                    entity_type=entity_type.value,
                )

                new_entity = Entity(
                    id=entity_id,
                    interaction_id=interaction.id,
                    name=name,
                    entity_type=entity_type,
                    confidence=confidence,
                    provenance=provenance,
                    created_at=now,
                )
                entities.append(new_entity)
                existing_names.add(name.lower())

        # ---------------------------------------------------------------
        # Step 4: Copy the original facts list.
        # ---------------------------------------------------------------
        facts = [fact.model_copy() for fact in original.facts]

        # ---------------------------------------------------------------
        # Step 5: Process new facts from the LLM.
        # Create Fact objects with LLM provenance and generate IDs.
        # ---------------------------------------------------------------
        new_facts_list = llm_review.get("new_facts", [])
        if isinstance(new_facts_list, list):
            # Build an entity name→ID lookup for resolving fact references
            entity_lookup = {e.name.lower().strip(): e.id for e in entities}

            for item in new_facts_list:
                if not isinstance(item, dict):
                    continue

                subject_name = str(item.get("subject", "")).strip()
                predicate = str(item.get("predicate", "")).strip()
                object_value = str(item.get("object", "")).strip()

                if not subject_name or not predicate or not object_value:
                    continue

                # Resolve subject to entity ID.
                # ORIGINAL CODE (disabled 2026-04-07): If subject name wasn't in
                # the entity lookup, the raw name string was used as entity ID,
                # causing FOREIGN KEY constraint failures in the facts table.
                # subject_id = entity_lookup.get(subject_name.lower(), subject_name)
                #
                # FIX: Auto-create the missing entity so the FK is satisfied.
                # This covers cases where the LLM references an entity in a fact
                # but forgets to include it in new_entities.
                subject_id = entity_lookup.get(subject_name.lower())
                if subject_id is None:
                    auto_entity_id = generate_entity_id(
                        interaction_id=interaction.id,
                        name=subject_name,
                        entity_type=EntityType.OTHER.value,
                    )
                    auto_entity = Entity(
                        id=auto_entity_id,
                        interaction_id=interaction.id,
                        name=subject_name,
                        entity_type=EntityType.OTHER,
                        confidence=0.6,
                        provenance=f"{provenance}:auto-from-fact",
                        created_at=now,
                    )
                    entities.append(auto_entity)
                    entity_lookup[subject_name.lower()] = auto_entity_id
                    subject_id = auto_entity_id

                # Resolve object: check if it's an entity name
                object_entity_id = entity_lookup.get(object_value.lower())
                object_literal = object_value if object_entity_id is None else None

                # Determine confidence
                confidence = item.get("confidence", 0.7)
                if not isinstance(confidence, (int, float)):
                    confidence = 0.7
                confidence = max(0.0, min(1.0, float(confidence)))

                # Generate deterministic fact ID
                obj_for_id = object_entity_id or object_literal or object_value
                fact_id = generate_fact_id(
                    subject_entity_id=subject_id,
                    predicate=predicate,
                    object_value=obj_for_id,
                )

                new_fact = Fact(
                    id=fact_id,
                    interaction_id=interaction.id,
                    subject_entity_id=subject_id,
                    predicate=predicate,
                    object_entity_id=object_entity_id,
                    object_literal=object_literal,
                    confidence=confidence,
                    source_span=provenance,
                    created_at=now,
                )
                facts.append(new_fact)

        # ---------------------------------------------------------------
        # Step 6: Build and return the enhanced ExtractionResult.
        # Summary and chunks are unchanged -- the LLM only affects
        # entities and facts.
        # ---------------------------------------------------------------
        enhanced = ExtractionResult(
            interaction_id=original.interaction_id,
            entities=entities,
            facts=facts,
            summary=original.summary,
            chunks=original.chunks,
        )

        logger.info(
            "llm_verification_complete",
            interaction_id=interaction.id,
            original_entities=len(original.entities),
            enhanced_entities=len(entities),
            original_facts=len(original.facts),
            enhanced_facts=len(facts),
            rejected_count=len(rejected_list) if isinstance(rejected_list, list) else 0,
        )

        return enhanced
