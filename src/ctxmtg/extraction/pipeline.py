# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Basic Extraction Pipeline
=========================

This module implements the ExtractionPipeline interface, orchestrating
the complete extraction process for one interaction. It is the "main"
entry point for the extraction subsystem -- the ingestion worker calls
pipeline.process(interaction) and gets back a fully structured
ExtractionResult containing entities, facts, a summary, and text chunks.

Pipeline steps (in order):
    1. Run spaCy NER → extract named entities (people, orgs, locations)
    2. Run regex extraction → catch emails, URLs, dates, phone numbers
    3. Merge and deduplicate entities from both providers
    4. Assign entity IDs (per-interaction, using generate_entity_id)
    5. Build rich entity context (summary, co-entities -- max 500 chars)
    6. Build entity tags (source_type, instance -- max 20 pairs)
    7. Extract facts (subject-verb-object triples via dependency parse)
    8. Summarize the interaction content (TextRank)
    9. Chunk text for embedding (paragraph → sentence boundaries)

The pipeline is constructed with a DomainProfile that controls:
    - Which entity types to extract (profile.ner.entity_types)
    - Custom regex patterns (profile.ner.custom_patterns)
    - Chunk size and overlap (profile.embedding.chunk_size/overlap)

Why per-interaction entity IDs?
Each entity gets a unique ID scoped to its interaction. The same
real-world person "Alice" in two meetings = two different entity IDs.
Entity merging across interactions is deferred to the Consolidator
micro-agent in Phase 3. Queries find entities by name (case-insensitive),
not by entity ID. See research/round-2/03-unified-schema-design.md.

Depends on:
    - ctxmtg.extraction.spacy_ner (SpacyNERProvider)
    - ctxmtg.extraction.regex_extractor (RegexExtractor)
    - ctxmtg.extraction.fact_extractor (SimpleFactExtractor)
    - ctxmtg.extraction.summarizer (TextRankSummarizer)
    - ctxmtg.interfaces.extraction (ExtractionPipeline ABC)
    - ctxmtg.models.interaction (Interaction, Entity, ExtractionResult)
    - ctxmtg.models.profile (DomainProfile)
    - ctxmtg.storage.id_gen (generate_entity_id, generate_fact_id)

Used by:
    - ctxmtg.ingestion.worker (calls pipeline.process() for each interaction)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

# ---------------------------------------------------------------
# Import our extraction sub-components
# ---------------------------------------------------------------
from ctxmtg.extraction.fact_extractor import SimpleFactExtractor
from ctxmtg.extraction.regex_extractor import RegexExtractor
from ctxmtg.extraction.spacy_ner import SpacyNERProvider
from ctxmtg.extraction.summarizer import TextRankSummarizer

# ---------------------------------------------------------------
# Import the pipeline interface and data models
# ---------------------------------------------------------------
from ctxmtg.exceptions import ExtractionError
from ctxmtg.interfaces.extraction import ExtractionPipeline
from ctxmtg.models.interaction import Entity, ExtractionResult, Interaction
from ctxmtg.models.profile import DomainProfile
from ctxmtg.storage.id_gen import generate_entity_id

if TYPE_CHECKING:
    from ctxmtg.extraction.intelligence import IntelligenceContext
    from ctxmtg.extraction.llm_verifier import LLMExtractionVerifier

# ---------------------------------------------------------------
# Logger for the pipeline orchestrator. Logs step timings and
# result counts, but never logs content (PII concern).
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.extraction.pipeline")

# ---------------------------------------------------------------
# Constants for entity context and tags limits.
# These limits are enforced by Pydantic validators on Entity,
# but we also enforce them here to catch issues early.
# ---------------------------------------------------------------
MAX_CONTEXT_CHARS = 500  # Max serialized context dict size
MAX_TAGS_PAIRS = 20  # Max key-value pairs in tags dict


# =====================================================================
# BasicExtractionPipeline -- Full Extraction Orchestrator
# =====================================================================


class BasicExtractionPipeline(ExtractionPipeline):
    """
    Full extraction pipeline: NER + regex + facts + summary + chunking.

    Orchestrates the complete extraction process for a single interaction.
    Constructed with a DomainProfile that controls entity types, custom
    patterns, and chunking behaviour.

    The pipeline uses four sub-components:
    - SpacyNERProvider: finds people, orgs, locations, events via NLP
    - RegexExtractor: finds emails, URLs, dates, phones via patterns
    - SimpleFactExtractor: finds subject-verb-object relationships
    - TextRankSummarizer: picks the most representative sentences

    All four share the same spaCy model instance to avoid redundant
    loading and processing.

    Usage:
        from ctxmtg.models.profile import DomainProfile

        profile = DomainProfile(name="test", version="1.0")
        pipeline = BasicExtractionPipeline(profile)
        result = pipeline.process(interaction)
        # result.entities, result.facts, result.summary, result.chunks
    """

    def __init__(
        self,
        profile: DomainProfile,
        llm_verifier: LLMExtractionVerifier | None = None,
    ) -> None:
        """
        Initialize the extraction pipeline with a domain profile.

        Creates and configures all sub-components (NER, regex, facts,
        summarizer). The spaCy model is loaded once and shared across
        components that need it.

        Args:
            profile: The DomainProfile controlling extraction behaviour.
                     Specifies entity types, custom patterns, and chunk sizes.
            llm_verifier: Optional LLM-based extraction verifier. If provided,
                          called after Phase 1 extraction to verify and enhance
                          results. If None, Phase 1 behavior is unchanged.
        """
        # Store the profile for later reference
        self._profile = profile

        # Store the optional LLM verifier (Phase 2 enhancement)
        self._llm_verifier = llm_verifier

        # ---------------------------------------------------------------
        # Initialize the spaCy NER provider. This loads the spaCy model
        # which is shared with the fact extractor and summarizer.
        # ---------------------------------------------------------------
        self._spacy_ner = SpacyNERProvider()
        try:
            self._spacy_ner.load()
        except ExtractionError:
            raise
        except Exception as exc:
            logger.error(
                "pipeline_init_failed",
                error_code="CTXMTG-EXT-008",
                profile_name=profile.name,
                error=str(exc),
            )
            raise ExtractionError(
                f"Failed to initialize extraction pipeline: {exc}",
                error_code="CTXMTG-EXT-008",
            ) from exc

        # ---------------------------------------------------------------
        # Initialize the regex extractor with custom patterns from the
        # domain profile. These catch emails, URLs, dates, etc.
        # ---------------------------------------------------------------
        self._regex_extractor = RegexExtractor(custom_patterns=profile.ner.custom_patterns or None)

        # ---------------------------------------------------------------
        # Initialize the fact extractor. It reuses the spaCy model from
        # the NER provider (no second model load).
        # ---------------------------------------------------------------
        self._fact_extractor = SimpleFactExtractor(nlp=self._spacy_ner.get_nlp())

        # ---------------------------------------------------------------
        # Initialize the TextRank summarizer. Also reuses the spaCy model.
        # ---------------------------------------------------------------
        self._summarizer = TextRankSummarizer(nlp=self._spacy_ner.get_nlp())

        # ---------------------------------------------------------------
        # Extract configuration from the profile for easy access.
        # ---------------------------------------------------------------

        # Which entity types to extract (empty list = all types)
        self._entity_types: list[str] | None = (
            profile.ner.entity_types if profile.ner.entity_types else None
        )

        # Chunk size and overlap for text splitting
        self._chunk_size: int = profile.embedding.chunk_size
        self._chunk_overlap: int = profile.embedding.chunk_overlap

        logger.info(
            "extraction_pipeline_initialized",
            profile_name=profile.name,
            entity_types=self._entity_types,
            chunk_size=self._chunk_size,
        )

    def process(
        self,
        interaction: Interaction,
        intelligence_ctx: IntelligenceContext | None = None,
    ) -> ExtractionResult:
        """
        Run the full extraction pipeline on an interaction.

        Executes all pipeline steps in order:
        1. spaCy NER → entities
        2. Regex extraction → more entities
        3. Merge + deduplicate entities
        3b. Feed merged entities into intelligence rolling context
        4. Assign per-interaction entity IDs
        5. Build entity context (summary, co-entities, hive enrichment)
        6. Build entity tags (source_type, instance)
        7. Extract facts (SVO triples)
        8. Summarize content
        9. Chunk text for embedding
        10. Optional LLM verification (with intelligence context)

        Args:
            interaction: The Interaction to process. Must have non-empty
                         content for meaningful extraction.
            intelligence_ctx: Optional IntelligenceContext providing
                              cross-instance hive hints for prompt
                              enrichment.  When None, the pipeline
                              behaves identically to Phase 3 (no
                              intelligence enrichment).

        Returns:
            An ExtractionResult with entities, facts, summary, and chunks.
        """
        content = interaction.content
        interaction_id = interaction.id

        # Handle empty content gracefully
        if not content or not content.strip():
            logger.warning(
                "empty_content_for_extraction",
                error_code="CTXMTG-EXT-001",
                interaction_id=interaction_id,
            )
            return ExtractionResult(
                interaction_id=interaction_id,
                entities=[],
                facts=[],
                summary="",
                chunks=[],
            )

        # ---------------------------------------------------------------
        # Step 1: Run spaCy NER to extract named entities.
        # This processes the text through spaCy's full pipeline.
        # ---------------------------------------------------------------
        spacy_entities = self._spacy_ner.extract_entities(content, entity_types=self._entity_types)

        # Also get the spaCy Doc for reuse in later steps (facts, summary)
        doc = self._spacy_ner.process_text(content)

        # ---------------------------------------------------------------
        # Step 2: Run regex extraction for structured patterns.
        # Catches emails, URLs, dates, phone numbers, versions.
        # ---------------------------------------------------------------
        regex_entities = self._regex_extractor.extract_entities(
            content, entity_types=self._entity_types
        )

        # ---------------------------------------------------------------
        # Step 3: Merge and deduplicate entities from both providers.
        # Same name (case-insensitive) = same entity. We keep the one
        # with higher confidence.
        # ---------------------------------------------------------------
        merged_entities = self._merge_entities(spacy_entities, regex_entities)

        # ---------------------------------------------------------------
        # Step 3a: Filter out garbage entities using profile rules.
        # Added 2026-04-07 to prevent timestamp fragments, header labels,
        # and too-short names from reaching the database.
        # ---------------------------------------------------------------
        merged_entities = self._filter_entities(merged_entities)

        # ---------------------------------------------------------------
        # Step 3b: Feed merged entities into the intelligence rolling
        # context so later chunks in the same session can see what
        # entities were found in earlier chunks.  Only runs when an
        # IntelligenceContext is provided (Phase 4+ behaviour).
        # ---------------------------------------------------------------
        if intelligence_ctx is not None:
            intelligence_ctx.add_batch_entities(merged_entities)

        # ---------------------------------------------------------------
        # Step 4: Assign per-interaction entity IDs using generate_entity_id.
        # Each entity gets a deterministic UUIDv5 based on interaction_id
        # + name + type. This means re-processing the same interaction
        # produces the same entity IDs (idempotent).
        # ---------------------------------------------------------------
        now = datetime.now(timezone.utc)
        for entity in merged_entities:
            entity.id = generate_entity_id(
                interaction_id=interaction_id,
                name=entity.name,
                entity_type=entity.entity_type.value,
            )
            entity.interaction_id = interaction_id
            entity.created_at = now

        # ---------------------------------------------------------------
        # Step 5: Build rich entity context for vector embedding.
        # Each entity gets a context dict with:
        #   - summary: surrounding sentence or clause
        #   - co_entities: other entities mentioned nearby
        #   - hive_intelligence: enrichment from hive hints (Phase 4+)
        # Truncated to MAX_CONTEXT_CHARS when serialized.
        # ---------------------------------------------------------------
        self._build_entity_context(merged_entities, content, intelligence_ctx)

        # ---------------------------------------------------------------
        # Step 6: Build entity tags for SQL filtering.
        # Each entity gets tags like source_type, source_instance.
        # Max MAX_TAGS_PAIRS key-value pairs.
        # ---------------------------------------------------------------
        self._build_entity_tags(merged_entities, interaction)

        # ---------------------------------------------------------------
        # Step 7: Extract facts (subject-verb-object triples).
        # Uses spaCy dependency parse to find relationships between
        # entities. Reuses the Doc from step 1.
        # ---------------------------------------------------------------
        facts = self._fact_extractor.extract_facts_from_doc(doc, merged_entities)

        # Assign interaction_id to all facts
        for fact in facts:
            fact.interaction_id = interaction_id
            fact.created_at = now

        # ---------------------------------------------------------------
        # Step 8: Summarize the interaction content.
        # Uses TextRank to pick the most representative sentences.
        # Reuses the Doc from step 1.
        # ---------------------------------------------------------------
        summary = self._summarizer.summarize_from_doc(doc, max_length=300)

        # ---------------------------------------------------------------
        # Step 9: Chunk text for embedding.
        # Split on paragraph boundaries, then sentence boundaries
        # if chunks exceed the target size.
        # ---------------------------------------------------------------
        chunks = self._chunk_text(content)

        # ---------------------------------------------------------------
        # Build the complete extraction result.
        # ---------------------------------------------------------------
        result = ExtractionResult(
            interaction_id=interaction_id,
            entities=merged_entities,
            facts=facts,
            summary=summary if summary else None,
            chunks=chunks,
        )

        # ---------------------------------------------------------------
        # Step 10 (optional): LLM verification and enhancement.
        # If an LLM verifier is configured, run it to verify entities,
        # reject false positives, and add missed entities/facts.
        # If not configured, this step is skipped (Phase 1 behavior).
        # When intelligence_ctx is available, the intelligence context
        # string is passed to the verifier for prompt enrichment.
        # ---------------------------------------------------------------
        if self._llm_verifier is not None:
            # Build the intelligence context string for LLM injection.
            # Empty string if no intelligence is available (graceful).
            intel_context_str: str | None = None
            if intelligence_ctx is not None:
                entity_names = [e.name for e in result.entities]
                intel_context_str = intelligence_ctx.build_prompt_context(
                    entity_names
                ) or None

            result = self._llm_verifier.verify_and_enhance(
                interaction, result, intelligence_context=intel_context_str
            )

        logger.info(
            "extraction_pipeline_complete",
            interaction_id=interaction_id,
            entity_count=len(result.entities),
            fact_count=len(result.facts),
            chunk_count=len(result.chunks),
            summary_length=len(result.summary) if result.summary else 0,
        )

        return result

    def _filter_entities(self, entities: list[Entity]) -> list[Entity]:
        """
        Filter out garbage entities using profile-defined rules.

        Applied after NER + regex + merge, before ID assignment.
        Checks min/max name length, reject patterns, and reject names.
        """
        filters = self._profile.ner.entity_filters
        if not filters:
            return entities

        # Pre-compile reject patterns
        compiled_patterns = []
        for pat in filters.reject_patterns:
            try:
                compiled_patterns.append(re.compile(pat))
            except re.error:
                logger.warning("invalid_reject_pattern", pattern=pat)

        reject_names_lower = {n.lower() for n in filters.reject_names}

        original_count = len(entities)
        filtered = []
        for entity in entities:
            name = entity.name.strip()

            # Length checks
            if len(name) < filters.min_name_length:
                continue
            if len(name) > filters.max_name_length:
                continue

            # Exact name rejection (case-insensitive)
            if name.lower() in reject_names_lower:
                continue

            # Pattern rejection
            rejected = False
            for pattern in compiled_patterns:
                if pattern.fullmatch(name):
                    rejected = True
                    break
            if rejected:
                continue

            filtered.append(entity)

        dropped = original_count - len(filtered)
        if dropped > 0:
            logger.info(
                "entities_filtered",
                original=original_count,
                kept=len(filtered),
                dropped=dropped,
            )

        return filtered

    def _merge_entities(
        self,
        spacy_entities: list[Entity],
        regex_entities: list[Entity],
    ) -> list[Entity]:
        """
        Merge entities from spaCy and regex extractors, deduplicating by name.

        When both extractors find the same entity name (case-insensitive),
        we keep the one with higher confidence. This avoids duplicate
        entities in the output.

        Args:
            spacy_entities: Entities from SpacyNERProvider.
            regex_entities: Entities from RegexExtractor.

        Returns:
            A deduplicated list of Entity objects.
        """
        # Use a dict keyed by lowercase name for deduplication.
        # When there's a conflict, keep the higher-confidence one.
        seen: dict[str, Entity] = {}

        # Process spaCy entities first (they often have richer type info)
        for entity in spacy_entities:
            name_key = entity.name.lower().strip()
            if name_key not in seen or entity.confidence > seen[name_key].confidence:
                seen[name_key] = entity

        # Then process regex entities (they may fill gaps)
        for entity in regex_entities:
            name_key = entity.name.lower().strip()
            if name_key not in seen or entity.confidence > seen[name_key].confidence:
                seen[name_key] = entity

        return list(seen.values())

    def _build_entity_context(
        self,
        entities: list[Entity],
        content: str,
        intelligence_ctx: IntelligenceContext | None = None,
    ) -> None:
        """
        Build rich entity context for vector embedding.

        For each entity, creates a context dict with:
        - summary: the sentence or clause containing the entity
        - co_entities: other entity names that appear in the same text
        - hive_intelligence: cross-instance enrichment (Phase 4+, optional)

        The context is truncated to MAX_CONTEXT_CHARS when serialized
        to JSON. This context is what gets vector-embedded, enabling
        semantic search on entities.

        Modifies entities in-place.

        Args:
            entities: List of Entity objects to enrich.
            content: The full interaction text (for finding surrounding context).
            intelligence_ctx: Optional IntelligenceContext for hive enrichment.
                              When None, behaviour is identical to Phase 3.
        """
        # Build a list of all entity names for co-entity detection
        all_entity_names = [e.name for e in entities]

        for entity in entities:
            # Find the entity's surrounding context in the content.
            # Look for the entity name and extract surrounding text.
            context_summary = self._find_surrounding_context(entity.name, content)

            # Build co-entities list: other entities mentioned in the text
            co_entities = [name for name in all_entity_names if name.lower() != entity.name.lower()]

            # Build the context dict
            context: dict[str, object] = {
                "summary": context_summary,
                "co_entities": co_entities[:10],  # Limit to 10 co-entities
            }

            # ----------------------------------------------------------
            # Hive intelligence enrichment (Phase 4+).
            # If intelligence context is available, look up the entity
            # in the hive hints and prepend enrichment data.  This gives
            # the vector embedding richer cross-instance context.
            # ----------------------------------------------------------
            if intelligence_ctx is not None:
                enrichment = intelligence_ctx.get_entity_context_enrichment(
                    entity.name
                )
                if enrichment:
                    # Add hive intelligence as a compact summary prefix.
                    # Keeps the enrichment concise to fit MAX_CONTEXT_CHARS.
                    hive_summary = enrichment.get("summary", "")
                    hive_predicates = enrichment.get("top_predicates", [])
                    if hive_summary or hive_predicates:
                        hive_parts = []
                        if hive_predicates:
                            hive_parts.append(
                                "hive_predicates: " + ", ".join(hive_predicates[:3])
                            )
                        if hive_summary:
                            hive_parts.append(hive_summary[:80])
                        context["hive_intelligence"] = "; ".join(hive_parts)

            # Truncate if serialized context exceeds MAX_CONTEXT_CHARS.
            # Progressively shorten the summary until it fits.
            serialized = json.dumps(context)
            while len(serialized) > MAX_CONTEXT_CHARS and context_summary:
                # Shorten the summary by half each iteration
                context_summary = context_summary[: len(context_summary) // 2]
                context["summary"] = context_summary
                serialized = json.dumps(context)

            # If still too long, remove co_entities
            if len(serialized) > MAX_CONTEXT_CHARS:
                context["co_entities"] = []
                serialized = json.dumps(context)

            # Final safety check: if still too long, use minimal context
            if len(serialized) > MAX_CONTEXT_CHARS:
                context = {"summary": context_summary[:100]}

            entity.context = context

    def _build_entity_tags(self, entities: list[Entity], interaction: Interaction) -> None:
        """
        Build structured entity tags for SQL filtering.

        Adds key-value tags to each entity:
        - source_type: the interaction's source type (meeting, email, etc.)
        - instance: the source instance ("local" for Phase 1)

        Additional tags could come from domain profile custom patterns
        in future phases.

        Modifies entities in-place. Max MAX_TAGS_PAIRS pairs.

        Args:
            entities: List of Entity objects to tag.
            interaction: The parent Interaction for source metadata.
        """
        for entity in entities:
            tags: dict[str, str] = {
                "source_type": interaction.source_type.value,
                "instance": interaction.source_instance,
            }

            # Add the interaction title as a tag if available
            if interaction.title:
                tags["source_title"] = interaction.title[:50]

            # Ensure we don't exceed MAX_TAGS_PAIRS
            if len(tags) > MAX_TAGS_PAIRS:
                # Keep only the first MAX_TAGS_PAIRS entries
                tags = dict(list(tags.items())[:MAX_TAGS_PAIRS])

            entity.tags = tags

    def _find_surrounding_context(self, entity_name: str, content: str) -> str:
        """
        Find the text surrounding an entity mention.

        Looks for the entity name in the content and extracts a
        window of ~150 characters around it. This gives context
        about what was said about the entity.

        Args:
            entity_name: The entity's name to search for.
            content: The full interaction text.

        Returns:
            A string of surrounding context, or empty string if
            the entity name isn't found.
        """
        # Find the entity name in the content (case-insensitive)
        lower_content = content.lower()
        lower_name = entity_name.lower()
        pos = lower_content.find(lower_name)

        if pos == -1:
            # Entity name not found in content (shouldn't happen but be safe)
            return ""

        # Extract a window around the entity mention.
        # Take up to 75 chars before and 75 chars after.
        window_size = 75
        start = max(0, pos - window_size)
        end = min(len(content), pos + len(entity_name) + window_size)
        context = content[start:end].strip()

        # Clean up: try to start/end at word boundaries
        if start > 0:
            # Don't start mid-word
            space_pos = context.find(" ")
            if space_pos > 0 and space_pos < 20:
                context = context[space_pos + 1 :]

        if end < len(content):
            # Don't end mid-word
            space_pos = context.rfind(" ")
            if space_pos > len(context) - 20:
                context = context[:space_pos]

        return context

    def _chunk_text(self, content: str) -> list[str]:
        """
        Split text into chunks for embedding.

        Uses a two-level splitting strategy:
        1. First split on paragraph boundaries (double newlines)
        2. If a paragraph exceeds chunk_size, split on sentence
           boundaries (periods followed by spaces)
        3. Apply overlap between consecutive chunks

        This preserves semantic coherence: paragraphs stay together
        when possible, and sentence boundaries are preferred over
        arbitrary character splits.

        Args:
            content: The full text to chunk.

        Returns:
            A list of text chunks, each approximately chunk_size
            characters long with chunk_overlap characters of overlap.
        """
        if not content.strip():
            return []

        chunk_size = self._chunk_size
        chunk_overlap = self._chunk_overlap

        # ---------------------------------------------------------------
        # Step 1: Split into paragraphs (double newline or 2+ newlines)
        # ---------------------------------------------------------------
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            # No paragraph breaks -- treat as one block
            paragraphs = [content.strip()]

        # ---------------------------------------------------------------
        # Step 2: Group paragraphs into chunks that fit within chunk_size.
        # If a single paragraph exceeds chunk_size, split it by sentences.
        # ---------------------------------------------------------------
        raw_chunks: list[str] = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(paragraph) <= chunk_size:
                # Paragraph fits in a chunk
                if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
                    # Can add to current chunk
                    if current_chunk:
                        current_chunk += " " + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # Start a new chunk
                    if current_chunk:
                        raw_chunks.append(current_chunk)
                    current_chunk = paragraph
            else:
                # Paragraph too long -- split by sentences
                if current_chunk:
                    raw_chunks.append(current_chunk)
                    current_chunk = ""

                sentences = self._split_sentences(paragraph)
                sent_chunk = ""
                for sentence in sentences:
                    if len(sent_chunk) + len(sentence) + 1 <= chunk_size:
                        if sent_chunk:
                            sent_chunk += " " + sentence
                        else:
                            sent_chunk = sentence
                    else:
                        if sent_chunk:
                            raw_chunks.append(sent_chunk)
                        sent_chunk = sentence

                if sent_chunk:
                    current_chunk = sent_chunk

        # Don't forget the last chunk
        if current_chunk:
            raw_chunks.append(current_chunk)

        # ---------------------------------------------------------------
        # Step 3: Apply overlap between consecutive chunks.
        # Take the last chunk_overlap characters of the previous chunk
        # and prepend to the next chunk.
        # ---------------------------------------------------------------
        if chunk_overlap > 0 and len(raw_chunks) > 1:
            overlapped_chunks: list[str] = [raw_chunks[0]]
            for i in range(1, len(raw_chunks)):
                prev = raw_chunks[i - 1]
                # Take the last chunk_overlap chars of the previous chunk
                overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
                # Try to start overlap at a word boundary
                space_pos = overlap_text.find(" ")
                if space_pos > 0:
                    overlap_text = overlap_text[space_pos + 1 :]
                # Prepend overlap to the current chunk
                overlapped_chunks.append(overlap_text + " " + raw_chunks[i])
            return overlapped_chunks

        return raw_chunks

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using simple heuristics.

        Uses period-space and newline as sentence boundaries. This is
        simpler than spaCy's sentence segmenter but avoids the overhead
        of running the full pipeline on each paragraph.

        Args:
            text: Text to split into sentences.

        Returns:
            A list of sentence strings.
        """
        # Split on period followed by space or newline, or just newline
        import re

        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [s.strip() for s in sentences if s.strip()]
