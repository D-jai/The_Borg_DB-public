# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Extraction Interface ABCs
=========================

This module defines the abstract base classes for the text extraction
pipeline: NER (Named Entity Recognition), fact extraction, summarization,
and the full pipeline orchestrator.

The extraction pipeline is the system's "reading comprehension" --
it takes raw text (meetings, emails, documents) and extracts structured
knowledge: who was mentioned (entities), what was said about them
(facts), and a brief summary of the content.

Phase 1 implements these with spaCy + regex (no LLM required).
Phase 2 adds LLM-assisted extraction for higher quality results.
The interfaces remain the same -- only the implementations change.

Depends on:
    - abc (Python's Abstract Base Class machinery)
    - ctxmtg.models.interaction (Entity, Fact, Interaction, ExtractionResult)
    - ctxmtg.models.profile (DomainProfile -- controls extraction behavior)

Used by:
    - ctxmtg.extraction.spacy_ner (implements NERProvider)
    - ctxmtg.extraction.regex_extractor (implements NERProvider)
    - ctxmtg.extraction.fact_extractor (implements FactExtractor)
    - ctxmtg.extraction.summarizer (implements Summarizer)
    - ctxmtg.extraction.pipeline (implements ExtractionPipeline)
    - ctxmtg.ingestion.worker (uses ExtractionPipeline to process interactions)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# ---------------------------------------------------------------
# Import the data models that flow through the extraction pipeline.
# The pipeline reads Interactions and produces Entities, Facts,
# and ExtractionResults. The DomainProfile controls what to extract.
# ---------------------------------------------------------------
from ctxmtg.models.interaction import (
    Entity,
    ExtractionResult,
    Fact,
    Interaction,
)
from ctxmtg.models.profile import DomainProfile

# =====================================================================
# NERProvider ABC -- Named Entity Recognition Interface
# =====================================================================


class NERProvider(ABC):
    """
    Named Entity Recognition provider.

    Extracts named entities (people, organizations, locations, etc.)
    from raw text. Different implementations use different strategies:
    - SpacyNERProvider: uses spaCy's NER pipeline (Phase 1)
    - RegexExtractor: uses regex patterns for structured entities
      like emails, URLs, dates (Phase 1)
    - LLMNERProvider: uses a local LLM for domain-specific NER (Phase 2)

    Multiple NER providers can be layered together in the extraction
    pipeline -- spaCy finds standard entities, regex finds structured
    patterns, and the LLM catches domain-specific entities that both miss.

    Usage:
        ner = SpacyNERProvider(model_name="en_core_web_sm")
        entities = ner.extract_entities(
            text="Alice from Acme Corp proposed OAuth2.",
            entity_types=["person", "org", "tool"],
        )
    """

    @abstractmethod
    def extract_entities(
        self, text: str, entity_types: list[str] | None = None
    ) -> list[Entity]:
        """
        Extract named entities from text.

        Scans the input text and identifies named entities (people,
        organizations, projects, tools, etc.). Each entity includes
        a confidence score and provenance information.

        Args:
            text: The raw text to extract entities from.
            entity_types: Optional list of entity types to look for.
                          If None, extract all supported entity types.
                          If provided, only extract these specific types
                          (e.g., ["person", "org"]).

        Returns:
            A list of Entity objects found in the text. Each entity
            has a name, type, confidence score, and provenance string
            identifying which extractor found it.
        """
        ...


# =====================================================================
# FactExtractor ABC -- Subject-Predicate-Object Triple Extraction
# =====================================================================


class FactExtractor(ABC):
    """
    Subject-predicate-object fact extraction.

    Extracts structured knowledge triples from text, given a set of
    known entities. For example, from "Alice proposed OAuth2", it
    extracts the triple: (Alice, proposed, OAuth2).

    Facts link entities together through predicates (verbs or
    relationships). They form the structured knowledge graph that
    the SQL store manages.

    Phase 1: SimpleFactExtractor uses spaCy dependency parsing
    to find subject-verb-object patterns (template-based).
    Phase 2: LLMFactExtractor uses a local LLM for more complex
    fact extraction including implicit relationships.

    Usage:
        extractor = SimpleFactExtractor()
        facts = extractor.extract_facts(
            text="Alice proposed migrating to OAuth2.",
            entities=[alice_entity, oauth2_entity],
        )
    """

    @abstractmethod
    def extract_facts(self, text: str, entities: list[Entity]) -> list[Fact]:
        """
        Extract facts from text, given known entities.

        Analyzes the text to find relationships between the provided
        entities. Each fact is a subject-predicate-object triple where
        the subject is always an entity, the predicate is a verb or
        relationship, and the object is either another entity or a
        literal string.

        Args:
            text: The raw text to extract facts from.
            entities: List of Entity objects already found in the text
                      by the NER provider. Facts link these entities
                      through predicates.

        Returns:
            A list of Fact objects representing the relationships
            found between entities in the text. Each fact includes
            a confidence score and the source text span.
        """
        ...


# =====================================================================
# Summarizer ABC -- Text Summarization Interface
# =====================================================================


class Summarizer(ABC):
    """
    Text summarization.

    Produces a concise summary of input text. Phase 1 uses an
    extractive approach (TextRank: picks the most important sentences)
    that requires no LLM. Phase 2 adds abstractive summarization
    via the local LLM for more natural-sounding summaries.

    Summaries are stored alongside interactions and used for:
    - Quick preview of interactions in search results
    - Entity context building (what was the entity mentioned in?)
    - Farming pattern descriptions

    Usage:
        summarizer = TextRankSummarizer()
        summary = summarizer.summarize(
            text="Alice proposed migrating to OAuth2. Bob raised concerns...",
            max_length=200,
        )
    """

    @abstractmethod
    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        Produce a summary of the input text.

        Generates a concise summary that captures the key points
        of the input text. The summary length is bounded by max_length
        characters.

        Args:
            text: The raw text to summarize.
            max_length: Maximum length of the summary in characters.
                        Default is 200 characters.

        Returns:
            A summary string of at most max_length characters.
            Returns an empty string if the input is empty or
            cannot be summarized.
        """
        ...


# =====================================================================
# ExtractionPipeline ABC -- Full Extraction Orchestrator
# =====================================================================


class ExtractionPipeline(ABC):
    """
    Full extraction pipeline: NER + facts + summary + chunking.

    Constructed with a DomainProfile that controls entity_types,
    custom_patterns, and extraction behavior. The profile is fixed
    for the lifetime of the pipeline instance (swap profiles by
    creating a new pipeline instance).

    The pipeline orchestrates the complete extraction process:
    1. Run NER (spaCy + regex + domain patterns) → entities
    2. Merge and deduplicate entities within the interaction
    3. Build rich entity context for vector embedding
    4. Extract facts (subject-predicate-object triples)
    5. Summarize the interaction content
    6. Chunk the text for embedding

    The result is an ExtractionResult containing everything needed
    to store the processed interaction in both SQL and vector stores.

    Usage:
        profile = ProfileLoader.load("general")
        pipeline = BasicExtractionPipeline(profile)
        result = pipeline.process(interaction)
        # result.entities, result.facts, result.summary, result.chunks
    """

    @abstractmethod
    def __init__(self, profile: DomainProfile) -> None:
        """
        Initialize with a domain profile.

        The profile controls extraction behavior:
        - profile.ner.entity_types: which entity types to extract
        - profile.ner.custom_patterns: domain-specific regex patterns
        - profile.embedding.chunk_size: target chunk size for embedding
        - profile.embedding.chunk_overlap: overlap between chunks

        The profile is fixed for the lifetime of this pipeline instance.
        To switch profiles, create a new pipeline instance.

        Args:
            profile: The DomainProfile that controls extraction behavior.
        """
        ...

    @abstractmethod
    def process(self, interaction: Interaction) -> ExtractionResult:
        """
        Run the full extraction pipeline on an interaction.

        Takes a raw Interaction and produces a complete ExtractionResult
        containing entities, facts, a summary, and text chunks ready
        for embedding. This is the main entry point for the extraction
        subsystem.

        Args:
            interaction: The Interaction to process. Must have non-empty
                         content for extraction to produce results.

        Returns:
            An ExtractionResult containing:
            - entities: all named entities found in the interaction
            - facts: subject-predicate-object triples linking entities
            - summary: a concise summary of the interaction content
            - chunks: text segments ready for vector embedding
        """
        ...
