# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
spaCy Named Entity Recognition Provider
========================================

This module implements the NERProvider interface using spaCy's
built-in Named Entity Recognition (NER) pipeline. It loads a
spaCy language model (default: en_core_web_sm) and extracts
entities from raw text, mapping spaCy's entity labels to our
EntityType enum.

spaCy's NER is the primary entity extraction mechanism in Phase 1.
It finds standard entity types: people (PERSON), organizations (ORG),
locations (GPE/LOC), dates (DATE), events (EVENT), etc. For entities
that spaCy misses (emails, URLs, phone numbers), the RegexExtractor
provides a complementary extraction layer.

In the system architecture, this is the first step of the extraction
pipeline. The pipeline runs spaCy NER, then regex extraction, merges
the results, and proceeds to fact extraction and summarization.

spaCy label → EntityType mapping:
    PERSON   → EntityType.PERSON
    ORG      → EntityType.ORG
    GPE      → EntityType.LOCATION  (geo-political entities: countries, cities)
    LOC      → EntityType.LOCATION  (non-GPE locations: mountains, rivers)
    FAC      → EntityType.LOCATION  (facilities: buildings, airports)
    EVENT    → EntityType.EVENT
    PRODUCT  → EntityType.TOOL      (products, software tools)
    WORK_OF_ART → EntityType.PROJECT (creative works, project names)
    LAW      → EntityType.OTHER     (legal documents, laws)
    DATE     → EntityType.EVENT     (temporal references, mapped to event)
    NORP     → EntityType.ORG       (nationalities, religious/political groups)
    LANGUAGE → EntityType.OTHER

Depends on:
    - spacy (NLP pipeline and NER model)
    - ctxmtg.interfaces.extraction (NERProvider ABC)
    - ctxmtg.models.interaction (Entity, EntityType)
    - ctxmtg.constants (DEFAULT_SPACY_MODEL)
    - ctxmtg.exceptions (ExtractionError)

Used by:
    - ctxmtg.extraction.pipeline (BasicExtractionPipeline uses this)
"""

from __future__ import annotations

# ---------------------------------------------------------------
# Import spaCy for NLP processing. The model is loaded lazily
# (on first use or explicit load) to avoid slow import times.
# ---------------------------------------------------------------
import spacy
import structlog
from spacy.language import Language
from spacy.tokens import Doc

# ---------------------------------------------------------------
# Import the interface this class implements and the data models
# it produces. Entity is the output type; EntityType is the enum
# we map spaCy labels to.
# ---------------------------------------------------------------
from ctxmtg.constants import DEFAULT_SPACY_MODEL
from ctxmtg.exceptions import ExtractionError
from ctxmtg.interfaces.extraction import NERProvider
from ctxmtg.models.interaction import Entity, EntityType

# ---------------------------------------------------------------
# Logger for this module. We log entity counts, model loading,
# and any errors -- but NEVER log entity names (PII concern).
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.extraction.spacy_ner")


# =====================================================================
# SpaCy NER label → EntityType mapping
# =====================================================================

# This dict maps spaCy's built-in NER labels to our EntityType enum.
# spaCy uses the OntoNotes label scheme which has ~18 entity types.
# We consolidate these into our simpler 8-type taxonomy.
SPACY_LABEL_MAP: dict[str, EntityType] = {
    "PERSON": EntityType.PERSON,  # People, including fictional
    "ORG": EntityType.ORG,  # Companies, agencies, institutions
    "GPE": EntityType.LOCATION,  # Countries, cities, states
    "LOC": EntityType.LOCATION,  # Non-GPE locations (mountains, rivers)
    "FAC": EntityType.LOCATION,  # Buildings, airports, highways
    "EVENT": EntityType.EVENT,  # Named events (wars, sports events)
    "PRODUCT": EntityType.TOOL,  # Objects, vehicles, software
    "WORK_OF_ART": EntityType.PROJECT,  # Titles of books, songs, projects
    "NORP": EntityType.ORG,  # Nationalities, religious/political groups
    "LAW": EntityType.OTHER,  # Named documents, laws, treaties
    "LANGUAGE": EntityType.OTHER,  # Any named language
    "DATE": EntityType.EVENT,  # Absolute or relative dates/periods
}


# =====================================================================
# SpacyNERProvider -- spaCy-based NER Implementation
# =====================================================================


class SpacyNERProvider(NERProvider):
    """
    Named Entity Recognition provider powered by spaCy.

    Loads a spaCy language model (default: en_core_web_sm, ~12MB)
    and uses its NER component to extract entities from text. Each
    entity gets a confidence score derived from spaCy's internal
    scoring, and a provenance string identifying which model found it.

    The provider maps spaCy's 18+ entity labels to our 8-type
    EntityType enum, filtering out types the caller doesn't need.

    Usage:
        ner = SpacyNERProvider()
        ner.load()  # loads the spaCy model
        entities = ner.extract_entities(
            "Alice from Acme Corp visited New York.",
            entity_types=["person", "org", "location"],
        )
        # → [Entity(name="Alice", entity_type=PERSON), ...]

    Why confidence scores?
    spaCy doesn't expose per-entity confidence directly in all
    versions, so we use a heuristic: longer entity spans and
    entities matching common spaCy labels get higher confidence.
    See _estimate_confidence() for details.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialize the spaCy NER provider.

        Does NOT load the model yet -- call load() explicitly or
        let extract_entities() load it lazily on first call.

        Args:
            model_name: Name of the spaCy model to load.
                        Defaults to DEFAULT_SPACY_MODEL ("en_core_web_sm").
        """
        # Store the model name for lazy loading
        self._model_name: str = model_name or DEFAULT_SPACY_MODEL

        # The loaded spaCy language pipeline (None until load() is called)
        self._nlp: Language | None = None

        # Provenance string for entities found by this provider.
        # Includes the model name so we know which model found each entity.
        self._provenance: str = f"spacy:{self._model_name}"

    def load(self) -> None:
        """
        Load the spaCy language model into memory.

        This can be called explicitly before extraction, or it will
        be called automatically on the first extract_entities() call.
        Loading takes ~1-2 seconds for en_core_web_sm.

        Raises:
            ExtractionError: If the model cannot be loaded (not installed,
                             corrupted, etc.).
        """
        try:
            # Load the full spaCy pipeline. en_core_web_sm includes
            # tokenizer, tagger, parser, NER, and lemmatizer.
            self._nlp = spacy.load(self._model_name)
            logger.info(
                "spacy_model_loaded",
                model=self._model_name,
                pipeline_components=self._nlp.pipe_names,
            )
        except OSError as exc:
            # Model not found -- usually means it hasn't been downloaded.
            # python -m spacy download en_core_web_sm
            logger.error(
                "spacy_model_not_found",
                error_code="CTXMTG-EXT-001",
                model=self._model_name,
                error=str(exc),
            )
            raise ExtractionError(
                f"Failed to load spaCy model '{self._model_name}'. "
                f"Did you run: python -m spacy download {self._model_name}?",
                error_code="CTXMTG-EXT-001",
            ) from exc

    def _ensure_loaded(self) -> Language:
        """
        Ensure the spaCy model is loaded, loading it lazily if needed.

        Returns:
            The loaded spaCy Language pipeline.

        Raises:
            ExtractionError: If loading fails.
        """
        if self._nlp is None:
            self.load()
        # After load(), _nlp should never be None. Assert for safety.
        assert self._nlp is not None
        return self._nlp

    def get_nlp(self) -> Language:
        """
        Get the loaded spaCy Language pipeline.

        Public accessor for other modules (like FactExtractor and
        Summarizer) that need access to the same spaCy model to
        avoid loading it multiple times.

        Returns:
            The loaded spaCy Language pipeline.
        """
        return self._ensure_loaded()

    def process_text(self, text: str) -> Doc:
        """
        Process text through the spaCy pipeline and return the Doc.

        Public method so other extractors (fact_extractor, summarizer)
        can reuse the same processed Doc without re-processing.

        Args:
            text: The raw text to process.

        Returns:
            A spaCy Doc object with tokens, entities, and dependency parse.
        """
        nlp = self._ensure_loaded()
        return nlp(text)

    def extract_entities(self, text: str, entity_types: list[str] | None = None) -> list[Entity]:
        """
        Extract named entities from text using spaCy NER.

        Processes the input text through spaCy's NER pipeline, maps
        each detected entity label to our EntityType enum, and filters
        by the requested entity types. Each entity gets a confidence
        score and provenance information.

        Args:
            text: The raw text to extract entities from. Can be any
                  length, but very long texts (>100K chars) may be slow.
            entity_types: Optional list of entity type strings to filter
                          by (e.g., ["person", "org"]). If None, all
                          supported types are returned.

        Returns:
            A list of Entity objects. Each has:
            - name: the entity text as found in the document
            - entity_type: one of our EntityType enum values
            - confidence: estimated confidence score (0.0-1.0)
            - provenance: string identifying this extractor
            Note: entities do NOT have id or interaction_id set --
            the pipeline assigns those later.
        """
        # Ensure the spaCy model is loaded
        nlp = self._ensure_loaded()

        # Process the text through spaCy's full pipeline
        doc = nlp(text)

        # Build the set of accepted entity types for filtering.
        # If entity_types is None, accept all types we can map to.
        accepted_types: set[str] | None = None
        if entity_types is not None:
            # Normalize to lowercase for case-insensitive matching
            accepted_types = {t.lower() for t in entity_types}

        # ---------------------------------------------------------------
        # Iterate over spaCy's detected entities and convert each one
        # to our Entity model. We skip entities whose spaCy label we
        # don't have a mapping for, and filter by requested types.
        # ---------------------------------------------------------------
        entities: list[Entity] = []
        seen_names: set[str] = set()  # Track seen names for dedup within this text

        for ent in doc.ents:
            # Look up the EntityType for this spaCy label
            entity_type = SPACY_LABEL_MAP.get(ent.label_)
            if entity_type is None:
                # Unknown spaCy label (e.g., MONEY, CARDINAL, ORDINAL, PERCENT).
                # These are numeric types we don't track as entities.
                continue

            # Filter by requested entity types if specified
            if accepted_types is not None and entity_type.value not in accepted_types:
                continue

            # Clean up the entity text: strip whitespace, remove
            # trailing possessives ('s, 's) that spaCy includes in
            # entity spans (e.g., "Daniel Kim's" → "Daniel Kim").
            entity_name = ent.text.strip()
            if entity_name.endswith("'s") or entity_name.endswith("\u2019s"):
                entity_name = entity_name[:-2].rstrip()
            if not entity_name:
                continue

            # Skip duplicates within the same text (case-insensitive).
            # The same person mentioned 5 times in a paragraph should
            # produce one entity, not five.
            name_key = entity_name.lower()
            if name_key in seen_names:
                continue
            seen_names.add(name_key)

            # Estimate a confidence score for this entity.
            # See _estimate_confidence() for the heuristic logic.
            confidence = self._estimate_confidence(ent)

            # Build the Entity object. Note: id and interaction_id are
            # left empty ("") -- the pipeline will assign them later
            # using generate_entity_id().
            entity = Entity(
                id="",  # Assigned by pipeline
                interaction_id="",  # Assigned by pipeline
                name=entity_name,
                entity_type=entity_type,
                confidence=confidence,
                provenance=self._provenance,
            )
            entities.append(entity)

        logger.info(
            "spacy_ner_complete",
            entity_count=len(entities),
            text_length=len(text),
        )
        return entities

    def _estimate_confidence(self, ent: spacy.tokens.Span) -> float:
        """
        Estimate a confidence score for a spaCy entity.

        spaCy's small model (en_core_web_sm) doesn't expose per-entity
        confidence scores directly. We use a heuristic based on:
        1. Entity label reliability: PERSON and ORG are more reliable
           than PRODUCT or WORK_OF_ART in the small model.
        2. Entity text length: very short entities (1-2 chars) are
           more likely to be false positives.

        This is imperfect but gives us a useful signal for downstream
        filtering. Phase 2 (LLM-assisted NER) will provide more
        accurate confidence scores.

        Args:
            ent: A spaCy Span object representing a detected entity.

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        # Base confidence varies by label reliability.
        # These values are rough estimates from spaCy benchmarks.
        label_confidence: dict[str, float] = {
            "PERSON": 0.90,
            "ORG": 0.85,
            "GPE": 0.90,
            "LOC": 0.80,
            "FAC": 0.75,
            "EVENT": 0.70,
            "PRODUCT": 0.65,
            "WORK_OF_ART": 0.60,
            "NORP": 0.80,
            "LAW": 0.70,
            "LANGUAGE": 0.85,
            "DATE": 0.85,
        }
        base = label_confidence.get(ent.label_, 0.60)

        # Penalize very short entities (likely noise).
        # Single characters or 2-char entities are suspicious.
        text_len = len(ent.text.strip())
        if text_len <= 2:
            base *= 0.6
        elif text_len <= 4:
            base *= 0.8

        # Clamp to valid range [0.0, 1.0]
        return min(max(base, 0.0), 1.0)
