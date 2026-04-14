# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Regex-Based Entity Extractor
=============================

This module implements the NERProvider interface using regular
expressions to extract structured entities that spaCy's NER
typically misses: email addresses, URLs, dates (ISO format),
phone numbers, and version numbers.

While spaCy excels at detecting people, organizations, and locations
from natural language context, it doesn't recognise structured
patterns like "alice@example.com" or "v2.3.1". This regex extractor
complements spaCy by catching those patterns.

In the extraction pipeline, this runs AFTER spaCy NER. The pipeline
then merges entities from both providers, deduplicating by name.

The extractor also supports domain-specific custom patterns loaded
from the DomainProfile's NERConfig. For example, a legal profile
might define a pattern for case numbers ("\\d{2}-CV-\\d{4}").

Built-in patterns:
    - EMAIL:     RFC 5322 simplified email pattern → EntityType.OTHER
    - URL:       http(s)://... or www. patterns    → EntityType.OTHER
    - DATE_ISO:  YYYY-MM-DD dates                  → EntityType.EVENT
    - PHONE:     Various phone number formats      → EntityType.OTHER
    - VERSION:   Semantic versioning (v1.2.3)       → EntityType.OTHER

Depends on:
    - re (Python stdlib -- regular expressions)
    - ctxmtg.interfaces.extraction (NERProvider ABC)
    - ctxmtg.models.interaction (Entity, EntityType)

Used by:
    - ctxmtg.extraction.pipeline (BasicExtractionPipeline uses this)
"""

from __future__ import annotations

import re

import structlog

# ---------------------------------------------------------------
# Import the interface this class implements and the data models.
# ---------------------------------------------------------------
from ctxmtg.interfaces.extraction import NERProvider
from ctxmtg.models.interaction import Entity, EntityType

# ---------------------------------------------------------------
# Logger for this module. Logs pattern match counts per type.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.extraction.regex_extractor")


# =====================================================================
# Built-in Regex Patterns
# =====================================================================

# Each pattern is a tuple of (compiled_regex, entity_type, provenance_label).
# These cover the most common structured entities that spaCy misses.

# Email addresses: simplified RFC 5322 pattern.
# Matches "alice@example.com", "bob.smith@acme.co.uk", etc.
_EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")

# URLs: http(s) or www patterns.
# Matches "https://example.com/path", "www.example.com", etc.
_URL_PATTERN = re.compile(r"(?:https?://|www\.)[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+")

# ISO dates: YYYY-MM-DD format.
# Matches "2024-03-15", "2023-12-01", etc.
_DATE_ISO_PATTERN = re.compile(r"\b\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b")

# Phone numbers: various formats including international.
# Matches "+1-555-123-4567", "(555) 123-4567", "555.123.4567", etc.
_PHONE_PATTERN = re.compile(
    r"(?:\+\d{1,3}[-.\s]?)?"  # Optional country code
    r"(?:\(?\d{2,4}\)?[-.\s]?)?"  # Optional area code
    r"\d{3,4}[-.\s]?\d{3,4}\b"  # Main number
)

# Version numbers: semantic versioning patterns.
# Matches "v1.2.3", "v2.0", "version 3.4.5-beta", etc.
_VERSION_PATTERN = re.compile(
    r"\bv(?:ersion\s*)?(\d+(?:\.\d+){1,3}(?:-[a-zA-Z0-9.]+)?)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------
# Bundle all built-in patterns with their EntityType and provenance.
# ---------------------------------------------------------------
_BUILTIN_PATTERNS: list[tuple[re.Pattern[str], EntityType, str]] = [
    (_EMAIL_PATTERN, EntityType.OTHER, "regex:email_pattern"),
    (_URL_PATTERN, EntityType.OTHER, "regex:url_pattern"),
    (_DATE_ISO_PATTERN, EntityType.EVENT, "regex:date_iso_pattern"),
    (_PHONE_PATTERN, EntityType.OTHER, "regex:phone_pattern"),
    (_VERSION_PATTERN, EntityType.OTHER, "regex:version_pattern"),
]


# =====================================================================
# RegexExtractor -- Regex-based NER Implementation
# =====================================================================


class RegexExtractor(NERProvider):
    """
    Regex-based entity extractor for structured patterns.

    Extracts entities that NLP-based NER providers typically miss:
    email addresses, URLs, ISO dates, phone numbers, and version
    numbers. Also supports custom regex patterns from domain profiles.

    This provider complements SpacyNERProvider in the extraction
    pipeline. spaCy handles natural language entities (people, places);
    regex handles structured patterns (emails, URLs, dates).

    Confidence scores for regex matches are high (0.95) because regex
    patterns are precise -- if the pattern matches, it's almost
    certainly the correct entity type. The score is slightly below
    1.0 to account for occasional false positives (e.g., a phone
    pattern matching a random number sequence).

    Usage:
        extractor = RegexExtractor()
        # Optionally add custom patterns from a domain profile:
        extractor = RegexExtractor(custom_patterns=[
            {"pattern": r"\\d{2}-CV-\\d{4}", "entity_type": "other"}
        ])
        entities = extractor.extract_entities("Contact alice@acme.com")
    """

    # Default confidence score for regex-matched entities.
    # Regex matches are very precise, so confidence is high.
    DEFAULT_CONFIDENCE = 0.95

    def __init__(
        self,
        custom_patterns: list[dict[str, str]] | None = None,
    ) -> None:
        """
        Initialize the regex extractor with optional custom patterns.

        Args:
            custom_patterns: Optional list of custom regex patterns from
                             a DomainProfile's NERConfig. Each dict must
                             have "pattern" (regex string) and
                             "entity_type" (EntityType value string).
        """
        # Start with all built-in patterns
        self._patterns: list[tuple[re.Pattern[str], EntityType, str]] = list(_BUILTIN_PATTERNS)

        # Compile and add custom patterns from the domain profile
        if custom_patterns:
            for cp in custom_patterns:
                self._add_custom_pattern(cp)

    def _add_custom_pattern(self, pattern_config: dict[str, str]) -> None:
        """
        Compile and add a custom regex pattern.

        Args:
            pattern_config: Dict with "pattern" (regex) and "entity_type"
                            (string matching an EntityType value).

        Logs a warning and skips if the pattern is invalid.
        """
        pattern_str = pattern_config.get("pattern", "")
        entity_type_str = pattern_config.get("entity_type", "other")

        if not pattern_str:
            logger.warning(
                "empty_custom_pattern",
                error_code="CTXMTG-EXT-003",
                config=str(pattern_config),
            )
            return

        # Try to map the entity_type string to our EntityType enum
        try:
            entity_type = EntityType(entity_type_str.lower())
        except ValueError:
            # Unknown entity type -- default to OTHER
            entity_type = EntityType.OTHER
            logger.warning(
                "unknown_entity_type_in_pattern",
                error_code="CTXMTG-EXT-003",
                entity_type=entity_type_str,
                defaulting_to="other",
            )

        # Compile the regex pattern
        try:
            compiled = re.compile(pattern_str)
        except re.error as exc:
            logger.warning(
                "invalid_custom_pattern",
                error_code="CTXMTG-EXT-003",
                pattern=pattern_str,
                error=str(exc),
            )
            return

        # Build a provenance label for this custom pattern
        provenance = f"regex:custom:{pattern_str[:30]}"

        self._patterns.append((compiled, entity_type, provenance))
        logger.info(
            "custom_pattern_added",
            pattern_preview=pattern_str[:30],
            entity_type=entity_type.value,
        )

    def extract_entities(self, text: str, entity_types: list[str] | None = None) -> list[Entity]:
        """
        Extract entities from text using regex patterns.

        Runs all registered patterns (built-in + custom) against the
        input text. Each match becomes an Entity with the type defined
        by the pattern and a high confidence score.

        Deduplicates by entity name within the same text (case-insensitive).

        Args:
            text: The raw text to scan for structured entities.
            entity_types: Optional list of entity type strings to filter by.
                          If None, returns all matched entities.

        Returns:
            A list of Entity objects. Each has name, entity_type,
            confidence, and provenance set. id and interaction_id are
            empty strings (assigned by the pipeline later).
        """
        # Build filter set if entity types are specified
        accepted_types: set[str] | None = None
        if entity_types is not None:
            accepted_types = {t.lower() for t in entity_types}

        entities: list[Entity] = []
        seen_names: set[str] = set()  # Dedup tracker

        # ---------------------------------------------------------------
        # Run each pattern against the text and collect matches.
        # ---------------------------------------------------------------
        for pattern, entity_type, provenance in self._patterns:
            # Filter by requested entity types if specified
            if accepted_types is not None and entity_type.value not in accepted_types:
                continue

            # Find all matches for this pattern
            for match in pattern.finditer(text):
                # Get the matched text. For version patterns with groups,
                # use the first group; otherwise use the full match.
                if match.lastindex and match.lastindex >= 1:
                    entity_name = match.group(1).strip()
                else:
                    entity_name = match.group(0).strip()

                if not entity_name:
                    continue

                # Deduplicate by lowercase name
                name_key = entity_name.lower()
                if name_key in seen_names:
                    continue
                seen_names.add(name_key)

                # Build the Entity with high confidence (regex is precise)
                entity = Entity(
                    id="",  # Assigned by pipeline
                    interaction_id="",  # Assigned by pipeline
                    name=entity_name,
                    entity_type=entity_type,
                    confidence=self.DEFAULT_CONFIDENCE,
                    provenance=provenance,
                )
                entities.append(entity)

        logger.info(
            "regex_extraction_complete",
            entity_count=len(entities),
            text_length=len(text),
        )
        return entities
