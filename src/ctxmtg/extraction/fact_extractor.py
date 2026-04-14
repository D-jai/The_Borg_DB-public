# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Simple Fact Extractor (Dependency Parse)
=========================================

This module implements the FactExtractor interface using spaCy's
dependency parse to extract subject-verb-object (SVO) triples from
text. These triples become Fact records: structured knowledge atoms
like "Alice proposed OAuth2" or "Charlie leads implementation".

This is a template-based approach (no LLM required). It works by:
1. Parsing each sentence with spaCy's dependency parser
2. Finding verbs (predicates) in each sentence
3. For each verb, finding its subject and object
4. Matching subjects and objects to known entities
5. Creating Fact triples for matched entity pairs

The approach is intentionally conservative -- it catches obvious
subject-verb-object relationships but misses complex constructions
like passive voice ("OAuth2 was proposed by Alice"), relative
clauses, or implicit relationships. Phase 2 adds LLM-assisted
fact extraction for more nuanced understanding.

Why dependency parsing instead of pattern matching?
Dependency parsing understands grammatical structure, so it can
handle varied word order and clause structure. Pattern matching
("X verb Y") is too brittle for natural language.

Depends on:
    - spacy (dependency parsing via loaded Language model)
    - ctxmtg.interfaces.extraction (FactExtractor ABC)
    - ctxmtg.models.interaction (Entity, Fact)
    - ctxmtg.storage.id_gen (generate_fact_id for deterministic IDs)

Used by:
    - ctxmtg.extraction.pipeline (BasicExtractionPipeline uses this)
"""

from __future__ import annotations

import structlog
from spacy.language import Language
from spacy.tokens import Doc, Token

# ---------------------------------------------------------------
# Import the interface and data models
# ---------------------------------------------------------------
from ctxmtg.interfaces.extraction import FactExtractor
from ctxmtg.models.interaction import Entity, Fact
from ctxmtg.storage.id_gen import generate_fact_id

# ---------------------------------------------------------------
# Logger: logs fact counts, not content (no PII in logs).
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.extraction.fact_extractor")


# =====================================================================
# SimpleFactExtractor -- Dependency-Parse-Based Fact Extraction
# =====================================================================


class SimpleFactExtractor(FactExtractor):
    """
    Extracts subject-predicate-object facts using spaCy dependency parse.

    For each sentence in the text, this extractor:
    1. Finds the main verb (ROOT of the dependency tree)
    2. Finds the nominal subject (nsubj) and direct object (dobj)
    3. Matches subject and object text to known entity names
    4. Creates a Fact triple linking the entity pair through the verb

    This produces "obvious" facts -- clear subject-verb-object
    statements. It misses complex constructions, but catches enough
    to populate the knowledge graph for Phase 1 queries.

    Usage:
        extractor = SimpleFactExtractor(nlp=spacy_model)
        facts = extractor.extract_facts(
            text="Alice proposed migrating to OAuth2.",
            entities=[alice_entity, oauth2_entity],
        )
        # → [Fact(subject="Alice", predicate="proposed", object="OAuth2")]
    """

    def __init__(self, nlp: Language | None = None) -> None:
        """
        Initialize the fact extractor.

        Args:
            nlp: A loaded spaCy Language model. If None, the extractor
                 will try to use a pre-processed Doc passed to it.
                 Typically, the pipeline shares the same spaCy model
                 across NER, fact extraction, and summarization.
        """
        self._nlp = nlp

    def extract_facts(self, text: str, entities: list[Entity]) -> list[Fact]:
        """
        Extract subject-predicate-object facts from text.

        Analyses the text using spaCy dependency parsing to find
        verb-mediated relationships between known entities. Only
        creates facts where BOTH the subject and the predicate can
        be identified; the object may be another entity or a literal.

        Args:
            text: The raw text to extract facts from.
            entities: List of Entity objects already extracted from
                      this text. Facts link these entities through
                      predicates.

        Returns:
            A list of Fact objects. Each has subject_entity_id,
            predicate, and either object_entity_id or object_literal.
            The interaction_id is empty ("") -- assigned by pipeline.
        """
        if not text or not entities:
            return []

        # Build a lookup from entity name (lowercase) to Entity object.
        # This lets us quickly check if a subject/object token matches
        # a known entity.
        entity_lookup: dict[str, Entity] = {}
        for ent in entities:
            entity_lookup[ent.name.lower()] = ent
            # Also add individual tokens for multi-word entities.
            # E.g., "Alice Smith" should match "Alice" alone.
            for token in ent.name.split():
                token_lower = token.lower().strip()
                if len(token_lower) > 2:  # Skip very short tokens
                    entity_lookup[token_lower] = ent

        # Process text with spaCy if we have a model
        if self._nlp is not None:
            doc = self._nlp(text)
        else:
            # Fallback: return empty if no model available
            logger.warning(
                "no_spacy_model_for_fact_extraction",
                error_code="CTXMTG-EXT-004",
            )
            return []

        return self._extract_from_doc(doc, entity_lookup)

    def extract_facts_from_doc(self, doc: Doc, entities: list[Entity]) -> list[Fact]:
        """
        Extract facts from a pre-processed spaCy Doc.

        This avoids re-processing text that was already run through
        spaCy for NER. The pipeline calls this instead of
        extract_facts() when it already has a Doc.

        Args:
            doc: A spaCy Doc (already processed through the pipeline).
            entities: List of Entity objects for entity matching.

        Returns:
            A list of Fact objects extracted from the Doc.
        """
        if not entities:
            return []

        # Build the entity lookup dict
        entity_lookup: dict[str, Entity] = {}
        for ent in entities:
            entity_lookup[ent.name.lower()] = ent
            for token in ent.name.split():
                token_lower = token.lower().strip()
                if len(token_lower) > 2:
                    entity_lookup[token_lower] = ent

        return self._extract_from_doc(doc, entity_lookup)

    def _extract_from_doc(self, doc: Doc, entity_lookup: dict[str, Entity]) -> list[Fact]:
        """
        Internal: extract facts from a spaCy Doc with an entity lookup.

        Iterates over sentences, finds verbs, and extracts SVO triples
        where subjects or objects match known entities.

        Args:
            doc: The processed spaCy Doc.
            entity_lookup: Dict mapping lowercase entity names to Entity objects.

        Returns:
            List of Fact objects.
        """
        facts: list[Fact] = []
        seen_triples: set[str] = set()  # Dedup tracker for triples

        # ---------------------------------------------------------------
        # Process each sentence independently. This is important because
        # dependency parse is sentence-level -- subjects and objects
        # from different sentences should not be mixed.
        # ---------------------------------------------------------------
        for sent in doc.sents:
            # Find all verbs in this sentence
            for token in sent:
                # Look for verb tokens that are predicates.
                # We consider ROOT verbs and other verbs with subjects.
                if token.pos_ != "VERB":
                    continue

                # Extract the predicate (the verb, possibly with particles)
                predicate = self._get_predicate(token)
                if not predicate:
                    continue

                # Find the subject of this verb
                subject_entity = self._find_subject(token, entity_lookup)

                # Find the object of this verb
                object_entity, object_literal = self._find_object(token, entity_lookup)

                # We need at least a subject entity to create a fact
                if subject_entity is None:
                    continue

                # We need either an object entity or a literal
                if object_entity is None and object_literal is None:
                    continue

                # Determine the object value for ID generation and dedup
                object_value = object_entity.id if object_entity else (object_literal or "")

                # Build a dedup key from the triple components
                dedup_key = f"{subject_entity.id}:{predicate}:{object_value}"
                if dedup_key in seen_triples:
                    continue
                seen_triples.add(dedup_key)

                # Extract the source span (the sentence text) for provenance
                source_span = sent.text.strip()
                # Truncate long spans to keep storage reasonable
                if len(source_span) > 200:
                    source_span = source_span[:197] + "..."

                # Generate a deterministic fact ID
                fact_id = generate_fact_id(
                    subject_entity_id=subject_entity.id,
                    predicate=predicate,
                    object_value=object_value,
                )

                # Build the Fact object
                fact = Fact(
                    id=fact_id,
                    interaction_id="",  # Assigned by pipeline
                    subject_entity_id=subject_entity.id,
                    predicate=predicate,
                    object_entity_id=(object_entity.id if object_entity else None),
                    object_literal=(object_literal if object_entity is None else None),
                    confidence=0.7,  # Template-based extraction = moderate confidence
                    source_span=source_span,
                )
                facts.append(fact)

        # ---------------------------------------------------------------
        # Second pass: possessive relationships (poss dependency).
        # Captures "Daniel Kim's internship" → (Daniel Kim, has, internship).
        # The possessor's 's is stripped at NER time, but the dependency
        # edge still exists in the parse tree. This recovers the
        # ownership signal that would otherwise be lost.
        # ---------------------------------------------------------------
        for sent in doc.sents:
            for token in sent:
                if token.dep_ != "poss":
                    continue

                # The possessor token (e.g., "Kim" in "Daniel Kim's")
                # Try to match its subtree text to a known entity.
                possessor_text = token.text.strip().lower()
                # Also try the possessor with its left children
                # (e.g., "Daniel" + "Kim" for compound names).
                full_possessor = " ".join(
                    t.text for t in token.subtree
                    if t.dep_ not in ("case", "punct")
                ).strip()
                # Strip possessive suffix if still present in subtree.
                for suffix in ("'s", "\u2019s"):
                    if full_possessor.endswith(suffix):
                        full_possessor = full_possessor[:-2].rstrip()

                possessor_entity = (
                    entity_lookup.get(full_possessor.lower())
                    or entity_lookup.get(possessor_text)
                )
                if possessor_entity is None:
                    continue

                # The possessed noun is the token's head (e.g., "internship").
                possessed = token.head.text.strip()
                if not possessed or len(possessed) <= 1:
                    continue

                # Build a dedup key and check for duplicates.
                dedup_key = f"{possessor_entity.id}:has:{possessed}"
                if dedup_key in seen_triples:
                    continue
                seen_triples.add(dedup_key)

                fact_id = generate_fact_id(
                    subject_entity_id=possessor_entity.id,
                    predicate="has",
                    object_value=possessed,
                )
                fact = Fact(
                    id=fact_id,
                    interaction_id="",
                    subject_entity_id=possessor_entity.id,
                    predicate="has",
                    object_literal=possessed,
                    confidence=0.65,  # Possessive inference = slightly lower confidence
                    source_span=sent.text.strip()[:200],
                )
                facts.append(fact)

        logger.info("fact_extraction_complete", fact_count=len(facts))
        return facts

    def _get_predicate(self, verb_token: Token) -> str | None:
        """
        Extract the predicate string from a verb token.

        Uses the verb's lemma (base form) as the predicate. Also
        includes any particle (e.g., "give up" → "give_up").

        Args:
            verb_token: A spaCy Token with pos_=="VERB".

        Returns:
            The predicate string (lemma form), or None if empty.
        """
        # Use the lemma for normalisation ("proposed" → "propose")
        predicate = verb_token.lemma_.lower().strip()

        # Check for verb particles ("give up", "carry out", etc.)
        for child in verb_token.children:
            if child.dep_ == "prt":  # Particle dependency
                predicate = f"{predicate}_{child.lemma_.lower()}"

        # Filter out very common verbs that don't carry meaning.
        # "be", "have", "do" rarely create useful facts on their own.
        stop_verbs = {"be", "have", "do", "can", "will", "would", "should", "could"}
        if predicate in stop_verbs:
            return None

        return predicate if predicate else None

    def _find_subject(self, verb_token: Token, entity_lookup: dict[str, Entity]) -> Entity | None:
        """
        Find the subject of a verb and match it to a known entity.

        Looks for nominal subjects (nsubj, nsubjpass) in the verb's
        children. If the subject text matches a known entity name,
        returns that entity.

        Args:
            verb_token: The verb Token to find a subject for.
            entity_lookup: Dict mapping lowercase names to Entity objects.

        Returns:
            The matched Entity, or None if no entity matches the subject.
        """
        # Look through the verb's direct children for subjects
        for child in verb_token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                # Try to match the subject to a known entity.
                # First try the full subtree text (for multi-word subjects).
                subtree_text = (
                    " ".join(t.text for t in child.subtree if t.dep_ not in ("det", "punct"))
                    .strip()
                    .lower()
                )

                if subtree_text in entity_lookup:
                    return entity_lookup[subtree_text]

                # Then try just the token text
                if child.text.lower() in entity_lookup:
                    return entity_lookup[child.text.lower()]

        # If no direct subject, check the verb's head (for compound
        # clauses where the subject is attached to a parent verb).
        if verb_token.dep_ in ("conj", "xcomp", "ccomp"):
            return self._find_subject(verb_token.head, entity_lookup)

        return None

    def _find_object(
        self, verb_token: Token, entity_lookup: dict[str, Entity]
    ) -> tuple[Entity | None, str | None]:
        """
        Find the object of a verb and try to match it to an entity.

        Looks for direct objects (dobj), attributes (attr), and
        prepositional objects (pobj) in the verb's children. Returns
        either a matched entity or a literal string.

        Args:
            verb_token: The verb Token to find an object for.
            entity_lookup: Dict mapping lowercase names to Entity objects.

        Returns:
            A tuple of (entity_or_none, literal_or_none). Exactly one
            of the two will be non-None if an object is found.
        """
        # Look for direct objects and attributes
        for child in verb_token.children:
            if child.dep_ in ("dobj", "attr", "oprd"):
                # Try to match to an entity
                subtree_text = " ".join(
                    t.text for t in child.subtree if t.dep_ not in ("det", "punct")
                ).strip()

                if subtree_text.lower() in entity_lookup:
                    return entity_lookup[subtree_text.lower()], None

                if child.text.lower() in entity_lookup:
                    return entity_lookup[child.text.lower()], None

                # No entity match -- return as literal (if meaningful)
                if len(subtree_text) > 2:
                    return None, subtree_text

        # Check prepositional objects ("migrating TO OAuth2")
        for child in verb_token.children:
            if child.dep_ == "prep":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        subtree_text = " ".join(
                            t.text for t in pobj.subtree if t.dep_ not in ("det", "punct")
                        ).strip()

                        if subtree_text.lower() in entity_lookup:
                            return entity_lookup[subtree_text.lower()], None

                        if pobj.text.lower() in entity_lookup:
                            return entity_lookup[pobj.text.lower()], None

                        # Return as literal if meaningful
                        if len(subtree_text) > 2:
                            return None, subtree_text

        return None, None
