# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Extraction Package
==================

This package handles all information extraction from raw text:
Named Entity Recognition (NER), fact extraction (subject-predicate-
object triples), and text summarization. These are the first
processing steps after content is ingested into the system.

The extraction pipeline turns unstructured meeting transcripts,
emails, and documents into structured knowledge that can be stored,
queried, and mined for patterns.

Processing flow:
    Raw text → NER (entities) → Fact extraction (relationships)
             → Summarization (key points)
             → All stored in SQL + vector stores

Submodules:
    - spacy_ner.py       : spaCy-based NER provider
    - regex_extractor.py : Regex + gazetteer pattern matching
    - fact_extractor.py  : Subject-predicate-object extraction
    - summarizer.py      : TextRank extractive summarizer
    - pipeline.py        : Orchestrates NER + facts + summary
"""
