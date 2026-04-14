# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Ingestion Package
=================

This package orchestrates the processing of new content into the
knowledge system. When a user ingests a file, email, meeting
transcript, or any other text, the ingestion worker coordinates
the full pipeline:

    1. Content normalization (handled by intake/ package)
    2. Extraction: NER, fact extraction, summarization
    3. Embedding: text chunking + vector generation
    4. Storage: write structured data to SQLite, vectors to LanceDB

The ingestion worker is synchronous (CPU-bound work) and processes
one interaction at a time. For batch ingestion, items are queued
and processed sequentially to avoid memory pressure on edge devices.

Submodules:
    - worker.py  : Coordinates extraction + embedding + storage
    - loaders/   : File format loaders (PDF, DOCX, TXT, etc.)
"""
