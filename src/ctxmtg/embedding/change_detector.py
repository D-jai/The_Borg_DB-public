# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Content Change Detector
=======================

This module implements incremental embedding updates using BLAKE3
content hashing. When new text arrives for embedding, the detector
hashes it and checks if we've already embedded identical content.
If so, we skip re-embedding and reuse the existing embedding ID.

Why this matters: embedding is the most CPU-expensive operation in
the pipeline (~50ms per chunk on a Raspberry Pi). Skipping unchanged
content saves significant time during re-ingestion workflows, where
the user re-imports the same files with minor edits.

BLAKE3 was chosen over SHA-256 because it is:
    - 3-5x faster on the same hardware
    - Built for content hashing (unlike SHA-256 which targets crypto)
    - Produces 256-bit hashes, plenty for uniqueness
(See blake3 in pyproject.toml core dependencies.)

The detector maintains an in-memory dictionary mapping content hashes
to embedding IDs. This dictionary is ephemeral -- it lives only for
the lifetime of the detector instance. For persistent tracking, the
caller should load/save the mapping from the SQL store.

Depends on:
    - blake3 (fast content hashing library)

Used by:
    - ctxmtg.ingestion.worker (checks if content changed before re-embedding)
    - ctxmtg.embedding.onnx_embedder (optionally wraps the provider)
"""

from __future__ import annotations

import blake3
import structlog

# ---------------------------------------------------------------
# Module-level logger -- only logs hash metadata, never content.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.embedding.change_detector")


class ContentChangeDetector:
    """
    Detects whether content has changed since its last embedding.

    Uses BLAKE3 hashing to fingerprint text content. Maintains an
    in-memory mapping of content_hash → embedding_id. When new content
    arrives, we hash it and check the mapping:
        - Hash found → content unchanged → skip re-embedding, return
          the existing embedding ID.
        - Hash not found → new or changed content → mark for embedding.

    This is the incremental update mechanism described in the Round 2
    research (research/round-2/03-unified-schema-design.md § incremental).

    Usage:
        detector = ContentChangeDetector()
        # Register content that has already been embedded
        detector.register("some text", "emb-id-001")
        # Check if new content needs embedding
        if not detector.has_embedding("some text"):
            # Content is new -- embed it
            vector = embedder.embed_single("some text")
            detector.register("some text", "emb-id-002")
        else:
            # Content unchanged -- reuse existing embedding
            emb_id = detector.get_embedding_id("some text")
    """

    def __init__(self) -> None:
        """
        Create a new detector with an empty hash→ID mapping.

        The mapping starts empty. Callers should populate it with
        known content→embedding_id pairs via register() before
        using has_embedding()/get_embedding_id() to check new content.
        """
        # Maps BLAKE3 hash (hex string) → embedding ID (string).
        # Both keys and values are plain strings for easy serialisation.
        self._hash_to_embedding_id: dict[str, str] = {}

    # =================================================================
    # Public API
    # =================================================================

    def hash_content(self, content: str) -> str:
        """
        Compute the BLAKE3 hash of a text string.

        The text is UTF-8 encoded before hashing. The result is a
        64-character hex string (256 bits) that uniquely identifies
        the content. Even a single-character change produces a
        completely different hash.

        Args:
            content: The text to hash.

        Returns:
            A 64-character hexadecimal hash string.
        """
        # BLAKE3 expects bytes, so we encode the string to UTF-8.
        # The hexdigest is a fixed-length 64-char hex string.
        return blake3.blake3(content.encode("utf-8")).hexdigest()

    def register(self, content: str, embedding_id: str) -> str:
        """
        Record that a piece of content has been embedded.

        Computes the BLAKE3 hash of the content and stores the
        mapping hash → embedding_id. If the same content was
        already registered, the embedding_id is updated.

        Args:
            content:      The text that was embedded.
            embedding_id: The ID of the embedding record (from the
                          vector store or embedding metadata table).

        Returns:
            The BLAKE3 hash of the content (useful for logging).
        """
        content_hash = self.hash_content(content)
        self._hash_to_embedding_id[content_hash] = embedding_id

        logger.debug(
            "content_registered",
            content_hash=content_hash[:16],  # truncate for brevity
            embedding_id=embedding_id,
        )

        return content_hash

    def has_embedding(self, content: str) -> bool:
        """
        Check whether content has already been embedded.

        Hashes the content and looks up the hash in the mapping.
        Returns True if an embedding ID is registered for this hash.

        Args:
            content: The text to check.

        Returns:
            True if the content's hash has a registered embedding ID,
            False otherwise (meaning the content is new or changed).
        """
        content_hash = self.hash_content(content)
        return content_hash in self._hash_to_embedding_id

    def get_embedding_id(self, content: str) -> str | None:
        """
        Get the embedding ID for previously embedded content.

        Returns the embedding ID if the content was registered
        (i.e., has_embedding() would return True). Returns None
        if the content is not in the mapping.

        Args:
            content: The text whose embedding ID to retrieve.

        Returns:
            The embedding ID string, or None if not found.
        """
        content_hash = self.hash_content(content)
        return self._hash_to_embedding_id.get(content_hash)

    def register_hash(self, content_hash: str, embedding_id: str) -> None:
        """
        Register a pre-computed hash directly (for bulk loading).

        Use this when you already have the BLAKE3 hash (e.g., loaded
        from the database) and don't want to re-hash the content.

        Args:
            content_hash: A pre-computed BLAKE3 hex hash string.
            embedding_id: The ID of the corresponding embedding.
        """
        self._hash_to_embedding_id[content_hash] = embedding_id

    def remove(self, content: str) -> bool:
        """
        Remove a content→embedding mapping.

        Used when content is deleted or its embedding is invalidated
        (e.g., model version changed and re-embedding is needed).

        Args:
            content: The text to remove from the mapping.

        Returns:
            True if the content was found and removed, False if it
            wasn't in the mapping.
        """
        content_hash = self.hash_content(content)
        if content_hash in self._hash_to_embedding_id:
            del self._hash_to_embedding_id[content_hash]
            return True
        return False

    def clear(self) -> None:
        """
        Remove all registered hash→embedding_id mappings.

        Useful when the embedding model changes and all content
        needs to be re-embedded from scratch.
        """
        count = len(self._hash_to_embedding_id)
        self._hash_to_embedding_id.clear()
        logger.info("change_detector_cleared", removed_count=count)

    @property
    def size(self) -> int:
        """
        Return the number of registered content→embedding mappings.

        This tells you how many distinct pieces of content the
        detector is tracking. Useful for health monitoring.

        Returns:
            The number of entries in the hash mapping.
        """
        return len(self._hash_to_embedding_id)

    def get_all_hashes(self) -> dict[str, str]:
        """
        Return a copy of all hash→embedding_id mappings.

        Useful for persisting the mapping to disk or the SQL store
        so it can be reloaded on restart.

        Returns:
            A dict mapping content_hash → embedding_id.
        """
        return dict(self._hash_to_embedding_id)

    def load_mappings(self, mappings: dict[str, str]) -> int:
        """
        Bulk-load hash→embedding_id mappings (e.g., from database).

        Merges the provided mappings into the existing mapping.
        Existing entries with the same hash are overwritten.

        Args:
            mappings: Dict mapping content_hash → embedding_id.

        Returns:
            The number of mappings loaded.
        """
        self._hash_to_embedding_id.update(mappings)
        logger.info("mappings_loaded", count=len(mappings))
        return len(mappings)
