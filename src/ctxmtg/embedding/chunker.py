# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Text Chunker
=============

This module splits long text into smaller, overlapping chunks that
are suitable for embedding. Embedding models have a maximum input
length (e.g., 256 tokens for the default all-MiniLM-L6-v2 config),
so any text longer than that must be split before embedding.

The chunking strategy preserves semantic boundaries:
    1. First, split on paragraph boundaries (double newlines).
    2. If a paragraph is still too long, split on sentence boundaries
       (periods, question marks, exclamation marks followed by space).
    3. If a sentence is still too long, split on word boundaries
       (spaces) as a last-resort fallback.

Each chunk includes character-offset metadata (start, end) so the
system can trace any embedding back to the exact portion of the
original text it came from. This is critical for provenance tracking.

Overlap between consecutive chunks prevents information loss at
chunk boundaries. The default overlap of 32 characters means that
each chunk repeats about 25 words of context from the previous chunk.

Depends on:
    - re (regular expressions for splitting on sentence boundaries)

Used by:
    - ctxmtg.embedding.onnx_embedder (chunks text before embedding)
    - ctxmtg.ingestion.worker (chunks interaction content for storage)
    - ctxmtg.extraction.pipeline (chunks text for the extraction step)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from ctxmtg.exceptions import EmbeddingError

# ---------------------------------------------------------------
# Module-level logger -- no PII, only chunk metadata in logs.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.embedding.chunker")


# ---------------------------------------------------------------
# TextChunk dataclass: represents one chunk with offset metadata.
# The start/end offsets are character positions in the original text,
# allowing the caller to reconstruct exactly which portion of the
# source this chunk came from.
# ---------------------------------------------------------------
@dataclass
class TextChunk:
    """
    A text chunk with character-offset metadata.

    Attributes:
        text:  The chunk's text content.
        start: Character offset of the chunk's first character in the
               original text (0-based, inclusive).
        end:   Character offset of the chunk's last character + 1 in
               the original text (0-based, exclusive). So
               original_text[start:end] == text (approximately,
               overlap trimming may cause slight differences).
        index: The sequential index of this chunk (0, 1, 2, …).
    """

    text: str
    start: int
    end: int
    index: int


# ---------------------------------------------------------------
# Regex for splitting text into sentences. Matches a sentence-
# ending punctuation mark (. ? !) followed by one or more spaces,
# then a capital letter or end of string. This captures the split
# point without consuming the capital letter.
# ---------------------------------------------------------------
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# ---------------------------------------------------------------
# Regex for paragraph boundaries: two or more consecutive newlines,
# possibly with whitespace between them.
# ---------------------------------------------------------------
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


class TextChunker:
    """
    Splits text into overlapping chunks for embedding.

    The chunker tries to respect natural text boundaries so that
    each chunk contains a coherent piece of text:
        1. Split on paragraph boundaries first (double newlines).
        2. If a paragraph exceeds chunk_size, split on sentences.
        3. If a sentence exceeds chunk_size, split on words.

    Adjacent chunks overlap by chunk_overlap characters so that
    information at the boundary of two chunks is captured in both.
    This means no important context is lost at split points.

    Usage:
        chunker = TextChunker(chunk_size=256, chunk_overlap=32)
        chunks = chunker.chunk("Long text goes here...")
        for c in chunks:
            print(f"Chunk {c.index}: [{c.start}:{c.end}] {c.text[:50]}...")
    """

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
    ) -> None:
        """
        Configure the chunker.

        Args:
            chunk_size:    Target maximum number of characters per chunk.
                           Chunks may be slightly larger if a sentence
                           cannot be split without breaking words.
            chunk_overlap: Number of characters to repeat at the start
                           of each chunk from the end of the previous
                           chunk. Must be less than chunk_size.

        Raises:
            ValueError: If chunk_overlap >= chunk_size or either is <= 0.
        """
        # Validate parameters to catch configuration mistakes early.
        if chunk_size <= 0:
            raise EmbeddingError(
                f"chunk_size must be > 0, got {chunk_size}",
                error_code="CTXMTG-EXT-007",
            )
        if chunk_overlap < 0:
            raise EmbeddingError(
                f"chunk_overlap must be >= 0, got {chunk_overlap}",
                error_code="CTXMTG-EXT-007",
            )
        if chunk_overlap >= chunk_size:
            raise EmbeddingError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})",
                error_code="CTXMTG-EXT-007",
            )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    # =================================================================
    # Public API
    # =================================================================

    def chunk(self, text: str) -> list[TextChunk]:
        """
        Split text into overlapping chunks with character offsets.

        Splitting strategy (in priority order):
        1. Paragraph boundaries (double newlines)
        2. Sentence boundaries (. ! ? followed by whitespace)
        3. Word boundaries (spaces) as last resort

        Args:
            text: The full text to split into chunks.

        Returns:
            A list of TextChunk objects, each with the chunk text and
            its start/end character offsets in the original text. If
            the input text is empty, returns an empty list.
        """
        # Handle trivial cases first
        if not text or not text.strip():
            return []

        # If the entire text fits in one chunk, return it directly.
        if len(text) <= self._chunk_size:
            return [TextChunk(text=text, start=0, end=len(text), index=0)]

        # Step 1: Split text into paragraphs (preserving positions)
        segments = self._split_into_segments(text)

        # Step 2: Merge small segments and split oversized ones,
        #         then assemble into overlapping chunks.
        chunks = self._assemble_chunks(text, segments)

        logger.debug(
            "text_chunked",
            text_length=len(text),
            chunk_count=len(chunks),
            chunk_size=self._chunk_size,
            overlap=self._chunk_overlap,
        )

        return chunks

    @property
    def chunk_size(self) -> int:
        """Return the configured chunk size."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Return the configured chunk overlap."""
        return self._chunk_overlap

    # =================================================================
    # Internal helpers
    # =================================================================

    def _split_into_segments(self, text: str) -> list[tuple[int, int]]:
        """
        Split text into (start, end) offset pairs at paragraph boundaries.

        First tries paragraph boundaries (double newlines). For segments
        that are too long, further splits on sentence boundaries, and
        finally on word boundaries.

        Returns:
            List of (start, end) tuples representing non-overlapping
            segments of the original text.
        """
        # Split on paragraph boundaries first
        raw_segments = self._split_by_pattern(text, _PARAGRAPH_SPLIT_RE)

        # Further split any segment that exceeds chunk_size
        refined: list[tuple[int, int]] = []
        for seg_start, seg_end in raw_segments:
            seg_text = text[seg_start:seg_end]
            if len(seg_text) <= self._chunk_size:
                # Segment fits in one chunk -- keep it as-is
                refined.append((seg_start, seg_end))
            else:
                # Paragraph too long -- split on sentence boundaries
                sentence_segs = self._split_by_pattern(seg_text, _SENTENCE_SPLIT_RE)
                for ss_start, ss_end in sentence_segs:
                    sent_text = seg_text[ss_start:ss_end]
                    if len(sent_text) <= self._chunk_size:
                        # Sentence fits -- record its absolute offsets
                        refined.append((seg_start + ss_start, seg_start + ss_end))
                    else:
                        # Sentence still too long -- split on words
                        word_segs = self._split_by_words(sent_text, self._chunk_size)
                        for ws_start, ws_end in word_segs:
                            refined.append(
                                (seg_start + ss_start + ws_start, seg_start + ss_start + ws_end)
                            )

        return refined

    @staticmethod
    def _split_by_pattern(
        text: str,
        pattern: re.Pattern[str],
    ) -> list[tuple[int, int]]:
        """
        Split text by a regex pattern, returning (start, end) offsets.

        The split delimiter (matched text) is excluded from segments.
        Consecutive delimiters produce empty segments which are filtered.

        Args:
            text:    The text to split.
            pattern: Compiled regex marking split points.

        Returns:
            List of (start, end) offset pairs for non-empty segments.
        """
        segments: list[tuple[int, int]] = []
        last_end = 0

        for match in pattern.finditer(text):
            # Everything from last_end to the start of this match
            # is a segment (the text between delimiters).
            seg_start = last_end
            seg_end = match.start()
            if seg_end > seg_start:
                segments.append((seg_start, seg_end))
            last_end = match.end()

        # Don't forget the final segment after the last delimiter
        if last_end < len(text):
            segments.append((last_end, len(text)))

        return segments

    @staticmethod
    def _split_by_words(
        text: str,
        max_size: int,
    ) -> list[tuple[int, int]]:
        """
        Last-resort splitter: break text on word boundaries (spaces).

        Walks through the text word by word, accumulating until the
        current chunk would exceed max_size, then starts a new chunk.

        Args:
            text:     The text to split.
            max_size: Maximum character length per chunk.

        Returns:
            List of (start, end) offset pairs.
        """
        segments: list[tuple[int, int]] = []

        # Find word boundaries using split positions
        words_with_offsets: list[tuple[int, int]] = []
        pos = 0
        for word in text.split(" "):
            if word:  # skip empty strings from multiple spaces
                words_with_offsets.append((pos, pos + len(word)))
            pos += len(word) + 1  # +1 for the space

        # Greedily pack words into chunks
        chunk_start = 0
        chunk_end = 0
        for word_start, word_end in words_with_offsets:
            # If this word would push us over the limit, finalize the
            # current chunk (unless it's the first word in the chunk)
            if chunk_end > chunk_start and (word_end - chunk_start) > max_size:
                segments.append((chunk_start, chunk_end))
                chunk_start = word_start

            chunk_end = word_end

        # Don't forget the last chunk
        if chunk_end > chunk_start:
            segments.append((chunk_start, chunk_end))

        # Handle edge case: if text starts with spaces, ensure
        # we don't lose the content
        if not segments and text.strip():
            segments.append((0, len(text)))

        return segments

    def _assemble_chunks(
        self,
        text: str,
        segments: list[tuple[int, int]],
    ) -> list[TextChunk]:
        """
        Merge small segments and apply overlap to produce final chunks.

        Walks through the segments, accumulating text until the next
        segment would push the accumulated length beyond chunk_size.
        Then emits the accumulated text as a chunk and starts a new
        accumulation with overlap from the previous chunk.

        Args:
            text:     The original full text (for extracting substrings).
            segments: Non-overlapping (start, end) offset pairs from
                      _split_into_segments().

        Returns:
            List of TextChunk objects with overlap applied.
        """
        if not segments:
            return []

        chunks: list[TextChunk] = []
        chunk_index = 0

        # We accumulate segments into a "current chunk" buffer.
        # When adding the next segment would overflow chunk_size,
        # we emit the buffer as a chunk and start fresh.
        current_start = segments[0][0]
        current_end = segments[0][1]

        for i in range(1, len(segments)):
            _seg_start, seg_end = segments[i]

            # Calculate the total length if we include this segment
            # (including any whitespace gap between current_end and seg_start)
            tentative_length = seg_end - current_start

            if tentative_length <= self._chunk_size:
                # The next segment fits -- extend the current chunk
                current_end = seg_end
            else:
                # Emit the current chunk
                chunk_text = text[current_start:current_end]
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        start=current_start,
                        end=current_end,
                        index=chunk_index,
                    )
                )
                chunk_index += 1

                # Start the next chunk with overlap from the previous one.
                # The overlap region is taken from the end of the current chunk.
                overlap_start = max(
                    current_end - self._chunk_overlap,
                    current_start,  # don't go before the chunk's own start
                )
                current_start = overlap_start
                current_end = seg_end

        # Emit the final accumulated chunk
        chunk_text = text[current_start:current_end]
        if chunk_text.strip():
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    start=current_start,
                    end=current_end,
                    index=chunk_index,
                )
            )

        return chunks
