# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
TextRank Extractive Summarizer
===============================

This module implements the Summarizer interface using the TextRank
algorithm for extractive summarization. It picks the most important
sentences from the input text based on their similarity to other
sentences, without generating any new text.

How TextRank works (in plain English):
1. Split the text into sentences using spaCy's sentence segmenter
2. Compute a similarity score between every pair of sentences
   (using word overlap / cosine similarity of word vectors)
3. Build a graph where sentences are nodes and similarity scores
   are edge weights
4. Run the PageRank algorithm on this graph to find the "most
   connected" sentences (sentences similar to many others)
5. Return the top-ranked sentences as the summary

This is the same algorithm that Google uses for web page ranking,
but applied to sentences instead of web pages. A sentence that
shares vocabulary with many other sentences is likely a good summary
candidate.

Why extractive (not abstractive)?
Extractive summarization picks existing sentences verbatim -- no
hallucination risk, no LLM required. Phase 2 adds LLM-based
abstractive summarization for more natural-sounding summaries.

Depends on:
    - spacy (sentence segmentation and word vectors)
    - numpy (matrix operations for TextRank)
    - ctxmtg.interfaces.extraction (Summarizer ABC)

Used by:
    - ctxmtg.extraction.pipeline (BasicExtractionPipeline uses this)
"""

from __future__ import annotations

import numpy as np
import structlog
from spacy.language import Language
from spacy.tokens import Doc

# ---------------------------------------------------------------
# Import the Summarizer interface
# ---------------------------------------------------------------
from ctxmtg.interfaces.extraction import Summarizer

# ---------------------------------------------------------------
# Logger: logs summary lengths, sentence counts.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.extraction.summarizer")


# =====================================================================
# TextRankSummarizer -- Extractive Summarization via TextRank
# =====================================================================


class TextRankSummarizer(Summarizer):
    """
    Extractive text summarizer using the TextRank algorithm.

    Selects the most representative sentences from input text by
    building a sentence similarity graph and running PageRank to
    identify the most "central" sentences. These central sentences
    are the ones most connected to the overall meaning of the text.

    The algorithm requires no training data and no LLM -- just
    spaCy for sentence segmentation and numpy for matrix math.

    Usage:
        summarizer = TextRankSummarizer(nlp=spacy_model)
        summary = summarizer.summarize(
            text="Alice proposed... Bob raised... Charlie will...",
            max_length=200,
        )
        # → "Alice proposed migrating the authentication service to OAuth2."

    Parameters:
        damping_factor: Controls how much PageRank "leaks" between
                        iterations. Standard value is 0.85 (same as
                        original PageRank paper).
        max_iterations: Maximum PageRank iterations before convergence.
        convergence_threshold: Stop iterating when scores change less
                               than this between iterations.
    """

    # ---------------------------------------------------------------
    # PageRank hyperparameters (standard values from the literature).
    # ---------------------------------------------------------------
    DAMPING_FACTOR = 0.85  # Same as Google's original PageRank
    MAX_ITERATIONS = 100  # Convergence usually happens in 20-30
    CONVERGENCE_THRESHOLD = 1e-4  # Stop when changes are tiny

    def __init__(self, nlp: Language | None = None) -> None:
        """
        Initialize the TextRank summarizer.

        Args:
            nlp: A loaded spaCy Language model for sentence segmentation
                 and word vectors. If None, summarize() will return
                 a simple truncation instead of a proper summary.
        """
        self._nlp = nlp

    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        Produce a summary of the input text.

        Uses TextRank to identify the most representative sentences.
        The summary contains the top-ranked sentences in their original
        order, truncated to max_length characters.

        Args:
            text: The raw text to summarize. Works best with 3+ sentences.
            max_length: Maximum length of the summary in characters.

        Returns:
            A summary string of at most max_length characters. Returns
            empty string for empty input, or the full text if it's
            already shorter than max_length.
        """
        # Handle edge cases: empty or very short text
        if not text or not text.strip():
            return ""

        text = text.strip()

        # If text is already short enough, return it as-is
        if len(text) <= max_length:
            return text

        # If no spaCy model, fall back to simple truncation
        if self._nlp is None:
            return self._simple_truncate(text, max_length)

        # Process with spaCy for sentence segmentation
        doc = self._nlp(text)
        return self._summarize_from_doc(doc, max_length)

    def summarize_from_doc(self, doc: Doc, max_length: int = 200) -> str:
        """
        Summarize from a pre-processed spaCy Doc.

        Avoids re-processing text that was already run through spaCy.
        The pipeline calls this instead of summarize() when it already
        has a Doc object.

        Args:
            doc: A spaCy Doc (already processed).
            max_length: Maximum summary length in characters.

        Returns:
            A summary string of at most max_length characters.
        """
        text = doc.text.strip()
        if not text:
            return ""
        if len(text) <= max_length:
            return text

        return self._summarize_from_doc(doc, max_length)

    def _summarize_from_doc(self, doc: Doc, max_length: int) -> str:
        """
        Internal: run TextRank on a spaCy Doc to produce a summary.

        Steps:
        1. Extract sentences from the Doc
        2. Build a similarity matrix between all sentence pairs
        3. Run PageRank on the similarity matrix
        4. Select top-ranked sentences in original order
        5. Truncate to max_length

        Args:
            doc: Processed spaCy Doc.
            max_length: Character limit for the summary.

        Returns:
            Summary string.
        """
        # Extract sentences as a list of Span objects
        sentences = list(doc.sents)

        # Need at least 2 sentences for TextRank to be meaningful
        if len(sentences) <= 1:
            return self._simple_truncate(doc.text.strip(), max_length)

        # ---------------------------------------------------------------
        # Step 1: Build the sentence similarity matrix.
        # Each cell (i, j) = similarity between sentence i and sentence j.
        # We use word overlap (Jaccard-like) as the similarity metric.
        # ---------------------------------------------------------------
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._sentence_similarity(sentences[i], sentences[j])
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim

        # ---------------------------------------------------------------
        # Step 2: Run PageRank on the similarity matrix.
        # This finds which sentences are most "central" in the graph --
        # i.e., most similar to many other sentences.
        # ---------------------------------------------------------------
        scores = self._pagerank(similarity_matrix)

        # ---------------------------------------------------------------
        # Step 3: Rank sentences by their PageRank score.
        # Then select top sentences in their ORIGINAL order (not by score)
        # so the summary reads naturally.
        # ---------------------------------------------------------------
        ranked_indices = list(np.argsort(scores)[::-1])

        # Select sentences greedily until we hit max_length
        selected_indices: list[int] = []
        current_length = 0

        for idx in ranked_indices:
            sent_text = sentences[idx].text.strip()
            # Check if adding this sentence would exceed the limit
            new_length = (
                current_length
                + len(sent_text)
                + (
                    2 if current_length > 0 else 0  # Space between sentences
                )
            )
            if new_length <= max_length:
                selected_indices.append(idx)
                current_length = new_length
            # Stop once we have enough content
            if current_length >= max_length * 0.8:
                break

        # If no sentences fit, just take the highest-ranked one (truncated)
        if not selected_indices:
            best_idx = ranked_indices[0]
            return self._simple_truncate(sentences[best_idx].text.strip(), max_length)

        # Sort selected indices to maintain original text order
        selected_indices.sort()

        # Combine selected sentences
        summary = " ".join(sentences[idx].text.strip() for idx in selected_indices)

        # Final truncation safety net
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(" ", 1)[0]

        logger.info(
            "textrank_summary_complete",
            input_sentences=n,
            selected_sentences=len(selected_indices),
            summary_length=len(summary),
        )

        return summary

    def _sentence_similarity(self, sent1, sent2) -> float:
        """
        Compute similarity between two sentences using word overlap.

        Uses a simple but effective approach: count the number of
        meaningful words shared between the two sentences, normalised
        by the combined sentence length. This is similar to a Jaccard
        index but on word lemmas.

        We use lemmas (base forms) so "proposed" and "proposing" are
        treated as the same word. We also filter out stop words and
        punctuation, which don't contribute to meaning.

        Args:
            sent1: First spaCy Span (sentence).
            sent2: Second spaCy Span (sentence).

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        # Extract meaningful words (lemmas) from each sentence.
        # Skip stop words (the, a, is, etc.) and punctuation.
        words1 = {
            token.lemma_.lower()
            for token in sent1
            if not token.is_stop and not token.is_punct and len(token.text) > 1
        }
        words2 = {
            token.lemma_.lower()
            for token in sent2
            if not token.is_stop and not token.is_punct and len(token.text) > 1
        }

        # If either sentence has no meaningful words, similarity is 0
        if not words1 or not words2:
            return 0.0

        # Count shared words (intersection)
        overlap = words1 & words2

        # Normalise by the sum of both sets' sizes (prevents large
        # sentences from dominating). Adding 1e-10 avoids division by zero.
        denominator = len(words1) + len(words2) + 1e-10
        return 2.0 * len(overlap) / denominator

    def _pagerank(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Run the PageRank algorithm on a similarity matrix.

        PageRank iteratively computes importance scores for each node
        (sentence) in the graph. Nodes connected to many important
        nodes get higher scores. This is the same algorithm Google
        uses for ranking web pages.

        The formula for each iteration:
            score[i] = (1 - d) + d * sum(
                similarity[i][j] * score[j] / out_degree[j]
                for all j != i
            )
        where d is the damping factor (0.85).

        Args:
            similarity_matrix: Square matrix of sentence similarities.

        Returns:
            Array of PageRank scores, one per sentence.
        """
        n = similarity_matrix.shape[0]

        # Normalise columns (each column sums to 1 for valid transition probabilities)
        col_sums = similarity_matrix.sum(axis=0)
        # Avoid division by zero for isolated nodes
        col_sums[col_sums == 0] = 1.0
        normalised = similarity_matrix / col_sums

        # Initialize all scores equally
        scores = np.ones(n) / n

        # Iterative power method
        for _iteration in range(self.MAX_ITERATIONS):
            prev_scores = scores.copy()

            # PageRank formula: (1-d)/n + d * M * scores
            scores = (1 - self.DAMPING_FACTOR) / n + self.DAMPING_FACTOR * normalised @ scores

            # Check convergence: stop if scores barely changed
            delta = np.abs(scores - prev_scores).sum()
            if delta < self.CONVERGENCE_THRESHOLD:
                break

        return scores  # type: ignore[no-any-return]

    def _simple_truncate(self, text: str, max_length: int) -> str:
        """
        Simple fallback: truncate text to max_length at a word boundary.

        Used when spaCy is not available or the text has too few
        sentences for TextRank to be useful.

        Args:
            text: Text to truncate.
            max_length: Maximum character length.

        Returns:
            Truncated text, cut at the last complete word.
        """
        if len(text) <= max_length:
            return text

        # Cut at max_length, then back up to the last space
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.5:
            truncated = truncated[:last_space]

        return truncated
