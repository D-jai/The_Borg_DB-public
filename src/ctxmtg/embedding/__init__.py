# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Embedding Package
=================

This package manages text embedding -- converting text into dense
numerical vectors that capture semantic meaning. These vectors are
stored in LanceDB and used for semantic similarity search during
query execution.

The embedding pipeline:
    1. Text chunking: splits long text into overlapping segments
    2. Change detection: BLAKE3 hashing to skip unchanged content
    3. ONNX inference: converts text chunks into embedding vectors

The default model (all-MiniLM-L6-v2) produces 384-dimensional
vectors and runs efficiently on CPU, including Raspberry Pi.
See research/round-1/03-embedding-and-vectorization.md for the
model selection analysis.

Submodules:
    - onnx_embedder.py   : ONNX Runtime embedding provider
    - change_detector.py  : BLAKE3 content hashing for incremental updates
    - chunker.py          : Text chunking strategies
"""
