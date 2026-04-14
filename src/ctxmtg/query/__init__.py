# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Query Package
=============

This package handles the full query pipeline: from a user's natural
language question to a ranked list of answers drawn from both the SQL
and vector stores.

Query pipeline stages:
    1. Intent classification: determine what kind of query this is
       (factual lookup, semantic search, aggregation, temporal, etc.)
    2. Query planning: generate SQL queries and/or vector search
       parameters based on the classified intent
    3. Execution: run the planned queries against both stores
    4. Fusion: combine results from SQL and vector stores using
       Reciprocal Rank Fusion (RRF)
    5. Reranking: lightweight TF-IDF reranking for final ordering

Retrieval modes (user-selectable):
    - PARALLEL:      Both stores independently, RRF fusion (Phase 1, all tiers)
    - VECTOR_TO_SQL: Vector discovery → LLM bridge → SQL facts (Tier 1+)
    - SQL_TO_VECTOR: SQL briefing → LLM bridge → vector chunks (Tier 1+)
    - BIDIRECTIONAL: Both paths in parallel, full synthesis (Tier 2+)

The query server runs asynchronously to handle concurrent requests,
while the underlying stores use connection pooling for efficiency.

Submodules:
    - intent.py              : Intent classification regex patterns and helpers
    - interpreter.py         : RuleBasedQueryInterpreter (regex + entity matching)
    - planner.py             : TemplateQueryPlanner (SQL templates per intent)
    - executor.py            : QueryExecutor (parallel SQL + vector execution)
    - fusion.py              : RRFFuser (Reciprocal Rank Fusion, k=60)
    - reranker.py            : TFIDFReranker (lightweight TF-IDF reranking)
    - briefing.py            : SQLBriefingBuilder (Pass 1 profiling queries)
    - informed_retrieval.py  : VectorToSQLRetriever, SQLToVectorRetriever
    - bidirectional.py       : BidirectionalRetriever
    - server.py              : Query server (async, always-running -- Phase 2)
"""

from ctxmtg.query.autocomplete import AutocompleteEngine
from ctxmtg.query.bidirectional import BidirectionalRetriever
from ctxmtg.query.briefing import SQLBriefingBuilder
from ctxmtg.query.cross_encoder_reranker import CrossEncoderReranker
from ctxmtg.query.executor import QueryExecutor
from ctxmtg.query.fusion import RRFFuser
from ctxmtg.query.informed_retrieval import SQLToVectorRetriever, VectorToSQLRetriever
from ctxmtg.query.interpreter import RuleBasedQueryInterpreter
from ctxmtg.query.llm_fusion import LLMFuser
from ctxmtg.query.llm_interpreter import LLMQueryInterpreter
from ctxmtg.query.planner import TemplateQueryPlanner
from ctxmtg.query.quality_logger import QueryQualityLogger
from ctxmtg.query.reranker import TFIDFReranker
from ctxmtg.query.reranker_factory import create_reranker
from ctxmtg.query.synthesizer import LLMSynthesizer
from ctxmtg.query.tiered_reranker import TieredReranker

__all__ = [
    "AutocompleteEngine",
    "BidirectionalRetriever",
    "CrossEncoderReranker",
    "LLMFuser",
    "LLMQueryInterpreter",
    "LLMSynthesizer",
    "QueryExecutor",
    "QueryQualityLogger",
    "RRFFuser",
    "RuleBasedQueryInterpreter",
    "SQLBriefingBuilder",
    "SQLToVectorRetriever",
    "TFIDFReranker",
    "TemplateQueryPlanner",
    "TieredReranker",
    "VectorToSQLRetriever",
    "create_reranker",
]
