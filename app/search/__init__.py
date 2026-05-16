"""Shared search helpers and services."""

from app.search.filter_options import SearchFilterOptions
from app.search.filters import get_filter_value, parse_search_filter_options
from app.search.fusion import (
    reciprocal_rank_fusion,
    weighted_hybrid_fusion,
    weighted_rrf_fusion,
)
from app.search.virtual_chunks import (
    append_missing_abstract_chunks,
    build_abstract_chunk,
)
from app.search.local_search import (
    ChunkSearchService,
    LocalSearchService,
    PaperSearchService,
)
from app.search.query_builder import build_paradedb_query
from app.search.types import RankedPaper

__all__ = [
    "append_missing_abstract_chunks",
    "build_abstract_chunk",
    "build_paradedb_query",
    "ChunkSearchService",
    "get_filter_value",
    "LocalSearchService",
    "PaperSearchService",
    "RankedPaper",
    "SearchFilterOptions",
    "parse_search_filter_options",
    "reciprocal_rank_fusion",
    "weighted_hybrid_fusion",
    "weighted_rrf_fusion",
]
