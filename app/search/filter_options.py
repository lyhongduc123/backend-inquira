"""Shared filter option types for search workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SearchFilterOptions:
    """Unified filter options for paper search."""

    author_name: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    venue: Optional[str] = None
    min_citation_count: Optional[int] = None
    max_citation_count: Optional[int] = None
    journal_quartile: Optional[str] = None
    field_of_study: Optional[List[str]] = None
