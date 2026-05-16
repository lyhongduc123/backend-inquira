"""Compatibility exports for processor result types.

New code should import ranking/search result types from ``app.search.types``.
"""

from app.search.types import RankedPaper

__all__ = ["RankedPaper"]
