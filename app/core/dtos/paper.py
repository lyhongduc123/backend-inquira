"""Compatibility exports for paper transfer types.

New code should import from ``app.domain.papers.types``.
"""

from app.domain.papers.types import PaperDTO, PaperEnrichedDTO

__all__ = ["PaperDTO", "PaperEnrichedDTO"]
