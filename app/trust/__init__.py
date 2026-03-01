"""Trust and reputation computation module."""

from app.trust.compute_scores import (
    TrustScoreComputer,
    compute_paper_trust_scores,
    compute_institution_reputation,
    compute_author_reputation,
)
from app.trust.scoring_service import ScoringService
# Backward compatibility: import from new location
from app.papers.journal_service import JournalService

# Deprecated alias - use app.papers.journal_service.JournalService instead
JournalLookupService = JournalService

__all__ = [
    "TrustScoreComputer",
    "compute_paper_trust_scores",
    "compute_institution_reputation",
    "compute_author_reputation",
    "ScoringService",
    "JournalLookupService",  # Deprecated, kept for backward compatibility
    "JournalService",
]
