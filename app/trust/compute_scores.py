"""
Trust Score Computation Utilities

Provides functions to compute and update trust/reputation scores for:
- Authors (individual researcher reputation)
- Institutions (organizational trust)
- Papers (combined author + institutional trust)
- Citations (network analysis metrics)

Usage:
    python -m app.trust.compute_scores --all
    python -m app.trust.compute_scores --authors-only
    python -m app.trust.compute_scores --paper-id <paper_id>
"""

import math
from typing import Any, Dict, List, Optional
import logging
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.models import DBAuthor, DBInstitution, DBPaper, DBAuthorPaper, DBCitation

logger = logging.getLogger(__name__)


class TrustScoreComputer:
    """Computes trust and reputation scores across the database."""
    
    def __init__(self, db: Session):
        self.db = db
    
def compute_author_reputation(author_data: Dict[str, Any]) -> float:
    """
    Compute composite reputation score for an author (0-100).
    
    Pure function - takes author data dict, returns score.
    
    Factors:
    - H-index (weight: 30%)
    - Total citations (weight: 20%)
    - Field-weighted citation impact (weight: 20%)
    - Verification status (weight: 10%)
    - Retraction penalty (weight: -20% per retraction)
    - Collaboration diversity (weight: 10%)
    
    Args:
        author_data: Dict with keys: h_index, total_citations, field_weighted_citation_impact,
                     verified, retracted_papers_count, collaboration_diversity_score
        
    Returns:
        Reputation score (0-100)
    """
    if not author_data:
        return 0.0
    
    score = 0.0
    
    # 1. H-index (0-30 points, capped at 50)
    h_index = author_data.get('h_index', 0) or 0
    h_score = min(30, h_index * 0.6)
    score += h_score
    
    # 2. Citation count (0-20 points, logarithmic)
    total_citations = author_data.get('total_citations', 0) or 0
    if total_citations > 0:
        citation_score = min(20, math.log10(total_citations) * 4)
        score += citation_score
    
    # 3. FWCI (0-20 points)
    fwci = author_data.get('field_weighted_citation_impact')
    if fwci:
        fwci_score = min(20, fwci * 10)
        score += fwci_score
    
    # 4. Verification bonus (10 points)
    if author_data.get('verified', False):
        score += 10
    
    # 5. Retraction penalty (-20 per retraction)
    retraction_count = author_data.get('retracted_papers_count', 0) or 0
    retraction_penalty = retraction_count * 20
    score -= retraction_penalty
    
    # 6. Collaboration diversity (0-10 points)
    collab_diversity = author_data.get('collaboration_diversity_score')
    if collab_diversity:
        collab_score = collab_diversity * 10
        score += collab_score
    
    return max(0.0, min(100.0, score))
    
def compute_institution_reputation(institution_data: Dict[str, Any]) -> float:
    """
    Compute composite reputation score for an institution (0-100).
    
    Pure function - takes institution data dict, returns score.
    
    Factors:
    - Total citations (weight: 30%)
    - H-index (weight: 25%)
    - Average paper quality (FWCI) (weight: 25%)
    - Retraction rate penalty (weight: -20%)
    - Paper volume (weight: 10%)
    
    Args:
        institution_data: Dict with keys: total_citations, h_index, avg_paper_quality,
                         retraction_rate, total_papers
        
    Returns:
        Reputation score (0-100)
    """
    if not institution_data:
        return 0.0
    
    score = 0.0
    
    # 1. Citation count (0-30 points, logarithmic)
    total_citations = institution_data.get('total_citations', 0) or 0
    if total_citations > 0:
        citation_score = min(30, math.log10(total_citations) * 6)
        score += citation_score
    
    # 2. H-index (0-25 points, capped at 100)
    h_index = institution_data.get('h_index', 0) or 0
    h_score = min(25, h_index * 0.25)
    score += h_score
    
    # 3. Average paper quality (0-25 points)
    avg_quality = institution_data.get('avg_paper_quality')
    if avg_quality:
        quality_score = min(25, avg_quality * 12.5)
        score += quality_score
    
    # 4. Retraction rate penalty (1% = -1 point)
    retraction_rate = institution_data.get('retraction_rate', 0) or 0
    retraction_penalty = retraction_rate * 100
    score -= retraction_penalty
    
    # 5. Paper volume (0-10 points, logarithmic)
    total_papers = institution_data.get('total_papers', 0) or 0
    if total_papers > 0:
        volume_score = min(10, math.log10(total_papers) * 3)
        score += volume_score
    
    return max(0.0, min(100.0, score))
    

    
def compute_paper_trust_scores(
    author_papers_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute trust scores for a paper based on its authors and institutions.
    
    Pure function - takes author-paper data list, returns score dictionary.
    
    Args:
        author_papers_data: List of dicts with keys:
            - author_reputation_score: float | None
            - institution_reputation_score: float | None
            - institution_id: int | None
            - country_code: str | None
        
    Returns:
        Dict with computed scores: author_trust_score, institutional_trust_score,
        network_diversity_score, author_count, is_single_author, etc.
    """
    if not author_papers_data:
        return {
            "author_trust_score": None,
            "institutional_trust_score": None,
            "network_diversity_score": 0.0,
            "author_count": 0,
            "is_single_author": False,
            "institutions_distinct_count": 0,
            "countries_distinct_count": 0,
        }
    
    # Author trust score (average)
    author_scores = [
        ap['author_reputation_score']
        for ap in author_papers_data
        if ap.get('author_reputation_score') is not None
    ]
    
    # Institutional trust score (average)
    institution_scores = [
        ap['institution_reputation_score']
        for ap in author_papers_data
        if ap.get('institution_reputation_score') is not None
    ]
    
    # Network diversity score
    unique_institutions = len(
        set(ap.get('institution_id') for ap in author_papers_data if ap.get('institution_id'))
    )
    
    # Country diversity
    unique_countries = len(
        set(ap.get('country_code') for ap in author_papers_data if ap.get('country_code'))
    )
    
    return {
        "author_trust_score": sum(author_scores) / len(author_scores) if author_scores else None,
        "institutional_trust_score": sum(institution_scores) / len(institution_scores) if institution_scores else None,
        "network_diversity_score": min(1.0, unique_institutions / len(author_papers_data)) if author_papers_data else 0.0,
        "author_count": len(author_papers_data),
        "is_single_author": len(author_papers_data) == 1,
        "institutions_distinct_count": unique_institutions,
        "countries_distinct_count": unique_countries,
    }
    

