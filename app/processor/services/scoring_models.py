"""
Advanced scoring models for papers, authors, and institutions.
Combines OpenAlex and Semantic Scholar metadata for comprehensive ranking.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from app.models.papers import DBPaper
from app.models.authors import DBAuthor
from app.models.institutions import DBInstitution
import math


@dataclass
class ScoringWeights:
    """Configurable weights for different scoring components."""

    # Paper-level weights
    citation_quality: float = 0.25
    venue_prestige: float = 0.20
    institution_reputation: float = 0.15
    influential_citations: float = 0.15
    recency_factor: float = 0.10
    open_access_bonus: float = 0.05
    field_normalization: float = 0.10

    # Author-level weights
    author_h_index: float = 0.20
    author_citation_count: float = 0.15
    coauthor_network: float = 0.15
    author_productivity: float = 0.10

    # Exploration/diversity
    diversity_boost: float = 0.15
    novelty_score: float = 0.10


@dataclass
class AuthorMetrics:
    """Aggregated author metrics from OpenAlex."""

    author_id: str
    name: str
    h_index: Optional[int] = None
    citation_count: int = 0
    paper_count: int = 0
    institutions: Optional[List[str]] = None
    collaboration_count: int = 0
    influential_work_count: int = 0

    def __post_init__(self):
        if self.institutions is None:
            self.institutions = []


@dataclass
class InstitutionMetrics:
    """Institution reputation metrics."""

    institution_id: str
    name: str
    works_count: int = 0
    cited_by_count: int = 0
    country_code: Optional[str] = None
    type: Optional[str] = None

    # Derived metrics
    avg_citations_per_work: float = 0.0
    h_index_estimate: Optional[int] = None


class CitationQualityScorer:
    """Score citation quality using multiple factors."""

    @staticmethod
    def calculate(
        citation_count: int,
        publication_year: Optional[int],
        fwci: Optional[float],
        cited_by_percentile: Optional[Dict[str, int]],
        is_in_top_10_percent: bool = False,
    ) -> float:
        """
        Calculate citation quality score (0-100).

        Args:
            citation_count: Total citations
            publication_year: Year of publication
            fwci: Field-Weighted Citation Impact from OpenAlex
            cited_by_percentile: Percentile ranking
            is_in_top_10_percent: Whether in top 10% of field

        Returns:
            Quality score 0-100
        """
        score = 0.0

        # Base citation count with logarithmic scaling
        if citation_count > 0:
            # Log scale to prevent extremely high citations from dominating
            score += min(40, math.log10(citation_count + 1) * 10)

        # FWCI (Field-Weighted Citation Impact) - highly valuable metric
        if fwci is not None:
            # FWCI > 1 means above world average
            # Cap at 30 points max
            score += min(30, fwci * 10)

        # Citation velocity (citations per year since publication)
        if publication_year and citation_count > 0:
            years_since_pub = max(1, datetime.now().year - publication_year)
            velocity = citation_count / years_since_pub
            score += min(15, math.log10(velocity + 1) * 5)

        # Percentile ranking bonus
        if cited_by_percentile:
            percentile_max = cited_by_percentile.get("max", 0)
            if percentile_max >= 99:
                score += 10
            elif percentile_max >= 95:
                score += 7
            elif percentile_max >= 90:
                score += 5

        # Top 10% bonus
        if is_in_top_10_percent:
            score += 5

        return min(100, score)


class VenuePrestigeScorer:
    """Score venue/journal prestige using SJR data."""

    @staticmethod
    def calculate(
        venue_name: Optional[str],
        venue_type: Optional[str],
        is_oa: bool = False,
        publisher: Optional[str] = None,
        journal_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate venue prestige score (0-100).
        Prioritizes SJR data (journal_data) over simple venue name matching.

        Args:
            venue_name: Name of publication venue
            venue_type: Type (journal, conference, etc.)
            is_oa: Whether open access
            publisher: Publisher name
            journal_data: SJR data from DBJournal (sjr_score, sjr_best_quartile, h_index, cites_per_doc_2years)

        Returns:
            Prestige score 0-100
        """
        # If we have SJR data, use it as primary source
        if journal_data:
            score = 20.0  # Base for matched journal

            # SJR Quartile (most important indicator)
            quartile = journal_data.get("sjr_best_quartile", "").upper()
            if quartile == "Q1":
                score += 50  # Top quartile
            elif quartile == "Q2":
                score += 35
            elif quartile == "Q3":
                score += 20
            elif quartile == "Q4":
                score += 10

            # H-Index (journal reputation)
            h_index = journal_data.get("h_index", 0)
            if h_index > 200:
                score += 20
            elif h_index > 100:
                score += 15
            elif h_index > 50:
                score += 10
            elif h_index > 20:
                score += 5

            # Impact factor equivalent (cites per doc)
            cites_per_doc = journal_data.get("cites_per_doc_2years", 0)
            if cites_per_doc > 10:
                score += 15
            elif cites_per_doc > 5:
                score += 10
            elif cites_per_doc > 2:
                score += 5

            # SJR Score (overall quality)
            sjr_score = journal_data.get("sjr_score", 0)
            if sjr_score > 2.0:
                score += 10
            elif sjr_score > 1.0:
                score += 5

            # Diamond OA bonus (free for authors and readers)
            if journal_data.get("is_open_access_diamond", False):
                score += 5

            return min(100, score)

        # Fallback: Use venue name if no SJR data
        score = 25.0  # Lower base for unmatched journals

        if venue_name:
            # Heuristic matching for known top venues
            venue_lower = venue_name.lower()
            top_keywords = [
                "nature",
                "science",
                "cell",
                "lancet",
                "nejm",
                "pnas",
                "jama",
            ]
            if any(kw in venue_lower for kw in top_keywords):
                score += 40
            # Conference proceedings
            elif "ieee" in venue_lower or "acm" in venue_lower:
                score += 25
            elif "conference" in venue_lower or "proceedings" in venue_lower:
                score += 15

        # Venue type bonus
        if venue_type == "journal":
            score += 10
        elif venue_type == "conference":
            score += 5

        # Open access bonus
        if is_oa:
            score += 5

        return min(100, score)


class InstitutionReputationScorer:
    """Score institution reputation."""

    @staticmethod
    def calculate(
        institutions: List[DBInstitution],
        countries_distinct_count: int = 0,
        institutions_distinct_count: int = 0,
    ) -> float:
        """
        Calculate institution reputation score (0-100).

        Args:
            institutions: List of institution dicts from OpenAlex
            countries_distinct_count: Number of distinct countries
            institutions_distinct_count: Number of distinct institutions

        Returns:
            Reputation score 0-100
        """
        base_scores = []
        for inst in institutions:
            i_score = 40.0  # Baseline
            
            if inst.type == "education":
                i_score += 10
   
            citations = inst.total_citations or 0
            if citations > 1_000_000:
                i_score += 30
            elif citations > 100_000:
                i_score += 20
            elif citations > 10_000:
                i_score += 10
                
            base_scores.append(i_score)

        max_base = max(base_scores) if base_scores else 40.0

        country_bonus = min(10, countries_distinct_count * 2)
        collaboration_bonus = min(5, institutions_distinct_count)

        return min(100, max_base + country_bonus + collaboration_bonus)

class AuthorReputationScorer:
    """Score author reputation."""

    @staticmethod
    def calculate(
        author = DBAuthor,
    ) -> float:
        """
        Calculate author reputation score (0-100).

        Args:
            authors: List of author dicts from Semantic Scholar
            authorships: List of authorship dicts from OpenAlex

        Returns:
            Reputation score 0-100
        """
        max_author_score = 0.0

        author_score = 20.0
        h_index = getattr(author, "h_index", 0)
        if h_index:
            author_score += min(30, h_index * 2)
        
        total_citation = getattr(author, "total_citations", 0)
        if total_citation > 0:
            author_score += math.log10(total_citation + 1) * 4
        total_paper = getattr(author, "paper_count", 0)
        if total_paper > 0:
            author_score += min(15, math.log10(total_paper + 1) * 3)

        max_author_score = max(max_author_score, author_score)
        
        return min(100, max_author_score)


class DiversityScorer:
    """Promote diversity and explorability in results."""

    @staticmethod
    def calculate(
        publication_year: Optional[int],
        topics: List[Dict[str, Any]],
        current_results_topics: Optional[List[str]] = None,
        is_open_access: bool = False,
    ) -> float:
        """
        Calculate diversity/novelty score to promote exploration (0-100).

        Args:
            publication_year: Year of publication
            topics: Paper topics
            current_results_topics: Topics already in results (for diversity)
            is_open_access: Whether open access

        Returns:
            Diversity score 0-100
        """
        score = 50.0  # Baseline

        # Recency bonus (recent papers for exploration)
        if publication_year:
            years_old = datetime.now().year - publication_year
            if years_old <= 2:
                score += 20
            elif years_old <= 5:
                score += 10
            elif years_old <= 10:
                score += 5

        # Topic diversity (different from current results)
        if topics and current_results_topics:
            paper_topics = {t.get("display_name", "") for t in topics}
            overlap = len(paper_topics & set(current_results_topics))
            diversity_bonus = max(0, (len(paper_topics) - overlap) * 5)
            score += min(20, diversity_bonus)

        # Open access promotes accessibility and exploration
        if is_open_access:
            score += 10

        return min(100, score)


class ComprehensiveScorer:
    """Main scoring class that combines all factors."""

    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = weights or ScoringWeights()
        self.citation_scorer = CitationQualityScorer()
        self.venue_scorer = VenuePrestigeScorer()
        self.institution_scorer = InstitutionReputationScorer()
        self.author_scorer = AuthorReputationScorer()

    