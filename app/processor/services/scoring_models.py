"""
Advanced scoring models for papers, authors, and institutions.
Combines OpenAlex and Semantic Scholar metadata for comprehensive ranking.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
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
        is_in_top_10_percent: bool = False
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
            percentile_max = cited_by_percentile.get('max', 0)
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
    """Score venue/journal prestige."""
    
    # Known high-impact venues (can be extended with a database)
    PRESTIGIOUS_VENUES = {
        'Nature', 'Science', 'Cell', 'The Lancet', 'NEJM',
        'PNAS', 'Nature Genetics', 'Nature Medicine'
    }
    
    @staticmethod
    def calculate(
        venue_name: Optional[str],
        venue_type: Optional[str],
        is_oa: bool = False,
        publisher: Optional[str] = None
    ) -> float:
        """
        Calculate venue prestige score (0-100).
        
        Args:
            venue_name: Name of publication venue
            venue_type: Type (journal, conference, etc.)
            is_oa: Whether open access
            publisher: Publisher name
        
        Returns:
            Prestige score 0-100
        """
        score = 30.0  # Base score
        
        if venue_name:
            # Check against prestigious venues
            if venue_name in VenuePrestigeScorer.PRESTIGIOUS_VENUES:
                score += 50
            # Partial matches for well-known venues
            elif any(pv in venue_name for pv in VenuePrestigeScorer.PRESTIGIOUS_VENUES):
                score += 30
            # Conference proceedings
            elif 'conference' in venue_name.lower() or 'proceedings' in venue_name.lower():
                score += 15
        
        # Venue type bonus
        if venue_type == 'journal':
            score += 10
        elif venue_type == 'conference':
            score += 5
        
        # Open access bonus (slightly)
        if is_oa:
            score += 5
        
        return min(100, score)


class InstitutionReputationScorer:
    """Score institution reputation."""
    
    @staticmethod
    def calculate(
        institutions: List[Dict[str, Any]],
        countries_distinct_count: int = 0,
        institutions_distinct_count: int = 0
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
        if not institutions:
            return 20.0  # Low baseline for no affiliation
        
        score = 0.0
        
        # Analyze each institution
        for inst in institutions[:3]:  # Top 3 institutions
            inst_score = 40.0  # Base institutional score
            
            # Check for type (R1, etc.)
            inst_type = inst.get('type', '')
            if inst_type == 'education':
                inst_score += 10
            
            # Country diversity bonus
            if inst.get('country_code') in ['US', 'GB', 'DE', 'FR', 'JP', 'CA', 'AU', 'CH', 'NL', 'SE']:
                inst_score += 10
            
            # Use cited_by_count or works_count as proxy for reputation
            cited_by = inst.get('cited_by_count', 0)
            if cited_by > 1000000:
                inst_score += 20
            elif cited_by > 100000:
                inst_score += 10
            
            score = max(score, inst_score)
        
        # International collaboration bonus
        if countries_distinct_count > 1:
            score += min(10, countries_distinct_count * 2)
        
        # Multi-institution collaboration
        if institutions_distinct_count > 1:
            score += min(5, institutions_distinct_count)
        
        return min(100, score)


class InfluentialCitationScorer:
    """Score influential citations from Semantic Scholar."""
    
    @staticmethod
    def calculate(
        citation_count: int,
        influential_citation_count: int,
        reference_count: int = 0
    ) -> float:
        """
        Calculate influential citation score (0-100).
        
        Args:
            citation_count: Total citations
            influential_citation_count: Influential citations (from Semantic Scholar)
            reference_count: Number of references
        
        Returns:
            Influence score 0-100
        """
        if citation_count == 0:
            return 0.0
        
        # Influential citation ratio
        influence_ratio = influential_citation_count / max(1, citation_count)
        
        # Base score from ratio (up to 50 points)
        score = influence_ratio * 50
        
        # Absolute influential citation count (logarithmic)
        if influential_citation_count > 0:
            score += min(30, math.log10(influential_citation_count + 1) * 10)
        
        # Reference thoroughness (well-cited papers should cite others)
        if reference_count > 0:
            thoroughness = min(1.0, reference_count / 30)  # 30 refs = full score
            score += thoroughness * 20
        
        return min(100, score)


class AuthorReputationScorer:
    """Score author reputation."""
    
    @staticmethod
    def calculate(
        authors: List[Dict[str, Any]],
        authorships: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """
        Calculate author reputation score (0-100).
        
        Args:
            authors: List of author dicts from Semantic Scholar
            authorships: List of authorship dicts from OpenAlex
        
        Returns:
            Reputation score 0-100
        """
        if not authors and not authorships:
            return 30.0  # Low baseline
        
        max_author_score = 0.0
        
        # Process Semantic Scholar authors
        if authors:
            for author in authors[:3]:  # Top 3 authors
                author_score = 20.0
                
                # H-index
                h_index = author.get('hIndex', 0)
                if h_index:
                    author_score += min(30, h_index * 2)
                
                # Citation count
                citation_count = author.get('citationCount', 0)
                if citation_count > 0:
                    author_score += min(20, math.log10(citation_count + 1) * 4)
                
                # Paper count (productivity)
                paper_count = author.get('paperCount', 0)
                if paper_count > 0:
                    author_score += min(15, math.log10(paper_count + 1) * 5)
                
                max_author_score = max(max_author_score, author_score)
        
        # Process OpenAlex authorships (institutional affiliations)
        if authorships:
            for authorship in authorships[:3]:
                author = authorship.get('author', {})
                institutions = authorship.get('institutions', [])
                
                authorship_score = 20.0
                
                # Number of institutions (indicates established researcher)
                if institutions:
                    authorship_score += min(10, len(institutions) * 3)
                
                # Author position (first/last author bonus)
                position = authorship.get('author_position', '')
                if position in ['first', 'last']:
                    authorship_score += 15
                elif position == 'middle':
                    authorship_score += 5
                
                max_author_score = max(max_author_score, authorship_score)
        
        # Multi-author collaboration bonus
        if authorships and len(authorships) > 1:
            max_author_score += min(15, len(authorships) * 2)
        
        return min(100, max_author_score)


class FieldNormalizationScorer:
    """Normalize scores by field to ensure fairness across disciplines."""
    
    @staticmethod
    def calculate(
        topics: List[Dict[str, Any]],
        concepts: List[Dict[str, Any]],
        fwci: Optional[float] = None
    ) -> float:
        """
        Calculate field-normalized score (0-100).
        
        Args:
            topics: Topics from OpenAlex
            concepts: Concepts from OpenAlex
            fwci: Field-Weighted Citation Impact (already normalized)
        
        Returns:
            Field-normalized score 0-100
        """
        score = 50.0  # Baseline
        
        # FWCI is already field-normalized, use it directly
        if fwci is not None:
            # FWCI of 1.0 = world average (50 points)
            # FWCI of 2.0 = 2x world average (75 points)
            # FWCI of 4.0+ = 100 points
            score = min(100, 25 + (fwci * 25))
        
        # Topic diversity bonus
        if topics:
            topic_count = len(topics)
            score += min(10, topic_count * 2)
        
        # Concept specificity
        if concepts:
            # Papers with specific concepts (lower level) are more focused
            high_level_concepts = sum(1 for c in concepts if c.get('level', 0) <= 1)
            if high_level_concepts < len(concepts) / 2:
                score += 5  # Specificity bonus
        
        return min(100, score)


class PredatoryFilterScorer:
    """Filter and score against predatory/low-quality indicators."""
    
    PREDATORY_INDICATORS = {
        'retracted', 'predatory', 'questionable', 'withdrawn'
    }
    
    @staticmethod
    def calculate(
        is_retracted: bool = False,
        is_paratext: bool = False,
        venue_name: Optional[str] = None,
        has_doi: bool = True,
        indexed_in: Optional[List[str]] = None
    ) -> float:
        """
        Calculate quality filter score (0-100).
        Higher score = more trustworthy.
        
        Args:
            is_retracted: Whether paper is retracted
            is_paratext: Whether paper is paratext
            venue_name: Venue name to check
            has_doi: Whether has DOI
            indexed_in: List of indexes (PubMed, Crossref, etc.)
        
        Returns:
            Quality score 0-100 (0 = predatory, 100 = highly trustworthy)
        """
        # Start with high trust
        score = 100.0
        
        # Critical failures
        if is_retracted:
            return 0.0  # Retracted papers should not appear
        
        if is_paratext:
            score -= 50  # Paratext is not primary research
        
        # No DOI is concerning
        if not has_doi:
            score -= 30
        
        # Venue name check
        if venue_name:
            venue_lower = venue_name.lower()
            if any(indicator in venue_lower for indicator in PredatoryFilterScorer.PREDATORY_INDICATORS):
                score -= 70
        
        # Indexing is a good sign
        if indexed_in:
            if 'pubmed' in indexed_in:
                score = min(100, score + 10)
            if 'crossref' in indexed_in:
                score = min(100, score + 5)
        else:
            score -= 20  # Not indexed anywhere is concerning
        
        return max(0, score)


class DiversityScorer:
    """Promote diversity and explorability in results."""
    
    @staticmethod
    def calculate(
        publication_year: Optional[int],
        topics: List[Dict[str, Any]],
        current_results_topics: Optional[List[str]] = None,
        is_open_access: bool = False
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
            paper_topics = {t.get('display_name', '') for t in topics}
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
        self.influence_scorer = InfluentialCitationScorer()
        self.author_scorer = AuthorReputationScorer()
        self.field_scorer = FieldNormalizationScorer()
        self.predatory_scorer = PredatoryFilterScorer()
        self.diversity_scorer = DiversityScorer()
    
    def score_paper(
        self,
        openalex_data: Optional[Dict[str, Any]] = None,
        semantic_data: Optional[Dict[str, Any]] = None,
        current_results_topics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive paper score using both OpenAlex and Semantic Scholar data.
        
        Args:
            openalex_data: Paper data from OpenAlex
            semantic_data: Paper data from Semantic Scholar
            current_results_topics: Topics already in results (for diversity)
        
        Returns:
            Dictionary with component scores and final score
        """
        scores = {}
        
        # Extract common fields
        citation_count = 0
        publication_year = None
        
        if openalex_data:
            citation_count = openalex_data.get('cited_by_count', 0)
            publication_year = openalex_data.get('publication_year')
        elif semantic_data:
            citation_count = semantic_data.get('citationCount', 0)
            publication_year = semantic_data.get('year')
        
        # 1. Citation Quality
        if openalex_data:
            fwci = openalex_data.get('fwci')
            cited_by_percentile = openalex_data.get('cited_by_percentile_year', {})
            is_top_10 = openalex_data.get('citation_normalized_percentile', {}).get('is_in_top_10_percent', False)
            
            scores['citation_quality'] = self.citation_scorer.calculate(
                citation_count=citation_count,
                publication_year=publication_year,
                fwci=fwci,
                cited_by_percentile=cited_by_percentile,
                is_in_top_10_percent=is_top_10
            )
        else:
            scores['citation_quality'] = 50.0
        
        # 2. Venue Prestige
        venue_name = None
        venue_type = None
        is_oa = False
        
        if openalex_data:
            primary_location = openalex_data.get('primary_location', {})
            venue_name = primary_location.get('source', {}).get('display_name')
            venue_type = primary_location.get('source', {}).get('type')
            is_oa = openalex_data.get('open_access', {}).get('is_oa', False)
        elif semantic_data:
            venue_name = semantic_data.get('venue')
            is_oa = semantic_data.get('isOpenAccess', False)
        
        scores['venue_prestige'] = self.venue_scorer.calculate(
            venue_name=venue_name,
            venue_type=venue_type,
            is_oa=is_oa
        )
        
        # 3. Institution Reputation
        if openalex_data:
            authorships = openalex_data.get('authorships', [])
            institutions = []
            for authorship in authorships:
                institutions.extend(authorship.get('institutions', []))
            
            scores['institution_reputation'] = self.institution_scorer.calculate(
                institutions=institutions,
                countries_distinct_count=openalex_data.get('countries_distinct_count', 0),
                institutions_distinct_count=openalex_data.get('institutions_distinct_count', 0)
            )
        else:
            scores['institution_reputation'] = 50.0
        
        # 4. Influential Citations
        influential_count = 0
        reference_count = 0
        
        if semantic_data:
            influential_count = semantic_data.get('influentialCitationCount', 0)
            reference_count = semantic_data.get('referenceCount', 0)
        elif openalex_data:
            reference_count = openalex_data.get('referenced_works_count', 0)
        
        scores['influential_citations'] = self.influence_scorer.calculate(
            citation_count=citation_count,
            influential_citation_count=influential_count,
            reference_count=reference_count
        )
        
        # 5. Author Reputation
        authors = []
        authorships = []
        
        if semantic_data:
            authors = semantic_data.get('authors', [])
        if openalex_data:
            authorships = openalex_data.get('authorships', [])
        
        scores['author_reputation'] = self.author_scorer.calculate(
            authors=authors,
            authorships=authorships
        )
        
        # 6. Field Normalization
        if openalex_data:
            topics = openalex_data.get('topics', [])
            concepts = openalex_data.get('concepts', [])
            fwci = openalex_data.get('fwci')
            
            scores['field_normalization'] = self.field_scorer.calculate(
                topics=topics,
                concepts=concepts,
                fwci=fwci
            )
        else:
            scores['field_normalization'] = 50.0
        
        # 7. Predatory Filter
        is_retracted = False
        is_paratext = False
        has_doi = False
        indexed_in = []
        
        if openalex_data:
            is_retracted = openalex_data.get('is_retracted', False)
            is_paratext = openalex_data.get('is_paratext', False)
            has_doi = bool(openalex_data.get('doi'))
            indexed_in = openalex_data.get('indexed_in', [])
        elif semantic_data:
            has_doi = bool(semantic_data.get('externalIds', {}).get('DOI'))
        
        scores['predatory_filter'] = self.predatory_scorer.calculate(
            is_retracted=is_retracted,
            is_paratext=is_paratext,
            venue_name=venue_name,
            has_doi=has_doi,
            indexed_in=indexed_in
        )
        
        # If predatory filter fails, return very low score
        if scores['predatory_filter'] < 30:
            scores['final_score'] = scores['predatory_filter'] / 10
            return scores
        
        # 8. Diversity/Explorability
        topics = []
        if openalex_data:
            topics = openalex_data.get('topics', [])
        
        scores['diversity'] = self.diversity_scorer.calculate(
            publication_year=publication_year,
            topics=topics,
            current_results_topics=current_results_topics,
            is_open_access=is_oa
        )
        
        # Calculate weighted final score
        final_score = (
            scores['citation_quality'] * self.weights.citation_quality +
            scores['venue_prestige'] * self.weights.venue_prestige +
            scores['institution_reputation'] * self.weights.institution_reputation +
            scores['influential_citations'] * self.weights.influential_citations +
            scores['author_reputation'] * (self.weights.author_h_index + self.weights.author_citation_count) +
            scores['field_normalization'] * self.weights.field_normalization +
            scores['diversity'] * self.weights.diversity_boost
        )
        
        # Apply predatory filter as a multiplier
        final_score *= (scores['predatory_filter'] / 100)
        
        scores['final_score'] = min(100, final_score)
        
        return scores
