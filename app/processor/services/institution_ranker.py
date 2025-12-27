"""
Institution-level ranking and reputation scoring.
Aggregates data from multiple papers to build institution profiles.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import math


@dataclass
class InstitutionProfile:
    """Comprehensive institution profile."""
    institution_id: str
    display_name: str
    country_code: Optional[str] = None
    institution_type: Optional[str] = None
    
    # Aggregated metrics
    total_papers: int = 0
    total_citations: int = 0
    highly_cited_papers: int = 0  # Top 10% in field
    h_index_estimate: int = 0
    
    # Field strength
    field_distribution: Dict[str, int] = field(default_factory=dict)
    top_fields: List[str] = field(default_factory=list)
    
    # Collaboration metrics
    international_collaborations: int = 0
    unique_coauthor_institutions: set = field(default_factory=set)
    
    # Quality metrics
    avg_fwci: float = 0.0
    open_access_ratio: float = 0.0
    
    # Reputation score
    reputation_score: float = 0.0
    
    # Notable authors
    top_authors: List[Dict[str, Any]] = field(default_factory=list)


class InstitutionRanker:
    """Build and rank institution profiles from paper metadata."""
    
    def __init__(self):
        self.institutions: Dict[str, InstitutionProfile] = {}
    
    def add_paper_data(
        self,
        openalex_data: Dict[str, Any],
        semantic_data: Optional[Dict[str, Any]] = None
    ):
        """
        Add paper data to institution profiles.
        
        Args:
            openalex_data: OpenAlex paper data
            semantic_data: Semantic Scholar paper data (optional)
        """
        authorships = openalex_data.get('authorships', [])
        
        for authorship in authorships:
            institutions = authorship.get('institutions', [])
            author_info = authorship.get('author', {})
            
            for inst_data in institutions:
                inst_id = inst_data.get('id', '')
                if not inst_id:
                    continue
                
                # Get or create institution profile
                if inst_id not in self.institutions:
                    self.institutions[inst_id] = InstitutionProfile(
                        institution_id=inst_id,
                        display_name=inst_data.get('display_name', 'Unknown'),
                        country_code=inst_data.get('country_code'),
                        institution_type=inst_data.get('type')
                    )
                
                profile = self.institutions[inst_id]
                
                # Update paper count
                profile.total_papers += 1
                
                # Update citations
                citations = openalex_data.get('cited_by_count', 0)
                profile.total_citations += citations
                
                # Check if highly cited (top 10%)
                is_highly_cited = openalex_data.get('citation_normalized_percentile', {}).get('is_in_top_10_percent', False)
                if is_highly_cited:
                    profile.highly_cited_papers += 1
                
                # Update FWCI
                fwci = openalex_data.get('fwci')
                if fwci is not None:
                    # Running average
                    profile.avg_fwci = (profile.avg_fwci * (profile.total_papers - 1) + fwci) / profile.total_papers
                
                # Update field distribution
                topics = openalex_data.get('topics', [])
                for topic in topics[:3]:  # Top 3 topics
                    topic_name = topic.get('display_name', '')
                    if topic_name:
                        profile.field_distribution[topic_name] = profile.field_distribution.get(topic_name, 0) + 1
                
                # Track collaboration
                other_institutions = [i.get('id') for i in institutions if i.get('id') != inst_id]
                profile.unique_coauthor_institutions.update(other_institutions)
                
                # International collaboration
                if openalex_data.get('countries_distinct_count', 0) > 1:
                    profile.international_collaborations += 1
                
                # Open access tracking
                is_oa = openalex_data.get('open_access', {}).get('is_oa', False)
                if is_oa:
                    oa_count = profile.open_access_ratio * profile.total_papers + 1
                    profile.open_access_ratio = oa_count / profile.total_papers
                else:
                    oa_count = profile.open_access_ratio * profile.total_papers
                    profile.open_access_ratio = oa_count / profile.total_papers
                
                # Track notable authors
                if semantic_data and semantic_data.get('authors'):
                    for author in semantic_data['authors']:
                        h_index = author.get('hIndex', 0)
                        if h_index > 20:  # Notable threshold
                            author_entry = {
                                'author_id': author.get('authorId'),
                                'name': author.get('name'),
                                'h_index': h_index,
                                'citation_count': author.get('citationCount', 0)
                            }
                            # Avoid duplicates
                            if not any(a['author_id'] == author_entry['author_id'] for a in profile.top_authors):
                                profile.top_authors.append(author_entry)
    
    def calculate_h_index(self, inst_id: str) -> int:
        """
        Estimate institution H-index from papers.
        Note: This is a simplified estimation. True H-index requires all papers.
        
        Args:
            inst_id: Institution ID
        
        Returns:
            Estimated H-index
        """
        profile = self.institutions.get(inst_id)
        if not profile:
            return 0
        
        # Simplified estimation: h-index is approximately sqrt(total_citations)
        # This is a rough approximation
        estimated_h = int(math.sqrt(profile.total_citations))
        
        # Cap at number of papers
        return min(estimated_h, profile.total_papers)
    
    def calculate_field_strength(self, inst_id: str, field: str) -> float:
        """
        Calculate institution's strength in a specific field (0-100).
        
        Args:
            inst_id: Institution ID
            field: Field name
        
        Returns:
            Field strength score
        """
        profile = self.institutions.get(inst_id)
        if not profile or not profile.field_distribution:
            return 0.0
        
        field_papers = profile.field_distribution.get(field, 0)
        if field_papers == 0:
            return 0.0
        
        # Percentage of papers in this field
        field_ratio = field_papers / profile.total_papers
        
        # Combine with absolute count (log scale)
        absolute_score = min(50, math.log10(field_papers + 1) * 15)
        ratio_score = field_ratio * 50
        
        return absolute_score + ratio_score
    
    def calculate_reputation_score(self, inst_id: str) -> float:
        """
        Calculate comprehensive institution reputation score (0-100).
        
        Args:
            inst_id: Institution ID
        
        Returns:
            Reputation score
        """
        profile = self.institutions.get(inst_id)
        if not profile:
            return 0.0
        
        score = 0.0
        
        # 1. Research output (20 points)
        if profile.total_papers > 0:
            output_score = min(20, math.log10(profile.total_papers + 1) * 5)
            score += output_score
        
        # 2. Citation impact (25 points)
        if profile.total_citations > 0:
            citation_score = min(25, math.log10(profile.total_citations + 1) * 3)
            score += citation_score
        
        # 3. H-index (20 points)
        h_index = self.calculate_h_index(inst_id)
        h_score = min(20, h_index / 5)
        score += h_score
        
        # 4. Highly cited papers ratio (15 points)
        if profile.total_papers > 0:
            highly_cited_ratio = profile.highly_cited_papers / profile.total_papers
            score += highly_cited_ratio * 15
        
        # 5. Field-weighted citation impact (10 points)
        if profile.avg_fwci > 0:
            fwci_score = min(10, profile.avg_fwci * 5)
            score += fwci_score
        
        # 6. International collaboration (5 points)
        if profile.total_papers > 0:
            intl_ratio = profile.international_collaborations / profile.total_papers
            score += intl_ratio * 5
        
        # 7. Notable authors (5 points)
        if profile.top_authors:
            notable_score = min(5, len(profile.top_authors) * 0.5)
            score += notable_score
        
        profile.reputation_score = min(100, score)
        return profile.reputation_score
    
    def get_top_fields(self, inst_id: str, limit: int = 5) -> List[str]:
        """
        Get institution's top research fields.
        
        Args:
            inst_id: Institution ID
            limit: Maximum fields to return
        
        Returns:
            List of top field names
        """
        profile = self.institutions.get(inst_id)
        if not profile or not profile.field_distribution:
            return []
        
        sorted_fields = sorted(
            profile.field_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        profile.top_fields = [field for field, _ in sorted_fields[:limit]]
        return profile.top_fields
    
    def rank_institutions(self) -> List[InstitutionProfile]:
        """
        Rank all institutions by reputation score.
        
        Returns:
            List of institutions sorted by reputation
        """
        # Calculate scores for all institutions
        for inst_id in self.institutions:
            self.calculate_reputation_score(inst_id)
            self.get_top_fields(inst_id)
        
        # Sort by reputation score
        ranked = sorted(
            self.institutions.values(),
            key=lambda x: x.reputation_score,
            reverse=True
        )
        
        return ranked
    
    def get_institution_summary(self, inst_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of institution metrics.
        
        Args:
            inst_id: Institution ID
        
        Returns:
            Dictionary with institution summary
        """
        profile = self.institutions.get(inst_id)
        if not profile:
            return None
        
        return {
            'institution_id': profile.institution_id,
            'name': profile.display_name,
            'country': profile.country_code,
            'type': profile.institution_type,
            'metrics': {
                'total_papers': profile.total_papers,
                'total_citations': profile.total_citations,
                'h_index_estimate': self.calculate_h_index(inst_id),
                'highly_cited_papers': profile.highly_cited_papers,
                'avg_fwci': round(profile.avg_fwci, 2),
                'open_access_ratio': round(profile.open_access_ratio * 100, 1),
                'international_collaboration_ratio': round(
                    (profile.international_collaborations / max(1, profile.total_papers)) * 100, 1
                ),
                'unique_partner_institutions': len(profile.unique_coauthor_institutions)
            },
            'top_fields': self.get_top_fields(inst_id),
            'reputation_score': round(profile.reputation_score, 2),
            'notable_authors_count': len(profile.top_authors)
        }


class FieldStrengthAnalyzer:
    """Analyze field-specific strength across institutions."""
    
    def __init__(self, ranker: InstitutionRanker):
        self.ranker = ranker
    
    def get_top_institutions_in_field(
        self,
        field: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top institutions in a specific field.
        
        Args:
            field: Field name
            limit: Maximum institutions to return
        
        Returns:
            List of institutions with field strength scores
        """
        field_scores = []
        
        for inst_id, profile in self.ranker.institutions.items():
            field_strength = self.ranker.calculate_field_strength(inst_id, field)
            
            if field_strength > 0:
                field_scores.append({
                    'institution_id': inst_id,
                    'name': profile.display_name,
                    'country': profile.country_code,
                    'field_strength': round(field_strength, 2),
                    'papers_in_field': profile.field_distribution.get(field, 0),
                    'reputation_score': profile.reputation_score
                })
        
        # Sort by field strength
        field_scores.sort(key=lambda x: x['field_strength'], reverse=True)
        
        return field_scores[:limit]
    
    def compare_institutions(
        self,
        inst_id1: str,
        inst_id2: str
    ) -> Dict[str, Any]:
        """
        Compare two institutions across metrics.
        
        Args:
            inst_id1: First institution ID
            inst_id2: Second institution ID
        
        Returns:
            Comparison dictionary
        """
        profile1 = self.ranker.institutions.get(inst_id1)
        profile2 = self.ranker.institutions.get(inst_id2)
        
        if not profile1 or not profile2:
            return {}
        
        return {
            'institution_1': {
                'name': profile1.display_name,
                'reputation_score': profile1.reputation_score,
                'total_papers': profile1.total_papers,
                'total_citations': profile1.total_citations,
                'avg_fwci': profile1.avg_fwci,
                'top_fields': profile1.top_fields
            },
            'institution_2': {
                'name': profile2.display_name,
                'reputation_score': profile2.reputation_score,
                'total_papers': profile2.total_papers,
                'total_citations': profile2.total_citations,
                'avg_fwci': profile2.avg_fwci,
                'top_fields': profile2.top_fields
            },
            'comparison': {
                'reputation_difference': profile1.reputation_score - profile2.reputation_score,
                'citation_ratio': profile1.total_citations / max(1, profile2.total_citations),
                'productivity_ratio': profile1.total_papers / max(1, profile2.total_papers),
                'quality_difference': profile1.avg_fwci - profile2.avg_fwci
            }
        }
