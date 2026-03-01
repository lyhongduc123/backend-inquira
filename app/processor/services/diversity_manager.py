"""
Diversity and explorability mechanisms for search results.
Ensures result sets are diverse and promote discovery of novel work.
"""
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass
import random
import math


@dataclass
class DiversityConfig:
    """Configuration for diversity mechanisms."""
    # Temporal diversity
    recency_weight: float = 0.3  # Weight for recent papers
    temporal_spread: bool = True  # Ensure papers from different eras
    
    # Topic diversity
    topic_diversity_weight: float = 0.3  # Weight for topic diversity
    min_topic_coverage: int = 3  # Minimum number of distinct topics
    
    # Author diversity
    author_diversity_weight: float = 0.2  # Weight for author diversity
    max_same_author_papers: int = 2  # Max papers from same first author
    
    # Institution diversity
    institution_diversity_weight: float = 0.2  # Weight for institution diversity
    max_same_institution_papers: int = 3  # Max papers from same institution
    
    # Exploration vs Exploitation
    exploration_ratio: float = 0.2  # Ratio of results to prioritize exploration

                            
class DiversityManager:
    """Manage diversity and explorability in search results."""
    
    def __init__(self, config: Optional[DiversityConfig] = None):
        self.config = config or DiversityConfig()
    
    def diversify_results(
        self,
        papers: List[Dict[str, Any]],
        scores: Dict[str, Dict[str, float]],
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Re-rank papers to promote diversity while maintaining relevance.
        
        Args:
            papers: List of paper dictionaries (OpenAlex + Semantic Scholar merged)
            scores: Dictionary mapping paper_id to score components
            limit: Maximum papers to return
        
        Returns:
            Diversified list of papers
        """
        if not papers:
            return []
        
        # Split into high-relevance and exploration candidates
        exploration_count = int(limit * self.config.exploration_ratio)
        relevance_count = limit - exploration_count
        
        # Sort by final score
        sorted_papers = sorted(
            papers,
            key=lambda p: scores.get(p.get('id', ''), {}).get('final_score', 0),
            reverse=True
        )
        
        # Take top relevance papers
        selected_papers = []
        used_authors = set()
        used_institutions = set()
        used_topics = set()
        
        # Phase 1: Select high-relevance diverse papers
        for paper in sorted_papers:
            if len(selected_papers) >= relevance_count:
                break
            
            # Check diversity constraints
            if self._passes_diversity_check(
                paper,
                used_authors,
                used_institutions,
                used_topics
            ):
                selected_papers.append(paper)
                self._update_diversity_tracking(
                    paper,
                    used_authors,
                    used_institutions,
                    used_topics
                )
        
        # Phase 2: Add exploration papers (novel, recent, or underrepresented)
        exploration_candidates = [p for p in sorted_papers if p not in selected_papers]
        exploration_papers = self._select_exploration_papers(
            exploration_candidates,
            scores,
            exploration_count,
            used_topics
        )
        
        selected_papers.extend(exploration_papers)
        
        return selected_papers[:limit]
    
    def _passes_diversity_check(
        self,
        paper: Dict[str, Any],
        used_authors: Set[str],
        used_institutions: Set[str],
        used_topics: Set[str]
    ) -> bool:
        """Check if paper passes diversity constraints."""
        
        # Check author diversity
        first_author = self._get_first_author(paper)
        if first_author:
            author_count = sum(1 for a in used_authors if a == first_author)
            if author_count >= self.config.max_same_author_papers:
                return False
        
        # Check institution diversity
        institutions = self._get_institutions(paper)
        if institutions:
            for inst in institutions:
                inst_count = sum(1 for i in used_institutions if i == inst)
                if inst_count >= self.config.max_same_institution_papers:
                    return False
        
        return True
    
    def _update_diversity_tracking(
        self,
        paper: Dict[str, Any],
        used_authors: Set[str],
        used_institutions: Set[str],
        used_topics: Set[str]
    ):
        """Update tracking sets with paper information."""
        
        # Track first author
        first_author = self._get_first_author(paper)
        if first_author:
            used_authors.add(first_author)
        
        # Track institutions
        institutions = self._get_institutions(paper)
        used_institutions.update(institutions)
        
        # Track topics
        topics = self._get_topics(paper)
        used_topics.update(topics)
    
    def _select_exploration_papers(
        self,
        candidates: List[Dict[str, Any]],
        scores: Dict[str, Dict[str, float]],
        count: int,
        used_topics: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Select papers for exploration (novel topics, recent work).
        
        Args:
            candidates: Candidate papers
            scores: Score components
            count: Number to select
            used_topics: Topics already covered
        
        Returns:
            List of exploration papers
        """
        exploration_scores = []
        
        for paper in candidates:
            paper_id = paper.get('id', '')
            exp_score = 0.0
            
            # Recency bonus
            year = paper.get('publication_year') or paper.get('year')
            if year:
                years_old = 2025 - year
                if years_old <= 2:
                    exp_score += 30
                elif years_old <= 5:
                    exp_score += 20
                elif years_old <= 10:
                    exp_score += 10
            
            # Topic novelty bonus
            paper_topics = set(self._get_topics(paper))
            new_topics = paper_topics - used_topics
            exp_score += len(new_topics) * 10
            
            # Diversity bonus from scores
            if paper_id in scores:
                exp_score += scores[paper_id].get('diversity', 0) * 0.5
            
            # Influential but less cited papers (hidden gems)
            if paper_id in scores:
                influential = scores[paper_id].get('influential_citations', 0)
                citation = scores[paper_id].get('citation_quality', 0)
                if influential > citation:
                    exp_score += 15  # High influence relative to citations
            
            exploration_scores.append((paper, exp_score))
        
        # Sort by exploration score
        exploration_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top exploration papers
        return [paper for paper, _ in exploration_scores[:count]]
    
    def _get_first_author(self, paper: Dict[str, Any]) -> Optional[str]:
        """Get first author ID from paper."""
        # OpenAlex format
        authorships = paper.get('authorships', [])
        if authorships:
            for authorship in authorships:
                if authorship.get('author_position') == 'first':
                    return authorship.get('author', {}).get('id')
            # Fallback to first in list
            return authorships[0].get('author', {}).get('id')
        
        # DB format or Semantic Scholar format
        authors = paper.get('authors', [])
        if authors:
            # Check if sorted by author_position (DB format)
            authors_sorted = sorted(
                [a for a in authors if a.get('author_position') is not None],
                key=lambda x: x.get('author_position', 999)
            )
            if authors_sorted:
                return str(authors_sorted[0].get('authorId'))
            # Fallback to first author in list
            return str(authors[0].get('authorId'))
        
        return None
    
    def _get_institutions(self, paper: Dict[str, Any]) -> List[str]:
        """Get institution IDs from paper."""
        institutions = []
        
        # OpenAlex format
        authorships = paper.get('authorships', [])
        for authorship in authorships:
            for inst in authorship.get('institutions', []):
                inst_id = inst.get('id')
                if inst_id:
                    institutions.append(inst_id)
        
        # DB format - get institution_id from authors
        if not institutions:
            authors = paper.get('authors', [])
            for author in authors:
                inst_id = author.get('institution_id')
                if inst_id:
                    institutions.append(str(inst_id))
        
        return institutions
    
    def _get_topics(self, paper: Dict[str, Any]) -> List[str]:
        """Get topic names from paper."""
        topics = []
        
        # OpenAlex topics
        for topic in paper.get('topics', [])[:5]:
            topic_name = topic.get('display_name')
            if topic_name:
                topics.append(topic_name)
        
        # Semantic Scholar fields of study
        for field in paper.get('s2FieldsOfStudy', [])[:5]:
            field_name = field.get('category')
            if field_name:
                topics.append(field_name)
        
        return topics
    
    def calculate_temporal_diversity(
        self,
        papers: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate temporal diversity score (0-100).
        Higher score = better spread across time periods.
        
        Args:
            papers: List of papers
        
        Returns:
            Temporal diversity score
        """
        if not papers:
            return 0.0
        
        years = []
        for paper in papers:
            year = paper.get('publication_year') or paper.get('year')
            if year:
                years.append(year)
        
        if not years:
            return 0.0
        
        # Calculate spread metrics
        min_year = min(years)
        max_year = max(years)
        year_range = max_year - min_year
        
        # Count distinct decades
        decades = len(set(y // 10 for y in years))
        
        # Score based on range and decade diversity
        range_score = min(50, year_range * 2)
        decade_score = min(50, decades * 10)
        
        return range_score + decade_score
    
    def calculate_topic_diversity(
        self,
        papers: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate topic diversity score (0-100).
        
        Args:
            papers: List of papers
        
        Returns:
            Topic diversity score
        """
        all_topics = set()
        
        for paper in papers:
            topics = self._get_topics(paper)
            all_topics.update(topics)
        
        # More unique topics = higher diversity
        topic_count = len(all_topics)
        
        # Ideal is 1-2 topics per paper
        ideal_topics = len(papers) * 1.5
        
        if topic_count >= ideal_topics:
            return 100.0
        else:
            return (topic_count / ideal_topics) * 100
    
    def calculate_author_diversity(
        self,
        papers: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate author diversity score (0-100).
        
        Args:
            papers: List of papers
        
        Returns:
            Author diversity score
        """
        all_authors = set()
        
        for paper in papers:
            # OpenAlex
            for authorship in paper.get('authorships', []):
                author_id = authorship.get('author', {}).get('id')
                if author_id:
                    all_authors.add(author_id)
            
            # Semantic Scholar
            for author in paper.get('authors', []):
                author_id = author.get('authorId')
                if author_id:
                    all_authors.add(author_id)
        
        author_count = len(all_authors)
        
        # Ideal is ~3 authors per paper (varied collaboration)
        ideal_authors = len(papers) * 3
        
        if author_count >= ideal_authors:
            return 100.0
        else:
            return (author_count / ideal_authors) * 100
    
    def calculate_overall_diversity(
        self,
        papers: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate overall diversity metrics.
        
        Args:
            papers: List of papers
        
        Returns:
            Dictionary with diversity metrics
        """
        return {
            'temporal_diversity': self.calculate_temporal_diversity(papers),
            'topic_diversity': self.calculate_topic_diversity(papers),
            'author_diversity': self.calculate_author_diversity(papers),
            'overall_diversity': (
                self.calculate_temporal_diversity(papers) * 0.33 +
                self.calculate_topic_diversity(papers) * 0.34 +
                self.calculate_author_diversity(papers) * 0.33
            )
        }


class ExplorationBooster:
    """Boost scores for exploration-worthy papers."""
    
    @staticmethod
    def boost_underrepresented_work(
        paper: Dict[str, Any],
        current_results: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate boost for underrepresented work (0-20 points).
        
        Args:
            paper: Paper to evaluate
            current_results: Papers already in results
        
        Returns:
            Boost score
        """
        boost = 0.0
        
        # Check if from underrepresented region
        institutions = []
        for authorship in paper.get('authorships', []):
            for inst in authorship.get('institutions', []):
                country = inst.get('country_code')
                if country:
                    institutions.append(country)
        
        # Boost for non-Western countries
        underrepresented_regions = ['IN', 'BR', 'CN', 'ZA', 'NG', 'MX', 'AR']
        if any(country in underrepresented_regions for country in institutions):
            boost += 10
        
        # Check if topic is underrepresented in current results
        paper_topics = set()
        for topic in paper.get('topics', [])[:3]:
            topic_name = topic.get('display_name')
            if topic_name:
                paper_topics.add(topic_name)
        
        result_topics = set()
        for result in current_results:
            for topic in result.get('topics', [])[:3]:
                topic_name = topic.get('display_name')
                if topic_name:
                    result_topics.add(topic_name)
        
        unique_topics = paper_topics - result_topics
        if len(unique_topics) >= 2:
            boost += 10
        
        return boost
    
    @staticmethod
    def boost_emerging_research(paper: Dict[str, Any]) -> float:
        """
        Calculate boost for emerging research (0-15 points).
        
        Args:
            paper: Paper to evaluate
        
        Returns:
            Boost score
        """
        boost = 0.0
        
        # Very recent papers with growing citations
        year = paper.get('publication_year') or paper.get('year')
        citations = paper.get('cited_by_count') or paper.get('citationCount', 0)
        
        if year and year >= 2023:
            # Recent paper
            boost += 8
            
            # With decent citations despite recency = emerging
            if citations > 10:
                boost += 7
        elif year and year >= 2020:
            # Moderately recent
            boost += 4
            
            if citations > 50:
                boost += 5
        
        return boost
