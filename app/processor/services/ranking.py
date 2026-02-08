from typing import List, Dict, Any, Optional

from app.models.papers import DBPaperChunk
from app.chunks.schemas import ChunkRetrieved
from app.retriever.paper_schemas import Paper
from collections import defaultdict
from sentence_transformers import CrossEncoder

from .scoring_models import ComprehensiveScorer, ScoringWeights
from .institution_ranker import InstitutionRanker
from .diversity_manager import DiversityManager, DiversityConfig

import torch

from app.extensions.logger import create_logger

logger = create_logger(__name__)


class RankingService:
    """
    Advanced paper ranking service with multi-factor scoring.
    Combines citation quality, venue prestige, author reputation,
    institution trust, and diversity mechanisms.
    """

    def __init__(
        self,
        scoring_weights: Optional[ScoringWeights] = None,
        diversity_config: Optional[DiversityConfig] = None,
    ):
        """
        Initialize ranking service with configurable weights.

        Args:
            scoring_weights: Weights for scoring components
            diversity_config: Configuration for diversity mechanisms
        """
        self.scorer = ComprehensiveScorer(scoring_weights)
        self.diversity_manager = DiversityManager(diversity_config)
        self.institution_ranker = InstitutionRanker()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cross_encoder = CrossEncoder("BAAI/bge-reranker-base", device=device)

    def rerank_chunks(
        self, query: str, chunks: List[ChunkRetrieved]
    ) -> List[ChunkRetrieved]:
        """
        Rerank chunks using cross-encoder for fine-grained relevance.

        Uses BAAI/bge-reranker-base to score query-chunk pairs for better
        relevance ordering than simple vector similarity.

        Args:
            query: The search query
            chunks: List of ChunkRetrieved objects to rerank
        Returns:
            List of ChunkRetrieved objects reranked by cross-encoder scores
        """
        if not chunks:
            return []
        pairs = [[query, chunk.text] for chunk in chunks]

        scores = self.cross_encoder.predict(pairs)
        min_s, max_s = min(scores), max(scores)
        norm_scores = [
            (s - min_s) / (max_s - min_s + 1e-8)
            for s in scores
        ]

        for chunk, score in zip(chunks, norm_scores):
            chunk.relevance_score = score

        reranked_chunks = sorted(
            chunks, key=lambda c: c.relevance_score, reverse=True
        )

        return reranked_chunks

    def rank_papers(
        self,
        query: str,
        papers: List[Paper],
        chunks: List[ChunkRetrieved],
        enable_diversity: bool = True,
        limit: int = 25,
        relevance_threshold: float = 0.3,
    ) -> List[Paper]:
        """
        Rank papers based on comprehensive multi-factor scoring.

        Args:
            query: The search query
            papers: List of Paper objects to rank
            chunks: List of paper chunks with relevance scores
            enable_diversity: Whether to apply diversity mechanisms
            limit: Maximum papers to return
            relevance_threshold: Minimum semantic relevance score (0-1) to include paper.
                Papers below this threshold are filtered out as irrelevant.
                Example: 0.3 filters out papers like "real walrus" when querying "Walrus decentralized"

        Returns:
            List of Paper objects ranked by comprehensive score
        """
        if not papers:
            return []

        # Aggregate chunk scores (max relevance score per paper)
        paper_chunk_scores = defaultdict(float)
        for chunk in chunks:
            current_score = chunk.relevance_score
            print(
                f"Chunk ID: {chunk.id}, Paper ID: {chunk.paper_id}, Relevance Score: {current_score}"
            )
            if current_score is not None:
                paper_chunk_scores[str(chunk.paper_id)] = max(
                    paper_chunk_scores[str(chunk.paper_id)], float(current_score)
                )

        # If no chunks have scores, use all papers (fallback behavior)
        if not paper_chunk_scores:
            relevant_papers = papers[:limit]
        else:
            relevant_papers = [
                paper
                for paper in papers
                if paper_chunk_scores.get(str(paper.paper_id), 0) >= relevance_threshold
            ]

        if not relevant_papers:
            logger.warning(
                f"Relevance threshold {relevance_threshold} filtered out all {len(papers)} papers. "
                f"Max chunk score: {max(paper_chunk_scores.values()) if paper_chunk_scores else 0:.3f}"
            )
            return []

        # Score each relevant paper
        paper_scores = {}
        paper_dicts = []

        for paper in relevant_papers:
            paper_dict = self._paper_to_dict(paper)
            paper_dicts.append(paper_dict)

            openalex_data = paper_dict.get("openalex_data")
            semantic_data = paper_dict.get("semantic_data")

            scores = self.scorer.score_paper(
                openalex_data=openalex_data, semantic_data=semantic_data
            )

            # Add chunk relevance bonus (from vector similarity)
            chunk_score = paper_chunk_scores.get(str(paper.paper_id), 0)
            if chunk_score > 0:
                # Normalize chunk score (typically 0-1) to 0-100 and blend
                normalized_chunk = min(100, chunk_score * 100)
                scores["final_score"] = (
                    scores["final_score"] * 0.7 + normalized_chunk * 0.3
                )

            paper_scores[paper.id] = scores

            # Update institution ranker
            if openalex_data:
                self.institution_ranker.add_paper_data(openalex_data, semantic_data)

        # Apply diversity if enabled
        if enable_diversity:
            # Convert to format diversity manager expects
            diverse_dicts = self.diversity_manager.diversify_results(
                papers=paper_dicts, scores=paper_scores, limit=limit
            )

            # Convert back to Paper objects in order
            diverse_ids = [p.get("id") for p in diverse_dicts]
            paper_map = {p.id: p for p in relevant_papers}
            ranked_papers = [paper_map[pid] for pid in diverse_ids if pid in paper_map]
        else:
            # Simple sort by final score
            ranked_papers = sorted(
                relevant_papers,
                key=lambda p: paper_scores.get(p.id, {}).get("final_score", 0),
                reverse=True,
            )[:limit]

        # Attach scores to papers for transparency (use existing relevance_score field)
        for paper in ranked_papers:
            score_data = paper_scores.get(paper.id, {})
            if hasattr(paper, "relevance_score"):
                paper.relevance_score = score_data.get("final_score", 0)

        return ranked_papers

    def calculate_relevance(
        self, query: str, paper: Paper, chunk_score: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate comprehensive relevance score of a paper to the query.

        Args:
            query: The search query
            paper: The Paper object
            chunk_score: Relevance score from vector similarity

        Returns:
            Dictionary with score components
        """
        paper_dict = self._paper_to_dict(paper)

        scores = self.scorer.score_paper(
            openalex_data=paper_dict.get("openalex_data"),
            semantic_data=paper_dict.get("semantic_data"),
        )

        # Add chunk score
        if chunk_score > 0:
            normalized_chunk = min(100, chunk_score * 100)
            scores["chunk_relevance"] = normalized_chunk
            scores["final_score"] = scores["final_score"] * 0.7 + normalized_chunk * 0.3

        return scores

    def get_institution_rankings(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get ranked list of institutions based on aggregated paper data.

        Args:
            limit: Maximum institutions to return

        Returns:
            List of institution summaries with rankings
        """
        ranked_institutions = self.institution_ranker.rank_institutions()

        summaries = []
        for inst_profile in ranked_institutions[:limit]:
            summary = self.institution_ranker.get_institution_summary(
                inst_profile.institution_id
            )
            if summary:
                summaries.append(summary)

        return summaries

    def get_diversity_metrics(self, papers: List[Paper]) -> Dict[str, float]:
        """
        Calculate diversity metrics for a list of papers.

        Args:
            papers: List of papers

        Returns:
            Dictionary with diversity metrics
        """
        paper_dicts = [self._paper_to_dict(p) for p in papers]
        return self.diversity_manager.calculate_overall_diversity(paper_dicts)

    def _paper_to_dict(self, paper: Paper) -> Dict[str, Any]:
        """
        Convert Paper object to dictionary format for scoring.

        Args:
            paper: Paper object

        Returns:
            Dictionary with paper data
        """
        # Extract year from publication_date
        year = None
        if hasattr(paper, "publication_date") and paper.publication_date:
            year = (
                paper.publication_date.year
                if hasattr(paper.publication_date, "year")
                else None
            )

        paper_dict = {
            "id": paper.id,
            "title": paper.title,
            "publication_year": year,
            "year": year,
        }

        # Try to get OpenAlex ID from external_ids
        openalex_id = None
        if hasattr(paper, "external_ids") and paper.external_ids:
            openalex_id = paper.external_ids.get("OpenAlex") or paper.external_ids.get(
                "openalex"
            )

        if openalex_id:
            # Ideally, fetch full OpenAlex data from database or API
            # For now, use what's available on the Paper object
            openalex_data = {
                "id": openalex_id,
                "cited_by_count": getattr(paper, "citation_count", 0),
                "publication_year": year,
                "fwci": getattr(paper, "fwci", None),
                "is_retracted": getattr(paper, "is_retracted", False),
                "open_access": {"is_oa": getattr(paper, "is_open_access", False)},
                "authorships": [],  # Not stored in Paper schema
                "topics": getattr(paper, "topics", []),
                "concepts": getattr(paper, "concepts", []),
                "primary_location": {},  # Not stored in Paper schema
            }
            paper_dict["openalex_data"] = openalex_data

        # Try to get Semantic Scholar data
        s2_id = None
        if hasattr(paper, "external_ids") and paper.external_ids:
            s2_id = paper.external_ids.get("CorpusId")

        if s2_id:
            semantic_data = {
                "paperId": s2_id,
                "citationCount": getattr(paper, "citation_count", 0),
                "influentialCitationCount": getattr(
                    paper, "influential_citation_count", 0
                ),
                "referenceCount": getattr(paper, "reference_count", 0),
                "authors": getattr(paper, "authors", []),
                "year": getattr(paper, "year", None),
                "venue": getattr(paper, "venue", None),
                "isOpenAccess": getattr(paper, "is_open_access", False),
            }
            paper_dict["semantic_data"] = semantic_data

        return paper_dict
