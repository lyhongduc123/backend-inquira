from typing import List, Dict, Any, Optional, Union
import gc

from app.models.papers import DBPaper, DBPaperChunk
from app.chunks.schemas import ChunkRetrieved
from app.core.dtos import PaperDTO
from app.processor.schemas import RankedPaper
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

        # Lazy load cross_encoder only when needed to save memory
        self._cross_encoder = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cuda_failed = False  # Track if CUDA has failed

    def _get_cross_encoder(self):
        """Lazily load the cross-encoder model on first use with CUDA error handling."""
        if self._cross_encoder is None:
            device = "cpu" if self._cuda_failed else self._device
            try:
                logger.debug(f"Loading CrossEncoder model on {device}")
                self._cross_encoder = CrossEncoder("BAAI/bge-reranker-base", device=device)
                
                # Test the model with a small batch
                if device == "cuda":
                    try:
                        test_scores = self._cross_encoder.predict([["test", "test"]])
                        logger.debug("CUDA test successful")
                    except RuntimeError as e:
                        if "CUDA" in str(e) or "CUBLAS" in str(e):
                            logger.warning(f"CUDA test failed: {e}. Falling back to CPU")
                            self._cuda_failed = True
                            self._cross_encoder = None
                            # Clean up CUDA memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                gc.collect()
                            return self._get_cross_encoder()  # Retry with CPU
                        raise
                        
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder: {e}")
                if device == "cuda" and not self._cuda_failed:
                    logger.warning("Retrying with CPU")
                    self._cuda_failed = True
                    self._cross_encoder = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    return self._get_cross_encoder()
                raise
        return self._cross_encoder

    @property
    def cross_encoder(self):
        """Property to access cross_encoder with lazy loading."""
        return self._get_cross_encoder()

    def rerank_chunks(
        self, query: str, chunks: List[ChunkRetrieved], batch_size: int = 16
    ) -> List[ChunkRetrieved]:
        """
        Rerank chunks using cross-encoder for fine-grained relevance.

        Uses BAAI/bge-reranker-base to score query-chunk pairs for better
        relevance ordering than simple vector similarity.

        Args:
            query: The search query
            chunks: List of ChunkRetrieved objects to rerank
            batch_size: Batch size for processing (default 16, reduced if CUDA errors occur)
        Returns:
            List of ChunkRetrieved objects reranked by cross-encoder scores
        """
        if not chunks:
            return []
        
        pairs = [[query, chunk.text] for chunk in chunks]
        
        try:
            # Try with specified batch size
            scores = self._predict_with_batching(pairs, batch_size)
        except RuntimeError as e:
            if "CUDA" in str(e) or "CUBLAS" in str(e) or "out of memory" in str(e).lower():
                logger.warning(f"CUDA error during prediction: {e}. Retrying with smaller batch or CPU")
                # Clear CUDA cache and retry
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Force CPU fallback
                self._cuda_failed = True
                self._cross_encoder = None
                
                try:
                    scores = self._predict_with_batching(pairs, batch_size)
                except Exception as inner_e:
                    logger.error(f"Failed to rerank even on CPU: {inner_e}")
                    # Return chunks with original scores
                    return chunks
            else:
                raise
        
        min_s, max_s = min(scores), max(scores)
        norm_scores = [(s - min_s) / (max_s - min_s + 1e-8) for s in scores]

        for chunk, score in zip(chunks, norm_scores):
            chunk.relevance_score = score

        reranked_chunks = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)

        return reranked_chunks
    
    def _predict_with_batching(self, pairs: List[List[str]], batch_size: int) -> List[float]:
        """Process pairs in batches to avoid memory issues."""
        if len(pairs) <= batch_size:
            return self.cross_encoder.predict(pairs) # type: ignore
        
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_scores = self.cross_encoder.predict(batch)
            all_scores.extend(batch_scores)
        
        return all_scores

    def rank_papers(
        self,
        query: str,
        papers: List[DBPaper],
        chunks: List[ChunkRetrieved],
        enable_diversity: bool = True,
        limit: int = 25,
        relevance_threshold: float = 0.3,
    ) -> List[RankedPaper]:
        """
        Rank papers based on comprehensive multi-factor scoring.
        NOTE: Now only sorts papers without filtering - all input papers are returned ranked.

        Args:
            query: The search query
            papers: List of Paper objects to rank
            chunks: List of paper chunks with relevance scores
            enable_diversity: Whether to apply diversity mechanisms
            limit: Maximum papers to return (DEPRECATED - kept for compatibility, not used)
            relevance_threshold: Minimum semantic relevance score (DEPRECATED - kept for compatibility, not used)

        Returns:
            List of Paper objects ranked by comprehensive score (all input papers)
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

        # Use all papers - no filtering by relevance threshold
        relevant_papers = papers

        # Score each relevant paper
        paper_scores = {}
        paper_dicts = []

        # Track topics for diversity scoring
        current_results_topics = []

        for paper in relevant_papers:
            paper_authors = paper.paper_authors or []
            authors_list = [
                {
                    "authorId": ap.author_id,
                    "author_position": ap.author_position,
                    "institution_id": ap.institution_id,
                }
                for ap in paper_authors
            ]
            paper_dict = {
                "id": paper.id,
                "paper_id": paper.paper_id,
                "title": paper.title,
                "topics": paper.topics or [],
                "authors": authors_list,
            }
            paper_dicts.append(paper_dict)
            
            paper_topics = paper.topics
            if paper_topics:
                current_results_topics.extend(
                    [
                        t.get("display_name", "")
                        for t in paper_topics
                        if t.get("display_name")
                    ]
                )

            # Call score_paper with db_paper directly (no more openalex_data/semantic_data dicts)
            scores = self.scorer.score_paper(
                db_paper=paper,
                journal_data=None,  # Will be fetched from DBJournal if needed
                trust_scores=None,  # Will use db_paper.author_trust_score, institutional_trust_score
                current_results_topics=current_results_topics,
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
            # Simple sort by final score - return all papers, no limit
            ranked_papers = sorted(
                relevant_papers,
                key=lambda p: paper_scores.get(p.id, {}).get("final_score", 0),
                reverse=True,
            )

        # Wrap DBPaper objects with ranking scores
        ranked_with_scores = []
        for paper in ranked_papers:
            score_data = paper_scores.get(paper.id, {})
            
            ranked_paper = RankedPaper(
                id=paper.id,
                paper_id=paper.paper_id,
                paper=paper,
                relevance_score=score_data.get("final_score", 0),
                ranking_scores=score_data,
            )
            ranked_with_scores.append(ranked_paper)

        return ranked_with_scores

    def calculate_relevance(
        self, query: str, paper: DBPaper, chunk_score: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate comprehensive relevance score of a paper to the query.

        Args:
            query: The search query
            paper: The DBPaper object
            chunk_score: Relevance score from vector similarity

        Returns:
            Dictionary with score components
        """
        scores = self.scorer.score_paper(
            db_paper=paper,
            journal_data=None,  # Will be fetched if needed
            trust_scores=None,  # Will use db_paper fields
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

    def get_diversity_metrics(self, papers: List[PaperDTO]) -> Dict[str, float]:
        """
        Calculate diversity metrics for a list of papers.

        Args:
            papers: List of papers

        Returns:
            Dictionary with diversity metrics
        """
        paper_dicts = [self._paper_to_dict(p) for p in papers]
        return self.diversity_manager.calculate_overall_diversity(paper_dicts)

    def _paper_to_dict(self, paper: Union[DBPaper, PaperDTO]) -> Dict[str, Any]:
        """
        Convert Paper object to dictionary format for diversity manager.
        Only extracts essential fields - no longer builds openalex_data/semantic_data dicts.

        Args:
            paper: Paper object (DBPaper or PaperDTO)

        Returns:
            Dictionary with basic paper data for diversity analysis
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
            "title": getattr(paper, "title", None),
            "year": year,
            "venue": getattr(paper, "venue", None),
            "topics": getattr(paper, "topics", []),
            "citation_count": getattr(paper, "citation_count", 0),
            "is_open_access": getattr(paper, "is_open_access", False),
        }

        return paper_dict
