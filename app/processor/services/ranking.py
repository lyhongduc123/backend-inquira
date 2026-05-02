import math
from typing import List, Dict, Any, Optional, Union
import gc
import os

import requests

from app.models.papers import DBPaper, DBPaperChunk
from app.domain.chunks.schemas import ChunkRetrieved
from app.core.dtos import PaperDTO
from app.core.config import settings
from app.processor.schemas import RankedPaper
from collections import defaultdict
from sentence_transformers import CrossEncoder

from .scoring_models import ComprehensiveScorer, ScoringWeights
from .institution_ranker import InstitutionRanker

import torch

from app.extensions.logger import create_logger

logger = create_logger(__name__)


class RankingService:
    """
    Advanced paper ranking service with multi-factor scoring.
    Combines citation quality, venue prestige, author reputation,
    institution trust, and diversity mechanisms.
    """

    QUARTILE_BONUS = {
        "Q1": 30, 
        "Q2": 15,  
        "Q3": 5,
        "Q4": 0,
    }

    def __init__(
        self,
        scoring_weights: Optional[ScoringWeights] = None,
    ):
        """
        Initialize ranking service with configurable weights.

        Args:
            scoring_weights: Weights for scoring components
            diversity_config: Configuration for diversity mechanisms
        """
        self.scorer = ComprehensiveScorer(scoring_weights)
        self.institution_ranker = InstitutionRanker()

        # Lazy load cross_encoder only when needed to save memory
        self._nli_model = None
        self._cross_encoder = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cuda_failed = False
        
        self._cf_api_token = settings.CF_API_TOKEN
        self._cf_rerank_url = self._build_cf_rerank_url()
        self._use_cloudflare_reranker = bool(self._cf_api_token and self._cf_rerank_url)

    def _build_cf_rerank_url(self) -> Optional[str]:
        """Build Cloudflare reranker HTTP route from env/config (cf_breakdown style)."""
        explicit_url = os.getenv("CF_RERANK_URL")
        if explicit_url:
            return explicit_url.rstrip("/")

        account_id = settings.R2_ACCOUNT_ID
        if not account_id:
            return None

        model_id = os.getenv("CF_RERANK_MODEL_ID", "@cf/baai/bge-reranker-base")
        return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_id}"

    def _get_cross_encoder(self):
        """Lazily load the cross-encoder model on first use with CUDA error handling."""
        if self._cross_encoder is None:
            device = "cpu" if self._cuda_failed else self._device
            try:
                logger.debug(f"Loading CrossEncoder model on {device}")
                self._cross_encoder = CrossEncoder(
                    "BAAI/bge-reranker-base", device=device
                )

                # Test the model with a small batch
                if device == "cuda":
                    try:
                        test_scores = self._cross_encoder.predict([["test", "test"]])
                        logger.debug("CUDA test successful")
                    except RuntimeError as e:
                        if "CUDA" in str(e) or "CUBLAS" in str(e):
                            logger.warning(
                                f"CUDA test failed: {e}. Falling back to CPU"
                            )
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

    def _get_nli_model(self):
        """Lazily load the NLI model for label classification."""
        if not hasattr(self, "_nli_model"):
            try:
                logger.debug("Loading NLI model for label classification")
                self._nli_model = CrossEncoder(
                    "cross-encoder/nli-deberta-v3-xsmall", device=self._device
                )
            except Exception as e:
                logger.error(f"Failed to load NLI model: {e}")
                self._nli_model = None
        return self._nli_model

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

        if self._use_cloudflare_reranker:
            try:
                return self._rerank_chunks_cloudflare(query=query, chunks=chunks)
            except Exception as e:
                logger.error(f"Cloudflare reranker failed: {e}")
                # Fall back to local reranking if Cloudflare fails

        pairs = [[query, chunk.text] for chunk in chunks]

        try:
            scores = self._predict_with_batching(pairs, batch_size)
        except RuntimeError as e:
            if (
                "CUDA" in str(e)
                or "CUBLAS" in str(e)
                or "out of memory" in str(e).lower()
            ):
                logger.warning(
                    f"CUDA error during prediction: {e}. Retrying with smaller batch or CPU"
                )
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
                    return chunks
            else:
                raise

        for chunk, score in zip(chunks, scores):
            chunk.relevance_score = float(score)
        reranked_chunks = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)

        return reranked_chunks

    def _predict_with_batching(
        self, pairs: List[List[str]], batch_size: int
    ) -> List[float]:
        """Process pairs in batches to avoid memory issues."""
        if len(pairs) <= batch_size:
            return self.cross_encoder.predict(pairs)  # type: ignore

        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            batch_scores = self.cross_encoder.predict(batch)
            all_scores.extend(batch_scores)

        return all_scores

    def _rerank_chunks_cloudflare(
        self, query: str, chunks: List['ChunkRetrieved'] 
    ) -> List['ChunkRetrieved']:
        """Rerank chunks using Cloudflare HTTP route."""
        if not chunks:
            return []

        documents = [chunk.text for chunk in chunks]
        scores = self._predict_with_batching_cloudflare(
            query=query, 
            documents=documents
        )

        if len(scores) != len(chunks):
            raise RuntimeError(
                f"Cloudflare score count mismatch: got {len(scores)} valid scores for {len(chunks)} chunks"
            )
        for chunk_idx, score in enumerate(scores):
            chunks[chunk_idx].relevance_score = score
            
        return sorted(chunks, key=lambda c: c.relevance_score, reverse=True)

    def _predict_with_batching_cloudflare(
        self, query: str, documents: List[str], batch_size: int = 50
    ) -> List[float]:
        """Process rerank requests in batches and reconstruct the exact order using 'id'."""
        all_scores: List[float] = [0.0] * len(documents)
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            contexts = [{"text": doc} for doc in batch_docs]
            batch_results = self._cloudflare_rerank_batch(
                query=query, 
                documents=contexts
            )
            for item in batch_results:
                batch_id = item.get("id")
                score = item.get("score", 0.0)
                if batch_id is not None and 0 <= batch_id < len(batch_docs):
                    global_idx = i + batch_id
                    all_scores[global_idx] = float(score)

        return all_scores

    def _cloudflare_rerank_batch(
        self, query: str, documents: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Call Cloudflare /ai/run route and parse rerank scores."""
        if not getattr(self, "_cf_rerank_url", None) or not getattr(self, "_cf_api_token", None):
            raise RuntimeError("Cloudflare reranker is not configured properly.")

        response = requests.post(
            self._cf_rerank_url, # type: ignore
            headers={
                "Authorization": f"Bearer {self._cf_api_token}",
                "Content-Type": "application/json",
            },
            json={"query": query, "contexts": documents},
            timeout=30,
        )
        response.raise_for_status()

        payload = response.json()
        result = payload.get("result", {})
        
        if isinstance(result, dict) and "response" in result:
            items = result["response"]
        elif isinstance(result, list): 
            items = result
        else:
            raise RuntimeError(f"Invalid Cloudflare rerank response format: {payload}")

        if isinstance(items, list):
            return items

        raise RuntimeError(f"Unexpected response items type: {type(items)}")

    def rank_papers(
        self,
        query: str,
        papers: List[DBPaper],
        chunks: List[ChunkRetrieved],
        weights: Dict[str, float],
    ) -> List[RankedPaper]:
        """
        Rank papers based on comprehensive multi-factor scoring.

        Returns:
            List of Paper objects ranked by comprehensive score (all input papers)
        """
        if not papers:
            return []

        paper_text_relevance = defaultdict(float)
        for chunk in chunks:
            paper_text_relevance[str(chunk.paper_id)] = max(
                paper_text_relevance[str(chunk.paper_id)], chunk.relevance_score or 0.0
            )

        ranked_results = []
        for paper in papers:
            chunk_relevance = paper_text_relevance.get(str(paper.paper_id), 0) * 100

            citation_score = min(70, math.log10((paper.citation_count or 0) + 1) * 20)
            venue_bonus = self.QUARTILE_BONUS.get(paper.journal.sjr_best_quartile, 0) if paper.journal else 0
            authority = citation_score + venue_bonus

            adjusted_authority = authority
            if chunk_relevance < 30:
                adjusted_authority = authority * (chunk_relevance / 30)

            final_score = (
                chunk_relevance * weights["relevance"]
                + adjusted_authority * weights["authority"]
            )

            ranked_paper = RankedPaper(
                id=paper.id,
                paper_id=paper.paper_id,
                paper=paper,
                relevance_score=final_score,
                ranking_scores={
                    "chunk_relevance": chunk_relevance,
                    "authority": authority,
                    "adjusted_authority": adjusted_authority,
                    "final_score": final_score,
                },
            )
            ranked_results.append(ranked_paper)

        sorted_results = sorted(
            ranked_results, key=lambda r: r.relevance_score, reverse=True
        )
        return sorted_results

    def rank_papers_v2(
        self,
        papers: List[DBPaper],
        chunks: List[ChunkRetrieved],
    ) -> List[RankedPaper]:
        """
        Rank papers using the v2 mathematical formula:
        S_final = S_rerank * min(1.3, 1 + 0.1 * log10(1 + C)) * exp(-0.15 * dt) * V
        """
        import math
        from datetime import datetime
        
        if not papers:
            return []
            
        current_year = datetime.now().year
        
        paper_chunks = defaultdict(list)
        for chunk in chunks:
            paper_chunks[str(chunk.paper_id)].append(chunk)
            
        ranked_results = []
        for paper in papers:
            p_chunks = paper_chunks.get(str(paper.paper_id), [])
            # Sort chunks descending by relevance score
            p_chunks.sort(key=lambda c: getattr(c, "relevance_score", 0.0) or 0.0, reverse=True)
            
            s_rerank_raw = sum(
                (getattr(c, "relevance_score", 0.0) or 0.0) / (i + 1)
                for i, c in enumerate(p_chunks)
            )
            s_rerank = max(0.3, min(1.0, s_rerank_raw)) if s_rerank_raw > 0 else 0.0
            
            citations = getattr(paper, "citation_count", 0) or 0
            citation_boost = min(1.3, 1 + 0.1 * math.log10(1 + citations))
            
            pub_year = getattr(paper, "year", None) or current_year
            delta_t = max(0, current_year - pub_year)
            time_decay = math.exp(-0.15 * delta_t)
            
            v_multiplier = 0.9
            if getattr(paper, "journal", None) and getattr(paper.journal, "sjr_best_quartile", None):
                quartile = str(paper.journal.sjr_best_quartile).upper()
                if quartile == "Q1":
                    v_multiplier = 1.15
                elif quartile == "Q2":
                    v_multiplier = 1.0
            
            final_score = s_rerank * citation_boost * time_decay * v_multiplier
            
            ranked_paper = RankedPaper(
                id=paper.id,
                paper_id=paper.paper_id,
                paper=paper,
                relevance_score=final_score,
                ranking_scores={
                    "s_rerank": s_rerank,
                    "citation_boost": citation_boost,
                    "time_decay": time_decay,
                    "v_multiplier": v_multiplier,
                    "final_score": final_score
                }
            )
            ranked_results.append(ranked_paper)
            
        return sorted(ranked_results, key=lambda r: r.relevance_score, reverse=True)
