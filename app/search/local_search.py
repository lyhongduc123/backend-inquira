"""Local database search services."""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from app.domain.chunks.types import Chunk, ChunkRetrieved
from app.extensions.logger import create_logger
from app.models.papers import DBPaper, DBPaperChunk
from app.processor.services.embeddings import EmbeddingService, get_embedding_service
from app.search.fusion import weighted_hybrid_fusion, weighted_rrf_fusion

if TYPE_CHECKING:
    from app.domain.chunks.repository import ChunkRepository
    from app.domain.papers.repository import PaperRepository
    from app.search.filter_options import SearchFilterOptions

logger = create_logger(__name__)


def _normalize_weights(primary: float, secondary: float) -> tuple[float, float]:
    total = primary + secondary
    if total <= 0:
        logger.warning("Invalid search weights provided. Falling back to defaults.")
        return 0.1, 0.9
    return primary / total, secondary / total


class PaperSearchService:
    """Local paper search over database BM25/vector primitives."""

    def __init__(
        self,
        repository: "PaperRepository",
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.repository = repository
        self.embedding_service = embedding_service or get_embedding_service()

    async def bm25_search(
        self,
        query: str,
        limit: int = 100,
        filter_options: Optional["SearchFilterOptions"] = None,
    ) -> List[tuple[DBPaper, float]]:
        papers_with_scores = await self.repository.bm25_search(
            query=query,
            limit=limit,
            filter_options=filter_options,
        )
        logger.info("BM25 paper search returned %s papers", len(papers_with_scores))
        return papers_with_scores

    async def semantic_search(
        self,
        query: str,
        limit: int = 100,
        filter_options: Optional["SearchFilterOptions"] = None,
    ) -> List[tuple[DBPaper, float]]:
        query_embedding = await self.embedding_service.create_embedding(
            query,
            task="search_query",
        )
        if not query_embedding:
            logger.error("Failed to generate query embedding for paper semantic search")
            return []

        papers_with_scores = await self.repository.semantic_search(
            query_embedding=query_embedding,
            limit=limit,
            filter_options=filter_options,
        )
        logger.info(
            "Semantic paper search returned %s papers",
            len(papers_with_scores),
        )
        return papers_with_scores

    async def hybrid_search(
        self,
        query: str,
        limit: int = 100,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        rrf_only: bool = False,
        filter_options: Optional["SearchFilterOptions"] = None,
    ) -> List[tuple[DBPaper, float]]:
        normalized_bm25, normalized_semantic = _normalize_weights(
            bm25_weight,
            semantic_weight,
        )
        candidate_limit = max(limit * 3, 200)

        query_embedding = await self.embedding_service.create_embedding(
            query,
            task="search_query",
        )
        if not query_embedding:
            logger.error("Failed to generate query embedding for paper hybrid search")
            return []

        bm25_candidates = await self.repository.bm25_search(
            query=query,
            limit=candidate_limit,
            filter_options=filter_options,
        )
        semantic_candidates = await self.repository.semantic_search(
            query_embedding=query_embedding,
            limit=candidate_limit,
            filter_options=filter_options,
        )

        results = weighted_hybrid_fusion(
            bm25_candidates,
            semantic_candidates,
            key=lambda paper: paper.paper_id,
            bm25_weight=normalized_bm25,
            semantic_weight=normalized_semantic,
            rrf_only=rrf_only,
            limit=limit,
        )
        logger.info("Hybrid paper search returned %s papers", len(results))
        return results


class ChunkSearchService:
    """Local chunk search over database BM25/vector primitives."""

    def __init__(
        self,
        repository: "ChunkRepository",
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.repository = repository
        self.embedding_service = embedding_service or get_embedding_service()

    def _to_retrieved_chunks(
        self,
        chunks_with_scores: List[tuple[DBPaperChunk, float]],
    ) -> List[ChunkRetrieved]:
        results: List[ChunkRetrieved] = []
        for chunk, score in chunks_with_scores:
            chunk_dict = Chunk.model_validate(chunk, from_attributes=True).model_dump()
            chunk_dict["relevance_score"] = score
            chunk_dict["embedding"] = None
            results.append(ChunkRetrieved.model_validate(chunk_dict))
        return results

    async def semantic_search(
        self,
        query: str,
        limit: int = 40,
        paper_ids: Optional[List[str]] = None,
    ) -> List[ChunkRetrieved]:
        query_embedding = await self.embedding_service.create_embedding(
            query,
            task="search_query",
        )
        if not query_embedding:
            logger.error("Failed to generate query embedding for chunk semantic search")
            return []

        chunks_with_scores = await self.repository.search_similar_chunks(
            query_embedding=query_embedding,
            limit=limit,
            paper_ids=paper_ids,
        )
        return self._to_retrieved_chunks(chunks_with_scores)

    async def hybrid_search(
        self,
        query: str,
        limit: int = 40,
        paper_ids: Optional[List[str]] = None,
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> List[ChunkRetrieved]:
        normalized_bm25, normalized_semantic = _normalize_weights(
            bm25_weight,
            semantic_weight,
        )
        candidate_limit = max(limit * 3, 200)

        query_embedding = await self.embedding_service.create_embedding(
            query,
            task="search_query",
        )
        if not query_embedding:
            logger.error("Failed to generate query embedding for chunk hybrid search")
            return []

        bm25_candidates = await self.repository.bm25_search(
            query=query,
            limit=candidate_limit,
            paper_ids=paper_ids,
        )
        semantic_candidates = await self.repository.search_similar_chunks(
            query_embedding=query_embedding,
            limit=candidate_limit,
            paper_ids=paper_ids,
        )

        chunks_with_scores = weighted_rrf_fusion(
            bm25_candidates,
            semantic_candidates,
            key=lambda chunk: chunk.chunk_id,
            bm25_weight=normalized_bm25,
            semantic_weight=normalized_semantic,
            limit=limit,
        )
        results = self._to_retrieved_chunks(chunks_with_scores)
        logger.info("Hybrid chunk search returned %s chunks", len(results))
        return results


class LocalSearchService:
    """Facade for local paper and chunk search services."""

    def __init__(
        self,
        paper_search: PaperSearchService,
        chunk_search: ChunkSearchService,
    ):
        self.papers = paper_search
        self.chunks = chunk_search
