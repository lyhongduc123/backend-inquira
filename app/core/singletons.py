"""
Singleton instances for stateless services.
These services have no state and can be safely shared across requests.
"""
from functools import lru_cache
from app.processor.services.transformer import TransformerService
from app.processor.services.ranking import RankingService
from app.processor.services.extractor import ExtractorService
from app.processor.services.chunker import ChunkingService
from app.processor.services.summarizer import SummarizerService
from app.processor.services.embeddings import EmbeddingService, get_embedding_service
from app.extensions.logger import create_logger

logger = create_logger(__name__)


@lru_cache(maxsize=1)
def get_transformer_service() -> TransformerService:
    """
    Get singleton TransformerService instance.
    Transforms API responses to DTOs - pure data transformation, no state.
    """
    logger.debug("Initializing singleton TransformerService")
    return TransformerService()


@lru_cache(maxsize=1)
def get_ranking_service() -> RankingService:
    """
    Get singleton RankingService instance.
    Ranks and reranks papers/chunks - stateless computation.
    """
    logger.debug("Initializing singleton RankingService")
    return RankingService()


@lru_cache(maxsize=1)
def get_extractor_service() -> ExtractorService:
    """
    Get singleton ExtractorService instance.
    Extracts text/structure from PDFs and XML - stateless processing.
    """
    from app.core.config import settings
    logger.debug(f"Initializing singleton ExtractorService (USE_CUDA={settings.USE_CUDA})")
    return ExtractorService(use_cuda=settings.USE_CUDA)


@lru_cache(maxsize=1)
def get_chunker_service() -> ChunkingService:
    """
    Get singleton ChunkingService instance.
    Chunks documents into semantic units - stateless processing.
    """
    logger.debug("Initializing singleton ChunkingService")
    return ChunkingService()


@lru_cache(maxsize=1)
def get_summarizer_service() -> SummarizerService:
    """
    Get singleton SummarizerService instance.
    Note: Uses chunker internally, which is also a singleton.
    """
    logger.debug("Initializing singleton SummarizerService")
    return SummarizerService(chunker=get_chunker_service())


# Note: EmbeddingService and LLMService already have their own singleton getters:
# - get_embedding_service() in app.processor.services.embeddings
# - get_llm_service() in app.llm
# We re-export them here for consistency

__all__ = [
    'get_transformer_service',
    'get_ranking_service',
    'get_extractor_service',
    'get_chunker_service',
    'get_summarizer_service',
    'get_embedding_service',  # Re-exported
]
