"""
Centralized dependency injection container.
Manages service and repository lifecycle with proper scoping.
"""
from functools import cached_property
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.papers.repository import PaperRepository
from app.authors.repository import AuthorRepository
from app.chunks.repository import ChunkRepository
from app.institutions.repository import InstitutionRepository
from app.conversations.repository import ConversationRepository

from app.papers.service import PaperService
from app.authors.service import AuthorService
from app.chunks.service import ChunkService
from app.institutions.service import InstitutionService
from app.conversations.service import ConversationService

# Lazy imports to avoid circular dependencies
# from app.retriever.service import RetrievalService
# from app.processor.paper_processor import PaperProcessor
# from app.rag_pipeline.pipeline import Pipeline
# from app.chat.services import ChatService

from app.core.singletons import (
    get_transformer_service,
    get_ranking_service,
    get_extractor_service,
    get_chunker_service,
    get_summarizer_service,
    get_embedding_service,
)
from app.llm import get_llm_service
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class ServiceContainer:
    """
    Centralized dependency injection container for all services and repositories.
    
    Scoping:
    - Request-scoped: Repositories, domain services (created per request)
    - Application-scoped: Stateless services (singletons, shared)
    
    Usage:
        container = ServiceContainer(db_session)
        paper = await container.paper_service.get_paper(paper_id)
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        logger.debug("ServiceContainer initialized")
    
    # ==================== REPOSITORIES (Request-scoped) ====================
    
    @cached_property
    def paper_repository(self) -> PaperRepository:
        """Paper repository for database operations"""
        return PaperRepository(self.db_session)
    
    @cached_property
    def author_repository(self) -> AuthorRepository:
        """Author repository for database operations"""
        return AuthorRepository(self.db_session)
    
    @cached_property
    def chunk_repository(self) -> ChunkRepository:
        """Chunk repository for database operations"""
        return ChunkRepository(self.db_session)
    
    @cached_property
    def institution_repository(self) -> InstitutionRepository:
        """Institution repository for database operations"""
        return InstitutionRepository(self.db_session)
    
    @cached_property
    def conversation_repository(self) -> ConversationRepository:
        """Conversation repository for database operations"""
        return ConversationRepository(self.db_session)
    
    # ==================== DOMAIN SERVICES (Request-scoped) ====================
    
    @cached_property
    def paper_service(self) -> PaperService:
        """Paper service for business logic"""
        return PaperService(repository=self.paper_repository)
    
    @cached_property
    def author_service(self) -> AuthorService:
        """Author service for business logic"""
        # AuthorService now uses transformer singleton
        service = AuthorService(db=self.db_session)
        service.transformer = self.transformer_service
        return service
    
    @cached_property
    def chunk_service(self) -> ChunkService:
        """Chunk service for business logic"""
        return ChunkService(repository=self.chunk_repository)
    
    @cached_property
    def institution_service(self) -> InstitutionService:
        """Institution service for business logic"""
        return InstitutionService(db=self.db_session)
    
    @cached_property
    def conversation_service(self) -> ConversationService:
        """Conversation service for business logic"""
        return ConversationService(db=self.db_session)
    
    # ==================== STATELESS SERVICES (Application-scoped singletons) ====================
    
    @property
    def transformer_service(self):
        """Singleton transformer service for DTO conversions"""
        return get_transformer_service()
    
    @property
    def ranking_service(self):
        """Singleton ranking service for paper/chunk ranking"""
        return get_ranking_service()
    
    @property
    def extractor_service(self):
        """Singleton extractor service for PDF/XML processing"""
        return get_extractor_service()
    
    @property
    def chunker_service(self):
        """Singleton chunker service for text chunking"""
        return get_chunker_service()
    
    @property
    def summarizer_service(self):
        """Singleton summarizer service for text summarization"""
        return get_summarizer_service()
    
    @property
    def embedding_service(self):
        """Singleton embedding service for vector embeddings"""
        return get_embedding_service()
    
    @property
    def llm_service(self):
        """Singleton LLM service for language model interactions"""
        return get_llm_service()
    
    # ==================== COMPLEX SERVICES (Request-scoped with dependencies) ====================
    
    @cached_property
    def retrieval_service(self):
        """Retrieval service for paper search and retrieval"""
        from app.retriever.service import RetrievalService
        
        return RetrievalService(
            db=self.db_session,
            paper_service=self.paper_service,
            chunk_service=self.chunk_service,
            embedding_service=self.embedding_service,
        )
    
    @cached_property
    def paper_processor(self):
        """Paper processor for PDF processing and chunking"""
        from app.processor.paper_processor import PaperProcessor
        
        return PaperProcessor(
            repository=self.paper_repository,
            chunk_repository=self.chunk_repository,
            retrieval_service=self.retrieval_service,
            extractor_service=self.extractor_service,
            chunker_service=self.chunker_service,
            embedding_service=self.embedding_service,
            summarizer_service=self.summarizer_service,
        )
    
    # ==================== ORCHESTRATORS (Request-scoped workflows) ====================
    
    @cached_property
    def pipeline(self):
        """Pipeline for multi-step orchestration workflows"""
        from app.rag_pipeline.pipeline import Pipeline
        
        return Pipeline(
            db_session=self.db_session,
            repository=self.paper_repository,
            retriever=self.retrieval_service,
            processor=self.paper_processor,
            llm_service=self.llm_service,
            ranking_service=self.ranking_service,
        )
    
    @cached_property
    def chat_service(self):
        """Chat service for conversational interactions"""
        from app.chat.services import ChatService
        
        return ChatService(
            db_session=self.db_session,
            rag_pipeline=self.pipeline,
            llm_service=self.llm_service,
        )
