"""
Centralized dependency injection container.
Manages service and repository lifecycle with proper scoping.
"""

from functools import cached_property
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

# Import from domain modules
from app.domain.papers.repository import PaperRepository
from app.domain.papers.service import PaperService
from app.domain.authors.repository import AuthorRepository
from app.domain.authors.service import AuthorService
from app.domain.institutions import InstitutionRepository, InstitutionService
from app.domain.conversations import ConversationRepository, ConversationService

from app.domain.chunks.repository import ChunkRepository
from app.domain.chunks.service import ChunkService
from app.core.singletons import (
    get_ranking_service,
    get_extractor_service,
    get_chunker_service,
    get_summarizer_service,
    get_embedding_service,
    get_zeroshot_tagger_service,
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
        service = PaperService(
            repository=self.paper_repository,
            retriever_service=self.retrieval_service,
            author_service=self.author_service,
            institution_service=self.institution_service,
            search_service=self.paper_search_service,
        )
        return service

    @cached_property
    def author_service(self) -> AuthorService:
        """Author service for business logic"""
        service = AuthorService(db=self.db_session, repository=self.author_repository)
        return service

    @cached_property
    def chunk_service(self) -> ChunkService:
        """Chunk service for business logic"""
        return ChunkService(
            repository=self.chunk_repository,
            search_service=self.chunk_search_service,
        )

    @cached_property
    def institution_service(self) -> InstitutionService:
        """Institution service for business logic"""
        return InstitutionService(
            db=self.db_session,
            repository=self.institution_repository
        )

    @cached_property
    def message_service(self):
        """Message service for business logic"""
        from app.domain.messages.service import MessageService
        return MessageService(db=self.db_session)
    
    @cached_property
    def conversation_service(self) -> ConversationService:
        """Conversation service for business logic"""
        return ConversationService(
            db=self.db_session,
            message_service=self.message_service
        )
    
    @cached_property
    def bookmark_service(self):
        """Bookmark service for business logic"""
        from app.domain.bookmarks.service import BookmarkService
        return BookmarkService(db=self.db_session)
    
    @cached_property
    def user_settings_service(self):
        """User settings service for business logic"""
        from app.domain.user_settings.service import UserSettingsService
        return UserSettingsService(db=self.db_session)
    
    @cached_property
    def pipeline_task_service(self):
        """Pipeline task service for task lifecycle management"""
        from app.domain.chat.pipeline_tasks import PipelineTaskService
        return PipelineTaskService(db=self.db_session)
    
    @cached_property
    def pipeline_event_store(self):
        """Pipeline event store for event persistence"""
        from app.domain.chat.pipeline_tasks import PipelineEventStore
        return PipelineEventStore(db=self.db_session)
    
    @cached_property
    def journal_service(self):
        """Journal service for journal lookup and enrichment"""
        from app.domain.papers.journal_service import JournalService
        return JournalService(db=self.db_session)
    
    @cached_property
    def conference_service(self):
        """Conference service for conference lookup and enrichment"""
        from app.domain.papers.conference_service import ConferenceService
        return ConferenceService(db_session=self.db_session)
    
    @cached_property
    def paper_linking_service(self):
        """Get or create PaperLinkingService singleton"""
        from app.domain.papers.linking_service import PaperLinkingService
        return PaperLinkingService(
            self.db_session,
            paper_repository=self.paper_repository,
            author_service=self.author_service,
            institution_service=self.institution_service
        )
    
    # ==================== CHAT HELPERS (Request-scoped utilities) ====================
    
    @cached_property
    def conversation_context_manager(self):
        """Context manager for conversation history"""
        from app.domain.conversations.context_manager import ConversationContextManager
        return ConversationContextManager(
            max_context_tokens=8000,
            max_messages=10,
        )
    
    @cached_property
    def conversation_summarization_service(self):
        """Service for conversation summarization"""
        from app.domain.conversations.summarization_service import ConversationSummarizationService
        return ConversationSummarizationService(llm_service=self.llm_service)
    
    @cached_property
    def chat_background_tasks(self):
        """Background task service for chat"""
        from app.domain.chat.background_tasks import ChatBackgroundTaskService
        return ChatBackgroundTaskService(
            summarization_service=self.conversation_summarization_service,
            context_manager=self.conversation_context_manager,
        )
    
    # ==================== STATELESS SERVICES (Application-scoped singletons) ====================

    @property
    def zeroshot_tagger_service(self):
        """Singleton zero-shot tagger service for paper tagging"""
        return get_zeroshot_tagger_service()

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

    @cached_property
    def paper_search_service(self):
        """Local database paper search service."""
        from app.search import PaperSearchService

        return PaperSearchService(
            repository=self.paper_repository,
            embedding_service=self.embedding_service,
        )

    @cached_property
    def chunk_search_service(self):
        """Local database chunk search service."""
        from app.search import ChunkSearchService

        return ChunkSearchService(
            repository=self.chunk_repository,
            embedding_service=self.embedding_service,
        )

    @cached_property
    def local_search_service(self):
        """Facade for local database search services."""
        from app.search import LocalSearchService

        return LocalSearchService(
            paper_search=self.paper_search_service,
            chunk_search=self.chunk_search_service,
        )

    # ==================== COMPLEX SERVICES (Request-scoped with dependencies) ====================

    @cached_property
    def retrieval_service(self):
        """Retrieval service for paper search and retrieval"""
        from app.retriever.service import RetrievalService

        return RetrievalService(
            db=self.db_session,
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
    
    @cached_property
    def preprocessing_service(self):
        """Preprocessing service for bulk paper ingestion"""
        from app.processor.preprocessing_service import PreprocessingService
        from app.processor.preprocessing_repository import PreprocessingRepository
        
        return PreprocessingService(
            db_session=self.db_session,
            paper_repository=self.paper_repository,
            preprocessing_repo=PreprocessingRepository(self.db_session),
            retriever=self.retrieval_service,
            paper_service=self.paper_service,
            processor=self.paper_processor,
            journal_service=self.journal_service,
            conference_service=self.conference_service,
            linking_service=self.paper_linking_service,
        )

    @cached_property
    def preprocessing_phase_service(self):
        """Phase service for queueable preprocessing stages."""
        from app.processor.preprocessing_repository import PreprocessingRepository
        from app.processor.services.preprocessing_phase_service import (
            PreprocessingPhaseService,
        )

        preprocessing_repo = PreprocessingRepository(self.db_session)
        return PreprocessingPhaseService(
            preprocessing_service=self.preprocessing_service,
            preprocessing_repository=preprocessing_repo,
            retriever=self.retrieval_service,
            linking_service=self.paper_linking_service,
            zeroshot_tagger_service=self.zeroshot_tagger_service,
        )

    @cached_property
    def preprocessing_single_phase_service(self):
        """Single phase preprocessing service for conditional phase execution."""
        from app.processor.preprocessing_single_phase import PreprocessingSinglePhaseService

        return PreprocessingSinglePhaseService(
            db_session=self.db_session,
            preprocessing_service=self.preprocessing_service,
        )

    # ==================== ORCHESTRATORS (Request-scoped workflows) ====================

    @cached_property
    def pipeline(self):
        """Unified RAG Pipeline with database and scoped search modes"""
        from app.rag_pipeline.pipeline import Pipeline

        return Pipeline(
            db_session=self.db_session,
            
        )

    @cached_property
    def database_pipeline(self):
        """Database-only search pipeline (routes to unified Pipeline.SEARCH_DATABASE)"""
        # Return the unified pipeline instance; callers use run_database_search_workflow()
        return self.pipeline

    @cached_property
    def scoped_pipeline(self):
        """Scoped paper pipeline (routes to unified Pipeline.SEARCH_SCOPED)"""
        # Return the unified pipeline instance; callers use run_scoped_search_workflow()
        return self.pipeline
        
    @cached_property
    def doi_title_pipeline(self):
        """DOI/Title-only lookup pipeline for precision retrieval"""
        from app.rag_pipeline.doi_title_pipeline import DoiTitlePipeline

        return DoiTitlePipeline(
            db_session=self.db_session,
            container=self,
        )
        

    @cached_property
    def chat_service(self):
        """Chat service for conversational interactions"""
        from app.domain.chat.services import ChatService

        return ChatService(
            db_session=self.db_session,
            rag_pipeline=self.pipeline,
            llm_service=self.llm_service,
            message_service=self.message_service,
            context_manager=self.conversation_context_manager,
            summarization_service=self.conversation_summarization_service,
            background_tasks=self.chat_background_tasks,
        )

    @property
    def chat_agent_service(self):
        """Chat Agent service for Agent Mode"""
        from app.domain.chat.agent.agent_service import ChatAgentService

        return ChatAgentService()
