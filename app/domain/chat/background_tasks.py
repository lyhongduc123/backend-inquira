"""
Background task service for post-response operations.
Handles validation and summarization without blocking the main response.
"""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.messages import DBMessage
from app.domain.validation.schemas import ValidationRequest
from app.domain.conversations.summarization_service import ConversationSummarizationService
from app.domain.conversations.context_manager import ConversationContextManager
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class ChatBackgroundTaskService:
    """Handles async post-response tasks for chat operations"""
    
    def __init__(
        self,
        summarization_service: ConversationSummarizationService,
        context_manager: ConversationContextManager,
    ):
        """
        Initialize background task service.
        
        Args:
            summarization_service: Service for conversation summarization
            context_manager: Manager for conversation context
        """
        self.summarization_service = summarization_service
        self.context_manager = context_manager
    
    async def validate_answer(
        self,
        validation_request: ValidationRequest,
        db_session: AsyncSession,
    ) -> None:
        """
        Validate answer and save results for benchmarking.
        Runs asynchronously after response is sent.
        
        Args:
            validation_request: Validation request with query, context, and answer
            db_session: Database session
        """
        try:
            # Import here to avoid circular dependency
            from app.domain.validation.service import validate_answer
            from app.domain.validation.repository import save_validation_result
            
            # Perform validation
            validation_result = await validate_answer(validation_request)
            
            # Save to database
            await save_validation_result(
                db_session, validation_request, validation_result
            )
            
            logger.info(
                f"Validation completed for message {validation_request.message_id}: "
                f"relevance={validation_result.relevance_score:.2f}, "
                f"hallucination={validation_result.has_hallucination}"
            )
        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            await db_session.rollback()
    
    async def check_and_summarize_conversation(
        self,
        conversation_id: str,
        db_session: AsyncSession,
    ) -> None:
        """
        Check if summarization is needed and perform it.
        Runs asynchronously after response is sent.
        
        Args:
            conversation_id: Conversation ID to check
            db_session: Database session
        """
        try:
            from app.domain.conversations.service import ConversationService
            
            conversation_service = ConversationService(db_session)
            db_conversation = await conversation_service.get_a_conversation(
                conversation_id=conversation_id
            )
            
            if not db_conversation:
                logger.warning(f"Conversation {conversation_id} not found for summarization")
                return

            result = await db_session.execute(
                select(DBMessage)
                .where(DBMessage.conversation_id == conversation_id)
                .order_by(DBMessage.created_at)
            )
            messages = result.scalars().all()
            
            if not messages:
                return
            
            existing_summary = (
                await self.summarization_service.get_conversation_summary(
                    db_conversation
                )
            )
            
            if self.context_manager.should_summarize_conversation(
                messages, existing_summary
            ):
                await self.summarization_service.update_conversation_summary(
                    db_conversation, messages, db_session
                )
                logger.info(
                    f"Conversation summarization completed for {conversation_id}"
                )
        except Exception as e:
            logger.error(f"Summarization failed: {e}", exc_info=True)
            await db_session.rollback()
    
    async def run_validation_with_new_session(
        self, validation_request: ValidationRequest
    ) -> None:
        """
        Validate answer using a new database session.
        For use with background task runners (FastAPI BackgroundTasks, asyncio.create_task).
        
        Args:
            validation_request: Validation request
        """
        from app.core.db.database import get_db_session
        
        try:
            async for session in get_db_session():
                try:
                    await self.validate_answer(validation_request, session)
                    break  # Success, exit the async for loop
                except Exception as e:
                    logger.error(f"Validation with new session failed: {e}", exc_info=True)
                    raise
        except Exception as e:
            logger.error(f"Failed to create validation session: {e}", exc_info=True)
    
    async def run_summarization_with_new_session(
        self, conversation_id: str
    ) -> None:
        """
        Check and summarize conversation using a new database session.
        For use with background task runners.
        
        Args:
            conversation_id: Conversation ID
        """
        from app.core.db.database import get_db_session
        
        try:
            async for session in get_db_session():
                try:
                    await self.check_and_summarize_conversation(conversation_id, session)
                    break  # Success, exit the async for loop
                except Exception as e:
                    logger.error(f"Summarization with new session failed: {e}", exc_info=True)
                    raise
        except Exception as e:
            logger.error(f"Failed to create summarization session: {e}", exc_info=True)
