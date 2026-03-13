"""
Conversation summarization service.

Handles automatic summarization of long conversations to maintain context
while staying within token limits.
"""

from typing import Any, List, Optional, Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.messages import DBMessage
from app.models.conversations import DBConversation
from app.extensions.logger import create_logger
from app.llm import get_llm_service

logger = create_logger(__name__)


class ConversationSummarizationService:
    """
    Service for summarizing conversations to maintain manageable context.
    
    Features:
    - Progressive summarization (chunk-based)
    - Preserves key information (decisions, facts, questions)
    - Updates conversation summary field
    """
    
    def __init__(self, llm_service=None):
        """
        Initialize summarization service.
        
        Args:
            llm_service: LLM service instance (uses singleton if not provided)
        """
        self.llm_service = llm_service or get_llm_service()
    
    def format_messages_for_summarization(
        self,
        messages: Sequence[DBMessage]
    ) -> str:
        """
        Format messages into text for summarization.
        
        Args:
            messages: List of messages to format
            
        Returns:
            Formatted conversation text
        """
        formatted_lines = []
        
        for msg in messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            formatted_lines.append(f"{role_label}: {msg.content}")
        
        return "\n\n".join(formatted_lines)
    
    async def summarize_conversation(
        self,
        messages: Sequence[DBMessage],
        existing_summary: Optional[str] = None
    ) -> str:
        """
        Generate a summary of the conversation.
        
        If existing_summary is provided, creates an incremental summary
        that builds on the previous one.
        
        This service focuses on orchestration: formatting messages and delegating
        the actual LLM interaction to the LLM service.
        
        Args:
            messages: Messages to summarize
            existing_summary: Previous summary (if any)
            
        Returns:
            Generated summary
        """
        if not messages:
            return ""
        
        # Format messages for summarization
        conversation_text = self.format_messages_for_summarization(messages)
        
        try:
            # Delegate to LLM service for actual summarization
            summary = await self.llm_service.summarize_conversation_context(
                conversation_text=conversation_text,
                existing_summary=existing_summary,
                temperature=0.3,  # Lower temperature for consistent summaries
                max_tokens=800,   # Roughly 500 words
            )
            
            logger.info(
                f"Generated {'incremental' if existing_summary else 'fresh'} "
                f"summary: {len(summary)} characters"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            # Fallback: return truncated conversation
            return self._create_fallback_summary(messages)
    
    def _create_fallback_summary(self, messages: Sequence[DBMessage]) -> str:
        """
        Create a basic summary when LLM summarization fails.
        
        Args:
            messages: Messages to summarize
            
        Returns:
            Basic summary
        """
        if not messages:
            return ""
        
        # Extract first and last user queries
        user_messages = [m for m in messages if m.role == "user"]
        
        if not user_messages:
            return "Conversation in progress."
        
        first_query = user_messages[0].content[:200]
        last_query = user_messages[-1].content[:200] if len(user_messages) > 1 else None
        
        summary_parts = [
            f"This conversation started with: {first_query}..."
        ]
        
        if last_query and last_query != first_query:
            summary_parts.append(f"Recent topic: {last_query}...")
        
        summary_parts.append(f"Total messages: {len(messages)}")
        
        return "\n".join(summary_parts)
    
    async def update_conversation_summary(
        self,
        conversation: DBConversation,
        messages: Sequence[DBMessage],
        db_session: AsyncSession
    ) -> str:
        """
        Update conversation summary in database.
        
        Args:
            conversation: Conversation to update
            messages: Messages to include in summary
            db_session: Database session
            
        Returns:
            Generated summary
        """
        existing_summary = None
        if hasattr(conversation, 'conversation_metadata') and conversation.conversation_metadata:
            existing_summary = conversation.conversation_metadata.get('summary')
        
        # Generate new summary
        new_summary = await self.summarize_conversation(messages, existing_summary)
        
        # Update conversation conversation_metadata
        if not hasattr(conversation, 'conversation_metadata') or conversation.conversation_metadata is None:
            conversation.conversation_metadata = {}
        
        conversation.conversation_metadata['summary'] = new_summary
        conversation.conversation_metadata['summarized_at'] = messages[-1].created_at.isoformat() if messages else None
        conversation.conversation_metadata['summarized_message_count'] = len(messages)
        
        # Save to database
        await db_session.commit()
        await db_session.refresh(conversation)
        
        logger.info(
            f"Updated summary for conversation {conversation.conversation_id}: "
            f"{len(new_summary)} characters, {len(messages)} messages"
        )
        
        return new_summary
    
    async def get_conversation_summary(
        self,
        conversation: Any
    ) -> Optional[str]:
        """
        Retrieve existing conversation summary.
        
        Args:
            conversation: Conversation to get summary for
            
        Returns:
            Summary text or None if no summary exists
        """
        if not hasattr(conversation, 'conversation_metadata') or not conversation.conversation_metadata:
            return None
        
        return conversation.conversation_metadata.get('summary')
