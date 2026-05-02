"""
Conversation context manager for handling conversation history and memory.

Manages conversation context, token counting, and smart truncation strategies.
"""

from typing import List, Dict, Any, Optional, Tuple, Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.messages import DBMessage
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class ConversationContextManager:
    """
    Manages conversation context for LLM interactions.
    
    Features:
    - Builds conversation history from database messages
    - Token-aware context window management
    - Smart truncation strategies (sliding window, summarization)
    - Configurable context limits
    """
    DEFAULT_MAX_CONTEXT_TOKENS = 8000 
    TOKENS_PER_MESSAGE_OVERHEAD = 4  
    CHARS_PER_TOKEN = 4  
    
    def __init__(
        self,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_messages: Optional[int] = None,
    ):
        """
        Initialize context manager.
        
        Args:
            max_context_tokens: Maximum tokens to include in context
            max_messages: Maximum number of messages (None = unlimited)
        """
        self.max_context_tokens = max_context_tokens
        self.max_messages = max_messages
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses character-based heuristic (4 chars ≈ 1 token).
        For production, consider using tiktoken library.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        return len(text) // self.CHARS_PER_TOKEN
    
    def format_message_for_llm(self, message: DBMessage) -> Dict[str, str]:
        """
        Format a database message for LLM API.
        
        Args:
            message: Database message
            
        Returns:
            Dict with 'role' and 'content' keys
        """
        return {
            "role": message.role,
            "content": message.content
        }
    
    def build_context_from_messages(
        self,
        messages: Sequence[DBMessage],
        include_summary: Optional[str] = None
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Build LLM-compatible context from messages with token management.
        
        Strategy:
        1. If conversation summary exists, include it first
        2. Include most recent messages that fit in token budget
        3. Always keep the last user message (current query)
        
        Args:
            messages: List of conversation messages (oldest to newest)
            include_summary: Optional conversation summary to prepend
            
        Returns:
            Tuple of (formatted_messages, total_tokens)
        """
        if not messages:
            return [], 0
        
        formatted_messages = []
        total_tokens = 0
        
        if include_summary:
            summary_msg = {
                "role": "system",
                "content": f"Previous conversation summary:\n{include_summary}"
            }
            summary_tokens = self.estimate_tokens(include_summary) + self.TOKENS_PER_MESSAGE_OVERHEAD
            formatted_messages.append(summary_msg)
            total_tokens += summary_tokens
            logger.info(f"Added conversation summary ({summary_tokens} tokens)")
        
        messages_to_include = []
        remaining_budget = self.max_context_tokens - total_tokens
        
        for message in reversed(messages):
            msg_formatted = self.format_message_for_llm(message)
            msg_tokens = self.estimate_tokens(message.content) + self.TOKENS_PER_MESSAGE_OVERHEAD
            
            # Always include the last message (most recent)
            if not messages_to_include:
                messages_to_include.append((msg_formatted, msg_tokens))
                remaining_budget -= msg_tokens
                continue
            
            # Check if we have budget for this message
            if remaining_budget >= msg_tokens:
                messages_to_include.append((msg_formatted, msg_tokens))
                remaining_budget -= msg_tokens
            else:
                logger.info(f"Context budget exceeded. Including {len(messages_to_include)} most recent messages.")
                break
            
            if self.max_messages and len(messages_to_include) >= self.max_messages:
                logger.info(f"Reached max message limit ({self.max_messages})")
                break
        
        messages_to_include.reverse()
        for msg, tokens in messages_to_include:
            formatted_messages.append(msg)
            total_tokens += tokens
        
        logger.info(
            f"Built context: {len(formatted_messages)} messages "
            f"({total_tokens}/{self.max_context_tokens} tokens)"
        )
        
        return formatted_messages, total_tokens
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        db_session: AsyncSession,
        include_current_query: bool = True,
        exclude_message_id: Optional[int] = None,
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Get conversation context from database.
        
        Args:
            conversation_id: Conversation ID
            db_session: Database session
            include_current_query: Whether to include the latest message
            exclude_message_id: Explicit message ID to exclude from history
            
        Returns:
            Tuple of (formatted_messages, total_tokens)
        """
        from app.domain.messages.repository import MessageRepository
        
        repo = MessageRepository(db_session)
        messages = await repo.list_by_conversation(
            conversation_id=conversation_id,
            include_inactive=False,
            limit=10
        )
        
        if not messages:
            return [], 0

        if exclude_message_id is not None:
            messages = [msg for msg in messages if msg.id != exclude_message_id]
        
        if not include_current_query and len(messages) > 0:
            messages = messages[:-1]
        
        conversation_summary = None
        
        return self.build_context_from_messages(messages, conversation_summary)
    
    async def get_conversation_top_history(
        self,
        conversation_id: str,
        db_session: AsyncSession,
        top_n: int = 5
    ) -> List[Dict[str, str]]:
        """
        Get top N most recent messages from conversation.
        
        Args:
            conversation_id: Conversation ID
            db_session: Database session
            top_n: Number of recent messages to retrieve
        """
        from app.domain.messages.repository import MessageRepository
        
        repo = MessageRepository(db_session)
        messages = await repo.list_by_conversation(
            conversation_id=conversation_id,
            include_inactive=False,
            limit=top_n
        )
        
        formatted_messages = [self.format_message_for_llm(msg) for msg in messages]
        
        logger.info(f"Retrieved top {len(formatted_messages)} messages for conversation {conversation_id}")
        
        return formatted_messages
    
    def should_summarize_conversation(
        self,
        messages: Sequence[DBMessage],
        current_summary: Optional[str] = None
    ) -> bool:
        """
        Determine if conversation should be summarized.
        
        Criteria:
        - More than 20 messages without summary
        - OR more than 10 messages since last summary
        - OR total tokens exceed 2x context window
        
        Args:
            messages: List of conversation messages
            current_summary: Existing summary (if any)
            
        Returns:
            True if summarization is recommended
        """
        if not messages:
            return False
        
        total_messages = len(messages)
        
        if not current_summary and total_messages > 20:
            return True
        
        if current_summary and total_messages > 10:
            return True

        total_tokens = sum(self.estimate_tokens(msg.content) for msg in messages)
        if total_tokens > (self.max_context_tokens * 2):
            return True
        
        return False
    
    def truncate_to_sliding_window(
        self,
        messages: Sequence[DBMessage],
        window_size: int = 10
    ) -> List[DBMessage]:
        """
        Keep only the most recent N messages (sliding window).
        
        Args:
            messages: List of messages
            window_size: Number of recent messages to keep
            
        Returns:
            Truncated list of messages
        """
        if len(messages) <= window_size:
            return list(messages)
        
        return list(messages[-window_size:])
