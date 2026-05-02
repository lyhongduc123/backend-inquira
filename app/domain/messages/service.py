"""
Service layer for message management
"""
from typing import Optional, List, Dict, Any, Literal
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.messages.repository import MessageRepository
from app.domain.messages.schemas import (
    MessageResponse,
    MessageWithPapersResponse,
)
from app.models.messages import DBMessage
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class MessageService:
    """Service for message operations with business logic"""
    
    def __init__(self, db: AsyncSession, repository: Optional["MessageRepository"] = None):
        if repository is None:
            repository = MessageRepository(db)
        self.repo = repository
    
    async def create_message(
        self,
        conversation_id: str,
        user_id: int,
        content: str,
        role: str = "user",
        status: str = "sent",
        paper_ids: Optional[List[str]] = None,
        paper_snapshots: Optional[List[Dict[str, Any]]] = None,
        progress_events: Optional[List[Dict[str, Any]]] = None,
        scoped_quote_refs: Optional[List[Dict[str, Any]]] = None,
        client_message_id: Optional[str] = None,
        pipeline_type: Optional[str] = None,
        completion_time_ms: Optional[int] = None,
    ) -> DBMessage:
        """
        Create a new message with optional paper links and metadata
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID
            content: Message content
            role: Message role (user or assistant)
            paper_ids: Optional list of paper IDs to link
            paper_snapshots: Optional paper metadata snapshots
            progress_events: Optional RAG pipeline progress events
            client_message_id: Optional client-side message ID for deduplication
            
        Returns:
            Created DBMessage
        """
        message_metadata = {}
        if paper_snapshots:
            message_metadata["paper_snapshots"] = paper_snapshots
        if progress_events:
            message_metadata["progress_events"] = progress_events
        if scoped_quote_refs:
            message_metadata["scoped_quote_refs"] = scoped_quote_refs
        if client_message_id:
            message_metadata["client_message_id"] = client_message_id

        message = await self.repo.create(
            conversation_id=conversation_id,
            user_id=user_id,
            content=content,
            role=role,
            message_metadata=message_metadata if message_metadata else None,
            is_active=True,
            status=status,
            pipeline_type=pipeline_type,
            completion_time_ms=completion_time_ms,
        )
        
        if paper_ids:
            logger.debug(f"Linking {len(paper_ids)} papers to message {message.id}")
            if role != "assistant":
                logger.warning(
                    "Linking papers to a non-assistant message, this may be unintended - Skipping linking."
                )
            else:
                await self.repo.link_papers_to_message(message.id, paper_ids)
        
        return message
    
    async def get_message(
        self,
        message_id: int,
        user_id: Optional[int] = None
    ) -> Optional[MessageWithPapersResponse]:
        """
        Get a single message by ID
        
        Args:
            message_id: Message ID
            user_id: Optional user ID for authorization check
            
        Returns:
            MessageWithPapersResponse if found, None otherwise
        """
        message = await self.repo.get_by_id(message_id, user_id)
        if not message:
            return None
        
        return self._to_response(message)
    
    async def check_existing_message(
        self,
        conversation_id: str,
        client_message_id: str
    ) -> Optional[DBMessage]:
        """
        Check if a message already exists by client message ID (for deduplication)
        
        Args:
            conversation_id: Conversation ID
            client_message_id: Client-side message ID
            
        Returns:
            DBMessage if found, None otherwise
        """
        return await self.repo.get_by_client_message_id(
            conversation_id, client_message_id
        )
    
    async def list_messages(
        self,
        conversation_id: str,
        page: int = 1,
        page_size: int = 100,
        include_inactive: bool = False
    ) -> List[MessageWithPapersResponse]:
        """
        List all messages for a conversation
        
        Args:
            conversation_id: Conversation ID
            page: Page number (1-indexed)
            page_size: Number of messages per page
            include_inactive: Whether to include inactive messages
            
        Returns:
            List of MessageWithPapersResponse
        """
        skip = (page - 1) * page_size
        messages = await self.repo.list_by_conversation(
            conversation_id=conversation_id,
            skip=skip,
            limit=page_size,
            include_inactive=include_inactive
        )
        
        return [self._to_response(msg) for msg in messages]
    
    async def update_message_status(
        self,
        message_id: int,
        status: Literal["sent", "pending", "failed"] = "sent",
    ) -> Optional[MessageResponse]:
        """
        Update message status
        
        Args:
            message_id: Message ID
            status: New status
            
        Returns:
            Updated MessageResponse if found, None otherwise
        """
        message = await self.repo.update_status(message_id, status)
        if not message:
            return None
        
        return MessageResponse.model_validate(message)
    
    async def add_metadata(
        self,
        message_id: int,
        metadata: Dict[str, Any]
    ) -> Optional[MessageResponse]:
        """
        Add or update message metadata
        
        Args:
            message_id: Message ID
            metadata: Metadata to add/update
            
        Returns:
            Updated MessageResponse if found, None otherwise
        """
        message = await self.repo.update_metadata(message_id, metadata)
        if not message:
            return None
        
        return MessageResponse.model_validate(message)
    
    async def delete_message(
        self,
        message_id: int,
        user_id: Optional[int] = None,
        soft_delete: bool = True,
        delete_assistant_reply_for_user: bool = True,
    ) -> bool:
        """
        Delete a message
        
        Args:
            message_id: Message ID
            user_id: Optional user ID for authorization check
            soft_delete: If True, mark as inactive; if False, delete from DB
            delete_assistant_reply_for_user: If True and target is a user message,
                also delete the first assistant message after it and before the next user message
            
        Returns:
            True if deleted, False if not found
        """
        return await self.repo.delete(
            message_id,
            user_id,
            soft_delete,
            delete_assistant_reply_for_user=delete_assistant_reply_for_user,
        )
    
    def _to_response(self, message: DBMessage) -> MessageWithPapersResponse:
        """
        Convert DBMessage to MessageWithPapersResponse
        
        Args:
            message: DBMessage to convert
            
        Returns:
            MessageWithPapersResponse
        """
        paper_snapshots = None
        progress_events = None
        scoped_quote_refs = None
        
        if message.message_metadata:
            paper_snapshots = message.message_metadata.get("paper_snapshots")
            progress_events = message.message_metadata.get("progress_events")
            scoped_quote_refs = message.message_metadata.get("scoped_quote_refs")
        
        return MessageWithPapersResponse(
            id=message.id,
            conversation_id=str(message.conversation_id),
            user_id=message.user_id,
            role=message.role,
            content=message.content,
            status=message.status,
            pipeline_type=message.pipeline_type,
            is_active=message.is_active,
            created_at=message.created_at,
            updated_at=message.updated_at,
            message_metadata=message.message_metadata,
            paper_snapshots=paper_snapshots,
            progress_events=progress_events,
            scoped_quote_refs=scoped_quote_refs,
        )
