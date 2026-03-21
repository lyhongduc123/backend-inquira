"""
Service layer for conversation management
"""

from typing import Literal, Optional, List, Dict, Any
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession
from .repository import ConversationRepository
from .schemas import (
    ConversationCreate,
    ConversationUpdate,
    ConversationDetail,
    ConversationSummary,
    Message,
)
from app.models.conversations import DBConversation
from app.models.messages import DBMessage
from app.domain.messages.service import MessageService

from app.extensions.logger import create_logger

logger = create_logger(__name__)


class ConversationService:
    def __init__(self, db: AsyncSession, message_service: Optional["MessageService"] = None):
        self.repo = ConversationRepository(db)
        # Accept via DI or create as fallback for backward compatibility
        if message_service is None:
            from app.domain.messages.service import MessageService
            message_service = MessageService(db)
        self.message_service = message_service

    async def create_conversation(
        self,
        user_id: int,
        title: Optional[str] = None,
        conversation_type: str = "multi_paper_rag",
        primary_paper_id: Optional[str] = None,
    ) -> ConversationDetail:
        """Create a new conversation"""
        if not title:
            title = "New Conversation"

        db_conversation = await self.repo.create(
            user_id=user_id,
            title=title,
            conversation_type=conversation_type,
            primary_paper_id=primary_paper_id,
        )

        return self._to_detail(db_conversation)

    async def get_or_create_conversation(
        self,
        user_id: int,
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> ConversationDetail:
        """Get existing conversation or create new one"""
        if conversation_id:
            db_conversation = await self.repo.get_by_id(conversation_id, user_id)
            if db_conversation:
                return self._to_detail(db_conversation)

        # Create new conversation
        return await self.create_conversation(user_id, title)

    async def list_conversations(
        self,
        user_id: int,
        page: int = 1,
        page_size: int = 20,
        archived: Optional[bool] = None,
    ) -> tuple[List[ConversationSummary], int]:
        """List conversations for user with pagination"""
        skip = (page - 1) * page_size
        conversations, total = await self.repo.list_by_user(
            user_id=user_id, archived=archived, skip=skip, limit=page_size
        )

        summaries = [self._to_summary(conv) for conv in conversations]
        return summaries, total
    
    async def get_a_conversation(
        self, conversation_id: str
    ) -> DBConversation | None:
        """Get conversation by ID without messages (for lightweight listing)"""
        db_conversation = await self.repo.get(conversation_id)
        if not db_conversation:
            return None

        return db_conversation

    async def get_conversation(
        self, conversation_id: str, user_id: int
    ) -> Optional[ConversationDetail]:
        """Get conversation by ID with all messages"""
        db_conversation = await self.repo.get_by_id(conversation_id, user_id)
        if not db_conversation:
            return None

        # Load messages for this conversation using MessageService
        message_responses = await self.message_service.list_messages(
            conversation_id=conversation_id,
            page=1,
            page_size=1000,  # Get all messages
            include_inactive=False
        )

        return self._to_detail_with_message_responses(db_conversation, message_responses)

    async def update_conversation(
        self, conversation_id: str, user_id: int, update_data: ConversationUpdate
    ) -> Optional[ConversationDetail]:
        """Update conversation"""
        db_conversation = await self.repo.update(
            conversation_id=conversation_id,
            user_id=user_id,
            title=update_data.title,
            is_archived=update_data.is_archived,
        )

        if not db_conversation:
            return None

        return self._to_detail(db_conversation)

    async def delete_conversation(self, conversation_id: str, user_id: int) -> bool:
        """Delete conversation"""
        return await self.repo.delete(conversation_id, user_id)

    async def add_message_to_conversation(
        self,
        conversation_id: str,
        user_id: int,
        message_text: str,
        role: str = "user",
        auto_title: bool = True,
        paper_ids: Optional[List[str]] = None,
        paper_snapshots: Optional[List[Dict[str, Any]]] = None,
        progress_events: Optional[List[Dict[str, Any]]] = None,
        scoped_quote_refs: Optional[List[Dict[str, Any]]] = None,
        client_message_id: Optional[str] = None,
        pipeline_type: Optional[str] = None,
        completion_time_ms: Optional[int] = None,
    ) -> int:
        """Save message to conversation and update metadata, optionally linking papers with snapshots and progress events"""
        
        # Delegate message creation to MessageService
        message = await self.message_service.create_message(
            conversation_id=conversation_id,
            user_id=user_id,
            content=message_text,
            role=role,
            paper_ids=paper_ids,
            paper_snapshots=paper_snapshots,
            progress_events=progress_events,
            scoped_quote_refs=scoped_quote_refs,
            client_message_id=client_message_id,
            pipeline_type=pipeline_type,
            completion_time_ms=completion_time_ms,
        )

        # Update conversation metadata
        await self.repo.increment_message_count(conversation_id)

        if auto_title:
            await self.repo.update_title_from_first_message(
                conversation_id=conversation_id, message_preview=message_text
            )
        
        return message.id
            
    async def update_message_status(
        self,
        message_id: int,
        status: Literal["sent", "pending", "failed"] = "sent",
    ) -> DBMessage | None:
        """Update message status (e.g. mark as inactive)"""
        result = await self.message_service.update_message_status(
            message_id=message_id,
            status=status,
        )
        # Return raw DBMessage for backward compatibility
        if result:
            return await self.message_service.repo.get_by_id(message_id)
        return None
        

    def _to_detail_with_message_responses(
        self,
        db_conversation: DBConversation,
        message_responses: Optional[List] = None,
    ) -> ConversationDetail:
        """Convert DB model to detail schema using MessageWithPapersResponse"""
        logger.debug(
            f"Converting conversation to detail",
            extra={
                "conversation_id": db_conversation.conversation_id,
                "has_messages": message_responses is not None,
            },
        )
        
        message_list = []
        if message_responses:
            for msg_resp in message_responses:
                msg_dict = {
                    "id": msg_resp.id,
                    "role": msg_resp.role,
                    "content": msg_resp.content,
                    "sources": None,  # Deprecated
                    "paper_snapshots": msg_resp.paper_snapshots,
                    "progress_events": msg_resp.progress_events,
                    "scoped_quote_refs": msg_resp.scoped_quote_refs,
                    "created_at": msg_resp.created_at,
                }
                message_list.append(msg_dict)

        return ConversationDetail(
            conversation_id=db_conversation.conversation_id,
            title=db_conversation.title,
            message_count=db_conversation.message_count,
            is_archived=db_conversation.is_archived,
            conversation_type=db_conversation.conversation_type,
            primary_paper_id=db_conversation.primary_paper_id,
            created_at=db_conversation.created_at,
            updated_at=db_conversation.updated_at,
            messages=message_list,
        )
    
    def _to_detail(
        self,
        db_conversation: DBConversation,
        messages: Optional[List[DBMessage]] = None,
    ) -> ConversationDetail:
        """Convert DB model to detail schema"""
        logger.debug(
            f"Converting conversation to detail",
            extra={
                "conversation_id": db_conversation.conversation_id,
                "has_messages": messages is not None,
            },
        )
        message_list = []
        if messages:
            message_list = []
            for msg in messages:
                # Use paper snapshots from message_metadata if available
                paper_snapshots = None
                progress_events = None
                if msg.message_metadata:
                    if "paper_snapshots" in msg.message_metadata:
                        paper_snapshots = msg.message_metadata["paper_snapshots"]
                    if "progress_events" in msg.message_metadata:
                        progress_events = msg.message_metadata["progress_events"]
                    scoped_quote_refs = msg.message_metadata.get("scoped_quote_refs")
                else:
                    scoped_quote_refs = None

                # Fallback to old sources format for backward compatibility
                # Only access msg.papers if it's already loaded (avoid lazy loading in async context)
                sources = None
                if not paper_snapshots:
                    # Check if the 'papers' relationship is already loaded
                    insp = inspect(msg)
                    if "papers" in insp.unloaded:
                        # Papers not loaded, skip backward compatibility
                        pass
                    else:
                        # Papers already loaded, safe to access
                        if msg.papers:
                            from app.core.dtos.paper import PaperDTO
                            papers_dto = PaperDTO.batch_from_db_models(msg.papers)
                            sources = [paper.model_dump(mode='json', by_alias=True) for paper in papers_dto]

                msg_dict = {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "sources": sources,  # Deprecated, kept for backward compatibility
                    "paper_snapshots": paper_snapshots,  # New unified format
                    "progress_events": progress_events,  # RAG pipeline progress
                    "scoped_quote_refs": scoped_quote_refs,
                    "created_at": msg.created_at,
                }
                message_list.append(msg_dict)

        return ConversationDetail(
            conversation_id=db_conversation.conversation_id,
            title=db_conversation.title,
            message_count=db_conversation.message_count,
            is_archived=db_conversation.is_archived,
            conversation_type=db_conversation.conversation_type,
            primary_paper_id=db_conversation.primary_paper_id,
            created_at=db_conversation.created_at,
            updated_at=db_conversation.updated_at,
            messages=message_list,
        )

    def _to_summary(self, db_conversation: DBConversation) -> ConversationSummary:
        """Convert DB model to summary schema"""
        return ConversationSummary(
            id=db_conversation.conversation_id,
            title=db_conversation.title,
            message_count=db_conversation.message_count,
            is_archived=db_conversation.is_archived,
            conversation_type=db_conversation.conversation_type,
            primary_paper_id=db_conversation.primary_paper_id,
            last_updated=db_conversation.updated_at,
        )

    async def get_paper_conversation(
        self, user_id: int, paper_id: str
    ) -> Optional[ConversationDetail]:
        """
        Get existing conversation for a specific paper.

        Returns conversation with full message history, or None if not exists.
        """
        db_conversation = await self.repo.get_by_paper(user_id, paper_id)

        if not db_conversation:
            return None

        # Load messages for this conversation using MessageService
        message_responses = await self.message_service.list_messages(
            conversation_id=db_conversation.conversation_id,
            page=1,
            page_size=1000,
            include_inactive=False
        )

        return self._to_detail_with_message_responses(db_conversation, message_responses)

    async def get_or_create_paper_conversation(
        self, user_id: int, paper_id: str, paper_title: str
    ) -> ConversationDetail:
        """
        Get existing conversation or create new one for paper.

        Useful for chat endpoints that auto-create conversations.
        """
        # Try to find existing
        existing = await self.get_paper_conversation(user_id, paper_id)
        if existing:
            return existing

        # Create new single-paper conversation
        title = f"Deep Dive: {paper_title[:50]}"
        if len(paper_title) > 50:
            title += "..."

        db_conversation = await self.repo.create(
            user_id=user_id,
            title=title,
            conversation_type="single_paper_detail",
            primary_paper_id=paper_id,
        )

        return self._to_detail(db_conversation)

    async def list_paper_conversations(
        self, user_id: int, paper_id: str, page: int = 1, page_size: int = 10
    ) -> tuple[List[ConversationSummary], int]:
        """
        List all conversations for a specific paper.

        Useful for showing conversation history for a paper.
        """
        skip = (page - 1) * page_size
        conversations, total = await self.repo.list_paper_conversations(
            user_id=user_id, paper_id=paper_id, skip=skip, limit=page_size
        )

        summaries = [self._to_summary(conv) for conv in conversations]
        return summaries, total
