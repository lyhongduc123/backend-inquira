"""
Service layer for conversation management
"""

from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from app.conversations.repository import ConversationRepository
from app.conversations.schemas import (
    ConversationCreate,
    ConversationUpdate,
    ConversationDetail,
    ConversationSummary,
    Message,
)
from app.models.conversations import DBConversation
from app.models.messages import DBMessage
from app.retriever.utils import batch_dbpaper_to_papers, batch_paper_to_dicts

from app.extensions.logger import create_logger

logger = create_logger(__name__)


class ConversationService:
    def __init__(self, db: AsyncSession):
        self.repo = ConversationRepository(db)

    async def create_conversation(
        self, user_id: int, title: Optional[str] = None
    ) -> ConversationDetail:
        """Create a new conversation"""
        if not title:
            title = "New Conversation"

        db_conversation = await self.repo.create(user_id=user_id, title=title)

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

    async def get_conversation(
        self, conversation_id: str, user_id: int
    ) -> Optional[ConversationDetail]:
        """Get conversation by ID with all messages"""
        db_conversation = await self.repo.get_by_id(conversation_id, user_id)
        if not db_conversation:
            return None

        # Load messages for this conversation
        messages = await self.repo.get_messages_by_conversation(conversation_id)

        return self._to_detail(db_conversation, messages)

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
    ) -> None:
        """Save message to conversation and update metadata, optionally linking papers used"""
        message = await self.repo.create_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role=role,
            content=message_text,
            status="sent",
        )

        if paper_ids:
            logger.debug(f"Linking {len(paper_ids)} papers to message {message.id}")
            if message.role != "assistant":
                logger.warning("Linking papers to a non-assistant message, this may be unintended - Skipping linking.")
            else:
                await self.repo.link_papers_to_message(message.id, paper_ids)

        await self.repo.increment_message_count(conversation_id)

        if auto_title:
            await self.repo.update_title_from_first_message(
                conversation_id=conversation_id, message_preview=message_text
            )

    def _to_detail(
        self, db_conversation: DBConversation, messages: Optional[List[DBMessage]] = None
    ) -> ConversationDetail:
        """Convert DB model to detail schema"""
        logger.debug(
            f"Converting conversation to detail",
            extra={"conversation_id": db_conversation.conversation_id, "has_messages": messages is not None}
        )
        message_list = []
        if messages:
            message_list = []
            for msg in messages:
                sources = batch_paper_to_dicts(batch_dbpaper_to_papers(msg.papers))
                msg_dict = {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "sources": sources,
                    "created_at": msg.created_at,
                }
                message_list.append(msg_dict)
                

        return ConversationDetail(
            id=db_conversation.conversation_id,
            title=db_conversation.title,
            message_count=db_conversation.message_count,
            is_archived=db_conversation.is_archived,
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
            last_updated=db_conversation.updated_at,
        )
