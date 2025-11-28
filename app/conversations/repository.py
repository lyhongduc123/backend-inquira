"""
Repository for conversation database operations
"""
from typing import Optional, List
from sqlalchemy import desc, select
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.conversations import DBConversation
from app.models.messages import DBMessage
import uuid


class ConversationRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(
        self,
        user_id: int,
        title: str = "New Conversation"
    ) -> DBConversation:
        """Create a new conversation"""
        conversation_id = str(uuid.uuid4())
        
        db_conversation = DBConversation(
            conversation_id=conversation_id,
            user_id=user_id,
            title=title,
            message_count=0,
            is_archived=False
        )
        
        self.db.add(db_conversation)
        await self.db.commit()
        await self.db.refresh(db_conversation)
        
        return db_conversation
    
    async def get_by_id(
        self,
        conversation_id: str,
        user_id: int
    ) -> Optional[DBConversation]:
        """Get conversation by ID for specific user"""
        result = await self.db.execute(
            select(DBConversation).where(
                DBConversation.conversation_id == conversation_id,
                DBConversation.user_id == user_id
            )
        )
        return result.scalar_one_or_none()
    
    async def list_by_user(
        self,
        user_id: int,
        archived: Optional[bool] = None,
        skip: int = 0,
        limit: int = 20
    ) -> tuple[List[DBConversation], int]:
        """List conversations for a user with pagination"""
        query = select(DBConversation).where(
            DBConversation.user_id == user_id
        )
        
        if archived is not None:
            query = query.where(DBConversation.is_archived == archived)
        
        # Get total count
        count_query = select(DBConversation).where(
            DBConversation.user_id == user_id
        )
        if archived is not None:
            count_query = count_query.where(DBConversation.is_archived == archived)
        
        total_result = await self.db.execute(count_query)
        total = len(total_result.all())
        
        # Get paginated results
        query = query.order_by(desc(DBConversation.updated_at))
        query = query.offset(skip).limit(limit)
        
        result = await self.db.execute(query)
        conversations = result.scalars().all()
        
        return list(conversations), total
    
    async def update(
        self,
        conversation_id: str,
        user_id: int,
        title: Optional[str] = None,
        is_archived: Optional[bool] = None
    ) -> Optional[DBConversation]:
        """Update conversation details"""
        conversation = await self.get_by_id(conversation_id, user_id)
        if not conversation:
            return None
        
        if title is not None:
            conversation.title = title
        if is_archived is not None:
            conversation.is_archived = is_archived
        
        await self.db.commit()
        await self.db.refresh(conversation)
        
        return conversation
    
    async def delete(
        self,
        conversation_id: str,
        user_id: int
    ) -> bool:
        """Delete conversation and all its messages"""
        conversation = await self.get_by_id(conversation_id, user_id)
        if not conversation:
            return False
        
        # Delete all messages first
        await self.db.execute(
            DBMessage.__table__.delete().where(
                DBMessage.conversation_id == conversation_id
            )
        )
        
        # Delete conversation
        await self.db.delete(conversation)
        await self.db.commit()
        
        return True
    
    async def increment_message_count(
        self,
        conversation_id: str
    ) -> None:
        """Increment message count for a conversation"""
        result = await self.db.execute(
            select(DBConversation).where(
                DBConversation.conversation_id == conversation_id
            )
        )
        conversation = result.scalar_one_or_none()
        
        if conversation:
            conversation.message_count += 1
            await self.db.commit()
    
    async def update_title_from_first_message(
        self,
        conversation_id: str,
        message_preview: str,
        max_length: int = 50
    ) -> None:
        """Auto-generate conversation title from first message"""
        result = await self.db.execute(
            select(DBConversation).where(
                DBConversation.conversation_id == conversation_id
            )
        )
        conversation = result.scalar_one_or_none()
        
        if conversation and conversation.message_count <= 1:
            # Only update if still has default title
            if conversation.title in ["New Conversation", ""]:
                title = message_preview[:max_length]
                if len(message_preview) > max_length:
                    title += "..."
                conversation.title = title
                await self.db.commit()
    
    async def create_message(
        self,
        conversation_id: str,
        user_id: int,
        role: str,
        content: str,
        status: str = "sent"
    ) -> DBMessage:
        """Create a new message in a conversation"""
        db_message = DBMessage(
            conversation_id=conversation_id,
            user_id=user_id,
            role=role,
            content=content,
            status=status,
            is_active=True
        )
        
        self.db.add(db_message)
        await self.db.commit()
        await self.db.refresh(db_message)
        
        return db_message
    
    async def get_messages_by_conversation(
        self,
        conversation_id: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[DBMessage]:
        """Get all messages for a conversation"""
        result = await self.db.execute(
            select(DBMessage)
            .where(DBMessage.conversation_id == conversation_id)
            .where(DBMessage.is_active == True)
            .order_by(DBMessage.created_at)
            .options(joinedload(DBMessage.papers))
            .offset(skip)
            .limit(limit)
        )
        return list(result.unique().scalars().all())
    
    async def link_papers_to_message(
        self,
        message_id: int,
        paper_ids: List[str]
    ):
        """Link papers to a message using DBMessagePaper join table. Accepts paper_id (string) and finds DBPaper.id."""
        from app.models.papers import DBPaper
        from app.models.message_papers import DBMessagePaper
        for paper_id in paper_ids:
            result = await self.db.execute(
                select(DBPaper).where(DBPaper.paper_id == paper_id)
            )
            db_paper = result.scalar_one_or_none()
            if db_paper:
                link = DBMessagePaper(message_id=message_id, paper_id=db_paper.id)
                self.db.add(link)
        await self.db.commit()
