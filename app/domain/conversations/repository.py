"""
Repository for conversation database operations
"""
from typing import Optional, List, Dict, Any
from sqlalchemy import desc, select, update
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
        title: str = "New Conversation",
        conversation_type: str = "multi_paper_rag",
        primary_paper_id: Optional[str] = None
    ) -> DBConversation:
        """Create a new conversation"""
        conversation_id = str(uuid.uuid4())
        
        db_conversation = DBConversation(
            conversation_id=conversation_id,
            user_id=user_id,
            title=title,
            message_count=0,
            is_archived=False,
            conversation_type=conversation_type,
            primary_paper_id=primary_paper_id
        )
        
        self.db.add(db_conversation)
        await self.db.commit()
        await self.db.refresh(db_conversation)
        
        return db_conversation
    
    async def get(
        self,
        conversation_id: str,
    ) -> Optional[DBConversation]:
        """Get conversation by ID for specific user"""
        result = await self.db.execute(
            select(DBConversation).where(
                DBConversation.conversation_id == conversation_id,
            )
        )
        return result.scalar_one_or_none()
    
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
        ).where(DBConversation.conversation_type != "single_paper_detail")
        
        if archived is not None:
            query = query.where(DBConversation.is_archived == archived)
        
        # Get total count
        count_query = select(DBConversation).where(
            DBConversation.user_id == user_id
        ).where(DBConversation.conversation_type != "single_paper_detail")
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
        update_data: Dict[str, Any]
    ) -> Optional[DBConversation]:
        """Update conversation details"""   
        try:     
            query = update(DBConversation).where(
                DBConversation.conversation_id == conversation_id
            ).values(**update_data)
            await self.db.execute(query)
            await self.db.commit()
        except Exception as e:
            await self.db.rollback()
            raise e

        return await self.get(conversation_id)
    
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
            DBMessage.__table__.delete().where( # type: ignore
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
    
    async def get_by_paper(
        self, 
        user_id: int, 
        paper_id: str
    ) -> Optional[DBConversation]:
        """
        Find single-paper conversation for user + paper.
        
        Returns the most recent non-archived conversation if multiple exist.
        """
        result = await self.db.execute(
            select(DBConversation)
            .where(
                DBConversation.user_id == user_id,
                DBConversation.primary_paper_id == paper_id,
                DBConversation.conversation_type == "single_paper_detail",
                DBConversation.is_archived == False
            )
            .order_by(desc(DBConversation.updated_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_paper_conversations(
        self,
        user_id: int,
        paper_id: str,
        skip: int = 0,
        limit: int = 10
    ) -> tuple[List[DBConversation], int]:
        """
        List all conversations for a specific paper (including archived).
        
        Useful for showing conversation history for a paper.
        """
        # Get total count
        count_query = select(DBConversation).where(
            DBConversation.user_id == user_id,
            DBConversation.primary_paper_id == paper_id,
            DBConversation.conversation_type == "single_paper_detail"
        )
        count_result = await self.db.execute(count_query)
        total = len(count_result.all())
        
        # Get paginated results
        result = await self.db.execute(
            select(DBConversation)
            .where(
                DBConversation.user_id == user_id,
                DBConversation.primary_paper_id == paper_id,
                DBConversation.conversation_type == "single_paper_detail"
            )
            .order_by(desc(DBConversation.updated_at))
            .offset(skip)
            .limit(limit)
        )
        conversations = result.scalars().all()
        
        return list(conversations), total
