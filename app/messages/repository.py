"""
Repository for message database operations
"""
from typing import Optional, List, Dict, Any
from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.messages import DBMessage
from app.models.papers import DBPaper
from app.models.message_papers import DBMessagePaper
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class MessageRepository:
    """Repository for message database operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(
        self,
        conversation_id: str,
        user_id: int,
        content: str,
        role: str = "user",
        message_metadata: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
        status: str = "pending",
    ) -> DBMessage:
        """
        Create a new message in a conversation
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID
            content: Message content
            role: Message role (user or assistant)
            message_metadata: Optional metadata (paper_snapshots, progress_events, client_message_id)
            is_active: Whether message is active
            status: Message status (pending, sent, failed)
            
        Returns:
            Created DBMessage
        """
        message = DBMessage(
            conversation_id=conversation_id,
            user_id=user_id,
            role=role,
            content=content,
            is_active=is_active,
            status=status,
            message_metadata=message_metadata or {},
        )
        
        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)
        
        return message
    
    async def get_by_id(
        self,
        message_id: int,
        user_id: Optional[int] = None
    ) -> Optional[DBMessage]:
        """
        Get message by ID
        
        Args:
            message_id: Message ID
            user_id: Optional user ID for authorization check
            
        Returns:
            DBMessage if found, None otherwise
        """
        query = select(DBMessage).where(DBMessage.id == message_id)
        
        if user_id is not None:
            query = query.where(DBMessage.user_id == user_id)
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_by_client_message_id(
        self,
        conversation_id: str,
        client_message_id: str
    ) -> Optional[DBMessage]:
        """
        Get message by client message ID for deduplication
        
        Args:
            conversation_id: Conversation ID
            client_message_id: Client-side message ID
            
        Returns:
            DBMessage if found, None otherwise
        """
        result = await self.db.execute(
            select(DBMessage).where(
                and_(
                    DBMessage.conversation_id == conversation_id,
                    DBMessage.message_metadata['client_message_id'].astext == client_message_id
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def list_by_conversation(
        self,
        conversation_id: str,
        skip: int = 0,
        limit: int = 100,
        include_inactive: bool = False
    ) -> List[DBMessage]:
        """
        Get all messages for a conversation
        
        Args:
            conversation_id: Conversation ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            include_inactive: Whether to include inactive messages
            
        Returns:
            List of DBMessage objects
        """
        query = select(DBMessage).where(
            DBMessage.conversation_id == conversation_id
        )
        
        if not include_inactive:
            query = query.where(DBMessage.is_active == True)
        
        query = query.order_by(DBMessage.created_at).offset(skip).limit(limit)
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def update_status(
        self,
        message_id: int,
        status: str
    ) -> Optional[DBMessage]:
        """
        Update message status
        
        Args:
            message_id: Message ID
            status: New status (pending, sent, failed)
            
        Returns:
            Updated DBMessage if found, None otherwise
        """
        result = await self.db.execute(
            select(DBMessage).where(DBMessage.id == message_id)
        )
        message = result.scalar_one_or_none()
        
        if message:
            message.status = status
            await self.db.commit()
            await self.db.refresh(message)
        
        return message
    
    async def update_metadata(
        self,
        message_id: int,
        metadata: Dict[str, Any]
    ) -> Optional[DBMessage]:
        """
        Update message metadata
        
        Args:
            message_id: Message ID
            metadata: New metadata to merge with existing
            
        Returns:
            Updated DBMessage if found, None otherwise
        """
        result = await self.db.execute(
            select(DBMessage).where(DBMessage.id == message_id)
        )
        message = result.scalar_one_or_none()
        
        if message:
            # Merge with existing metadata
            current_metadata = message.message_metadata or {}
            current_metadata.update(metadata)
            message.message_metadata = current_metadata
            
            await self.db.commit()
            await self.db.refresh(message)
        
        return message
    
    async def link_papers_to_message(
        self,
        message_id: int,
        paper_ids: List[str],
    ) -> None:
        """
        Link papers to a message via join table
        
        Args:
            message_id: Message ID
            paper_ids: List of paper IDs to link
        """
        for paper_id in paper_ids:
            result = await self.db.execute(
                select(DBPaper).where(DBPaper.paper_id == paper_id)
            )
            db_paper = result.scalar_one_or_none()
            
            if db_paper:
                link = DBMessagePaper(message_id=message_id, paper_id=db_paper.id)
                self.db.add(link)
        
        await self.db.commit()
    
    async def delete(
        self,
        message_id: int,
        user_id: Optional[int] = None,
        soft_delete: bool = True
    ) -> bool:
        """
        Delete a message (soft or hard delete)
        
        Args:
            message_id: Message ID
            user_id: Optional user ID for authorization check
            soft_delete: If True, mark as inactive; if False, delete from DB
            
        Returns:
            True if deleted, False if not found
        """
        query = select(DBMessage).where(DBMessage.id == message_id)
        
        if user_id is not None:
            query = query.where(DBMessage.user_id == user_id)
        
        result = await self.db.execute(query)
        message = result.scalar_one_or_none()
        
        if not message:
            return False
        
        if soft_delete:
            message.is_active = False
            await self.db.commit()
        else:
            await self.db.delete(message)
            await self.db.commit()
        
        return True
    
    async def count_by_conversation(
        self,
        conversation_id: str,
        include_inactive: bool = False
    ) -> int:
        """
        Count messages in a conversation
        
        Args:
            conversation_id: Conversation ID
            include_inactive: Whether to include inactive messages
            
        Returns:
            Message count
        """
        query = select(DBMessage).where(
            DBMessage.conversation_id == conversation_id
        )
        
        if not include_inactive:
            query = query.where(DBMessage.is_active == True)
        
        result = await self.db.execute(query)
        return len(result.all())
