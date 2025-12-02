"""
Conversation router for managing chat conversations
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db_session
from app.conversations.service import ConversationService
from app.conversations.schemas import (
    ConversationCreate,
    ConversationUpdate,
    ConversationDetail,
    ConversationListResponse,
    ConversationSummary
)
from app.auth.dependencies import get_current_user
from app.models.users import DBUser

router = APIRouter()


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    archived: Optional[bool] = Query(None, description="Filter by archive status"),
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
):
    """
    List all conversations for the current user
    
    - **page**: Page number for pagination
    - **page_size**: Number of items per page
    - **archived**: Filter archived/active conversations
    """
    service = ConversationService(db)
    
    conversations, total = await service.list_conversations(
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        archived=archived
    )
    
    return ConversationListResponse(
        conversations=conversations,
        total=total,
        page=page,
        page_size=page_size
    )


@router.post("", response_model=ConversationDetail)
async def create_conversation(
    request: ConversationCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
):
    """
    Create a new conversation
    
    - **title**: Optional conversation title
    """
    service = ConversationService(db)
    
    conversation = await service.create_conversation(
        user_id=current_user.id,
        title=request.title
    )
    
    return conversation


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
):
    """
    Get detailed conversation including all messages
    
    - **conversation_id**: ID of the conversation
    """
    service = ConversationService(db)
    
    conversation = await service.get_conversation(conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation


@router.patch("/{conversation_id}", response_model=ConversationDetail)
async def update_conversation(
    conversation_id: str,
    request: ConversationUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
):
    """
    Update conversation (rename, archive, etc.)
    
    - **conversation_id**: ID of the conversation
    - **title**: New title
    - **is_archived**: Archive status
    """
    service = ConversationService(db)
    
    conversation = await service.update_conversation(
        conversation_id=conversation_id,
        user_id=current_user.id,
        update_data=request
    )
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
):
    """
    Delete a conversation and all its messages
    
    - **conversation_id**: ID of the conversation
    """
    service = ConversationService(db)
    
    success = await service.delete_conversation(conversation_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"success": True, "message": "Conversation deleted"}
