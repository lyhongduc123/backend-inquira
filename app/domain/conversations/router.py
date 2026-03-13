"""
Conversation router for managing chat conversations
"""
from fastapi import APIRouter, Query, Depends, Request
from typing import Optional, TYPE_CHECKING
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from app.db.database import get_db_session
from .service import ConversationService
from .schemas import (
    ConversationCreate,
    ConversationUpdate,
    ConversationDetail,
    ConversationSummary,
    DeleteResponse
)
from app.auth.dependencies import get_current_user
from app.models.users import DBUser
from app.core.responses import PaginatedData
from app.core.exceptions import NotFoundException
from app.core.dependencies import get_container

if TYPE_CHECKING:
    from app.core.container import ServiceContainer

router = APIRouter()


@router.get("", response_model=PaginatedData[ConversationSummary])
async def list_conversations(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    archived: Optional[bool] = Query(None, description="Filter by archive status"),
    current_user: DBUser = Depends(get_current_user),
    container: "ServiceContainer" = Depends(get_container)
) -> PaginatedData[ConversationSummary]:
    """
    List all conversations for the current user
    
    - **page**: Page number for pagination
    - **page_size**: Number of items per page
    - **archived**: Filter archived/active conversations
    """
    conversations, total = await container.conversation_service.list_conversations(
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        archived=archived
    )
    
    from math import ceil
    total_pages = ceil(total / page_size) if page_size > 0 else 0
    
    return PaginatedData(
        items=conversations,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_previous=page > 1
    )
    
@router.post("", response_model=ConversationDetail)
async def create_conversation(
    http_request: Request,
    request: ConversationCreate,
    current_user: DBUser = Depends(get_current_user),
    container: "ServiceContainer" = Depends(get_container)
) -> ConversationDetail:
    """
    Create a new conversation
    
    - **title**: Optional conversation title
    """
    conversation = await container.conversation_service.create_conversation(
        user_id=current_user.id,
        title=request.title
    )
    
    return conversation

@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    http_request: Request,
    conversation_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> ConversationDetail:
    """
    Get detailed conversation including all messages
    
    - **conversation_id**: ID of the conversation
    """
    service = ConversationService(db)
    
    conversation = await service.get_conversation(conversation_id, current_user.id)
    if not conversation:
        raise NotFoundException(f"Conversation {conversation_id} not found")
    
    return conversation

@router.put("/{conversation_id}", response_model=ConversationDetail)
async def update_conversation(
    http_request: Request,
    conversation_id: str,
    request: ConversationUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> ConversationDetail:
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
        raise NotFoundException(f"Conversation {conversation_id} not found")
    
    return conversation


@router.delete("/{conversation_id}", response_model=DeleteResponse)
async def delete_conversation(
    http_request: Request,
    conversation_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> DeleteResponse:
    """
    Delete a conversation and all its messages
    
    - **conversation_id**: ID of the conversation
    """
    service = ConversationService(db)
    
    success = await service.delete_conversation(conversation_id, current_user.id)
    if not success:
        raise NotFoundException(f"Conversation {conversation_id} not found")
    
    return DeleteResponse(message="Conversation deleted successfully")
