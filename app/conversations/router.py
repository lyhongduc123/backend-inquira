"""
Conversation router for managing chat conversations
"""
from fastapi import APIRouter, Query, Depends, Request
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from app.db.database import get_db_session
from app.conversations.service import ConversationService
from app.conversations.schemas import (
    ConversationCreate,
    ConversationUpdate,
    ConversationDetail,
    ConversationSummary,
    DeleteResponse
)
from app.auth.dependencies import get_current_user
from app.models.users import DBUser
from app.core.responses import ApiResponse, PaginatedData, success_response, paginated_response
from app.core.exceptions import NotFoundException

router = APIRouter()


@router.get("", response_model=ApiResponse[PaginatedData[ConversationSummary]])
async def list_conversations(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    archived: Optional[bool] = Query(None, description="Filter by archive status"),
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> ApiResponse[PaginatedData[ConversationSummary]]:
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
    
    request_id = getattr(request.state, 'request_id', None)
    return paginated_response(
        items=conversations,
        total=total,
        page=page,
        page_size=page_size,
        request_id=request_id
    )
    
@router.post("", response_model=ApiResponse[ConversationDetail])
async def create_conversation(
    http_request: Request,
    request: ConversationCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> ApiResponse[ConversationDetail]:
    """
    Create a new conversation
    
    - **title**: Optional conversation title
    """
    service = ConversationService(db)
    
    conversation = await service.create_conversation(
        user_id=current_user.id,
        title=request.title
    )
    
    request_id = getattr(http_request.state, 'request_id', None)
    return success_response(conversation, request_id=request_id)

@router.get("/{conversation_id}", response_model=ApiResponse[ConversationDetail])
async def get_conversation(
    http_request: Request,
    conversation_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> ApiResponse[ConversationDetail]:
    """
    Get detailed conversation including all messages
    
    - **conversation_id**: ID of the conversation
    """
    service = ConversationService(db)
    
    conversation = await service.get_conversation(conversation_id, current_user.id)
    if not conversation:
        raise NotFoundException(f"Conversation {conversation_id} not found")
    
    request_id = getattr(http_request.state, 'request_id', None)
    return success_response(conversation, request_id=request_id)

@router.put("/{conversation_id}", response_model=ApiResponse[ConversationDetail])
async def update_conversation(
    http_request: Request,
    conversation_id: str,
    request: ConversationUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> ApiResponse[ConversationDetail]:
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
    
    request_id = getattr(http_request.state, 'request_id', None)
    return success_response(conversation, request_id=request_id)


@router.delete("/{conversation_id}", response_model=ApiResponse[DeleteResponse])
async def delete_conversation(
    http_request: Request,
    conversation_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> ApiResponse[DeleteResponse]:
    """
    Delete a conversation and all its messages
    
    - **conversation_id**: ID of the conversation
    """
    service = ConversationService(db)
    
    success = await service.delete_conversation(conversation_id, current_user.id)
    if not success:
        raise NotFoundException(f"Conversation {conversation_id} not found")
    
    request_id = getattr(http_request.state, 'request_id', None)
    return success_response(
        DeleteResponse(message="Conversation deleted successfully"),
        request_id=request_id
    )
