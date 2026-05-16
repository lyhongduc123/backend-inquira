"""Message router for message-level operations."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.core.exceptions import NotFoundException
from app.core.db.database import get_db_session
from app.models.users import DBUser

from .service import MessageService

router = APIRouter()


@router.delete("/{message_id}")
async def delete_message(
    message_id: int,
    soft_delete: bool = Query(True, description="Soft delete by marking message inactive"),
    delete_assistant_reply_for_user: bool = Query(
        True,
        description=(
            "If deleting a user message, also delete the first assistant reply "
            "before the next user message"
        ),
    ),
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user),
) -> dict[str, str]:
    """Delete one message for the current user."""
    service = MessageService(db)

    deleted = await service.delete_message(
        message_id=message_id,
        user_id=current_user.id,
        soft_delete=soft_delete,
        delete_assistant_reply_for_user=delete_assistant_reply_for_user,
    )
    if not deleted:
        raise NotFoundException(f"Message {message_id} not found")

    return {"message": "Message deleted successfully"}
