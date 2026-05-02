"""
Bookmark router for API endpoints
"""
from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, TYPE_CHECKING, Literal
from app.db.database import get_db_session
from app.auth.dependencies import get_current_user
from app.models.users import DBUser
from app.core.dependencies import get_container
from .service import BookmarkService
from .schemas import (
    BookmarkCreate, 
    BookmarkUpdate, 
    BookmarkResponse,
    BookmarkWithPaperResponse,
    BookmarkListResponse
)

if TYPE_CHECKING:
    from app.core.container import ServiceContainer


router = APIRouter()


@router.post("/", response_model=BookmarkResponse)
async def create_bookmark(
    request: BookmarkCreate,
    current_user: DBUser = Depends(get_current_user),
    container: "ServiceContainer" = Depends(get_container)
) -> BookmarkResponse:
    """Create a new bookmark"""
    bookmark = await container.bookmark_service.create_bookmark(
        user_id=current_user.id,
        paper_id=request.paper_id,
        notes=request.notes
    )
    return bookmark


@router.get("", response_model=BookmarkListResponse)
async def list_bookmarks(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    q: str | None = Query(None, min_length=1, description="Search query"),
    is_open_access: bool | None = Query(None),
    has_notes: bool | None = Query(None),
    sort_by: Literal["id", "citations", "year"] | None = Query(None),
    sort_order: Literal["asc", "desc"] = Query("desc"),
    current_user: DBUser = Depends(get_current_user),
    container: "ServiceContainer" = Depends(get_container)
) -> BookmarkListResponse:
    """List bookmarks for the current user with optional API-side filters."""
    result = await container.bookmark_service.list_bookmarks(
        user_id=current_user.id,
        skip=skip,
        limit=limit,
        query=q,
        is_open_access=is_open_access,
        has_notes=has_notes,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return result


@router.get("/{bookmark_id}", response_model=BookmarkWithPaperResponse)
async def get_bookmark(
    bookmark_id: int,
    current_user: DBUser = Depends(get_current_user),
    container: "ServiceContainer" = Depends(get_container)
) -> BookmarkWithPaperResponse:
    """Get a specific bookmark with paper details"""
    bookmark = await container.bookmark_service.get_bookmark(bookmark_id, current_user.id)
    return bookmark


@router.patch("/{bookmark_id}", response_model=BookmarkResponse)
async def update_bookmark(
    bookmark_id: int,
    request: BookmarkUpdate,
    current_user: DBUser = Depends(get_current_user),
    container: "ServiceContainer" = Depends(get_container)
) -> BookmarkResponse:
    """Update bookmark notes"""
    bookmark = await container.bookmark_service.update_bookmark(
        bookmark_id=bookmark_id,
        user_id=current_user.id,
        notes=request.notes
    )
    return bookmark


@router.delete("/{bookmark_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_bookmark(
    bookmark_id: int,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
):
    """Delete a bookmark"""
    service = BookmarkService(db)
    await service.delete_bookmark(bookmark_id, current_user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/check/{paper_id}", response_model=Dict[str, bool])
async def check_bookmark(
    paper_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> Dict[str, bool]:
    """Check if a paper is bookmarked"""
    service = BookmarkService(db)
    is_bookmarked = await service.check_bookmarked(current_user.id, paper_id)
    return {"is_bookmarked": is_bookmarked}
