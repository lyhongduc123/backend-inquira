"""
Bookmark router for API endpoints
"""
from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any
from app.db.database import get_db_session
from app.auth.dependencies import get_current_user
from app.models.users import DBUser
from app.bookmarks.service import BookmarkService
from app.bookmarks.schemas import (
    BookmarkCreate, 
    BookmarkUpdate, 
    BookmarkResponse,
    BookmarkWithPaperResponse
)


router = APIRouter()


@router.post("/", response_model=BookmarkResponse)
async def create_bookmark(
    request: BookmarkCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> BookmarkResponse:
    """Create a new bookmark"""
    service = BookmarkService(db)
    bookmark = await service.create_bookmark(
        user_id=current_user.id,
        paper_id=request.paper_id,
        notes=request.notes
    )
    return bookmark


@router.get("/", response_model=Dict[str, Any])
async def list_bookmarks(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """List all bookmarks for the current user"""
    service = BookmarkService(db)
    result = await service.list_bookmarks(
        user_id=current_user.id,
        skip=skip,
        limit=limit
    )
    return result


@router.get("/{bookmark_id}", response_model=BookmarkWithPaperResponse)
async def get_bookmark(
    bookmark_id: int,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> BookmarkWithPaperResponse:
    """Get a specific bookmark with paper details"""
    service = BookmarkService(db)
    bookmark = await service.get_bookmark(bookmark_id, current_user.id)
    return bookmark


@router.patch("/{bookmark_id}", response_model=BookmarkResponse)
async def update_bookmark(
    bookmark_id: int,
    request: BookmarkUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> BookmarkResponse:
    """Update bookmark notes"""
    service = BookmarkService(db)
    bookmark = await service.update_bookmark(
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
