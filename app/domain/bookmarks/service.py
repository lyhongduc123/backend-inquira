"""
Bookmark service for business logic
"""
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from .repository import BookmarkRepository
from .schemas import BookmarkResponse, BookmarkWithPaperResponse, BookmarkListResponse
from app.core.exceptions import NotFoundException, BadRequestException


class BookmarkService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = BookmarkRepository(db)

    async def create_bookmark(self, user_id: int, paper_id: str, notes: Optional[str] = None) -> BookmarkResponse:
        """Create a new bookmark"""
        # Check if already bookmarked
        existing = await self.repo.get_by_paper(user_id, paper_id)
        if existing:
            raise BadRequestException("Paper already bookmarked")
        
        bookmark = await self.repo.create(user_id, paper_id, notes)
        return BookmarkResponse(
            id=bookmark.id,
            paper_id=bookmark.paper_id,
            notes=bookmark.notes,
            created_at=bookmark.created_at,
            updated_at=bookmark.updated_at
        )

    async def get_bookmark(self, bookmark_id: int, user_id: int) -> BookmarkWithPaperResponse:
        """Get a specific bookmark with paper details"""
        bookmark = await self.repo.get_by_id(bookmark_id, user_id)
        if not bookmark:
            raise NotFoundException(f"Bookmark {bookmark_id} not found")
        
        paper_metadata = None
        if bookmark.paper:
            from app.domain.papers.schemas import PaperMetadata
            paper_metadata = PaperMetadata.from_db_model(bookmark.paper)
        
        return BookmarkWithPaperResponse(
            id=bookmark.id,
            paper_id=bookmark.paper_id,
            notes=bookmark.notes,
            created_at=bookmark.created_at,
            updated_at=bookmark.updated_at,
            paper=paper_metadata
        )

    async def list_bookmarks(self, user_id: int, skip: int = 0, limit: int = 50) -> BookmarkListResponse:
        """List all bookmarks for a user"""
        bookmarks, total = await self.repo.list_by_user(user_id, skip, limit)
        
        items = []
        for bookmark in bookmarks:
            paper_metadata = None
            if bookmark.paper:
                from app.domain.papers.schemas import PaperMetadata
                paper_metadata = PaperMetadata.from_db_model(bookmark.paper)
            
            items.append(BookmarkWithPaperResponse(
                id=bookmark.id,
                paper_id=bookmark.paper_id,
                notes=bookmark.notes,
                created_at=bookmark.created_at,
                updated_at=bookmark.updated_at,
                paper=paper_metadata
            ))
        
        return BookmarkListResponse(
            items=items,
            total=total,
            skip=skip,
            limit=limit
        )

    async def update_bookmark(self, bookmark_id: int, user_id: int, notes: Optional[str] = None) -> BookmarkResponse:
        """Update bookmark notes"""
        bookmark = await self.repo.update(bookmark_id, user_id, notes)
        if not bookmark:
            raise NotFoundException(f"Bookmark {bookmark_id} not found")
        
        return BookmarkResponse(
            id=bookmark.id,
            paper_id=bookmark.paper_id,
            notes=bookmark.notes,
            created_at=bookmark.created_at,
            updated_at=bookmark.updated_at
        )

    async def delete_bookmark(self, bookmark_id: int, user_id: int) -> None:
        """Delete a bookmark"""
        success = await self.repo.delete(bookmark_id, user_id)
        if not success:
            raise NotFoundException(f"Bookmark {bookmark_id} not found")

    async def check_bookmarked(self, user_id: int, paper_id: str) -> bool:
        """Check if a paper is bookmarked"""
        return await self.repo.check_exists(user_id, paper_id)
