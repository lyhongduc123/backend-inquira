"""
Bookmark repository for database operations
"""
from typing import Optional, List
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from app.models.bookmarks import DBBookmark
from app.models.papers import DBPaper
from app.models.authors import DBAuthorPaper


class BookmarkRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, user_id: int, paper_id: str, notes: Optional[str] = None) -> DBBookmark:
        """Create a new bookmark"""
        bookmark = DBBookmark(
            user_id=user_id,
            paper_id=paper_id,
            notes=notes,
            is_active=True
        )
        self.db.add(bookmark)
        await self.db.commit()
        await self.db.refresh(bookmark)
        return bookmark

    async def get_by_id(self, bookmark_id: int, user_id: int) -> Optional[DBBookmark]:
        """Get bookmark by ID for specific user"""
        result = await self.db.execute(
            select(DBBookmark)
            .where(
                and_(
                    DBBookmark.id == bookmark_id,
                    DBBookmark.user_id == user_id,
                    DBBookmark.is_active == True
                )
            )
            .options(
                joinedload(DBBookmark.paper)
                .selectinload(DBPaper.paper_authors)
                .selectinload(DBAuthorPaper.author),
                joinedload(DBBookmark.paper).joinedload(DBPaper.journal)
            )
        )
        return result.scalar_one_or_none()

    async def get_by_paper(self, user_id: int, paper_id: str) -> Optional[DBBookmark]:
        """Get bookmark for a specific paper and user"""
        result = await self.db.execute(
            select(DBBookmark).where(
                and_(
                    DBBookmark.user_id == user_id,
                    DBBookmark.paper_id == paper_id,
                    DBBookmark.is_active == True
                )
            )
        )
        return result.scalar_one_or_none()

    async def list_by_user(
        self, 
        user_id: int, 
        skip: int = 0, 
        limit: int = 50
    ) -> tuple[List[DBBookmark], int]:
        """List all bookmarks for a user with pagination"""
        query = select(DBBookmark).where(
            and_(
                DBBookmark.user_id == user_id,
                DBBookmark.is_active == True
            )
        ).options(
            joinedload(DBBookmark.paper)
            .selectinload(DBPaper.paper_authors)
            .selectinload(DBAuthorPaper.author),
            joinedload(DBBookmark.paper).joinedload(DBPaper.journal)
        )
        
        # Get total count
        count_result = await self.db.execute(query)
        total = len(count_result.all())
        
        # Get paginated results
        query = query.order_by(DBBookmark.created_at.desc())
        query = query.offset(skip).limit(limit)
        
        result = await self.db.execute(query)
        bookmarks = result.scalars().unique().all()
        
        return list(bookmarks), total

    async def update(self, bookmark_id: int, user_id: int, notes: Optional[str] = None) -> Optional[DBBookmark]:
        """Update bookmark notes"""
        bookmark = await self.get_by_id(bookmark_id, user_id)
        if not bookmark:
            return None
        
        if notes is not None:
            bookmark.notes = notes
        
        await self.db.commit()
        await self.db.refresh(bookmark)
        return bookmark

    async def delete(self, bookmark_id: int, user_id: int) -> bool:
        """Soft delete a bookmark"""
        bookmark = await self.get_by_id(bookmark_id, user_id)
        if not bookmark:
            return False
        
        bookmark.is_active = False
        await self.db.commit()
        return True

    async def check_exists(self, user_id: int, paper_id: str) -> bool:
        """Check if bookmark exists"""
        bookmark = await self.get_by_paper(user_id, paper_id)
        return bookmark is not None
