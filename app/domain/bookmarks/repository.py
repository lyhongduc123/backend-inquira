"""Bookmark repository for database operations."""

import re
from typing import Optional, List
from sqlalchemy import func, select, and_, or_, desc, asc, literal_column
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from app.models.bookmarks import DBBookmark
from app.models.papers import DBPaper
from app.models.authors import DBAuthorPaper
from app.models.authors import DBAuthor


class BookmarkRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self, user_id: int, paper_id: str, notes: Optional[str] = None
    ) -> DBBookmark:
        """Create a new bookmark"""
        bookmark = DBBookmark(
            user_id=user_id, paper_id=paper_id, notes=notes, is_active=True
        )
        self.db.add(bookmark)
        await self.db.commit()
        await self.db.refresh(bookmark)
        return bookmark

    async def get_by_id(
        self,
        bookmark_id: int,
        user_id: int,
        load_paper: bool = True,
    ) -> Optional[DBBookmark]:
        """Get bookmark by ID for specific user."""
        query = select(DBBookmark).where(
            and_(
                DBBookmark.id == bookmark_id,
                DBBookmark.user_id == user_id,
                DBBookmark.is_active == True,
            )
        )

        if load_paper:
            query = query.options(
                joinedload(DBBookmark.paper)
                .selectinload(DBPaper.authors)
                .selectinload(DBAuthorPaper.author),
                joinedload(DBBookmark.paper).joinedload(DBPaper.journal),
            )

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_paper(self, user_id: int, paper_id: str) -> Optional[DBBookmark]:
        """Get bookmark for a specific paper and user"""
        result = await self.db.execute(
            select(DBBookmark).where(
                and_(
                    DBBookmark.user_id == user_id,
                    DBBookmark.paper_id == paper_id,
                    DBBookmark.is_active == True,
                )
            )
        )
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 50,
        query: Optional[str] = None,
        is_open_access: Optional[bool] = None,
        has_notes: Optional[bool] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> tuple[List[DBBookmark], int]:
        """List bookmarks for a user with pagination and optional filters."""
        base_conditions = [
            DBBookmark.user_id == user_id,
            DBBookmark.is_active.is_(True),
        ]
        if has_notes is True:
            base_conditions.append(
                and_(
                    DBBookmark.notes.is_not(None),
                    func.length(func.trim(DBBookmark.notes)) > 0,
                )
            )
        elif has_notes is False:
            base_conditions.append(
                or_(
                    DBBookmark.notes.is_(None),
                    func.length(func.trim(DBBookmark.notes)) == 0,
                )
            )

        if is_open_access is not None:
            base_conditions.append(DBPaper.is_open_access.is_(is_open_access))
        normalized_query = query.strip() if query else ""
        search_rank = None

        if normalized_query:
            search_document = func.concat_ws(
                " ",
                func.coalesce(DBPaper.title, ""),
                func.coalesce(DBPaper.venue, ""),
                func.coalesce(DBBookmark.notes, ""),
            )
            ts_query = func.websearch_to_tsquery("english", normalized_query)
            text_match = func.to_tsvector("english", search_document).op("@@")(ts_query)
            author_match = DBPaper.id.in_(
                select(DBAuthorPaper.paper_id)
                .join(DBAuthor)
                .where(DBAuthor.name.ilike(f"%{normalized_query}%"))
            )

            base_conditions.append(or_(text_match, author_match))
            search_rank = func.ts_rank_cd(
                func.to_tsvector("english", search_document),
                ts_query,
            )

        base_filter = and_(*base_conditions)

        stmt = (
            select(DBBookmark)
            .join(DBPaper, DBPaper.paper_id == DBBookmark.paper_id, isouter=True)
            .where(base_filter)
            .options(
                joinedload(DBBookmark.paper)
                .selectinload(DBPaper.authors)
                .selectinload(DBAuthorPaper.author),
                joinedload(DBBookmark.paper).joinedload(DBPaper.journal),
            )
            .offset(skip)
            .limit(limit)
        )
        
        normalized_sort_by = (sort_by or "").strip().lower()
        normalized_sort_order = (sort_order or "desc").strip().lower()
        is_desc = normalized_sort_order != "asc"

        def with_direction(expr):
            return desc(expr) if is_desc else asc(expr)

        if normalized_sort_by == "citations":
            stmt = stmt.order_by(
                with_direction(DBPaper.citation_count),
                with_direction(DBBookmark.id),
            )
        elif normalized_sort_by == "year":
            stmt = stmt.order_by(
                with_direction(DBPaper.year),
                with_direction(DBBookmark.id),
            )
        elif normalized_sort_by == "id":
            stmt = stmt.order_by(with_direction(DBBookmark.id))
        elif normalized_query and search_rank is not None:
            stmt = stmt.order_by(
                with_direction(search_rank),
                with_direction(DBBookmark.id),
            )
        else:
            stmt = stmt.order_by(with_direction(DBBookmark.id))

        count_query = (
            select(func.count())
            .select_from(DBBookmark)
            .join(DBPaper, DBPaper.paper_id == DBBookmark.paper_id, isouter=True)
            .where(base_filter)
        )

        total = (await self.db.execute(count_query)).scalar_one()

        result = await self.db.execute(stmt)
        bookmarks = result.scalars().unique().all()

        return list(bookmarks), total

    async def update(
        self, bookmark_id: int, user_id: int, notes: Optional[str] = None
    ) -> Optional[DBBookmark]:
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
        bookmark = await self.get_by_id(bookmark_id, user_id, load_paper=False)
        if not bookmark:
            return False

        bookmark.is_active = False
        await self.db.commit()
        return True

    async def check_exists(self, user_id: int, paper_id: str) -> bool:
        """Check if bookmark exists"""
        bookmark = await self.get_by_paper(user_id, paper_id)
        return bookmark is not None
