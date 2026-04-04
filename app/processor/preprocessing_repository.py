"""
Repository for preprocessing-specific database operations.
Separates database queries from business logic in preprocessing service.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, exists, func
from app.models.papers import DBPaper
from app.models.authors import DBAuthor, DBAuthorPaper
from app.models.preprocessing_state import DBPreprocessingState
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class PreprocessingRepository:
    """Repository for preprocessing database operations"""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ==================== Preprocessing state operations ====================

    async def get_state_by_job_id(self, job_id: str) -> Optional[DBPreprocessingState]:
        """Get preprocessing state by job id."""
        stmt = select(DBPreprocessingState).where(DBPreprocessingState.job_id == job_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def create_state(self, job_id: str, target_count: int) -> DBPreprocessingState:
        """Create a new preprocessing state."""
        state = DBPreprocessingState(
            job_id=job_id,
            target_count=target_count,
            current_index=0,
            processed_count=0,
            skipped_count=0,
            error_count=0,
            is_completed=False,
            is_running=False,
            is_paused=False,
        )
        self.db.add(state)
        await self.db.commit()
        await self.db.refresh(state)
        return state

    async def save_state(self, state: DBPreprocessingState, refresh: bool = False) -> DBPreprocessingState:
        """Persist state changes and optionally refresh."""
        await self.db.commit()
        if refresh:
            await self.db.refresh(state)
        return state

    async def set_state_continuation_token(
        self, state: DBPreprocessingState, continuation_token: Optional[str]
    ) -> None:
        """Update continuation token for a state."""
        state.continuation_token = continuation_token  # type: ignore
        await self.db.commit()

    # ==================== Paper operations ====================

    async def paper_exists(self, paper_id: str) -> bool:
        """
        Check if a paper exists in the database.

        Args:
            paper_id: Paper identifier

        Returns:
            True if paper exists, False otherwise
        """
        stmt = select(exists().where(DBPaper.paper_id == paper_id))
        result = await self.db.scalar(stmt)
        return result or False

    async def get_unprocessed_papers(self, limit: int = 100) -> List[DBPaper]:
        """
        Get papers that have is_processed = False.

        Args:
            limit: Maximum number of papers to return

        Returns:
            List of unprocessed DBPaper objects
        """
        stmt = (
            select(DBPaper)
            .where(
                DBPaper.is_processed == False,
                DBPaper.is_open_access == True,
            )
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return list(result.scalars().all())


    async def get_papers_missing_embeddings(self, limit: int = 1000) -> List[DBPaper]:
        """
        Get papers that don't have embeddings but have abstracts.

        Args:
            limit: Maximum number of papers to return

        Returns:
            List of DBPaper objects missing embeddings
        """
        stmt = (
            select(DBPaper)
            .where(
                DBPaper.embedding.is_(None),
                DBPaper.abstract.isnot(None),
            )
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_papers_for_citation_linking(self, limit: int = 200) -> List[DBPaper]:
        """
        Get papers likely to have references for citation linking.

        Strategy:
        - Require a valid paper_id
        - Require reference_count > 0
        - Prefer newest papers first
        """
        stmt = (
            select(DBPaper)
            .where(
                DBPaper.paper_id.isnot(None),
                DBPaper.reference_count > 0,
            )
            .order_by(DBPaper.updated_at.desc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_papers_for_tagging(
        self,
        limit: int = 200,
        only_missing_tags: bool = True,
    ) -> List[DBPaper]:
        """
        Get papers eligible for zero-shot tag computation.

        Args:
            limit: Max papers to return
            only_missing_tags: When True, include only papers where tags are null
        """
        conditions = [
            DBPaper.paper_id.isnot(None),
            DBPaper.abstract.isnot(None),
        ]
        if only_missing_tags:
            conditions.append(DBPaper.paper_tags.is_(None))

        stmt = select(DBPaper).where(*conditions).order_by(DBPaper.updated_at.desc()).limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_paper_tags(self, paper_id: str, tags: List[Dict[str, Any]]) -> None:
        """Persist computed tags for a paper."""
        paper = await self.db.scalar(select(DBPaper).where(DBPaper.paper_id == paper_id))
        if not paper:
            return

        paper.paper_tags = tags  # type: ignore
        paper.updated_at = datetime.utcnow()  # type: ignore
        await self.db.commit()

    async def get_paper_count(self) -> int:
        """
        Get total count of papers in database.

        Returns:
            Total number of papers
        """
        stmt = select(func.count()).select_from(DBPaper)
        result = await self.db.scalar(stmt)
        return result or 0

    async def get_processed_paper_count(self) -> int:
        """
        Get count of processed papers.

        Returns:
            Number of processed papers
        """
        stmt = select(func.count()).select_from(DBPaper).where(DBPaper.is_processed == True)
        result = await self.db.scalar(stmt)
        return result or 0

    # ==================== Author operations ====================

    async def list_authors_for_metrics(
        self,
        limit: int = 500,
        offset: int = 0,
        only_unprocessed: bool = False,
    ) -> List[DBAuthor]:
        """
        List authors for metrics computation.

        Args:
            limit: Maximum authors to return
            offset: Pagination offset
            only_unprocessed: If True, return only authors with is_processed=False
        """
        stmt = select(DBAuthor).order_by(DBAuthor.id.asc()).offset(offset).limit(limit)
        if only_unprocessed:
            stmt = stmt.where(DBAuthor.is_processed == False)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_author_papers_for_metrics(self, author_db_id: int) -> List[DBPaper]:
        """
        Get all linked papers for an author for metrics computation.
        """
        stmt = (
            select(DBPaper)
            .join(DBAuthorPaper, DBAuthorPaper.paper_id == DBPaper.id)
            .where(DBAuthorPaper.author_id == author_db_id)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_author_metrics(
        self,
        author_db_id: int,
        update_data: Dict[str, Any],
    ) -> None:
        """
        Update author fields in-place by database id.
        """
        author = await self.db.get(DBAuthor, author_db_id)
        if not author:
            return

        for key, value in update_data.items():
            if hasattr(author, key):
                setattr(author, key, value)

        author.updated_at = datetime.utcnow()  # type: ignore
        await self.db.commit()

    async def count_all_authors(self) -> int:
        """Count all authors."""
        result = await self.db.scalar(select(func.count()).select_from(DBAuthor))
        return int(result or 0)
