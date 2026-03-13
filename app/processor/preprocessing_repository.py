"""
Repository for preprocessing-specific database operations.
Separates database queries from business logic in preprocessing service.
"""
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, exists, func
from app.models.papers import DBPaper
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class PreprocessingRepository:
    """Repository for preprocessing database operations"""

    def __init__(self, db: AsyncSession):
        self.db = db

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
