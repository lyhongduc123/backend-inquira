"""
Repository for author database operations.
Handles CRUD operations for authors and author-paper relationships.
"""
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.authors import DBAuthor, DBAuthorPaper, DBAuthorInstitution
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class AuthorRepository:
    """Repository for author database operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_author_by_id(self, author_id: str) -> Optional[DBAuthor]:
        """Get author by OpenAlex author ID"""
        result = await self.db.execute(
            select(DBAuthor).where(DBAuthor.author_id == author_id)
        )
        return result.scalar_one_or_none()
    
    async def get_author_by_orcid(self, orcid: str) -> Optional[DBAuthor]:
        """Get author by ORCID"""
        result = await self.db.execute(
            select(DBAuthor).where(DBAuthor.orcid == orcid)
        )
        return result.scalar_one_or_none()
    
    async def create_author(self, author_data: dict) -> DBAuthor:
        """
        Create a new author record.
        
        Args:
            author_data: Dictionary containing author fields
            
        Returns:
            Created DBAuthor object
        """
        db_author = DBAuthor(**author_data)
        self.db.add(db_author)
        await self.db.commit()
        await self.db.refresh(db_author)
        
        logger.info(f"Created author {author_data.get('name')} ({author_data.get('author_id')})")
        return db_author
    
    async def update_author(self, author_id: str, author_data: dict) -> Optional[DBAuthor]:
        """
        Update existing author record.
        
        Args:
            author_id: OpenAlex author ID
            author_data: Dictionary containing updated fields
            
        Returns:
            Updated DBAuthor object or None if not found
        """
        author = await self.get_author_by_id(author_id)
        if not author:
            return None
        
        for key, value in author_data.items():
            if hasattr(author, key):
                setattr(author, key, value)
        
        await self.db.commit()
        await self.db.refresh(author)
        
        logger.info(f"Updated author {author_id}")
        return author
    
    async def upsert_author(self, author_data: dict) -> DBAuthor:
        """
        Create or update author record.
        
        Args:
            author_data: Dictionary containing author fields
            
        Returns:
            DBAuthor object (created or updated)
        """
        author_id = author_data.get('author_id')
        if not author_id:
            raise ValueError("author_id is required for upsert")
        
        existing = await self.get_author_by_id(author_id)
        if existing:
            # Update only if new data has more information
            return await self.update_author(author_id, author_data) or existing
        
        return await self.create_author(author_data)
    
    async def create_author_paper_link(
        self,
        author_id: int,
        paper_id: int,
        author_position: Optional[int] = None,
        is_corresponding: bool = False,
        institution_id: Optional[int] = None,
        institution_raw: Optional[str] = None,
        author_string: Optional[str] = None
    ) -> DBAuthorPaper:
        """
        Create author-paper relationship.
        
        Args:
            author_id: Database ID of author
            paper_id: Database ID of paper
            author_position: Position in author list (1 = first author)
            is_corresponding: Whether this is a corresponding author
            institution_id: Database ID of institution at time of paper
            institution_raw: Raw affiliation string from paper
            author_string: Author name as appeared in paper
            
        Returns:
            Created DBAuthorPaper object
        """
        # Check if link already exists
        result = await self.db.execute(
            select(DBAuthorPaper).where(
                DBAuthorPaper.author_id == author_id,
                DBAuthorPaper.paper_id == paper_id
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            logger.debug(f"Author-paper link already exists: author_id={author_id}, paper_id={paper_id}")
            return existing
        
        db_author_paper = DBAuthorPaper(
            author_id=author_id,
            paper_id=paper_id,
            author_position=author_position,
            is_corresponding=is_corresponding,
            institution_id=institution_id,
            institution_raw=institution_raw,
            author_string=author_string
        )
        
        self.db.add(db_author_paper)
        await self.db.commit()
        await self.db.refresh(db_author_paper)
        
        logger.debug(f"Created author-paper link: author_id={author_id}, paper_id={paper_id}")
        return db_author_paper
    
    async def create_author_institution_link(
        self,
        author_id: int,
        institution_id: int,
        year: Optional[int] = None,
        is_current: bool = False
    ) -> DBAuthorInstitution:
        """
        Create or update author-institution relationship.
        
        Args:
            author_id: Database ID of author
            institution_id: Database ID of institution
            year: Year of affiliation
            is_current: Whether this is current affiliation
            
        Returns:
            Created or updated DBAuthorInstitution object
        """
        # Check if link already exists
        result = await self.db.execute(
            select(DBAuthorInstitution).where(
                DBAuthorInstitution.author_id == author_id,
                DBAuthorInstitution.institution_id == institution_id
            )
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update paper count and year range
            existing.paper_count += 1
            if year:
                if not existing.start_year or year < existing.start_year:
                    existing.start_year = year
                if not existing.end_year or year > existing.end_year:
                    existing.end_year = year
            if is_current:
                existing.is_current = True
            
            await self.db.commit()
            await self.db.refresh(existing)
            logger.debug(f"Updated author-institution link: author_id={author_id}, institution_id={institution_id}")
            return existing
        
        db_author_institution = DBAuthorInstitution(
            author_id=author_id,
            institution_id=institution_id,
            start_year=year,
            end_year=year,
            is_current=is_current,
            paper_count=1
        )
        
        self.db.add(db_author_institution)
        await self.db.commit()
        await self.db.refresh(db_author_institution)
        
        logger.debug(f"Created author-institution link: author_id={author_id}, institution_id={institution_id}")
        return db_author_institution
