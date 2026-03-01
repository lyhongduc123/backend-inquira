"""
Repository for author database operations.
Handles CRUD operations for authors and author-paper relationships.
"""

from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload, joinedload
from app.models.authors import DBAuthor, DBAuthorPaper, DBAuthorInstitution
from app.models.papers import DBPaper
from app.models.journals import DBJournal
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
        result = await self.db.execute(select(DBAuthor).where(DBAuthor.orcid == orcid))
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

        logger.info(
            f"Created author {author_data.get('name')} ({author_data.get('author_id')})"
        )
        return db_author

    async def update_author(
        self, author_id: str, author_data: dict
    ) -> Optional[DBAuthor]:
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
            if hasattr(author, key) and value is not None:
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
        author_id = author_data.get("author_id")
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
        author_string: Optional[str] = None,
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
                DBAuthorPaper.author_id == author_id, DBAuthorPaper.paper_id == paper_id
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            logger.debug(
                f"Author-paper link already exists: author_id={author_id}, paper_id={paper_id}"
            )
            return existing

        db_author_paper = DBAuthorPaper(
            author_id=author_id,
            paper_id=paper_id,
            author_position=author_position,
            is_corresponding=is_corresponding,
            institution_id=institution_id,
            institution_raw=institution_raw,
            author_string=author_string,
        )

        self.db.add(db_author_paper)
        await self.db.commit()
        await self.db.refresh(db_author_paper)

        logger.debug(
            f"Created author-paper link: author_id={author_id}, paper_id={paper_id}"
        )
        return db_author_paper

    async def create_author_institution_link(
        self,
        author_id: int,
        institution_id: int,
        year: Optional[int] = None,
        is_current: bool = False,
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
                DBAuthorInstitution.institution_id == institution_id,
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
            logger.debug(
                f"Updated author-institution link: author_id={author_id}, institution_id={institution_id}"
            )
            return existing

        db_author_institution = DBAuthorInstitution(
            author_id=author_id,
            institution_id=institution_id,
            start_year=year,
            end_year=year,
            is_current=is_current,
            paper_count=1,
        )

        self.db.add(db_author_institution)
        await self.db.commit()
        await self.db.refresh(db_author_institution)

        logger.debug(
            f"Created author-institution link: author_id={author_id}, institution_id={institution_id}"
        )
        return db_author_institution

    async def get_author_papers_with_metadata(self, author_id: str) -> List:
        """
        Get all papers for an author with full metadata (journals, institutions).

        Args:
            author_id: Author identifier

        Returns:
            List of DBPaper objects with relationships loaded
        """

        author = await self.get_author_by_id(author_id)
        if not author:
            return []

        stmt = (
            select(DBPaper)
            .join(DBAuthorPaper, DBPaper.id == DBAuthorPaper.paper_id)
            .where(DBAuthorPaper.author_id == author.id)
            .options(
                joinedload(DBPaper.journal), 
                selectinload(DBPaper.paper_authors).selectinload(DBAuthorPaper.author)
            )
            .order_by(DBPaper.publication_date.desc().nulls_last())
        )

        result = await self.db.execute(stmt)
        papers = result.unique().scalars().all()

        logger.debug(f"Found {len(papers)} papers for author {author_id}")
        return list(papers)

    async def get_author_paper_links(self, author_id: str) -> List[DBAuthorPaper]:
        """
        Get all author-paper relationship records for an author.
        Includes position, corresponding status, institution links.

        Args:
            author_id: Author identifier

        Returns:
            List of DBAuthorPaper objects
        """
        author = await self.get_author_by_id(author_id)
        if not author:
            return []

        stmt = (
            select(DBAuthorPaper)
            .where(DBAuthorPaper.author_id == author.id)
            .options(
                selectinload(DBAuthorPaper.paper),
                selectinload(DBAuthorPaper.institution),
            )
        )

        result = await self.db.execute(stmt)
        links = result.scalars().all()

        return list(links)

    async def get_quartile_breakdown(self, author_id: str) -> Dict[str, int]:
        """
        Get paper count by journal quartile for an author.

        Args:
            author_id: Author identifier

        Returns:
            Dict mapping quartile (Q1, Q2, Q3, Q4) to paper count
        """

        author = await self.get_author_by_id(author_id)
        if not author:
            return {}

        stmt = (
            select(DBJournal.sjr_best_quartile, func.count(DBPaper.id))
            .select_from(DBAuthorPaper)
            .join(DBPaper, DBAuthorPaper.paper_id == DBPaper.id)
            .join(DBJournal, DBPaper.journal_id == DBJournal.id)
            .where(
                DBAuthorPaper.author_id == author.id,
                DBJournal.sjr_best_quartile.isnot(None),
            )
            .group_by(DBJournal.sjr_best_quartile)
        )

        result = await self.db.execute(stmt)
        rows = result.all()

        quartiles = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
        for quartile, count in rows:
            if quartile in quartiles:
                quartiles[quartile] = count

        return quartiles

    async def get_co_authors(
        self, author_id: str, limit: int = 10, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get most frequent co-authors for an author.

        Args:
            author_id: Author identifier
            limit: Maximum co-authors to return

        Returns:
            List of co-author dicts with collaboration stats
        """
        author = await self.get_author_by_id(author_id)
        if not author:
            return []

        paper_ids_stmt = select(DBAuthorPaper.paper_id).where(
            DBAuthorPaper.author_id == author.id
        )
        paper_ids_result = await self.db.execute(paper_ids_stmt)
        paper_ids = [row[0] for row in paper_ids_result.all()]

        if not paper_ids:
            return []

        stmt = (
            select(
                DBAuthor.id,
                DBAuthor.author_id,
                DBAuthor.name,
                DBAuthor.h_index,
                DBAuthor.total_citations,
                DBAuthor.total_papers,
                func.count(DBAuthorPaper.paper_id).label("collaboration_count"),
            )
            .select_from(DBAuthorPaper)
            .join(DBAuthor, DBAuthorPaper.author_id == DBAuthor.id)
            .where(
                DBAuthorPaper.paper_id.in_(paper_ids),
                DBAuthor.id != author.id,  # Exclude the author themselves
            )
            .group_by(
                DBAuthor.id,
                DBAuthor.author_id,
                DBAuthor.name,
                DBAuthor.h_index,
                DBAuthor.total_citations,
                DBAuthor.total_papers,
            )
            .order_by(func.count(DBAuthorPaper.paper_id).desc())
            .offset(offset)
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        rows = result.all()

        co_authors = [
            {
                "author_id": row[1],
                "name": row[2],
                "h_index": row[3],
                "total_citations": row[4],
                "total_papers": row[5],
                "collaboration_count": row[6],
            }
            for row in rows
        ]

        return co_authors

    async def get_citing_authors(
        self, author_id: str, limit: int = 50, offset: int = 0
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Get authors who have cited this author's papers.
        Uses cached relationships from author_relationships table.
        
        Args:
            author_id: Author identifier
            limit: Maximum authors to return
            offset: Offset for pagination
            
        Returns:
            Tuple of (list of citing author dicts, total count)
        """
        author = await self.get_author_by_id(author_id)
        if not author:
            return [], 0
        
        from app.models.author_relationships import DBAuthorRelationship
        
        # Get count from cached relationships
        count_stmt = (
            select(func.count())
            .select_from(DBAuthorRelationship)
            .where(
                DBAuthorRelationship.author_id == author.id,
                DBAuthorRelationship.relationship_type == 'citing'
            )
        )
        total = await self.db.scalar(count_stmt) or 0
        
        if total == 0:
            return [], 0
        
        # Get citing authors from cached relationships
        stmt = (
            select(
                DBAuthor.author_id,
                DBAuthor.name,
                DBAuthor.h_index,
                DBAuthor.total_citations,
                DBAuthor.total_papers,
                DBAuthorRelationship.relationship_count.label("citation_count")
            )
            .select_from(DBAuthorRelationship)
            .join(DBAuthor, DBAuthorRelationship.related_author_id == DBAuthor.id)
            .where(
                DBAuthorRelationship.author_id == author.id,
                DBAuthorRelationship.relationship_type == 'citing'
            )
            .order_by(DBAuthorRelationship.relationship_count.desc())
            .offset(offset)
            .limit(limit)
        )
        
        result = await self.db.execute(stmt)
        rows = result.all()
        
        citing_authors = [
            {
                "author_id": row[0],
                "name": row[1],
                "h_index": row[2],
                "total_citations": row[3],
                "total_papers": row[4],
                "citation_count": row[5]
            }
            for row in rows
        ]
        
        return citing_authors, total
    
    async def get_referenced_authors(
        self, author_id: str, limit: int = 50, offset: int = 0
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Get authors that this author has referenced/cited.
        Uses cached relationships from author_relationships table.
        
        Args:
            author_id: Author identifier
            limit: Maximum authors to return
            offset: Offset for pagination
            
        Returns:
            Tuple of (list of referenced author dicts, total count)
        """
        author = await self.get_author_by_id(author_id)
        if not author:
            return [], 0
        
        from app.models.author_relationships import DBAuthorRelationship
        
        # Get count from cached relationships
        count_stmt = (
            select(func.count())
            .select_from(DBAuthorRelationship)
            .where(
                DBAuthorRelationship.author_id == author.id,
                DBAuthorRelationship.relationship_type == 'referenced'
            )
        )
        total = await self.db.scalar(count_stmt) or 0
        
        if total == 0:
            return [], 0
        
        # Get referenced authors from cached relationships
        stmt = (
            select(
                DBAuthor.author_id,
                DBAuthor.name,
                DBAuthor.h_index,
                DBAuthor.total_citations,
                DBAuthor.total_papers,
                DBAuthorRelationship.relationship_count.label("reference_count")
            )
            .select_from(DBAuthorRelationship)
            .join(DBAuthor, DBAuthorRelationship.related_author_id == DBAuthor.id)
            .where(
                DBAuthorRelationship.author_id == author.id,
                DBAuthorRelationship.relationship_type == 'referenced'
            )
            .order_by(DBAuthorRelationship.relationship_count.desc())
            .offset(offset)
            .limit(limit)
        )
        
        result = await self.db.execute(stmt)
        rows = result.all()
        
        referenced_authors = [
            {
                "author_id": row[0],
                "name": row[1],
                "h_index": row[2],
                "total_citations": row[3],
                "total_papers": row[4],
                "reference_count": row[5]
            }
            for row in rows
        ]
        
        return referenced_authors, total
    
    async def compute_author_relationships(self, author_id: str) -> Dict[str, int]:
        """
        Compute and cache author-to-author relationships for citing and referencing.
        This should be called as a background job after author enrichment.
        
        Args:
            author_id: Author identifier
            
        Returns:
            Dict with counts: {"citing_authors": int, "referenced_authors": int}
        """
        author = await self.get_author_by_id(author_id)
        if not author:
            return {"citing_authors": 0, "referenced_authors": 0}
        
        from app.models.author_relationships import DBAuthorRelationship
        
        logger.info(f"Computing author relationships for {author_id}")
        
        # Delete existing relationships for this author to recompute
        await self.db.execute(
            DBAuthorRelationship.__table__.delete().where(
                DBAuthorRelationship.author_id == author.id
            )
        )
        
        co_authors = await self.get_co_authors(author_id, limit=10000)
        for co_author in co_authors:
            related_author = await self.get_author_by_id(co_author["author_id"])
            if related_author:
                relationship = DBAuthorRelationship(
                    author_id=author.id,
                    related_author_id=related_author.id,
                    relationship_type="collaboration",
                    relationship_count=co_author["collaboration_count"]
                )
                self.db.add(relationship)
        
        # For citing and referenced authors, we need paper-level citation data
        # This would require the DBCitation table to be populated
        # For now, return placeholder counts
        # TODO: Implement when citation data is available
        
        await self.db.commit()
        
        return {
            "collaborations": len(co_authors),
            "citing_authors": 0,  # TODO: Implement with citation data
            "referenced_authors": 0  # TODO: Implement with citation data
        }
