"""
Repository for author database operations.
Handles CRUD operations for authors and author-paper relationships.
"""

from typing import Optional, List, Dict, Any
from psycopg2 import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update, delete, Integer, cast
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import selectinload, joinedload
from app.models.authors import DBAuthor, DBAuthorPaper, DBAuthorInstitution
from app.models.citations import DBCitation
from app.models.papers import DBPaper
from app.models.journals import DBJournal
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class AuthorRepository:
    """Repository for author database operations"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_author(self, author_id: str) -> Optional[DBAuthor]:
        """Get author by OpenAlex author ID"""
        result = await self.db.execute(
            select(DBAuthor).where(DBAuthor.author_id == author_id)
            .options(selectinload(DBAuthor.author_institutions).joinedload(DBAuthorInstitution.institution))
        )
        return result.scalar_one_or_none()

    async def get_author_by_orcid(self, orcid: str) -> Optional[DBAuthor]:
        """Get author by ORCID"""
        result = await self.db.execute(
            select(DBAuthor).where(DBAuthor.orcid == orcid)
            .options(selectinload(DBAuthor.author_institutions).joinedload(DBAuthorInstitution.institution))
        )
        return result.scalar_one_or_none()

    async def get_author_by_openalex_id(self, openalex_id: str) -> Optional[DBAuthor]:
        """Get author by OpenAlex ID."""
        result = await self.db.execute(
            select(DBAuthor).where(DBAuthor.openalex_id == openalex_id)
            .options(selectinload(DBAuthor.author_institutions).joinedload(DBAuthorInstitution.institution))
        )
        return result.scalar_one_or_none()
    
    async def has_openalex_conflicts(self, openalex_id: str) -> bool:
        """Check whether more than one author exists with the same OpenAlex ID."""
        if not openalex_id:
            return False

        count = await self.db.scalar(
            select(func.count()).select_from(DBAuthor).where(DBAuthor.openalex_id == openalex_id)
        )
        return int(count or 0) > 1
    
    async def mark_openalex_conflicts(self, openalex_id: str) -> int:
        """
        Mark all authors with the same OpenAlex ID as conflict.

        Returns:
            Number of affected rows.
        """
        if not openalex_id:
            return 0

        await self.db.execute(
            update(DBAuthor)
            .where(DBAuthor.openalex_id == openalex_id)
            .values(is_conflict=True)
        )
        await self.db.commit()
        count = await self.db.scalar(
            select(func.count()).select_from(DBAuthor).where(DBAuthor.openalex_id == openalex_id)
        )
        return int(count or 0)

    async def mark_orcid_conflicts(self, orcid: str) -> int:
        """
        Mark all authors with the same ORCID as conflict.

        Returns:
            Number of affected rows.
        """
        if not orcid:
            return 0

        await self.db.execute(
            update(DBAuthor)
            .where(DBAuthor.orcid == orcid)
            .values(is_conflict=True)
        )
        await self.db.commit()
        count = await self.db.scalar(
            select(func.count()).select_from(DBAuthor).where(DBAuthor.orcid == orcid)
        )
        return int(count or 0)

    async def has_orcid_duplicates(self, orcid: str) -> bool:
        """Check whether more than one author exists with the same ORCID."""
        if not orcid:
            return False

        count = await self.db.scalar(
            select(func.count()).select_from(DBAuthor).where(DBAuthor.orcid == orcid)
        )
        return int(count or 0) > 1

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
        author = await self.get_author(author_id)
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

        existing = await self.get_author(author_id)

        if existing:
            for key, value in author_data.items():
                if hasattr(existing, key) and value is not None:
                    setattr(existing, key, value)

            await self.db.commit()
            await self.db.refresh(existing)

            # If this update creates/maintains ORCID duplication, mark conflicts.
            existing_orcid = getattr(existing, "orcid", None)
            if existing_orcid and await self.has_orcid_duplicates(existing_orcid):
                await self.mark_orcid_conflicts(existing_orcid)

            return existing

        created = await self.create_author(author_data)

        created_orcid = author_data.get("orcid")
        if created_orcid and await self.has_orcid_duplicates(created_orcid):
            await self.mark_orcid_conflicts(created_orcid)

        return created

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
        stmt = insert(DBAuthorInstitution).values(
            author_id=author_id,
            institution_id=institution_id,
            start_year=year,
            end_year=year,
            is_current=is_current,
            paper_count=1,
        ).on_conflict_do_update(
            index_elements=["author_id", "institution_id"],
            set_={
                "paper_count": DBAuthorInstitution.paper_count + 1,
                "start_year": func.least(DBAuthorInstitution.start_year, year),
                "end_year": func.greatest(DBAuthorInstitution.end_year, year),
                "is_current": DBAuthorInstitution.is_current | is_current,
            },
        ).returning(DBAuthorInstitution)

        result = await self.db.execute(stmt)
        await self.db.commit()
        
        db_author_institution = result.scalar_one()
        
        logger.debug(f"Upserted author-institution link: author_id={author_id}, institution_id={institution_id}")
        return db_author_institution

    async def get_author_papers_with_metadata(self, author_id: str) -> List:
        """
        Get all papers for an author with full metadata (journals, institutions).

        Args:
            author_id: Author identifier

        Returns:
            List of DBPaper objects with relationships loaded
        """

        author = await self.get_author(author_id)
        if not author:
            return []

        stmt = (
            select(DBPaper)
            .join(DBAuthorPaper, DBPaper.id == DBAuthorPaper.paper_id)
            .where(DBAuthorPaper.author_id == author.id)
            .options(
                joinedload(DBPaper.journal), 
                selectinload(DBPaper.authors).selectinload(DBAuthorPaper.author)
            )
            .order_by(DBPaper.publication_date.desc().nulls_last())
        )

        result = await self.db.execute(stmt)
        papers = result.unique().scalars().all()

        logger.debug(f"Found {len(papers)} papers for author {author_id}")
        return list(papers)

    async def get_author_papers_with_metadata_paginated(
        self,
        author_id: str,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "year",
        sort_order: str = "desc",
    ) -> tuple[List[DBPaper], int]:
        """
        Get paginated papers for an author with sorting.

        Args:
            author_id: Author identifier
            limit: Page size
            offset: Pagination offset
            sort_by: "year" or "citation"
            sort_order: "asc" or "desc"

        Returns:
            Tuple of (papers, total)
        """
        author = await self.get_author(author_id)
        if not author:
            return [], 0

        base_query = (
            select(DBPaper)
            .join(DBAuthorPaper, DBPaper.id == DBAuthorPaper.paper_id)
            .where(DBAuthorPaper.author_id == author.id)
            .options(
                joinedload(DBPaper.journal),
                selectinload(DBPaper.authors).selectinload(DBAuthorPaper.author),
            )
        )

        count_stmt = select(func.count()).select_from(base_query.subquery())
        total = int(await self.db.scalar(count_stmt) or 0)

        sort_key = (sort_by or "year").lower()
        sort_dir = (sort_order or "desc").lower()
        descending = sort_dir != "asc"

        if sort_key == "citation":
            order_col = DBPaper.citation_count
        else:
            order_col = DBPaper.publication_date

        ordered_query = base_query.order_by(
            order_col.desc().nulls_last() if descending else order_col.asc().nulls_first(),
            DBPaper.created_at.desc(),
        )

        result = await self.db.execute(ordered_query.offset(offset).limit(limit))
        papers = list(result.unique().scalars().all())
        return papers, total

    async def get_author_paper_links(self, author_id: str) -> List[DBAuthorPaper]:
        """
        Get all author-paper relationship records for an author.
        Includes position, corresponding status, institution links.

        Args:
            author_id: Author identifier

        Returns:
            List of DBAuthorPaper objects
        """
        author = await self.get_author(author_id)
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

        author = await self.get_author(author_id)
        if not author:
            return {}

        stmt = (
            select(DBJournal.sjr_best_quartile, func.count(DBPaper.id))
            .select_from(DBAuthorPaper)
            .join(DBPaper, DBAuthorPaper.paper_id == DBPaper.id)
            .outerjoin(DBJournal, DBPaper.journal_id == DBJournal.id)
            .where(DBAuthorPaper.author_id == author.id)
            .group_by(DBJournal.sjr_best_quartile)
        )

        result = await self.db.execute(stmt)
        rows = result.all()

        quartiles = {"q1": 0, "q2": 0, "q3": 0, "q4": 0, "unknown": 0}
        for quartile, count in rows:
            if isinstance(quartile, str) and quartile.upper() in {"Q1", "Q2", "Q3", "Q4"}:
                quartiles[quartile.lower()] = int(count)
            else:
                quartiles["unknown"] += int(count)

        return quartiles

    async def get_counts_by_year(self, author_id: str) -> Dict[int, Dict[str, int]]:
        """
        Get counts of papers and citations by year for an author.

        Returns:
            Dict like {2024: {"papers": 12, "citations": 5}, ...}
        """
        author = await self.get_author(author_id)
        if not author:
            return {}

        year_expr = func.coalesce(DBPaper.year, cast(func.extract("year", DBPaper.publication_date), Integer))

        stmt = (
            select(
                year_expr.label("year"),
                func.count(DBPaper.id).label("papers"),
            )
            .select_from(DBAuthorPaper)
            .join(DBPaper, DBAuthorPaper.paper_id == DBPaper.id)
            .where(DBAuthorPaper.author_id == author.id)
            .group_by(year_expr)
            .order_by(year_expr.desc())
        )

        result = await self.db.execute(stmt)
        counts_by_year: Dict[int, Dict[str, int]] = {}
        for year, papers in result.all():
            if year is None:
                continue
            counts_by_year[int(year)] = {
                "papers": int(papers or 0),
                "citations": 0, 
            }
            
        stmt = (
            select(
                DBCitation.citation_year.label("year"),
                func.count(DBCitation.id).label("citations"),
            )
            .select_from(DBAuthorPaper)
            .join(DBPaper, DBAuthorPaper.paper_id == DBPaper.id)
            .join(DBCitation, DBCitation.cited_paper_id == DBPaper.id)
            .where(DBAuthorPaper.author_id == author.id)
            .where(DBCitation.citation_year.isnot(None))
            .group_by(DBCitation.citation_year)
            .order_by(DBCitation.citation_year.desc())
        )

        result = await self.db.execute(stmt)
        
        for year, citations in result.all():
            if year is None:
                continue
            year = int(year)
            if year not in counts_by_year:
                counts_by_year[year] = {
                    "papers": 0,
                    "citations": 0,
                }
            counts_by_year[year]["citations"] = int(citations or 0)
    
        return counts_by_year

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
        author = await self.get_author(author_id)
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
                DBAuthor.last_paper_indexed_at.isnot(None).label("is_enriched"),
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
                DBAuthor.last_paper_indexed_at,
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
                "is_enriched": bool(row[6]),
                "collaboration_count": row[7],
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
        author = await self.get_author(author_id)
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
                DBAuthor.last_paper_indexed_at.isnot(None).label("is_enriched"),
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
                "is_enriched": bool(row[5]),
                "citation_count": row[6]
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
        author = await self.get_author(author_id)
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
                DBAuthor.last_paper_indexed_at.isnot(None).label("is_enriched"),
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
                "is_enriched": bool(row[5]),
                "reference_count": row[6]
            }
            for row in rows
        ]
        
        return referenced_authors, total
    
    async def delete_author_relationships(self, author_id: int) -> None:
        """Delete existing relationships for an author"""
        from app.models.author_relationships import DBAuthorRelationship
        await self.db.execute(
            delete(DBAuthorRelationship).where(
                DBAuthorRelationship.author_id == author_id
            )
        )
        await self.db.commit()

    async def bulk_insert_author_relationships(self, relationships: list) -> None:
        """Bulk insert author relationships"""
        if relationships:
            self.db.add_all(relationships)
            await self.db.commit()
    
    async def get_author_statistics(self) -> dict:
        """
        Get comprehensive author statistics.
        
        Returns:
            Dict with statistics about all authors
        """
        total = await self.db.scalar(select(func.count()).select_from(DBAuthor))
        verified = await self.db.scalar(
            select(func.count()).select_from(DBAuthor).where(DBAuthor.verified == True)
        )
        with_orcid = await self.db.scalar(
            select(func.count()).select_from(DBAuthor).where(DBAuthor.orcid.isnot(None))
        )
        with_retracted = await self.db.scalar(
            select(func.count()).select_from(DBAuthor).where(DBAuthor.has_retracted_papers == True)
        )
        avg_h_index = await self.db.scalar(
            select(func.avg(DBAuthor.h_index)).select_from(DBAuthor).where(DBAuthor.h_index.isnot(None))
        )
        avg_citations = await self.db.scalar(
            select(func.avg(DBAuthor.total_citations)).select_from(DBAuthor).where(DBAuthor.total_citations.isnot(None))
        )
        
        return {
            "total_authors": total or 0,
            "verified_authors": verified or 0,
            "with_orcid": with_orcid or 0,
            "with_retracted_papers": with_retracted or 0,
            "average_h_index": float(avg_h_index) if avg_h_index else None,
            "average_citations": float(avg_citations) if avg_citations else None
        }
    
    async def list_authors(
        self,
        page: int,
        page_size: int,
        search: Optional[str] = None,
        verified_only: bool = False
    ) -> tuple[list[DBAuthor], int]:
        """
        List authors with pagination and filters.
        
        Args:
            page: Page number (1-indexed)
            page_size: Items per page
            search: Optional search term
            verified_only: Filter for verified authors only
            
        Returns:
            Tuple of (authors list, total count)
        """
        query = select(DBAuthor)
        
        if search:
            query = query.where(
                DBAuthor.name.ilike(f"%{search}%") | 
                DBAuthor.display_name.ilike(f"%{search}%")
            )
        
        if verified_only:
            query = query.where(DBAuthor.verified == True)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = await self.db.scalar(count_query) or 0
        
        # Apply pagination
        query = query.offset((page - 1) * page_size).limit(page_size)
        query = query.order_by(DBAuthor.total_citations.desc().nulls_last())
        query = query.options(selectinload(DBAuthor.author_institutions).joinedload(DBAuthorInstitution.institution))
        
        result = await self.db.execute(query)
        authors = result.unique().scalars().all()
        
        return list(authors), total
