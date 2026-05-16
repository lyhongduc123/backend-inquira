"""
Paper linking service for handling author and institution relationships.
Separates business logic from repository data access layer.
"""
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, date

from app.domain.papers.repository import PaperRepository
from app.domain.authors.service import AuthorService
from app.domain.institutions import InstitutionService
from app.domain.papers.types import PaperEnrichedDTO
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class PaperLinkingService:
    """
    Service for linking papers with author and institution relationships.
    
    This service handles the business logic for:
    - Creating/updating authors from paper metadata
    - Creating/updating institutions from affiliations
    - Linking authors to papers with position/corresponding metadata
    - Linking authors to institutions
    
    Separates orchestration logic from pure data access (repository).
    """
    
    def __init__(
        self,
        db: AsyncSession,
        paper_repository: Optional[PaperRepository] = None,
        author_service: Optional[AuthorService] = None,
        institution_service: Optional[InstitutionService] = None,
    ):
        """
        Initialize linking service with dependencies.
        
        Args:
            db: Database session
            paper_repository: Optional paper repository (created if not provided)
            author_service: Optional author service (created if not provided)
            institution_service: Optional institution service (created if not provided)
        """
        self.db = db
        self.paper_repository = paper_repository or PaperRepository(db)
        self.author_service = author_service or AuthorService(db)
        self.institution_service = institution_service or InstitutionService(db)

    async def link_authors_and_institutions_for_dbpaper(
        self, db_paper: Any, authors: List[Dict[str, Any]]
    ) -> None:
        """
        Link authors and institutions for an existing DBPaper.

        This method performs the same orchestration previously implemented
        in `PaperService.link_authors_and_institutions` but centralized here
        so all linking logic lives in `PaperLinkingService`.
        """
        if not authors:
            logger.debug(f"No authors data for paper {getattr(db_paper, 'paper_id', '<unknown>')}")
            return

        # Extract publication year for author-institution tracking
        pub_year = None
        if getattr(db_paper, "publication_date", None):
            pd = db_paper.publication_date
            if isinstance(pd, (datetime, date)):
                pub_year = pd.year
            elif isinstance(pd, int):
                pub_year = pd

        for position, author_data in enumerate(authors, start=1):
            # Upsert author from merged author data
            db_author = await self.author_service.ingest_author_profile(author_data)
            if not db_author:
                continue

            # Process institutions for this author
            institutions = author_data.get("institutions", [])
            institution_id = None

            if institutions:
                primary_institution = institutions[0]
                db_institution = await self.institution_service.upsert_from_openalex(
                    primary_institution
                )

                if db_institution:
                    institution_id = db_institution.id

                    # Link author to institution
                    await self.author_service.link_author_to_institution(
                        author=db_author,
                        institution_id=db_institution.id,
                        year=pub_year,
                        is_current=False,
                    )

            # Link author to paper with position
            await self.author_service.link_author_to_paper(
                author=db_author,
                paper_id=db_paper.id,
                author_data=author_data,
                institution_id=institution_id,
                author_position=position,
            )
    
    async def link_authors_and_institutions(
        self, 
        paper_id: int,
        paper_dto: PaperEnrichedDTO
    ) -> bool:
        """
        Link paper with authors and institutions from DTO.
        
        This is business logic that orchestrates multiple repository operations:
        1. Upsert authors from paper metadata
        2. Upsert institutions from affiliations
        3. Create author-paper relationships
        4. Create author-institution relationships
        
        Args:
            paper_id: Database ID of the paper
            paper_dto: Enriched paper DTO with author/institution data
            
        Returns:
            True if linking successful, False otherwise
        """
        if not paper_dto.authors:
            logger.debug(f"Paper {paper_id} has no authors to link")
            return True
        
        try:
            for author_dto in paper_dto.authors:
                author = await self.author_service.ingest_author_profile(
                    author_dto.model_dump()
                )
                
                if not author:
                    logger.warning(f"Failed to upsert author {author_dto.author_id} for paper {paper_id}")
                    continue
                
                # Note: Institution handling would require separate institution data
                # The current AuthorDTO structure doesn't include institution information
                # Institution linking should be handled separately if needed
                
                # Step 2: Link author to paper
                await self.author_service.link_author_to_paper(
                    author=author,
                    paper_id=paper_id,
                    author_data=author_dto.model_dump(),
                    institution_id=None  # No institution data in AuthorDTO
                )
            
            logger.info(f"Successfully linked paper {paper_id} with {len(paper_dto.authors)} authors")
            return True
            
        except Exception as e:
            logger.error(f"Error linking paper {paper_id} with authors/institutions: {e}")
            return False
    
    async def batch_link_authors_and_institutions(
        self,
        papers_data: List[tuple[int, PaperEnrichedDTO]]
    ) -> int:
        """
        Batch link multiple papers with authors and institutions.
        
        Args:
            papers_data: List of tuples (paper_id, paper_dto)
            
        Returns:
            Number of successfully linked papers
        """
        success_count = 0
        
        for paper_id, paper_dto in papers_data:
            try:
                result = await self.link_authors_and_institutions(
                    paper_id=paper_id,
                    paper_dto=paper_dto
                )
                if result:
                    success_count += 1
            except Exception as e:
                logger.error(f"Error batch linking paper {paper_id}: {e}")
                continue
        
        logger.info(f"Batch linked {success_count}/{len(papers_data)} papers with authors")
        return success_count
    
    async def batch_link_citations_references(
        self,
        citation_data: List[tuple[str, List[str]]]
    ) -> int:
        """
        Batch link citations and references between papers in database.
        
        Creates DBCitation records for paper citation relationships.
        
        Args:
            citation_data: List of tuples (citing_paper_id, list of cited_paper_ids)
            
        Returns:
            Number of successfully created citation links
        """
        from app.models.citations import DBCitation
        from sqlalchemy import select
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        
        success_count = 0
        
        try:
            # Get paper ID mappings (paper_id str -> database id)
            paper_ids_str = set()
            for citing_id, cited_ids in citation_data:
                paper_ids_str.add(citing_id)
                paper_ids_str.update(cited_ids)
            
            # Fetch all papers at once
            from app.models.papers import DBPaper
            result = await self.db.execute(
                select(DBPaper.id, DBPaper.paper_id, DBPaper.publication_date)
                .where(DBPaper.paper_id.in_(list(paper_ids_str)))
            )
            papers_map = {row.paper_id: (row.id, row.publication_date) for row in result.all()}
            
            # Prepare citation records
            citation_records = []
            for citing_paper_id, cited_paper_ids in citation_data:
                if citing_paper_id not in papers_map:
                    logger.warning(f"Citing paper {citing_paper_id} not found in database")
                    continue
                
                citing_db_id, citing_date = papers_map[citing_paper_id]
                citation_year = citing_date.year if citing_date else None
                
                for cited_paper_id in cited_paper_ids:
                    if cited_paper_id not in papers_map:
                        continue
                    
                    cited_db_id, cited_date = papers_map[cited_paper_id]
                    
                    # Calculate years since publication
                    years_since = None
                    if citation_year and cited_date:
                        years_since = citation_year - cited_date.year
                    
                    citation_records.append({
                        'citing_paper_id': citing_db_id,
                        'cited_paper_id': cited_db_id,
                        'citation_year': citation_year,
                        'years_since_publication': years_since,
                        'is_self_citation': False,  # Will be computed later if needed
                        'is_influential': False,
                    })
            
            if not citation_records:
                logger.warning("No valid citation records to insert")
                return 0
            
            # Batch insert with ON CONFLICT DO NOTHING (avoid duplicates)
            stmt = (
                pg_insert(DBCitation)
                .values(citation_records)
                .on_conflict_do_nothing(
                    index_elements=['citing_paper_id', 'cited_paper_id']
                )
            )
            
            await self.db.execute(stmt)
            await self.db.commit()
            
            success_count = len(citation_records)
            logger.info(f"Successfully created {success_count} citation links")
            
        except Exception as e:
            logger.error(f"Error batch linking citations: {e}")
            await self.db.rollback()
            return 0
        
        return success_count
