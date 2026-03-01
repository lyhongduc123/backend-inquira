"""
Paper enrichment service for handling author and institution relationships.
Separates business logic from repository data access layer.
"""
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession

from app.papers.repository import PaperRepository
from app.authors.service import AuthorService
from app.institutions.service import InstitutionService
from app.core.dtos.paper import PaperEnrichedDTO
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class PaperEnrichmentService:
    """
    Service for enriching papers with author and institution relationships.
    
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
        Initialize enrichment service with dependencies.
        
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
    
    async def enrich_paper_with_authors_institutions(
        self, 
        paper_id: int,
        paper_dto: PaperEnrichedDTO
    ) -> bool:
        """
        Enrich paper with authors and institutions from DTO.
        
        This is business logic that orchestrates multiple repository operations:
        1. Upsert authors from paper metadata
        2. Upsert institutions from affiliations
        3. Create author-paper relationships
        4. Create author-institution relationships
        
        Args:
            paper_id: Database ID of the paper
            paper_dto: Enriched paper DTO with author/institution data
            
        Returns:
            True if enrichment successful, False otherwise
        """
        if not paper_dto.authors:
            logger.debug(f"Paper {paper_id} has no authors to enrich")
            return True
        
        try:
            for author_dto in paper_dto.authors:
                author = await self.author_service.upsert_from_merged_author(
                    author_dto.model_dump()
                )
                
                if not author:
                    logger.warning(f"Failed to upsert author {author_dto.author_id} for paper {paper_id}")
                    continue
                
                # Note: Institution handling would require separate institution data
                # The current AuthorDTO structure doesn't include institution information
                # Institution enrichment should be handled separately if needed
                
                # Step 2: Link author to paper
                await self.author_service.link_author_to_paper(
                    author=author,
                    paper_id=paper_id,
                    author_data=author_dto.model_dump(),
                    institution_id=None  # No institution data in AuthorDTO
                )
            
            logger.info(f"Successfully enriched paper {paper_id} with {len(paper_dto.authors)} authors")
            return True
            
        except Exception as e:
            logger.error(f"Error enriching paper {paper_id} with authors/institutions: {e}")
            return False
