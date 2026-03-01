"""Paper service for business logic"""

from typing import List, Optional, Dict, Union, TYPE_CHECKING
from app.papers.schemas import (
    PaperUpdateRequest,
    PaperDetailResponse,
)
from app.models.papers import DBPaper
from app.core.dtos.paper import PaperDTO, PaperEnrichedDTO
from app.extensions.logger import create_logger
from app.extensions.bibliography import bibtex_to_multiple_styles
from sqlalchemy import select

if TYPE_CHECKING:
    from app.papers.repository import PaperRepository, LoadOptions

logger = create_logger(__name__)


class PaperService:
    """Service for paper operations"""

    def __init__(self, repository: "PaperRepository"):
        self.repository = repository
    
    async def batch_check_existing_papers(self, paper_ids: List[str]) -> Dict[str, bool]:
        """
        Efficiently check which papers already exist in database.
        Reduces N queries to 1 query.
        
        Args:
            paper_ids: List of paper IDs to check
        
        Returns:
            Dict mapping paper_id -> exists (True/False)
        """
        if not paper_ids:
            return {}
        
        stmt = select(DBPaper.paper_id).where(DBPaper.paper_id.in_(paper_ids))
        result = await self.repository.db.execute(stmt)
        existing_ids = set(result.scalars().all())
        
        return {pid: pid in existing_ids for pid in paper_ids}
    
    async def batch_check_processed_papers(self, paper_ids: List[str]) -> Dict[str, bool]:
        """
        Efficiently check which papers are already processed.
        Reduces N queries to 1 query.
        
        Args:
            paper_ids: List of paper IDs to check
        
        Returns:
            Dict mapping paper_id -> is_processed (True/False)
        """
        if not paper_ids:
            return {}
        
        stmt = select(DBPaper.paper_id, DBPaper.is_processed).where(
            DBPaper.paper_id.in_(paper_ids)
        )
        result = await self.repository.db.execute(stmt)
        processed_map = {row[0]: row[1] for row in result.all()}
        
        # Return False for papers that don't exist yet
        return {pid: processed_map.get(pid, False) for pid in paper_ids}

    async def get_paper(
        self, paper_id: str, load_options: Optional["LoadOptions"] = None
    ) -> Optional[PaperDetailResponse]:
        """Get a single paper by paper_id"""
        paper = await self.repository.get_single_paper(
            paper_id, load_options=load_options
        )
        if not paper:
            return None
        return PaperDetailResponse.model_validate(paper)

    async def get_paper_by_external_ids(
        self, external_ids: dict, source: str
    ) -> Optional[DBPaper]:
        """
        Check if paper exists in database by external IDs and source.
        Returns raw DBPaper model for repository-level operations.

        Args:
            external_ids: Dictionary of external identifiers (DOI, ArXiv, etc.)
            source: Source name (e.g., 'SemanticScholar', 'OpenAlex')

        Returns:
            DBPaper if exists, None otherwise
        """
        return await self.repository.get_paper_by_external_ids(external_ids, source)

    async def get_paper_by_db_id(self, id: int) -> Optional[PaperDetailResponse]:
        """Get a single paper by database ID"""
        paper = await self.repository.get_paper_by_db_id(id)
        if not paper:
            return None
        return PaperDetailResponse.model_validate(paper)

    async def list_papers(
        self,
        page: int = 1,
        page_size: int = 20,
        processed_only: bool = False,
        source: Optional[str] = None,
        load_options: Optional["LoadOptions"] = None,
    ) -> tuple[List[PaperDetailResponse], int]:
        """List papers with pagination"""
        skip = (page - 1) * page_size
        papers, total = await self.repository.get_papers(
            skip=skip,
            limit=page_size,
            processed_only=processed_only,
            source=source,
            load_options=load_options,
        )

        paper_summaries = [PaperDetailResponse.model_validate(p) for p in papers]
        return paper_summaries, total

    async def update_paper(
        self, paper_id: str, update_data: PaperUpdateRequest
    ) -> Optional[PaperDetailResponse]:
        """Update a paper"""
        # Filter out None values
        data = update_data.model_dump(exclude_unset=True)
        if not data:
            return await self.get_paper(paper_id)

        paper = await self.repository.update_paper(paper_id, data)
        if not paper:
            return None

        return PaperDetailResponse.model_validate(paper)

    async def delete_paper(self, paper_id: str) -> bool:
        """
        Delete a paper.
        Note: Chunks should be deleted separately via ChunkService before calling this.
        """
        return await self.repository.delete_paper(paper_id)

    async def create_paper_from_schema(
        self, paper: Union[PaperDTO, PaperEnrichedDTO],
        defer_enrichment: bool = False
    ) -> Optional[DBPaper]:
        """
        Create paper from Paper DTO using ON CONFLICT DO NOTHING.
        Handles DTO-to-DBPaper model transformation and enrichment orchestration.

        Enrichment steps:
        1. Save paper to database
        2. Enrich with authors and institutions (can be deferred)
        3. Enrich with journal data (can be deferred)

        Args:
            paper: PaperDTO or PaperEnrichedDTO from retriever
            defer_enrichment: If True, skip author/journal enrichment (for batch operations)

        Returns:
            Created DBPaper object, or None if paper already exists
        """

        try:
            create_schema = self._dto_to_model(paper)
            db_paper = await self.repository.save_paper(create_schema)

            if db_paper and not defer_enrichment:
                if paper.authors:
                    try:
                        authors_dict: List[Dict] = [
                            author.model_dump() if not isinstance(author, dict) else author
                            for author in paper.authors
                        ]
                        await self.repository.enrich_paper_with_authors_institutions(
                            db_paper, authors_dict
                        )
                    except Exception as e:
                        logger.error(f"Failed to enrich paper {paper.paper_id} with authors: {e}")
                else:
                    logger.warning(f"No authors found for paper {paper.paper_id} during enrichment")
                
                # Enrich with journal data
                try:
                    await self.repository.enrich_paper_with_journal(db_paper)
                except Exception as e:
                    logger.error(f"Failed to enrich paper {paper.paper_id} with journal: {e}")

            return db_paper

        except Exception as e:
            logger.error(f"Error creating paper {paper.paper_id}: {e}")
            try:
                await self.repository.rollback()
            except Exception as rollback_error:
                logger.error(f"Failed to rollback transaction: {rollback_error}")
            raise e

    def _dto_to_model(self, paper: Union[PaperDTO, PaperEnrichedDTO]):
        """
        Transform Paper DTO to DBPaper model.
        Business logic for data transformation.

        Args:
            paper: PaperDTO or PaperEnrichedDTO

        Returns:
            DBPaper model instance
        """
        if paper.citation_styles is not None:
            bibtex = paper.citation_styles.get("bibtex", None)
            if bibtex:
                try:
                    res = bibtex_to_multiple_styles(bibtex)
                    paper.citation_styles = {
                        **paper.citation_styles,
                        **res
                    }
                except Exception as e:
                    logger.error(f"Failed to convert BibTeX for paper {paper.paper_id}: {e}")
                    pass
        create_schema = PaperDTO.model_dump(
            paper, 
            exclude={"id", "authors", "created_at", "updated_at", "last_accessed_at"}
        )

        if create_schema.get("citation_count") is None:
            create_schema["citation_count"] = 0
        if create_schema.get("influential_citation_count") is None:
            create_schema["influential_citation_count"] = 0
        if create_schema.get("reference_count") is None:
            create_schema["reference_count"] = 0
        
        return create_schema

    async def update_processing_status(
        self, paper_id: str, status: str, error: Optional[str] = None
    ):
        """Update paper processing status"""
        await self.repository.update_paper_processing_status(paper_id, status, error)

    async def search_similar_papers(
        self, query_embedding: List[float], limit: int = 10
    ) -> List[DBPaper]:
        """
        Search for similar papers using summary embeddings.

        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return

        Returns:
            List of similar papers ordered by similarity
        """
        return await self.repository.search_similar_papers(query_embedding, limit)
    
    async def batch_create_papers_from_schema(
        self,
        papers: List[Union[PaperDTO, PaperEnrichedDTO]],
        enrich: bool = True
    ) -> List[DBPaper]:
        """
        Batch create multiple papers efficiently with optional enrichment.
        
        This method optimizes paper creation by:
        1. Batch checking existing papers (N queries -> 1 query)
        2. Batch saving papers (N queries -> 1 query)
        3. Batch enriching authors/institutions/journals (N*M queries -> 3 queries)
        
        Args:
            papers: List of PaperDTO or PaperEnrichedDTO from retriever
            enrich: If True, enrich with authors/institutions/journals
        
        Returns:
            List of created DBPaper objects (newly created only)
        """
        if not papers:
            return []
        
        try:
            # Step 1: Batch check which papers already exist
            paper_ids = [p.paper_id for p in papers]
            existing_map = await self.batch_check_existing_papers(paper_ids)
            
            # Filter out existing papers
            new_papers = [p for p in papers if not existing_map.get(p.paper_id, False)]
            
            if not new_papers:
                logger.info(f"All {len(papers)} papers already exist, skipping")
                return []
            
            logger.info(f"Creating {len(new_papers)} new papers (skipped {len(papers) - len(new_papers)} existing)")
            
            # Step 2: Batch save all papers without enrichment
            created_papers = []
            papers_with_metadata = []  # For batch enrichment later
            
            for paper in new_papers:
                try:
                    # Create paper without enrichment
                    db_paper = await self.create_paper_from_schema(paper, defer_enrichment=True)
                    if db_paper:
                        created_papers.append(db_paper)
                        
                        # Store for batch enrichment if requested
                        if enrich and hasattr(paper, 'authors') and paper.authors:
                            authors_dict = [
                                author.model_dump() if not isinstance(author, dict) else author
                                for author in paper.authors
                            ]
                            papers_with_metadata.append((db_paper, authors_dict))
                        
                except Exception as e:
                    logger.error(f"Failed to create paper {paper.paper_id}: {e}")
                    continue
            
            # Step 3: Batch enrich authors, institutions, and journals
            if enrich and papers_with_metadata:
                stats = await self.repository.batch_enrich_papers_with_authors_journals(papers_with_metadata)
                logger.info(
                    f"Batch enrichment stats: {stats['authors']} authors, "
                    f"{stats['institutions']} institutions, {stats['journals']} journals"
                )
            
            logger.info(f"Successfully created {len(created_papers)} papers")
            return created_papers
            
        except Exception as e:
            logger.error(f"Batch paper creation failed: {e}", exc_info=True)
            try:
                await self.repository.rollback()
            except Exception as rollback_error:
                logger.error(f"Failed to rollback: {rollback_error}")
            raise e
