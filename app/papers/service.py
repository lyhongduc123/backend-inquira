"""Paper service for business logic"""
from typing import List, Optional
from app.papers.repository import PaperRepository
from app.papers.schemas import PaperCreate, PaperUpdate, PaperDetail, PaperSummary
from app.models.papers import DBPaper
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class PaperService:
    """Service for paper operations"""
    
    def __init__(self, repository: PaperRepository):
        self.repository = repository
    
    async def get_paper(self, paper_id: str) -> Optional[PaperDetail]:
        """Get a single paper by paper_id"""
        paper = await self.repository.get_paper_by_id(paper_id)
        if not paper:
            return None
        return PaperDetail.model_validate(paper)
    
    async def get_paper_if_exists(
        self,
        external_ids: dict,
        source: str
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
    
    async def get_paper_by_db_id(self, id: int) -> Optional[PaperDetail]:
        """Get a single paper by database ID"""
        paper = await self.repository.get_paper_by_db_id(id)
        if not paper:
            return None
        return PaperDetail.model_validate(paper)
    
    async def list_papers(
        self,
        page: int = 1,
        page_size: int = 20,
        processed_only: bool = False,
        source: Optional[str] = None
    ) -> tuple[List[PaperSummary], int]:
        """List papers with pagination"""
        skip = (page - 1) * page_size
        papers, total = await self.repository.get_papers(
            skip=skip,
            limit=page_size,
            processed_only=processed_only,
            source=source
        )
        
        paper_summaries = [PaperSummary.model_validate(p) for p in papers]
        return paper_summaries, total
    
    async def update_paper(self, paper_id: str, update_data: PaperUpdate) -> Optional[PaperDetail]:
        """Update a paper"""
        # Filter out None values
        data = update_data.model_dump(exclude_unset=True)
        if not data:
            return await self.get_paper(paper_id)
        
        paper = await self.repository.update_paper(paper_id, data)
        if not paper:
            return None
        
        return PaperDetail.model_validate(paper)
    
    async def delete_paper(self, paper_id: str) -> bool:
        """
        Delete a paper.
        Note: Chunks should be deleted separately via ChunkService before calling this.
        """
        return await self.repository.delete_paper(paper_id)
    
    async def create_paper_from_schema(self, paper) -> DBPaper:
        """
        Create paper from Paper schema object.
        Returns raw DBPaper for processor operations.
        
        Args:
            paper: Paper schema object from retriever
            
        Returns:
            Created DBPaper object
        """
        return await self.repository.create_paper(paper)
    
    async def update_processing_status(
        self,
        paper_id: str,
        status: str,
        error: Optional[str] = None
    ):
        """Update paper processing status"""
        await self.repository.update_paper_processing_status(paper_id, status, error)
    
    async def search_similar_papers(
        self,
        query_embedding: List[float],
        limit: int = 10
    ) -> List[DBPaper]:
        """
        Search for similar papers using summary embeddings.
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            
        Returns:
            List of similar papers ordered by similarity
        """
        return await self.repository.search_similar_papers(
            query_embedding, limit
        )
