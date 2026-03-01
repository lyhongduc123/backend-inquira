"""
Paper router for CRUD operations
"""
from fastapi import APIRouter, Query, Depends, Request
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db_session
from app.core.dependencies import get_container
from app.core.container import ServiceContainer
from app.papers.repository import LoadOptions
from app.chunks.schemas import ChunkResponse
from app.papers.schemas import (
    PaperDetail,
    PaperUpdate,
    PaginatedCitationsResponse,
    PaginatedReferencesResponse,
)
from app.conversations.service import ConversationService
from app.conversations.schemas import ConversationDetail, ConversationSummary
from app.auth.dependencies import get_current_user
from app.models.users import DBUser
from app.core.responses import PaginatedData
from app.core.exceptions import NotFoundException

router = APIRouter()


@router.get("", response_model=PaginatedData[PaperDetail])
async def list_papers(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    processed_only: bool = Query(False, description="Show only processed papers"),
    source: Optional[str] = Query(None, description="Filter by source (openalex, semantic, arxiv)"),
    container: ServiceContainer = Depends(get_container),
    current_user: DBUser = Depends(get_current_user)
) -> PaginatedData[PaperDetail]:
    """
    List all papers with pagination
    
    - **page**: Page number for pagination
    - **page_size**: Number of items per page
    - **processed_only**: Filter to show only processed papers
    - **source**: Filter by paper source
    """
    papers, total = await container.paper_service.list_papers(
        page=page,
        page_size=page_size,
        processed_only=processed_only,
        source=source
    )
    
    from math import ceil
    total_pages = ceil(total / page_size) if page_size > 0 else 0
    
    return PaginatedData(
        items=papers,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_previous=page > 1
    )


@router.get("/{paper_id}", response_model=PaperDetail)
async def get_paper(
    request: Request,
    paper_id: str,
    container: ServiceContainer = Depends(get_container),
    current_user: DBUser = Depends(get_current_user)
) -> PaperDetail:
    """
    Get a single paper by paper_id
    
    - **paper_id**: The paper's unique identifier (e.g., W1234567890, arxiv:1234.5678)
    """
    paper = await container.paper_service.get_paper(paper_id, load_options=LoadOptions.all())
    if not paper:
        raise NotFoundException(f"Paper {paper_id} not found")
    
    await container.paper_repository.update_last_accessed(paper_id)
    
    return paper


@router.patch("/{paper_id}", response_model=PaperDetail)
async def update_paper(
    request: Request,
    paper_id: str,
    update_data: PaperUpdate,
    container: ServiceContainer = Depends(get_container),
    current_user: DBUser = Depends(get_current_user)
) -> PaperDetail:
    """
    Update a paper's metadata
    
    - **paper_id**: The paper's unique identifier
    - **update_data**: Fields to update (title, abstract, venue, etc.)
    """
    paper = await container.paper_service.update_paper(paper_id, update_data)
    if not paper:
        raise NotFoundException(f"Paper {paper_id} not found")
    
    return paper


@router.delete("/{paper_id}", response_model=dict)
async def delete_paper(
    request: Request,
    paper_id: str,
    container: ServiceContainer = Depends(get_container),
    current_user: DBUser = Depends(get_current_user)
) -> dict:
    """
    Delete a paper and all its chunks
    
    - **paper_id**: The paper's unique identifier
    """
    
    deleted = await container.paper_service.delete_paper(paper_id)
    if not deleted:
        raise NotFoundException(f"Paper {paper_id} not found")
    
    return {"message": f"Paper {paper_id} deleted successfully"}


@router.get("/{paper_id}/citations", response_model=PaginatedCitationsResponse)
async def get_paper_citations(
    request: Request,
    paper_id: str,
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=1000, description="Max results per page"),
    fields: Optional[str] = Query(None, description="Comma-separated S2 fields"),
    container: ServiceContainer = Depends(get_container),
    current_user: DBUser = Depends(get_current_user)
) -> PaginatedCitationsResponse:
    """
    Get papers that cite this paper (live from Semantic Scholar).
    
    - **paper_id**: The paper's unique identifier
    - **offset**: Pagination offset (0-based)
    - **limit**: Number of results (max 1000 per S2 API)
    - **fields**: Optional S2 fields (e.g., "title,authors,abstract,year")
    
    Returns paginated list of citing papers with fresh citation data.
    Includes citation context and whether the citation is influential.
    """
    from app.retriever.provider.semantic_scholar_provider import SemanticScholarProvider
    from app.retriever.provider.base import RetrievalConfig
    from app.core.config import settings

    paper = await container.paper_service.get_paper(paper_id)
    if not paper:
        raise NotFoundException(f"Paper {paper_id} not found")

    provider = SemanticScholarProvider(
        api_url="https://api.semanticscholar.org/graph/v1",
        config=RetrievalConfig()
    )
    
    citations_data = await provider.get_citations(
        paper_id, 
        offset=offset, 
        limit=limit,
        fields=fields
    )
    
    # Filter out citations with null paperId (S2 API sometimes returns these)
    if "data" in citations_data:
        citations_data["data"] = [
            cit for cit in citations_data["data"]
            if cit.get("citingPaper", {}).get("paperId") is not None
        ]
    
    from app.papers.schemas import PaginatedCitationsResponse
    citations_response = PaginatedCitationsResponse(**citations_data)
    
    return citations_response


@router.get("/{paper_id}/references", response_model=PaginatedReferencesResponse)
async def get_paper_references(
    request: Request,
    paper_id: str,
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=1000, description="Max results per page"),
    fields: Optional[str] = Query(None, description="Comma-separated S2 fields"),
    container: ServiceContainer = Depends(get_container),
    current_user: DBUser = Depends(get_current_user)
) -> PaginatedReferencesResponse:
    """
    Get papers referenced by this paper (live from Semantic Scholar).
    
    - **paper_id**: The paper's unique identifier
    - **offset**: Pagination offset (0-based)
    - **limit**: Number of results (max 1000 per S2 API)
    - **fields**: Optional S2 fields (e.g., "title,authors,abstract,year")
    
    Returns paginated list of referenced papers with fresh data.
    Includes citation context and whether the reference is influential.
    """
    from app.retriever.provider.semantic_scholar_provider import SemanticScholarProvider
    from app.retriever.provider.base import RetrievalConfig

    paper = await container.paper_service.get_paper(paper_id)
    if not paper:
        raise NotFoundException(f"Paper {paper_id} not found")

    provider = SemanticScholarProvider(
        api_url="https://api.semanticscholar.org/graph/v1",
        config=RetrievalConfig()
    )
    
    references_data = await provider.get_references(
        paper_id,
        offset=offset,
        limit=limit,
        fields=fields
    )
    
    # Filter out references with null paperId (similar to citations)
    if "data" in references_data:
        references_data["data"] = [
            ref for ref in references_data["data"]
            if ref.get("citedPaper", {}).get("paperId") is not None
        ]
    
    from app.papers.schemas import PaginatedReferencesResponse
    references_response = PaginatedReferencesResponse(**references_data)
    
    return references_response


@router.get("/{paper_id}/chunks", response_model=list[ChunkResponse])
async def get_paper_chunks(
    request: Request,
    paper_id: str,
    container: ServiceContainer = Depends(get_container),
    current_user: DBUser = Depends(get_current_user)
) -> list[ChunkResponse]:
    """
    Get all chunks for a paper
    
    - **paper_id**: The paper's unique identifier
    """
    # Verify paper exists
    paper = await container.paper_service.get_paper(paper_id)
    if not paper:
        raise NotFoundException(f"Paper {paper_id} not found")
    
    # Get chunks using container
    chunks = await container.chunk_service.get_paper_chunks(paper_id)
    
    return chunks


@router.get("/{paper_id}/conversation", response_model=Optional[ConversationDetail])
async def get_paper_conversation(
    request: Request,
    paper_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> Optional[ConversationDetail]:
    """
    Get or check for existing single-paper conversation.
    
    Returns the active conversation for this user + paper, or null if none exists.
    
    - **paper_id**: The paper's unique identifier
    """
    conversation_service = ConversationService(db)
    
    # Find conversation for this user + paper
    conversation = await conversation_service.get_paper_conversation(
        user_id=current_user.id,
        paper_id=paper_id
    )
    
    return conversation


@router.get("/{paper_id}/conversations", response_model=PaginatedData[ConversationSummary])
async def list_paper_conversations(
    request: Request,
    paper_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=50, description="Items per page"),
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> PaginatedData[ConversationSummary]:
    """
    List all conversations for this user about this paper.
    
    Useful if user has multiple deep-dive sessions over time.
    
    - **paper_id**: The paper's unique identifier
    - **page**: Page number for pagination
    - **page_size**: Number of items per page
    """
    conversation_service = ConversationService(db)
    
    conversations, total = await conversation_service.list_paper_conversations(
        user_id=current_user.id,
        paper_id=paper_id,
        page=page,
        page_size=page_size
    )
    
    from math import ceil
    total_pages = ceil(total / page_size) if page_size > 0 else 0
    
    return PaginatedData(
        items=conversations,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_previous=page > 1
    )
