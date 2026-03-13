"""
Paper router for CRUD operations
"""

from fastapi import APIRouter, Query, Depends, Request, Body
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db_session
from app.core.dependencies import get_container
from app.core.container import ServiceContainer
from .repository import LoadOptions
from app.domain.chunks.schemas import ChunkResponse
from .schemas import (
    PaperDetail,
    PaperUpdate,
    PaginatedCitationsResponse,
    PaginatedReferencesResponse,
    ComputeTagsRequest,
)
from app.domain.conversations.service import ConversationService
from app.domain.conversations.schemas import ConversationDetail, ConversationSummary
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
    source: Optional[str] = Query(
        None, description="Filter by source (openalex, semantic, arxiv)"
    ),
    container: ServiceContainer = Depends(get_container),
    current_user: DBUser = Depends(get_current_user),
) -> PaginatedData[PaperDetail]:
    """
    List all papers with pagination

    - **page**: Page number for pagination
    - **page_size**: Number of items per page
    - **processed_only**: Filter to show only processed papers
    - **source**: Filter by paper source
    """
    papers, total = await container.paper_service.list_papers(
        page=page, page_size=page_size, processed_only=processed_only, source=source
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
        has_previous=page > 1,
    )


@router.get("/{paper_id}", response_model=PaperDetail)
async def get_paper(
    request: Request,
    paper_id: str,
    container: ServiceContainer = Depends(get_container),
    current_user: DBUser = Depends(get_current_user),
) -> PaperDetail:
    """
    Get a single paper by paper_id

    - **paper_id**: The paper's unique identifier (e.g., W1234567890, arxiv:1234.5678)
    """
    paper = await container.paper_service.get_paper(
        paper_id, load_options=LoadOptions.all()
    )
    if not paper:
        raise NotFoundException(f"Paper {paper_id} not found")

    await container.paper_repository.update_last_accessed(paper_id)

    return paper


@router.get("/{paper_id}/citations", response_model=PaginatedCitationsResponse)
async def get_paper_citations(
    request: Request,
    paper_id: str,
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=1000, description="Max results per page"),
    fields: Optional[str] = Query(None, description="Comma-separated S2 fields"),
    container: ServiceContainer = Depends(get_container),
    current_user: DBUser = Depends(get_current_user),
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

    citations_data = await container.paper_service.get_paper_citations(
        paper_id, limit=1, offset=0
    )
    from .schemas import PaginatedCitationsResponse

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
    current_user: DBUser = Depends(get_current_user),
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
    references_data = await container.paper_service.get_paper_references(
        paper_id, offset=offset, limit=limit
    )

    from .schemas import PaginatedReferencesResponse

    references_response = PaginatedReferencesResponse(**references_data)

    return references_response


@router.get("/{paper_id}/chunks", response_model=list[ChunkResponse])
async def get_paper_chunks(
    request: Request,
    paper_id: str,
    container: ServiceContainer = Depends(get_container),
    current_user: DBUser = Depends(get_current_user),
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
    current_user: DBUser = Depends(get_current_user),
    container: "ServiceContainer" = Depends(get_container),
) -> Optional[ConversationDetail]:
    """
    Get or check for existing single-paper conversation.

    Returns the active conversation for this user + paper, or null if none exists.

    - **paper_id**: The paper's unique identifier
    """
    # Find conversation for this user + paper
    conversation = await container.conversation_service.get_paper_conversation(
        user_id=current_user.id, paper_id=paper_id
    )

    return conversation


@router.get(
    "/{paper_id}/conversations", response_model=PaginatedData[ConversationSummary]
)
async def list_paper_conversations(
    request: Request,
    paper_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=50, description="Items per page"),
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user),
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
        user_id=current_user.id, paper_id=paper_id, page=page, page_size=page_size
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
        has_previous=page > 1,
    )


@router.post("/{paper_id}/compute-tags")
async def compute_paper_tags(
    request: ComputeTagsRequest, container: ServiceContainer = Depends(get_container)
):
    if request.category == "methodology":
        labels = [
            "Literature Review",
            "Meta-Analysis",
            "Case Study",
            "Empirical Study",
            "Theoretical",
        ]
    elif request.category == "resource":
        labels = ["Open Source Code", "New Dataset", "Survey Results"]
    else:
        labels = ["Artificial Intelligence", "Medicine", "Economics"]

    tags = container.zeroshot_tagger_service.compute_tags(
        request.content, labels, category=request.category
    )

    confident_tags = [t for t in tags if t["confidence"] > 50.0]

    return {
        "paperId": request.paperId,
        "results": confident_tags,
    }
