"""
FastAPI dependency injection helpers.
Provides container and service dependencies for route handlers.
"""
from fastapi import Depends
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db_session
from app.core.container import ServiceContainer


def get_container(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> ServiceContainer:
    """
    FastAPI dependency for ServiceContainer.
    
    Provides access to all services and repositories through a centralized container.
    Container is request-scoped - created fresh for each request.
    
    Usage in route handlers:
        @router.get("/papers/{paper_id}")
        async def get_paper(
            paper_id: str,
            container: ServiceContainer = Depends(get_container)
        ):
            paper = await container.paper_service.get_paper(paper_id)
            return paper
    
    Benefits:
    - Single point of dependency resolution
    - Lazy initialization (services created only when accessed)
    - Consistent service lifecycle management
    - Easy testing (mock container instead of individual services)
    """
    cached_container = getattr(request.state, "service_container", None)
    if cached_container is not None:
        return cached_container

    container = ServiceContainer(db_session=db)
    request.state.service_container = container
    return container
