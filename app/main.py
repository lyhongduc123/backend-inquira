import time
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager

from app.retriever.paper_service import PaperRetrievalService, RetrievalServiceType
from app.retriever.provider import SemanticScholarProvider
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import colorama
colorama.just_fix_windows_console()

from app.db.database import get_db_session, init_db
from sqlalchemy.ext.asyncio import AsyncSession

# For initializing database models
import app.models

# Import routers
from app.chat import router as chat_router
from app.conversations import router as conversations_router
from app.auth import router as auth_router

# Import core components for error handling
from app.core.exceptions import BaseApiException
from app.core.responses import error_response, ErrorCode
from app.extensions.middleware import RequestIDMiddleware
from app.extensions.logger import create_logger

logger = create_logger(__name__)

app = FastAPI(
    title="Exegent API",
    description="AI-powered chatbot and research assistant",
    version="1.0.0"
)

# Add Request ID middleware
app.add_middleware(RequestIDMiddleware)

# CORS middleware for frontend
from app.core.config import settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],  # Frontend URL from config
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables on startup
    await init_db()
    yield

# Exception handlers
@app.exception_handler(BaseApiException)
async def api_exception_handler(request: Request, exc: BaseApiException) -> JSONResponse:
    """Handle custom API exceptions with structured error response"""
    request_id = getattr(request.state, 'request_id', None)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            code=exc.code,
            message=exc.detail,
            details=exc.details,
            request_id=request_id
        ).model_dump(mode='json')
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle FastAPI validation errors"""
    request_id = getattr(request.state, 'request_id', None)
    errors = exc.errors()
    return JSONResponse(
        status_code=422,
        content=error_response(
            code=ErrorCode.VALIDATION_ERROR,
            message="Validation error",
            details={"errors": errors},
            request_id=request_id
        ).model_dump(mode='json')
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""
    request_id = getattr(request.state, 'request_id', None)
    logger.error(f"Unhandled exception: {exc}", exc_info=True, extra={"request_id": request_id})
    return JSONResponse(
        status_code=500,
        content=error_response(
            code=ErrorCode.INTERNAL_ERROR,
            message="Internal server error",
            details={"type": type(exc).__name__} if logger.level <= 10 else None,  # Include type in debug mode
            request_id=request_id
        ).model_dump(mode='json')
    )


app.router.lifespan_context = lifespan

# Health check routes
@app.get("/")
def read_root():
    return {"message": "Hello from Exegent!", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/test")
async def api_test():
    res = await SemanticScholarProvider("test").get_snippet("Walrus Sui definitions")
    return res

@app.get("/test")
async def api_test_retrieval(db: AsyncSession = Depends(get_db_session)):
    service = PaperRetrievalService(db)
    papers = await service.search("Blockchain Technology: Core", limit=5, services=[RetrievalServiceType.SEMANTIC])
    return papers

# API v1 routes
app.include_router(
    auth_router,
    prefix="/api/v1/auth",
    tags=["authentication"]
)
app.include_router(
    chat_router, 
    prefix="/api/v1/chat", 
    tags=["chat"]
)
app.include_router(
    conversations_router, 
    prefix="/api/v1/conversations", 
    tags=["conversations"]
)

def start():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
