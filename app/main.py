from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import colorama
colorama.just_fix_windows_console()

from app.core.db.database import get_db_session, init_db

# For initializing database models
import app.models

# Import routers from domain modules
from app.domain.chat.router import router as chat_router
from app.domain.conversations.router import router as conversations_router
from app.domain.messages.router import router as messages_router
from app.domain.authors.router import router as authors_router
from app.domain.papers.router import router as papers_router
from app.domain.institutions.router import router as institutions_router

# Import routers from other modules
from app.auth import router as auth_router
from app.processor.router import router as preprocessing_router
from app.domain.validation import router as validation_router
from app.domain.bookmarks import router as bookmarks_router
from app.domain.user_settings import router as user_settings_router
from app.rag_pipeline.router import router as rag_pipeline_router

# Import core components for error handling
from app.core.exceptions import BaseApiException
from app.core.responses import error_response, ErrorCode
from app.extensions.middleware import RequestIDMiddleware
from app.extensions.logger import create_logger

from app.core.config import settings

logger = create_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables on startup
    await init_db()
    
    # Initialize background task queue
    from app.workers.task_queue import initialize_task_queue
    task_queue = await initialize_task_queue()
    logger.info("Background task queue initialized")
    
    yield
    
    # Cleanup on shutdown
    await task_queue.stop()
    logger.info("Background task queue stopped")

app = FastAPI(
    title=f"{settings.APP_NAME} API",
    description="AI-powered chatbot and research assistant",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)


app.add_middleware(RequestIDMiddleware)

@app.middleware("http")
async def add_response_headers(request: Request, call_next):
    """Add standard headers to all responses"""
    response = await call_next(request)
    
    # Get request ID from request state
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    # Add standard headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Timestamp"] = datetime.utcnow().isoformat() + "Z"
    
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Timestamp"], 
)

# Exception handlers - HTTP-native format (no wrapper)
@app.exception_handler(BaseApiException)
async def api_exception_handler(request: Request, exc: BaseApiException) -> JSONResponse:
    """Handle custom API exceptions with HTTP-native error response"""
    request_id = getattr(request.state, 'request_id', None)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.code.value,
            "details": exc.details,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        },
        headers={
            "X-Request-ID": request_id or "unknown",
            "X-Error-Code": exc.code.value
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle FastAPI validation errors with HTTP-native format"""
    request_id = getattr(request.state, 'request_id', None)
    errors = exc.errors()
    return JSONResponse(
        status_code=422,
        content={
            "code": ErrorCode.VALIDATION_ERROR.value,
            "details": {"errors": errors},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        },
        headers={
            "X-Request-ID": request_id or "unknown",
            "X-Error-Code": ErrorCode.VALIDATION_ERROR.value
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with HTTP-native format"""
    request_id = getattr(request.state, 'request_id', None)
    logger.error(f"Unhandled exception: {exc}", exc_info=True, extra={"request_id": request_id})
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "code": ErrorCode.INTERNAL_ERROR.value,
            "details": {"type": type(exc).__name__} if logger.level <= 10 else None,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        },
        headers={
            "X-Request-ID": request_id or "unknown",
            "X-Error-Code": ErrorCode.INTERNAL_ERROR.value
        }
    )


# Health check routes
@app.get("/")
def read_root():
    return {"message": f"Hello from {settings.APP_NAME}!", "version": settings.APP_VERSION}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# app.dependency_overrides[get_current_user] = get_fake_user

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
app.include_router(
    messages_router,
    prefix="/api/v1/messages",
    tags=["messages"],
)
app.include_router(
    papers_router,
    prefix="/api/v1/papers",
    tags=["papers"]
)
app.include_router(
    preprocessing_router,
    prefix="/api/v1/admin/preprocessing",
    tags=["admin", "preprocessing"]
)
app.include_router(
    authors_router,
    prefix="/api/v1/authors",
    tags=["authors"]
)
app.include_router(
    institutions_router,
    prefix="/api/v1/institutions",
    tags=["institutions"]
)
app.include_router(
    validation_router,
    prefix="/api/v1/admin/validation",
    tags=["admin", "validation"]
)
app.include_router(
    bookmarks_router,
    prefix="/api/v1/bookmarks",
    tags=["bookmarks"]
)
app.include_router(
    user_settings_router,
    prefix="/api/v1/user/settings",
    tags=["user", "settings"]
)
app.include_router(
    rag_pipeline_router,
    prefix="/api/v1/rag",
    tags=["rag", "debug"]
)

from app.retriever.router import router as retriever_debug_router
app.include_router(retriever_debug_router)

def start():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
