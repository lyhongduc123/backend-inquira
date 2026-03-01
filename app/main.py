import time
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from datetime import datetime

from app.auth.dependencies import get_current_user, get_fake_user
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
from app.chat.test_router import router as test_router
from app.conversations import router as conversations_router
from app.auth import router as auth_router
from app.papers import router as papers_router
from app.processor.router import router as preprocessing_router
from app.authors.router import router as authors_router
from app.institutions.router import router as institutions_router
from app.validation import router as validation_router
from app.bookmarks import router as bookmarks_router
from app.user_settings import router as user_settings_router

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

# Middleware to add standard response headers
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

# CORS middleware for frontend
from app.core.config import settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],  # Frontend URL from config
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Timestamp"],  # Expose custom headers
)

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

# Exception handlers - HTTP-native format (no wrapper)
@app.exception_handler(BaseApiException)
async def api_exception_handler(request: Request, exc: BaseApiException) -> JSONResponse:
    """Handle custom API exceptions with HTTP-native error response"""
    request_id = getattr(request.state, 'request_id', None)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
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
            "detail": "Validation error",
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


app.router.lifespan_context = lifespan

# Health check routes
@app.get("/")
def read_root():
    return {"message": "Hello from Exegent!", "version": "1.0.0"}

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
    prefix="/api/v1/admin/authors",
    tags=["admin", "authors"]
)
app.include_router(
    institutions_router,
    prefix="/api/v1/admin/institutions",
    tags=["admin", "institutions"]
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
    test_router,
    prefix="/api/v1/chat",
    tags=["test"]
)

def start():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
