import time
from fastapi import Depends, FastAPI, Request
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

app = FastAPI(
    title="Exegent API",
    description="AI-powered chatbot and research assistant",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables on startup
    await init_db()
    yield

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
