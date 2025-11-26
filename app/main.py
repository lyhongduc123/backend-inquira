import time
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from app.retriever import RetrievalServiceType, retriever
from app.llm import llm_service
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import colorama
colorama.just_fix_windows_console()

from app.db.database import init_db
import app.models

# Import routers
from app.chat import router as chat_router
from app.conversations import router as conversations_router

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

# API v1 routes
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

@app.get("/test")
async def test_endpoint():
    return await retriever.search(search_services=[RetrievalServiceType.SEMANTIC], query="machine learning")

def start():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
