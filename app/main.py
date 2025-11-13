import time
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from app.retriever import RetrievalServiceType, retriever
from app.llm import llm_service
from fastapi.responses import StreamingResponse
import uvicorn
import colorama
colorama.just_fix_windows_console()

from app.db.database import init_db

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

@app.get("/")
def read_root():
    return {"message": "Hello from Exegent!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

def start():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
