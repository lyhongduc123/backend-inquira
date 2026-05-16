"""
Test router for simulating streaming events
"""

from typing import AsyncGenerator
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.extensions.logger import create_logger
from app.core.db.database import get_db_session
from .pipeline import Pipeline
from .schemas import SearchWorkflowConfig
import json
from fastapi import BackgroundTasks

router = APIRouter(prefix="/debug", tags=["test"])
logger = create_logger(__name__)

@router.get("", response_model=None)
async def pipeline_debug(
    query: str, 
    db_session = Depends(get_db_session)
):
    pipeline = Pipeline(db_session)
    config = SearchWorkflowConfig(query=query)
  
    async def run_pipeline():
        async for event in pipeline.run_database_search_workflow(config):
            yield f"event: pipeline_event\ndata: ok\n\n"
            
    return StreamingResponse(run_pipeline(), media_type="text/event-stream")
 