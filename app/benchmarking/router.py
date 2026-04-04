"""
Benchmark Router
API endpoints for benchmarking and pipeline comparison.
"""
from fastapi import APIRouter

from app.benchmarking.schemas import (
    ChunkSearchBenchmarkRequest,
    ChunkSearchBenchmarkReport,
)
from app.benchmarking.chunk_benchmark_service import ChunkSearchBenchmarkService

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])


@router.post("/chunk-search/compare", response_model=ChunkSearchBenchmarkReport)
async def compare_chunk_search_modes(
    request: ChunkSearchBenchmarkRequest,
):
    """Run isolated benchmark: screened title+abstract retrieval vs flat chunk search."""
    service = ChunkSearchBenchmarkService()
    return await service.run(request)
