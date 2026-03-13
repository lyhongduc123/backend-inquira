from .pipeline import RAGResult, Pipeline
from .hybrid_pipeline import HybridPipeline
from .database_pipeline import DatabasePipeline

# Backwards compatibility alias
RAGPipeline = Pipeline

__all__ = ["RAGResult", "Pipeline", "RAGPipeline", "HybridPipeline", "DatabasePipeline"]