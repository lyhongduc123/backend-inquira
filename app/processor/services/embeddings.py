import asyncio
from typing import Any, List, Optional

import httpx
from openai import AsyncOpenAI
import ollama
from app.core.config import settings
from app.core.exceptions import ServiceUnavailableException
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using OpenAI, Ollama, or Nomic (Singleton)."""
    
    _instance: Optional['EmbeddingService'] = None
    _initialized: bool = False
    
    def __new__(cls, provider: Optional[str] = None):
        """Singleton pattern: always return the same instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize embedding service (only once due to singleton)
        
        Args:
            provider: "openai", "ollama", or "nomic". If None, uses settings.EMBEDDING_PROVIDER
        """
        if self.__class__._initialized:
            return
            
        self.provider = provider or getattr(settings, 'EMBEDDING_PROVIDER', 'openai').lower()
        self.ollama_client = None
        self.openai_client = None
        self.nomic_base_url = "https://api-atlas.nomic.ai"

        if self.provider == "ollama":
            self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            self.model = settings.OLLAMA_EMBEDDING_MODEL
            self.dimension = 768 
            logger.info(f"Initialized Ollama embedding service with model: {self.model}")
        elif self.provider == "openai":
            self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = "text-embedding-ada-002"
            self.dimension = 1536
            logger.info(f"Initialized OpenAI embedding service with model: {self.model}")
        elif self.provider == "nomic":
            self.model = settings.EMBEDDING_MODEL_NAME or "nomic-embed-text-v1.5"
            self.dimension = 768
            logger.info(f"Initialized Nomic embedding service with model: {self.model}")
        else:
            raise ServiceUnavailableException(f"Unsupported embedding provider: {self.provider}")
        
        self.__class__._initialized = True

    def _build_task_texts(self, texts: List[str], task: str) -> List[str]:
        if self.provider == "ollama" and "nomic" in self.model.lower():
            return [f"{task}: {text}" for text in texts]
        return texts

    async def __embed(self, texts: List[str], task: str) -> List[List[float]]:
        prefixed_texts = self._build_task_texts(texts, task)

        if self.provider == "ollama" and self.ollama_client:
            ollama_client = self.ollama_client
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama_client.embed(model=self.model, input=prefixed_texts),
            )
            embeddings = response.get("embeddings", [])
            if not isinstance(embeddings, list):
                raise ServiceUnavailableException("Invalid Ollama embeddings response format")
            return embeddings

        if self.provider == "openai" and self.openai_client:
            response = await self.openai_client.embeddings.create(
                model=self.model,
                input=prefixed_texts,
            )
            return [item.embedding for item in response.data]

        if self.provider == "nomic":
            headers = {
                "Authorization": f"Bearer {settings.NOMIC_API_KEY}",
                "Content-Type": "application/json",
            }
            payload: dict[str, Any] = {
                "texts": prefixed_texts,
                "model": self.model,
                "task_type": task,
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.nomic_base_url}/v1/embedding/text",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            embeddings = data.get("embeddings", [])
            if not isinstance(embeddings, list):
                raise ServiceUnavailableException("Invalid Nomic embeddings response format")
            return embeddings

        raise ServiceUnavailableException(f"No valid client initialized for provider: {self.provider}")
    
    async def create_embedding(self, text: str, task: str = "search_document") -> List[float]:
        """
        Create embedding for a single text with task-aware prefix.
        
        For nomic-embed-text, prefixes improve retrieval quality:
        - 'search_query:' for user queries (asymmetric search)
        - 'search_document:' for document chunks (default)
        - 'clustering:' for clustering tasks
        - 'classification:' for classification tasks
        
        Args:
            text: Text to embed
            task: Task type ('search_query', 'search_document', 'clustering', 'classification')
            
        Returns:
            Embedding vector
            
        Raises:
            ServiceUnavailableException: If embedding service fails
        """
        try:
            batch_embeddings = await self.__embed([text], task=task)
            if not batch_embeddings:
                raise ServiceUnavailableException("Empty embedding response")
            return batch_embeddings[0]
            
        except ServiceUnavailableException:
            raise
        except Exception as e:
            error_msg = f"Error creating embedding with {self.provider}: {str(e)}"
            logger.error(error_msg)
            raise ServiceUnavailableException(error_msg)
    
    async def create_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 5,
        task: str = "search_document"
    ) -> List[List[float]]:
        """
        Create embeddings for multiple texts in batches with task-aware prefix.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (keep small to avoid OOM)
            task: Task type ('search_query', 'search_document', 'clustering', 'classification')
            
        Returns:
            List of embedding vectors
            
        Raises:
            ServiceUnavailableException: If embedding service fails
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                batch_embeddings = await self.__embed(batch, task=task)
                embeddings.extend(batch_embeddings)
                logger.info(
                    f"Created embeddings for batch {i // batch_size + 1} ({len(batch)} texts)"
                )
                
            except Exception as e:
                error_msg = f"Error creating embeddings for batch {i // batch_size + 1}: {str(e)}"
                logger.error(error_msg)
                raise ServiceUnavailableException(error_msg)
        
        return embeddings
    
    async def create_embeddings_parallel(
        self,
        texts: List[str],
        max_concurrent: int = 5
    ) -> List[List[float]]:
        """
        Create embeddings for multiple texts in parallel
        
        Args:
            texts: List of texts to embed
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of embedding vectors
            
        Raises:
            ServiceUnavailableException: If any embedding fails
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def create_with_semaphore(text: str) -> List[float]:
            async with semaphore:
                return await self.create_embedding(text)
        
        tasks = [create_with_semaphore(text) for text in texts]
        
        try:
            embeddings = await asyncio.gather(*tasks)
            return embeddings
        except Exception as e:
            error_msg = f"Error in parallel embedding: {str(e)}"
            logger.error(error_msg)
            raise ServiceUnavailableException(error_msg)
    
    async def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings for the current model"""
        return self.dimension


_embedding_service_instance: Optional[EmbeddingService] = None

def get_embedding_service(provider: Optional[str] = None) -> EmbeddingService:
    """
    Get the singleton EmbeddingService instance.
    Recommended over direct instantiation.
    
    Args:
        provider: Optional provider override (only used on first call)
        
    Returns:
        EmbeddingService singleton instance
    """
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService(provider)
    return _embedding_service_instance