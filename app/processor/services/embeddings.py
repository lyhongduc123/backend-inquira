import asyncio
from typing import List, Optional, Union
from openai import AsyncOpenAI
import ollama
from app.core.config import settings
from app.core.exceptions import ServiceUnavailableException
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using OpenAI or Ollama (Singleton)"""
    
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
            provider: "openai" or "ollama". If None, uses settings.EMBEDDING_PROVIDER
        """
        if self.__class__._initialized:
            return
            
        self.provider = provider or getattr(settings, 'EMBEDDING_PROVIDER', 'openai').lower()
        
        if self.provider == "ollama":
            self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            self.openai_client = None
            self.model = settings.OLLAMA_EMBEDDING_MODEL
            self.dimension = 768 
            logger.info(f"Initialized Ollama embedding service with model: {self.model}")
        else:
            self.ollama_client = None
            self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = "text-embedding-ada-002"
            self.dimension = 1536
            logger.info(f"Initialized OpenAI embedding service with model: {self.model}")
        
        self.__class__._initialized = True
    
    async def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            ServiceUnavailableException: If embedding service fails
        """
        try:
            if self.provider == "ollama" and self.ollama_client:
                # Ollama embeddings are synchronous, run in executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.ollama_client.embeddings(model=self.model, prompt=text) # type: ignore
                )
                embedding = response['embedding']
                return embedding
            elif self.openai_client:
                # OpenAI async
                response = await self.openai_client.embeddings.create(
                    model=self.model,
                    input=text
                )
                embedding = response.data[0].embedding
                return embedding
            else:
                error_msg = f"No valid client initialized for provider: {self.provider}"
                logger.error(error_msg)
                raise ServiceUnavailableException(error_msg)
            
        except ServiceUnavailableException:
            raise
        except Exception as e:
            error_msg = f"Error creating embedding with {self.provider}: {str(e)}"
            logger.error(error_msg)
            raise ServiceUnavailableException(error_msg)
    
    async def create_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 5  # Reduced from 10 for better memory management
    ) -> List[List[float]]:
        """
        Create embeddings for multiple texts in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (keep small to avoid OOM)
            
        Returns:
            List of embedding vectors
            
        Raises:
            ServiceUnavailableException: If embedding service fails
        """
        if self.provider == "ollama":
            # Ollama supports batch embeddings via embed endpoint
            # Use smaller batches to avoid memory allocation errors
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                try:
                    # Use embed endpoint with multiple inputs
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.ollama_client.embed(model=self.model, input=batch)  # type: ignore
                    )
                    
                    # Response contains 'embeddings' (list of embeddings)   
                    batch_embeddings = response.get('embeddings', [])
                    embeddings.extend(batch_embeddings)
                    
                    logger.info(f"Created {len(embeddings)}/{len(texts)} embeddings with Ollama")
                    
                except Exception as e:
                    error_msg = f"Error creating Ollama embeddings for batch {i // batch_size + 1}: {str(e)}"
                    logger.error(error_msg)
                    raise ServiceUnavailableException(error_msg)
            
            return embeddings
        
        # OpenAI batch processing
        if not self.openai_client:
            error_msg = "OpenAI client not initialized"
            logger.error(error_msg)
            raise ServiceUnavailableException(error_msg)
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = await self.openai_client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Created embeddings for batch {i // batch_size + 1} ({len(batch)} texts)")
                
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
                return await self.create_embedding(text)  # Will raise if fails
        
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