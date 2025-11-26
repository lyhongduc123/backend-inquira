import asyncio
from typing import List, Optional, Union
from openai import AsyncOpenAI
import ollama
from app.core.config import settings
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using OpenAI or Ollama"""
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize embedding service
        
        Args:
            provider: "openai" or "ollama". If None, uses settings.EMBEDDING_PROVIDER
        """
        self.provider = provider or getattr(settings, 'EMBEDDING_PROVIDER', 'openai').lower()
        
        if self.provider == "ollama":
            self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            self.openai_client = None
            self.model = getattr(settings, 'OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
            self.dimension = 768 
            logger.info(f"Initialized Ollama embedding service with model: {self.model}")
        else:
            self.ollama_client = None
            self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = "text-embedding-ada-002"
            self.dimension = 1536
            logger.info(f"Initialized OpenAI embedding service with model: {self.model}")
    
    async def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
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
                logger.error(f"No valid client initialized for provider: {self.provider}")
                return None
            
        except Exception as e:
            logger.error(f"Error creating embedding with {self.provider}: {e}")
            return None
    
    async def create_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[Optional[List[float]]]:
        """
        Create embeddings for multiple texts in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (only used for OpenAI)
            
        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        if self.provider == "ollama":
            # Ollama doesn't support batch processing, process sequentially
            embeddings = []
            for i, text in enumerate(texts):
                embedding = await self.create_embedding(text)
                embeddings.append(embedding)
                if (i + 1) % 10 == 0:
                    logger.info(f"Created {i + 1}/{len(texts)} embeddings with Ollama")
            return embeddings
        
        # OpenAI batch processing
        if not self.openai_client:
            logger.error("OpenAI client not initialized")
            return [None] * len(texts)
        
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
                logger.error(f"Error creating embeddings for batch {i // batch_size + 1}: {e}")
                # Add None for failed embeddings
                embeddings.extend([None] * len(batch))
        
        return embeddings
    
    async def create_embeddings_parallel(
        self,
        texts: List[str],
        max_concurrent: int = 5
    ) -> List[Optional[List[float]]]:
        """
        Create embeddings for multiple texts in parallel
        
        Args:
            texts: List of texts to embed
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def create_with_semaphore(text: str) -> Optional[List[float]]:
            async with semaphore:
                return await self.create_embedding(text)
        
        tasks = [create_with_semaphore(text) for text in texts]
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        result = []
        for emb in embeddings:
            if isinstance(emb, Exception):
                logger.error(f"Error in parallel embedding: {emb}")
                result.append(None)
            else:
                result.append(emb)
        
        return result
    
    async def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings for the current model"""
        return self.dimension