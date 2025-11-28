import os
from pydantic_settings import BaseSettings
 

class Settings(BaseSettings):
    DATABASE_URL: str
    EMBEDDING_MODEL_NAME: str | None = None
    VECTOR_STORE_PATH: str | None = None

    SEMANTIC_API_URL: str
    SEMANTIC_API_KEY: str
    SCHOLAR_URL: str
    ARXIV_API_URL: str
    OPENALEX_URL: str

    OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    
    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_PROVIDER: str = "ollama"
    EMBEDDING_PROVIDER: str = "ollama" 

    LOG_DIR: str = "logs"
    LOG_TO_CONSOLE: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"



try:
    settings = Settings()  # type: ignore
except Exception as e:
    raise RuntimeError(f"Configuration error: {e}") from None