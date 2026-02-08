import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    DATABASE_SYNC_URL: str
    EMBEDDING_MODEL_NAME: str | None = None
    VECTOR_STORE_PATH: str | None = None

    # External API URLs and Keys
    SEMANTIC_API_URL: str
    SEMANTIC_API_KEY: str
    SCHOLAR_API_URL: str
    ARXIV_API_URL: str
    OPENALEX_API_URL: str
    OPENALEX_API_KEY: str

    # LLMs API Keys
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    OPENROUTER_API_KEY: str
    MISTRALAI_API_KEY: str

    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_PROVIDER: str = "ollama"
    EMBEDDING_PROVIDER: str = "ollama"

    # LLM Prompt Parameters (Environment-based overrides)
    LLM_DEFAULT_TEMPERATURE: float = 0.7
    LLM_FACTUAL_TEMPERATURE: float = 0.3
    LLM_CREATIVE_TEMPERATURE: float = 0.8
    LLM_ANALYTICAL_TEMPERATURE: float = 0.4
    LLM_MAX_TOKENS: int = 4000
    LLM_TOP_P: float = 0.95

    # LLM Model Configuration
    LLM_MODEL: list[str] = [
        "openrouter/openai/gpt-oss-120b:free",
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "openrouter/nousresearch/hermes-3-llama-3.1-405b:free",
        "openrouter/mistralai/mistral-small-3.1-24b-instruct:free",
        "gemini/gemini-2.5-flash-lite",
        "mistral/mistral-large-latest"
    ]

    # JWT Authentication
    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # OAuth Settings
    OAUTH_GOOGLE_CLIENT_ID: str
    OAUTH_GOOGLE_CLIENT_SECRET: str
    OAUTH_GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/google/callback"

    OAUTH_GITHUB_CLIENT_ID: str
    OAUTH_GITHUB_CLIENT_SECRET: str
    OAUTH_GITHUB_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/github/callback"

    FRONTEND_URL: str = "http://localhost:3000"

    LOG_DIR: str = "logs"
    LOG_TO_CONSOLE: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


try:
    settings = Settings()  # type: ignore
except Exception as e:
    raise RuntimeError(f"Configuration error: {e}") from None
