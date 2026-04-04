import os
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    DATABASE_SYNC_URL: str
    BEIR_TEST_DATABASE_URL: str | None = None 
    REDIS_URL: str = "redis://localhost:6379/0"
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

    LLM_MAX_TOKENS: int = 4000
    LLM_TOP_P: float = 0.95

    # LLM Model Configuration
    LLM_MODEL: list[str] = [
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-flash-lite",
        "openrouter/openai/gpt-oss-120b:free",
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "openrouter/nousresearch/hermes-3-llama-3.1-405b:free",
        "openrouter/mistralai/mistral-small-3.1-24b-instruct:free",
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

    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: Literal["strict", "lax", "none"] = "lax"
    COOKIE_DOMAIN: str | None = None  
    
    EMAIL_OTP_EXPIRE_MINUTES: int = 5
    EMAIL_OTP_RESEND_COOLDOWN_SECONDS: int = 60
    EMAIL_OTP_MAX_ATTEMPTS: int = 5
    RESEND_API_KEY: str | None = None
    RESEND_FROM_EMAIL: str | None = None
    RESEND_API_URL: str = "https://api.resend.com/emails"

    LOG_DIR: str = "logs"
    LOG_TO_CONSOLE: bool = False
    LOG_LEVEL: str = "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL (default: DEBUG for development)
    
    # CUDA/GPU settings
    USE_CUDA: bool = True 
    DOC_OCR_ENABLED: bool = False
    DOC_ASSETS_DIR: str = "preprocessing_logs/docling_assets"
    DOC_ASSETS_PERSIST_LOCAL: bool = False
    DOC_EXPORT_HIERARCHICAL_CHUNKS: bool = False
    DOC_ENABLE_PYMUPDF_CROPS: bool = True

    R2_ENABLED: bool = True
    R2_ACCOUNT_ID: str | None = None
    R2_ENDPOINT_URL: str | None = None
    R2_ACCESS_KEY_ID: str | None = None
    R2_SECRET_ACCESS_KEY: str | None = None
    R2_BUCKET: str | None = None
    R2_PUBLIC_BASE_URL: str | None = None
    R2_ASSET_PREFIX: str = "docling-assets"
    R2_PDF_PREFIX: str = "source-pdfs"
    
    CF_API_TOKEN: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


try:
    settings = Settings()  # type: ignore
except Exception as e:
    raise RuntimeError(f"Configuration error: {e}") from None
