from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # App settings
    APP_NAME: str = "Google Drive RAG API"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Google OAuth2 Credentials
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/callback"
    GOOGLE_SCOPES: str = (
        "https://www.googleapis.com/auth/drive.readonly,"
        "https://www.googleapis.com/auth/userinfo.email,"
        "https://www.googleapis.com/auth/userinfo.profile"
    )
    TOKEN_STORAGE_PATH: str = "./data/tokens.json"

    # Qdrant Vector Database
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "google_drive_rag"
    QDRANT_VECTOR_SIZE: int = 1536
    QDRANT_IN_MEMORY: bool = False

    # LLM & Embedding Settings
    EMBEDDING_PROVIDER: str = "openai"  # "openai" or "mock"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"

    # RAG & Chunking Configuration
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.5

    @property
    def google_scopes_list(self) -> List[str]:
        return [s.strip() for s in self.GOOGLE_SCOPES.split(",") if s.strip()]


settings = Settings()
