from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Auth Schemas
class AuthURLResponse(BaseModel):
    authorization_url: str
    state: str


class AuthCallbackRequest(BaseModel):
    code: str
    state: Optional[str] = None


class TokenInfo(BaseModel):
    user_id: str
    email: Optional[str] = None
    access_token: str
    refresh_token: Optional[str] = None
    token_uri: str = "https://oauth2.googleapis.com/token"
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    scopes: List[str] = []
    expiry: Optional[datetime] = None


class AuthStatusResponse(BaseModel):
    authenticated: bool
    user_id: Optional[str] = None
    email: Optional[str] = None
    scopes: List[str] = []
    message: str


# Google Drive File Schemas
class DriveFileMetadata(BaseModel):
    id: str
    name: str
    mime_type: str
    size: Optional[int] = None
    modified_time: Optional[datetime] = None
    web_view_link: Optional[str] = None
    icon_link: Optional[str] = None


class DriveFileListResponse(BaseModel):
    files: List[DriveFileMetadata]
    total_count: int


class SyncDriveRequest(BaseModel):
    user_id: str = "default_user"
    folder_id: Optional[str] = None  # None means root / all accessible files
    file_ids: Optional[List[str]] = None  # Specific files to sync
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    force_resync: bool = False


class SyncDriveResult(BaseModel):
    user_id: str
    files_processed: int
    chunks_created: int
    skipped_files: int
    errors: List[str] = []
    synced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# RAG & Search Schemas
class ChunkMetadata(BaseModel):
    file_id: str
    file_name: str
    mime_type: str
    chunk_index: int
    total_chunks: int
    char_count: int
    web_view_link: Optional[str] = None
    extra: Dict[str, Any] = {}


class SearchQuery(BaseModel):
    query: str
    top_k: int = 5
    file_ids: Optional[List[str]] = None
    score_threshold: Optional[float] = None


class SearchResultItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: ChunkMetadata


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    total_results: int


class AskQuestionRequest(BaseModel):
    question: str
    top_k: int = 5
    file_ids: Optional[List[str]] = None
    score_threshold: Optional[float] = None
    stream: bool = False


class SourceCitation(BaseModel):
    file_id: str
    file_name: str
    chunk_index: int
    score: float
    text_snippet: str
    web_view_link: Optional[str] = None


class AskQuestionResponse(BaseModel):
    question: str
    answer: str
    citations: List[SourceCitation]
    confidence_score: Optional[float] = None
    model_used: str


# System / Health
class HealthResponse(BaseModel):
    status: str
    version: str
    qdrant_connected: bool
    qdrant_collection_exists: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
