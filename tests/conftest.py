from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient

from src.app.core.config import settings
from src.app.schemas.rag import TokenInfo
from src.app.services.chunker import TextChunker
from src.app.services.embedding import EmbeddingService
from src.app.services.google_auth import GoogleAuthService
from src.app.services.google_drive import GoogleDriveService
from src.app.services.qdrant_store import QdrantVectorStore


@pytest.fixture(autouse=True)
def override_test_settings(monkeypatch, tmp_path):
    tokens_file = tmp_path / "test_tokens.json"
    monkeypatch.setattr(settings, "TOKEN_STORAGE_PATH", str(tokens_file))
    monkeypatch.setattr(settings, "QDRANT_IN_MEMORY", True)
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "mock")
    monkeypatch.setattr(settings, "QDRANT_COLLECTION", "test_collection")
    monkeypatch.setattr(settings, "QDRANT_VECTOR_SIZE", 64)
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "test-client-id")
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setattr(settings, "GOOGLE_REDIRECT_URI", "http://localhost:8000/api/v1/auth/callback")


@pytest.fixture
def mock_token_info():
    return TokenInfo(
        user_id="user_123",
        email="user@example.com",
        access_token="fake_access_token_xyz",
        refresh_token="fake_refresh_token_abc",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="test-client-id",
        client_secret="test-client-secret",
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )


@pytest.fixture
def auth_service(tmp_path, mock_token_info):
    service = GoogleAuthService(
        client_id="test-client-id",
        client_secret="test-client-secret",
        token_storage_path=str(tmp_path / "tokens.json"),
    )
    return service


@pytest.fixture
def in_memory_qdrant_store():
    client = QdrantClient(location=":memory:")
    return QdrantVectorStore(
        client=client,
        collection_name="test_collection",
        vector_size=64,
    )


@pytest.fixture
def mock_embedding_service():
    return EmbeddingService(
        provider="mock",
        vector_size=64,
    )


@pytest.fixture
def text_chunker():
    return TextChunker(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def mock_drive_service(mocker):
    service = GoogleDriveService(drive_resource=MagicMock())
    return service
