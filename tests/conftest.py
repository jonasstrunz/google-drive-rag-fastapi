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


@pytest.fixture
def client(in_memory_qdrant_store, mock_embedding_service, auth_service):
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from src.app.api.deps import get_auth_service, get_drive_service, get_rag_pipeline
    from src.app.main import create_app
    from src.app.schemas.rag import DriveFileMetadata
    from src.app.services.rag_pipeline import RAGPipeline

    app = create_app()

    mock_drive = MagicMock()
    file_metadata = DriveFileMetadata(
        id="file_abc_123",
        name="Architecture_Guide.md",
        mime_type="text/markdown",
        size=1500,
        web_view_link="https://drive.google.com/architecture",
    )
    mock_drive.list_files.return_value = [file_metadata]
    mock_drive.get_file_metadata.return_value = file_metadata
    mock_drive.extract_file_content.return_value = (
        "Google Drive RAG architecture integrates FastAPI endpoints with Qdrant vector database.",
        file_metadata,
    )

    pipeline = RAGPipeline(
        drive_service=mock_drive,
        vector_store=in_memory_qdrant_store,
        embedding_service=mock_embedding_service,
    )

    app.dependency_overrides[get_auth_service] = lambda: auth_service
    app.dependency_overrides[get_drive_service] = lambda: mock_drive
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    with TestClient(app) as test_client:
        yield test_client
