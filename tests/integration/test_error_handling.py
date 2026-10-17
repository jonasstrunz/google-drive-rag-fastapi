from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.app.api.deps import get_auth_service, get_drive_service, get_rag_pipeline
from src.app.main import create_app
from src.app.services.rag_pipeline import RAGPipeline


@pytest.fixture
def error_test_client(in_memory_qdrant_store, mock_embedding_service, auth_service):
    app = create_app()

    mock_drive = MagicMock()
    rag_pipeline = RAGPipeline(
        drive_service=mock_drive,
        vector_store=in_memory_qdrant_store,
        embedding_service=mock_embedding_service,
    )

    app.dependency_overrides[get_auth_service] = lambda: auth_service
    app.dependency_overrides[get_drive_service] = lambda: mock_drive
    app.dependency_overrides[get_rag_pipeline] = lambda: rag_pipeline

    with TestClient(app) as test_client:
        test_client.mock_drive = mock_drive
        yield test_client


def test_auth_callback_missing_exchange(error_test_client):
    # Test error handling when code is invalid
    res = error_test_client.post(
        "/api/v1/auth/callback",
        json={"code": "bad_code", "state": "bad_state"},
    )
    assert res.status_code == status.HTTP_400_BAD_REQUEST


def test_drive_unauthenticated(error_test_client):
    error_test_client.mock_drive.list_files.side_effect = PermissionError("User is not authenticated with Google Drive.")

    res = error_test_client.get("/api/v1/drive/files?user_id=unauthed_user")
    assert res.status_code == status.HTTP_401_UNAUTHORIZED
    assert "not authenticated" in res.json()["detail"]


def test_drive_file_not_found(error_test_client):
    error_test_client.mock_drive.get_file_metadata.side_effect = Exception("File not found")

    res = error_test_client.get("/api/v1/drive/files/nonexistent_file_id")
    assert res.status_code == status.HTTP_404_NOT_FOUND
