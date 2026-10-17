from unittest.mock import MagicMock, patch

from src.app.schemas.rag import TokenInfo


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["version"] == "0.1.0"
    assert data["qdrant_connected"] is True


def test_auth_login_url(client):
    with patch("src.app.services.google_auth.Flow.from_client_config") as mock_flow_cls:
        mock_flow = MagicMock()
        mock_flow.authorization_url.return_value = (
            "https://accounts.google.com/o/oauth2/auth?client_id=test",
            "state_abc",
        )
        mock_flow_cls.return_value = mock_flow

        res = client.get("/api/v1/auth/login")
        assert res.status_code == 200
        data = res.json()
        assert "authorization_url" in data
        assert data["state"] == "state_abc"


def test_auth_status_unauthenticated(client):
    res = client.get("/api/v1/auth/status?user_id=unknown_user")
    assert res.status_code == 200
    data = res.json()
    assert data["authenticated"] is False


def test_auth_callback_get(client, auth_service):
    with patch.object(auth_service, "exchange_code_for_tokens") as mock_exchange:
        mock_exchange.return_value = TokenInfo(
            user_id="user_123",
            email="test@gmail.com",
            access_token="tok_123",
            refresh_token="ref_123",
            token_uri="https://oauth2.googleapis.com/token",
            client_id="cid",
            client_secret="csec",
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )

        res = client.get("/api/v1/auth/callback?code=mock_code&state=mock_state")
        assert res.status_code == 200
        data = res.json()
        assert data["user_id"] == "user_123"
        assert data["email"] == "test@gmail.com"


def test_drive_list_files(client):
    res = client.get("/api/v1/drive/files")
    assert res.status_code == 200
    data = res.json()
    assert data["total_count"] == 1
    assert data["files"][0]["id"] == "file_abc_123"
    assert data["files"][0]["name"] == "Architecture_Guide.md"


def test_drive_sync_and_rag_flow(client):
    # 1. Sync drive files
    sync_res = client.post("/api/v1/drive/sync", json={"user_id": "test_user"})
    assert sync_res.status_code == 200
    sync_data = sync_res.json()
    assert sync_data["files_processed"] == 1
    assert sync_data["chunks_created"] >= 1

    # 2. Search vectors
    search_res = client.post(
        "/api/v1/rag/search",
        json={"query": "FastAPI and Qdrant integration", "top_k": 3},
    )
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data["total_results"] >= 1
    assert "FastAPI" in search_data["results"][0]["text"]

    # 3. Ask question
    ask_res = client.post(
        "/api/v1/rag/ask",
        json={"question": "What components are integrated?", "top_k": 3},
    )
    assert ask_res.status_code == 200
    ask_data = ask_res.json()
    assert ask_data["question"] == "What components are integrated?"
    assert len(ask_data["citations"]) >= 1
    assert ask_data["citations"][0]["file_name"] == "Architecture_Guide.md"
    assert "FastAPI" in ask_data["answer"]
