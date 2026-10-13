import io
import zipfile

from src.app.services.rag_pipeline import RAGPipeline


def test_config_endpoints_get_and_post(client):
    # Test GET config
    get_res = client.get("/api/v1/config/llm")
    assert get_res.status_code == 200
    data = get_res.json()
    assert "llm_provider" in data
    assert "embedding_provider" in data

    # Test POST config update
    post_res = client.post(
        "/api/v1/config/llm",
        json={
            "llm_provider": "openrouter",
            "openrouter_api_key": "sk-or-v1-test",
            "openrouter_chat_model": "anthropic/claude-3.5-sonnet",
            "embedding_provider": "openrouter",
            "openrouter_embedding_model": "openai/text-embedding-3-small",
        },
    )
    assert post_res.status_code == 200
    updated_data = post_res.json()
    assert updated_data["llm_provider"] == "openrouter"
    assert updated_data["has_openrouter_key"] is True


def test_upload_single_file(client, in_memory_qdrant_store, mock_embedding_service):
    pipeline = RAGPipeline(
        vector_store=in_memory_qdrant_store,
        embedding_service=mock_embedding_service,
    )

    from src.app.api.deps import get_rag_pipeline
    from src.app.main import app

    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    try:
        file_content = b"# Architecture Overview\n\nFastAPI connects to Qdrant directly."
        files = {"files": ("architecture.md", file_content, "text/markdown")}

        res = client.post("/api/v1/config/upload/files", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["total_files"] == 1
        assert data["total_chunks"] >= 1
        assert data["results"][0]["filename"] == "architecture.md"
        assert data["results"][0]["status"] == "indexed"

        # Verify it can be searched
        search_res = client.post("/api/v1/rag/search", json={"query": "Architecture Overview"})
        assert search_res.status_code == 200
        sdata = search_res.json()
        assert len(sdata["results"]) >= 1
        assert "FastAPI" in sdata["results"][0]["text"]
    finally:
        app.dependency_overrides.clear()


def test_upload_zip_archive(client, in_memory_qdrant_store, mock_embedding_service):
    pipeline = RAGPipeline(
        vector_store=in_memory_qdrant_store,
        embedding_service=mock_embedding_service,
    )

    from src.app.api.deps import get_rag_pipeline
    from src.app.main import app

    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    try:
        # Create in-memory zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as z:
            z.writestr("notes.txt", "Meeting notes: Discussing Google Drive OAuth2 integration.")
            z.writestr("specs.json", '{"project": "Google Drive RAG", "engine": "Qdrant"}')

        zip_bytes = zip_buffer.getvalue()
        files = {"files": ("bundle.zip", zip_bytes, "application/zip")}

        res = client.post("/api/v1/config/upload/files", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["total_files"] == 2
        assert data["total_chunks"] >= 2
    finally:
        app.dependency_overrides.clear()
