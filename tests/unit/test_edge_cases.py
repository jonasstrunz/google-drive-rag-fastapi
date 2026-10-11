from unittest.mock import MagicMock

from src.app.schemas.rag import AskQuestionRequest, ChunkMetadata
from src.app.services.chunker import TextChunker
from src.app.services.embedding import EmbeddingService
from src.app.services.rag_pipeline import RAGPipeline


def test_chunker_long_paragraph():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    # Generate a long text that exceeds chunk_size in a single paragraph
    long_text = "Word " * 100
    chunks = chunker.split_text(long_text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_embedding_service_openai_success(mocker):
    mock_openai = MagicMock()
    mock_item = MagicMock()
    mock_item.embedding = [0.1] * 64
    mock_response = MagicMock()
    mock_response.data = [mock_item]
    mock_openai.embeddings.create.return_value = mock_response

    service = EmbeddingService(
        provider="openai",
        api_key="fake_key",
        vector_size=64,
    )
    service._openai_client = mock_openai

    embeddings = service.embed_batch(["Hello world"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 64
    assert embeddings[0][0] == 0.1


def test_rag_pipeline_with_openai_client(in_memory_qdrant_store, mock_embedding_service, monkeypatch):
    from src.app.core.config import settings
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "fake-openai-key")

    mock_drive = MagicMock()
    pipeline = RAGPipeline(
        drive_service=mock_drive,
        vector_store=in_memory_qdrant_store,
        embedding_service=mock_embedding_service,
    )

    # Seed vector store with a document
    in_memory_qdrant_store.upsert_chunks(
        chunks=["FastAPI is an async framework with automatic OpenAPI docs."],
        embeddings=[mock_embedding_service.embed_text("FastAPI is an async framework.")],
        metadatas=[
            ChunkMetadata(
                file_id="f1",
                file_name="FastAPI.md",
                mime_type="text/markdown",
                chunk_index=0,
                total_chunks=1,
                char_count=50,
                web_view_link="https://docs.fastapi.com",
                extra={},
            )
        ],
    )

    # Mock OpenAI client response for answer_question
    mock_openai = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "FastAPI is an asynchronous web framework for Python."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai.chat.completions.create.return_value = mock_response

    pipeline._openai_client = mock_openai

    res = pipeline.answer_question(AskQuestionRequest(question="What is FastAPI?"))
    assert res.model_used == "gpt-4o-mini"
    assert "FastAPI is an asynchronous" in res.answer
    assert len(res.citations) == 1
