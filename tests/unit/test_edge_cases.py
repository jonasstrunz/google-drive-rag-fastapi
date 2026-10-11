from unittest.mock import MagicMock

from src.app.schemas.rag import AskQuestionRequest, ChunkMetadata
from src.app.services.chunker import TextChunker
from src.app.services.embedding import EmbeddingService
from src.app.services.llm import LLMService
from src.app.services.rag_pipeline import RAGPipeline


def test_chunker_long_paragraph():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
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


def test_embedding_service_openrouter_success(mocker):
    mock_openrouter = MagicMock()
    mock_item = MagicMock()
    mock_item.embedding = [0.2] * 64
    mock_response = MagicMock()
    mock_response.data = [mock_item]
    mock_openrouter.embeddings.create.return_value = mock_response

    service = EmbeddingService(
        provider="openrouter",
        api_key="fake_openrouter_key",
        vector_size=64,
    )
    service._openai_client = mock_openrouter

    embeddings = service.embed_batch(["Hello openrouter"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 64
    assert embeddings[0][0] == 0.2


def test_llm_service_openrouter(monkeypatch):
    from src.app.core.config import settings
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "fake_sk_or")

    llm = LLMService(
        provider="openrouter",
        openrouter_api_key="fake_sk_or",
        openrouter_model="anthropic/claude-3.5-sonnet",
    )

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Claude response from OpenRouter."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    llm._openrouter_client = mock_client

    fake_snippet = MagicMock()
    fake_snippet.metadata.file_name = "doc.txt"
    fake_snippet.metadata.chunk_index = 0
    fake_snippet.metadata.total_chunks = 1
    fake_snippet.text = "Some text"

    answer, model_used = llm.generate_answer("What is this?", [fake_snippet])
    assert answer == "Claude response from OpenRouter."
    assert "openrouter:anthropic/claude-3.5-sonnet" == model_used


def test_rag_pipeline_with_llm_service(in_memory_qdrant_store, mock_embedding_service, monkeypatch):
    from src.app.core.config import settings
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "fake-openai-key")

    mock_drive = MagicMock()
    llm_service = LLMService(openai_api_key="fake-openai-key")

    mock_openai = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "FastAPI is an asynchronous web framework for Python."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai.chat.completions.create.return_value = mock_response
    llm_service._openai_client = mock_openai

    pipeline = RAGPipeline(
        drive_service=mock_drive,
        vector_store=in_memory_qdrant_store,
        embedding_service=mock_embedding_service,
        llm_service=llm_service,
    )

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

    res = pipeline.answer_question(AskQuestionRequest(question="What is FastAPI?"))
    assert "openai:" in res.model_used
    assert "FastAPI is an asynchronous" in res.answer
    assert len(res.citations) == 1
