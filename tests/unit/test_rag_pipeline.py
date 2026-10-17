from unittest.mock import MagicMock

from src.app.schemas.rag import (
    AskQuestionRequest,
    DriveFileMetadata,
    SearchQuery,
    SyncDriveRequest,
)
from src.app.services.chunker import TextChunker
from src.app.services.rag_pipeline import RAGPipeline


def test_rag_pipeline_sync_and_query(in_memory_qdrant_store, mock_embedding_service):
    # Mock GoogleDriveService
    mock_drive = MagicMock()
    file_1 = DriveFileMetadata(
        id="f1",
        name="Project_Spec.txt",
        mime_type="text/plain",
        size=300,
        web_view_link="https://drive.google.com/f1",
    )
    mock_drive.list_files.return_value = [file_1]
    mock_drive.extract_file_content.return_value = (
        "The project uses FastAPI on the backend and Qdrant as the vector database for RAG search.",
        file_1,
    )

    chunker = TextChunker(chunk_size=100, chunk_overlap=10)

    pipeline = RAGPipeline(
        drive_service=mock_drive,
        vector_store=in_memory_qdrant_store,
        embedding_service=mock_embedding_service,
        chunker=chunker,
    )

    # 1. Sync
    sync_req = SyncDriveRequest(user_id="test_user")
    sync_result = pipeline.sync_google_drive(sync_req)

    assert sync_result.files_processed == 1
    assert sync_result.chunks_created >= 1
    assert len(sync_result.errors) == 0

    # 2. Search
    search_res = pipeline.search(SearchQuery(query="What database is used?", top_k=1))
    assert search_res.total_results == 1
    assert "Qdrant" in search_res.results[0].text
    assert search_res.results[0].metadata.file_name == "Project_Spec.txt"

    # 3. Ask Question (Fallback extractive when no OpenAI key)
    qa_res = pipeline.answer_question(AskQuestionRequest(question="What database is used?"))
    assert qa_res.question == "What database is used?"
    assert len(qa_res.citations) == 1
    assert qa_res.citations[0].file_name == "Project_Spec.txt"
    assert "Qdrant" in qa_res.answer


def test_rag_pipeline_no_results(in_memory_qdrant_store, mock_embedding_service):
    mock_drive = MagicMock()
    pipeline = RAGPipeline(
        drive_service=mock_drive,
        vector_store=in_memory_qdrant_store,
        embedding_service=mock_embedding_service,
    )

    qa_res = pipeline.answer_question(AskQuestionRequest(question="Where is the document?"))
    assert "No relevant documents found" in qa_res.answer
    assert len(qa_res.citations) == 0
