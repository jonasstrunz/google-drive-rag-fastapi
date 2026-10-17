from src.app.schemas.rag import ChunkMetadata
from src.app.services.chunker import TextChunker


def test_text_chunker_basic():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    sample_text = (
        "FastAPI is a modern, fast web framework for building APIs with Python.\n\n"
        "Qdrant is a vector similarity search engine and database.\n\n"
        "Google Drive API lets you integrate with Google Drive storage."
    )
    chunks = chunker.split_text(sample_text)
    assert len(chunks) >= 1
    assert any("FastAPI" in c for c in chunks)
    assert any("Qdrant" in c for c in chunks)


def test_text_chunker_empty():
    chunker = TextChunker()
    assert chunker.split_text("") == []
    assert chunker.split_text("   \n\n  ") == []


def test_embedding_service_mock(mock_embedding_service):
    vec1 = mock_embedding_service.embed_text("Machine learning with vector embeddings")
    vec2 = mock_embedding_service.embed_text("Machine learning with vector embeddings")
    vec3 = mock_embedding_service.embed_text("Something completely different")

    assert len(vec1) == 64
    assert vec1 == vec2  # Deterministic
    assert vec1 != vec3


def test_qdrant_vector_store_operations(in_memory_qdrant_store, mock_embedding_service):
    assert in_memory_qdrant_store.collection_exists() is True
    assert in_memory_qdrant_store.is_connected() is True

    chunks = [
        "FastAPI provides fast performance and automatic OpenAPI documentation.",
        "Qdrant provides scalable vector indexing and payload filtering.",
    ]
    embeddings = mock_embedding_service.embed_batch(chunks)
    metadatas = [
        ChunkMetadata(
            file_id="doc_1",
            file_name="FastAPI_Overview.md",
            mime_type="text/markdown",
            chunk_index=0,
            total_chunks=1,
            char_count=len(chunks[0]),
            web_view_link="https://docs.google.com/fastapi",
        ),
        ChunkMetadata(
            file_id="doc_2",
            file_name="Qdrant_Guide.md",
            mime_type="text/markdown",
            chunk_index=0,
            total_chunks=1,
            char_count=len(chunks[1]),
            web_view_link="https://docs.google.com/qdrant",
        ),
    ]

    # Upsert
    upserted_count = in_memory_qdrant_store.upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    assert upserted_count == 2
    assert in_memory_qdrant_store.count_points() == 2

    # Search
    query_vec = mock_embedding_service.embed_text("FastAPI documentation")
    results = in_memory_qdrant_store.search(query_vector=query_vec, top_k=2)
    assert len(results) == 2
    assert results[0].metadata.file_name in ["FastAPI_Overview.md", "Qdrant_Guide.md"]

    # Filtered search by file_id
    filtered_results = in_memory_qdrant_store.search(
        query_vector=query_vec,
        top_k=2,
        file_ids=["doc_2"],
    )
    assert len(filtered_results) == 1
    assert filtered_results[0].metadata.file_id == "doc_2"

    # Delete by file_id
    in_memory_qdrant_store.delete_by_file_id("doc_1")
    assert in_memory_qdrant_store.count_points() == 1
