from typing import List, Optional

from src.app.core.config import settings
from src.app.core.logging import logger
from src.app.schemas.rag import (
    AskQuestionRequest,
    AskQuestionResponse,
    ChunkMetadata,
    SearchQuery,
    SearchResponse,
    SourceCitation,
    SyncDriveRequest,
    SyncDriveResult,
)
from src.app.services.chunker import TextChunker
from src.app.services.embedding import EmbeddingService
from src.app.services.google_drive import GoogleDriveService
from src.app.services.llm import LLMService
from src.app.services.qdrant_store import QdrantVectorStore


class RAGPipeline:
    def __init__(
        self,
        drive_service: Optional[GoogleDriveService] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        llm_service: Optional[LLMService] = None,
        chunker: Optional[TextChunker] = None,
    ):
        self.drive_service = drive_service or GoogleDriveService()
        self.vector_store = vector_store or QdrantVectorStore()
        self.embedding_service = embedding_service or EmbeddingService()
        self.llm_service = llm_service or LLMService()
        self.chunker = chunker or TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    def sync_google_drive(self, request: SyncDriveRequest) -> SyncDriveResult:
        """Syncs and embeds files from Google Drive into Qdrant."""
        result = SyncDriveResult(
            user_id=request.user_id,
            files_processed=0,
            chunks_created=0,
            skipped_files=0,
            errors=[],
        )

        try:
            if request.file_ids:
                # Sync specific files
                files = [self.drive_service.get_file_metadata(fid, user_id=request.user_id) for fid in request.file_ids]
            else:
                # List files in folder or drive
                files = self.drive_service.list_files(
                    user_id=request.user_id,
                    folder_id=request.folder_id,
                )
        except Exception as e:
            logger.error(f"Failed to fetch files from Google Drive: {e}")
            result.errors.append(f"Google Drive access error: {str(e)}")
            return result

        chunk_size = request.chunk_size or settings.CHUNK_SIZE
        chunk_overlap = request.chunk_overlap or settings.CHUNK_OVERLAP
        custom_chunker = (
            TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if (request.chunk_size or request.chunk_overlap)
            else self.chunker
        )

        for file_meta in files:
            try:
                # Extract content
                text_content, meta = self.drive_service.extract_file_content(file_meta.id, user_id=request.user_id)
                if not text_content:
                    result.skipped_files += 1
                    continue

                # Remove existing chunks for this file if resyncing
                self.vector_store.delete_by_file_id(file_meta.id)

                # Chunk text
                chunks = custom_chunker.split_text(text_content)
                if not chunks:
                    result.skipped_files += 1
                    continue

                # Generate embeddings
                embeddings = self.embedding_service.embed_batch(chunks)

                # Prepare metadatas
                metadatas = [
                    ChunkMetadata(
                        file_id=meta.id,
                        file_name=meta.name,
                        mime_type=meta.mime_type,
                        chunk_index=i,
                        total_chunks=len(chunks),
                        char_count=len(chunk_text),
                        web_view_link=meta.web_view_link,
                        extra={"file_size": meta.size},
                    )
                    for i, chunk_text in enumerate(chunks)
                ]

                # Upsert into Qdrant
                upserted = self.vector_store.upsert_chunks(
                    chunks=chunks,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

                result.files_processed += 1
                result.chunks_created += upserted

            except Exception as e:
                logger.error(f"Failed processing file {file_meta.name} ({file_meta.id}): {e}")
                result.errors.append(f"File '{file_meta.name}': {str(e)}")

        return result

    def search(self, query: SearchQuery) -> SearchResponse:
        """Embeds query and retrieves most similar text chunks."""
        query_vector = self.embedding_service.embed_text(query.query)
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=query.top_k,
            file_ids=query.file_ids,
            score_threshold=query.score_threshold,
        )
        return SearchResponse(
            query=query.query,
            results=results,
            total_results=len(results),
        )

    def answer_question(self, req: AskQuestionRequest) -> AskQuestionResponse:
        """Full RAG QA flow: Retrieve context chunks and generate LLM response with citations."""
        search_query = SearchQuery(
            query=req.question,
            top_k=req.top_k,
            file_ids=req.file_ids,
            score_threshold=req.score_threshold,
        )
        search_res = self.search(search_query)

        citations: List[SourceCitation] = [
            SourceCitation(
                file_id=item.metadata.file_id,
                file_name=item.metadata.file_name,
                chunk_index=item.metadata.chunk_index,
                score=item.score,
                text_snippet=item.text[:200] + ("..." if len(item.text) > 200 else ""),
                web_view_link=item.metadata.web_view_link,
            )
            for item in search_res.results
        ]

        if not search_res.results:
            return AskQuestionResponse(
                question=req.question,
                answer="No relevant documents found in Google Drive to answer your question.",
                citations=[],
                confidence_score=0.0,
                model_used="none",
            )

        # Generate answer using unified LLM service
        answer, model_used = self.llm_service.generate_answer(
            question=req.question,
            results=search_res.results,
        )

        avg_score = (
            sum(c.score for c in citations) / len(citations) if citations else 0.0
        )

        return AskQuestionResponse(
            question=req.question,
            answer=answer,
            citations=citations,
            confidence_score=round(avg_score, 4),
            model_used=model_used,
        )

    def ingest_direct_text(
        self,
        text_content: str,
        filename: str,
        file_id: str,
        mime_type: str = "text/plain",
        file_size: Optional[int] = None,
        custom_chunker: Optional[TextChunker] = None,
    ) -> int:
        """Ingests raw text directly, chunks, embeds and upserts to Qdrant."""
        if not text_content.strip():
            return 0

        chunker = custom_chunker or self.chunker
        self.vector_store.delete_by_file_id(file_id)

        chunks = chunker.split_text(text_content)
        if not chunks:
            return 0

        embeddings = self.embedding_service.embed_batch(chunks)
        metadatas = [
            ChunkMetadata(
                file_id=file_id,
                file_name=filename,
                mime_type=mime_type,
                chunk_index=i,
                total_chunks=len(chunks),
                char_count=len(chunk_text),
                web_view_link=f"local://upload/{filename}",
                extra={"file_size": file_size or len(text_content.encode("utf-8")), "source": "direct_upload"},
            )
            for i, chunk_text in enumerate(chunks)
        ]

        return self.vector_store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
