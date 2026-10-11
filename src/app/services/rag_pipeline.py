from typing import List, Optional

import openai

from src.app.core.config import settings
from src.app.core.logging import logger
from src.app.schemas.rag import (
    AskQuestionRequest,
    AskQuestionResponse,
    ChunkMetadata,
    SearchQuery,
    SearchResponse,
    SearchResultItem,
    SourceCitation,
    SyncDriveRequest,
    SyncDriveResult,
)
from src.app.services.chunker import TextChunker
from src.app.services.embedding import EmbeddingService
from src.app.services.google_drive import GoogleDriveService
from src.app.services.qdrant_store import QdrantVectorStore

SYSTEM_PROMPT = """You are an intelligent knowledge assistant that answers questions based on documents retrieved from Google Drive.
Analyze the provided Context Sources carefully and answer the user's question accurately and concisely.

Rules:
1. Ground your answer in the provided Context Sources.
2. If the context does not contain enough information to answer the question, state clearly that the answer cannot be found in the indexed documents.
3. Cite source files by mentioning [File: <filename>] when referring to specific facts.
4. Be clear, direct, and helpful.
"""


class RAGPipeline:
    def __init__(
        self,
        drive_service: Optional[GoogleDriveService] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        chunker: Optional[TextChunker] = None,
    ):
        self.drive_service = drive_service or GoogleDriveService()
        self.vector_store = vector_store or QdrantVectorStore()
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunker = chunker or TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        self._openai_client = None
        if settings.OPENAI_API_KEY:
            self._openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

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

        # Build context prompt
        context_parts = []
        for i, item in enumerate(search_res.results, 1):
            source_header = f"[Source {i}: File '{item.metadata.file_name}', Chunk {item.metadata.chunk_index + 1}/{item.metadata.total_chunks}]"
            context_parts.append(f"{source_header}\n{item.text}")

        context_str = "\n\n---\n\n".join(context_parts)
        user_prompt = f"Context Sources:\n{context_str}\n\nQuestion: {req.question}\n\nAnswer:"

        answer = ""
        model_name = settings.OPENAI_CHAT_MODEL

        if self._openai_client and settings.OPENAI_API_KEY:
            try:
                response = self._openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                answer = response.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"OpenAI completion error: {e}")
                answer = self._generate_fallback_answer(req.question, search_res.results)
                model_name = "extractive-fallback"
        else:
            answer = self._generate_fallback_answer(req.question, search_res.results)
            model_name = "extractive-fallback"

        avg_score = (
            sum(c.score for c in citations) / len(citations) if citations else 0.0
        )

        return AskQuestionResponse(
            question=req.question,
            answer=answer,
            citations=citations,
            confidence_score=round(avg_score, 4),
            model_used=model_name,
        )

    @staticmethod
    def _generate_fallback_answer(question: str, results: List[SearchResultItem]) -> str:
        """Deterministic extractive response when no live LLM API key is present."""
        top_snippet = results[0]
        return (
            f"Based on [{top_snippet.metadata.file_name}]:\n\n"
            f"{top_snippet.text}"
        )
