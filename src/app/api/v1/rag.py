from fastapi import APIRouter, Depends, HTTPException, status

from src.app.api.deps import get_rag_pipeline
from src.app.core.logging import logger
from src.app.schemas.rag import (
    AskQuestionRequest,
    AskQuestionResponse,
    SearchQuery,
    SearchResponse,
)
from src.app.services.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/rag", tags=["RAG & Vector Search"])


@router.post("/search", response_model=SearchResponse)
def search_vectors(
    query: SearchQuery,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """Semantic vector search across indexed Google Drive documents in Qdrant."""
    try:
        return rag_pipeline.search(query)
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector search error: {str(e)}",
        )


@router.post("/ask", response_model=AskQuestionResponse)
def ask_question(
    request: AskQuestionRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """Full RAG Question-Answering endpoint with context grounding and source citations."""
    try:
        return rag_pipeline.answer_question(request)
    except Exception as e:
        logger.error(f"RAG QA pipeline failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question answering failed: {str(e)}",
        )
