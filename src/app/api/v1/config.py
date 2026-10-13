import io
import os
import uuid
import zipfile
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from src.app.api.deps import get_rag_pipeline
from src.app.core.config import settings
from src.app.core.logging import logger
from src.app.schemas.rag import (
    BatchUploadResponse,
    FileUploadResult,
    LLMConfig,
    LLMConfigStatusResponse,
)
from src.app.services.chunker import TextChunker
from src.app.services.google_drive import GoogleDriveService
from src.app.services.llm import LLMService
from src.app.services.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/config", tags=["Configuration & Ingestion"])

# Global in-memory overrides for active runtime LLM configuration
ACTIVE_CONFIG: LLMConfig = LLMConfig(
    llm_provider=settings.LLM_PROVIDER,
    openai_api_key=settings.OPENAI_API_KEY,
    openai_chat_model=settings.OPENAI_CHAT_MODEL,
    openrouter_api_key=settings.OPENROUTER_API_KEY,
    openrouter_chat_model=settings.OPENROUTER_CHAT_MODEL,
    embedding_provider=settings.EMBEDDING_PROVIDER,
    openai_embedding_model=settings.OPENAI_EMBEDDING_MODEL,
    openrouter_embedding_model=settings.OPENROUTER_EMBEDDING_MODEL,
)


@router.get("/llm", response_model=LLMConfigStatusResponse)
def get_llm_config():
    """Retrieve active LLM and Embedding provider configuration without exposing raw secret keys."""
    return LLMConfigStatusResponse(
        llm_provider=ACTIVE_CONFIG.llm_provider,
        embedding_provider=ACTIVE_CONFIG.embedding_provider,
        openai_chat_model=ACTIVE_CONFIG.openai_chat_model,
        openai_embedding_model=ACTIVE_CONFIG.openai_embedding_model,
        openrouter_chat_model=ACTIVE_CONFIG.openrouter_chat_model,
        openrouter_embedding_model=ACTIVE_CONFIG.openrouter_embedding_model,
        has_openai_key=bool(ACTIVE_CONFIG.openai_api_key or settings.OPENAI_API_KEY),
        has_openrouter_key=bool(ACTIVE_CONFIG.openrouter_api_key or settings.OPENROUTER_API_KEY),
        status="configured",
    )


@router.post("/llm", response_model=LLMConfigStatusResponse)
def update_llm_config(
    config: LLMConfig,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """Update active LLM provider, models, and API keys at runtime."""
    global ACTIVE_CONFIG
    ACTIVE_CONFIG = config

    # Update pipeline services with new credentials/providers
    rag_pipeline.llm_service = LLMService(
        provider=config.llm_provider,
        openai_api_key=config.openai_api_key or settings.OPENAI_API_KEY,
        openrouter_api_key=config.openrouter_api_key or settings.OPENROUTER_API_KEY,
        openai_model=config.openai_chat_model,
        openrouter_model=config.openrouter_chat_model,
    )

    if config.embedding_provider == "openrouter":
        rag_pipeline.embedding_service.provider = "openrouter"
        rag_pipeline.embedding_service.api_key = config.openrouter_api_key or settings.OPENROUTER_API_KEY
        rag_pipeline.embedding_service.model = config.openrouter_embedding_model
    elif config.embedding_provider == "openai":
        rag_pipeline.embedding_service.provider = "openai"
        rag_pipeline.embedding_service.api_key = config.openai_api_key or settings.OPENAI_API_KEY
        rag_pipeline.embedding_service.model = config.openai_embedding_model
    else:
        rag_pipeline.embedding_service.provider = "mock"

    logger.info(f"Updated active LLM config: LLM={config.llm_provider}, Embed={config.embedding_provider}")

    return get_llm_config()


@router.post("/upload/files", response_model=BatchUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """Directly uploads single documents or a ZIP archive to extract, chunk, and index into Qdrant."""
    results: List[FileUploadResult] = []
    total_chunks = 0

    custom_chunker = None
    if chunk_size or chunk_overlap:
        custom_chunker = TextChunker(
            chunk_size=chunk_size or settings.CHUNK_SIZE,
            chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
        )

    for upload_file in files:
        filename = upload_file.filename or "uploaded_file"
        content_bytes = await upload_file.read()
        file_size = len(content_bytes)

        # Check if file is a ZIP archive
        if filename.lower().endswith(".zip") or upload_file.content_type == "application/zip":
            try:
                with zipfile.ZipFile(io.BytesIO(content_bytes)) as z:
                    for zip_info in z.infolist():
                        if zip_info.is_dir() or zip_info.filename.startswith("__MACOSX"):
                            continue
                        zip_sub_name = os.path.basename(zip_info.filename)
                        if not zip_sub_name:
                            continue

                        sub_bytes = z.read(zip_info)
                        sub_text = GoogleDriveService.extract_text_from_bytes(sub_bytes, zip_sub_name)

                        sub_file_id = f"upload_zip_{uuid.uuid4().hex[:12]}_{zip_sub_name}"
                        chunks = rag_pipeline.ingest_direct_text(
                            text_content=sub_text,
                            filename=zip_info.filename,
                            file_id=sub_file_id,
                            file_size=len(sub_bytes),
                            custom_chunker=custom_chunker,
                        )
                        total_chunks += chunks
                        results.append(
                            FileUploadResult(
                                filename=zip_info.filename,
                                file_id=sub_file_id,
                                size=len(sub_bytes),
                                chunks_created=chunks,
                                status="indexed" if chunks > 0 else "empty_or_skipped",
                            )
                        )
            except Exception as e:
                logger.error(f"Failed extracting ZIP archive {filename}: {e}")
                results.append(
                    FileUploadResult(
                        filename=filename,
                        file_id=f"err_{uuid.uuid4().hex[:8]}",
                        size=file_size,
                        chunks_created=0,
                        status="failed",
                        error=str(e),
                    )
                )
        else:
            # Single file upload
            try:
                extracted_text = GoogleDriveService.extract_text_from_bytes(
                    bytes_data=content_bytes,
                    filename=filename,
                    mime_type=upload_file.content_type,
                )
                file_id = f"upload_{uuid.uuid4().hex[:12]}_{filename}"
                chunks = rag_pipeline.ingest_direct_text(
                    text_content=extracted_text,
                    filename=filename,
                    file_id=file_id,
                    mime_type=upload_file.content_type or "text/plain",
                    file_size=file_size,
                    custom_chunker=custom_chunker,
                )
                total_chunks += chunks
                results.append(
                    FileUploadResult(
                        filename=filename,
                        file_id=file_id,
                        size=file_size,
                        chunks_created=chunks,
                        status="indexed" if chunks > 0 else "empty_or_skipped",
                    )
                )
            except Exception as e:
                logger.error(f"Failed processing uploaded file {filename}: {e}")
                results.append(
                    FileUploadResult(
                        filename=filename,
                        file_id=f"err_{uuid.uuid4().hex[:8]}",
                        size=file_size,
                        chunks_created=0,
                        status="failed",
                        error=str(e),
                    )
                )

    return BatchUploadResponse(
        total_files=len(results),
        total_chunks=total_chunks,
        results=results,
    )
