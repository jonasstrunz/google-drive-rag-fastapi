from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.app.api.deps import get_drive_service, get_rag_pipeline
from src.app.core.logging import logger
from src.app.schemas.rag import (
    DriveFileListResponse,
    DriveFileMetadata,
    SyncDriveRequest,
    SyncDriveResult,
)
from src.app.services.google_drive import GoogleDriveService
from src.app.services.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/drive", tags=["Google Drive"])


@router.get("/files", response_model=DriveFileListResponse)
def list_drive_files(
    user_id: str = Query("default_user", description="User ID"),
    folder_id: Optional[str] = Query(None, description="Optional parent folder ID"),
    page_size: int = Query(50, ge=1, le=200, description="Page size"),
    drive_service: GoogleDriveService = Depends(get_drive_service),
):
    """List accessible files in Google Drive."""
    try:
        files = drive_service.list_files(user_id=user_id, folder_id=folder_id, page_size=page_size)
        return DriveFileListResponse(files=files, total_count=len(files))
    except PermissionError as pe:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(pe),
        )
    except Exception as e:
        logger.error(f"Failed listing drive files: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error accessing Google Drive: {str(e)}",
        )


@router.get("/files/{file_id}", response_model=DriveFileMetadata)
def get_drive_file_details(
    file_id: str,
    user_id: str = Query("default_user", description="User ID"),
    drive_service: GoogleDriveService = Depends(get_drive_service),
):
    """Fetch metadata for a specific Google Drive file."""
    try:
        return drive_service.get_file_metadata(file_id=file_id, user_id=user_id)
    except PermissionError as pe:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(pe),
        )
    except Exception as e:
        logger.error(f"Failed getting drive file metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found or inaccessible: {str(e)}",
        )


@router.post("/sync", response_model=SyncDriveResult)
def sync_drive_files(
    request: SyncDriveRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """Sync, chunk, embed, and store Google Drive files into Qdrant vector database."""
    try:
        result = rag_pipeline.sync_google_drive(request)
        return result
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drive synchronization failed: {str(e)}",
        )
