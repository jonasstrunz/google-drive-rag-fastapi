from src.app.services.google_auth import GoogleAuthService
from src.app.services.google_drive import GoogleDriveService
from src.app.services.rag_pipeline import RAGPipeline


def get_auth_service() -> GoogleAuthService:
    return GoogleAuthService()


def get_drive_service() -> GoogleDriveService:
    return GoogleDriveService()


def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline()
