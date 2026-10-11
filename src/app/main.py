from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.api.v1.auth import router as auth_router
from src.app.api.v1.drive import router as drive_router
from src.app.api.v1.rag import router as rag_router
from src.app.core.config import settings
from src.app.core.logging import logger, setup_logging
from src.app.schemas.rag import HealthResponse
from src.app.services.qdrant_store import QdrantVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} in {settings.APP_ENV} mode")
    try:
        # Check / initialize Qdrant
        store = QdrantVectorStore()
        logger.info(f"Qdrant connection verified. Collection exists: {store.collection_exists()}")
    except Exception as e:
        logger.warning(f"Could not connect to Qdrant at startup: {e}")
    yield
    logger.info("Shutting down application...")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="Google Drive RAG API using FastAPI, Qdrant Vector DB, and Google OAuth2 Flow",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    def health_check():
        qdrant_connected = False
        collection_exists = False
        try:
            store = QdrantVectorStore()
            qdrant_connected = store.is_connected()
            collection_exists = store.collection_exists()
        except Exception:
            pass

        return HealthResponse(
            status="healthy" if qdrant_connected else "degraded",
            version="0.1.0",
            qdrant_connected=qdrant_connected,
            qdrant_collection_exists=collection_exists,
        )

    # Include API Routers under /api/v1
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(drive_router, prefix="/api/v1")
    app.include_router(rag_router, prefix="/api/v1")

    return app


app = create_app()
