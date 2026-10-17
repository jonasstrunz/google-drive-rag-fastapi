import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

from src.app.core.config import settings
from src.app.core.logging import logger
from src.app.schemas.rag import ChunkMetadata, SearchResultItem


class QdrantVectorStore:
    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        in_memory: Optional[bool] = None,
    ):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.vector_size = vector_size or settings.QDRANT_VECTOR_SIZE

        if client:
            self.client = client
        else:
            is_in_memory = in_memory if in_memory is not None else settings.QDRANT_IN_MEMORY
            if is_in_memory:
                self.client = QdrantClient(location=":memory:")
                logger.info("Initialized in-memory Qdrant client")
            else:
                qdrant_url = url or settings.QDRANT_URL
                qdrant_api_key = api_key or settings.QDRANT_API_KEY
                self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                logger.info(f"Initialized Qdrant client connecting to {qdrant_url}")

        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Ensures the collection exists with cosine distance metric."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=self.vector_size,
                        distance=rest_models.Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection '{self.collection_name}' (size: {self.vector_size})")
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection: {e}")

    def collection_exists(self) -> bool:
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception:
            return False

    def is_connected(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def upsert_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[ChunkMetadata],
    ) -> int:
        """Upsert text chunks with embeddings and metadata to Qdrant."""
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            return 0

        points = []
        for text, vector, meta in zip(chunks, embeddings, metadatas):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{meta.file_id}_chunk_{meta.chunk_index}"))
            payload = {
                "text": text,
                "file_id": meta.file_id,
                "file_name": meta.file_name,
                "mime_type": meta.mime_type,
                "chunk_index": meta.chunk_index,
                "total_chunks": meta.total_chunks,
                "char_count": meta.char_count,
                "web_view_link": meta.web_view_link,
                "extra": meta.extra,
            }
            points.append(
                rest_models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info(f"Upserted {len(points)} chunks into '{self.collection_name}'")
        return len(points)

    def delete_by_file_id(self, file_id: str) -> None:
        """Deletes all chunks associated with a specific Google Drive file ID."""
        filter_condition = rest_models.Filter(
            must=[
                rest_models.FieldCondition(
                    key="file_id",
                    match=rest_models.MatchValue(value=file_id),
                )
            ]
        )
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition,
            )
        except Exception:
            try:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=rest_models.FilterSelector(filter=filter_condition),
                )
            except Exception as e:
                logger.warning(f"Delete by file_id error: {e}")
        logger.info(f"Deleted chunks for file_id '{file_id}' from '{self.collection_name}'")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        file_ids: Optional[List[str]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResultItem]:
        """Perform vector similarity search with optional filtering by file IDs."""
        query_filter = None
        if file_ids:
            query_filter = rest_models.Filter(
                must=[
                    rest_models.FieldCondition(
                        key="file_id",
                        match=rest_models.MatchAny(any=file_ids),
                    )
                ]
            )

        # Use query_points (compatible with qdrant-client >= 1.10+) or search
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            ).points
        except (AttributeError, Exception):
            # Fallback for search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )

        output: List[SearchResultItem] = []
        for r in results:
            payload = r.payload or {}
            metadata = ChunkMetadata(
                file_id=payload.get("file_id", ""),
                file_name=payload.get("file_name", ""),
                mime_type=payload.get("mime_type", ""),
                chunk_index=payload.get("chunk_index", 0),
                total_chunks=payload.get("total_chunks", 1),
                char_count=payload.get("char_count", len(payload.get("text", ""))),
                web_view_link=payload.get("web_view_link"),
                extra=payload.get("extra", {}),
            )
            output.append(
                SearchResultItem(
                    chunk_id=str(r.id),
                    text=payload.get("text", ""),
                    score=float(r.score),
                    metadata=metadata,
                )
            )
        return output

    def count_points(self) -> int:
        """Returns total vector count in collection."""
        try:
            res = self.client.count(collection_name=self.collection_name)
            return res.count
        except Exception:
            return 0
