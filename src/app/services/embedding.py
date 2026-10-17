import hashlib
import math
from typing import List, Optional

import openai

from src.app.core.config import settings
from src.app.core.logging import logger


class EmbeddingService:
    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        vector_size: Optional[int] = None,
    ):
        self.provider = provider or settings.EMBEDDING_PROVIDER
        self.vector_size = vector_size or settings.QDRANT_VECTOR_SIZE

        if self.provider == "openrouter":
            self.api_key = api_key or settings.OPENROUTER_API_KEY
            self.model = model or settings.OPENROUTER_EMBEDDING_MODEL
        else:
            self.api_key = api_key or settings.OPENAI_API_KEY
            self.model = model or settings.OPENAI_EMBEDDING_MODEL

        self._openai_client = None
        if self.provider == "openrouter" and self.api_key:
            default_headers = {}
            if settings.OPENROUTER_HTTP_REFERER:
                default_headers["HTTP-Referer"] = settings.OPENROUTER_HTTP_REFERER
            if settings.OPENROUTER_TITLE:
                default_headers["X-Title"] = settings.OPENROUTER_TITLE
            self._openai_client = openai.OpenAI(
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=self.api_key,
                default_headers=default_headers if default_headers else None,
            )
        elif self.provider == "openai" and self.api_key:
            self._openai_client = openai.OpenAI(api_key=self.api_key)

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for single string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of strings."""
        if not texts:
            return []

        if self.provider in ["openai", "openrouter"] and self._openai_client:
            try:
                response = self._openai_client.embeddings.create(
                    input=texts,
                    model=self.model,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                logger.error(f"{self.provider.capitalize()} embedding generation failed: {e}")
                # Fall back to mock if in development / test without key
                if settings.APP_ENV in ["development", "test"]:
                    logger.warning(f"Falling back to deterministic mock embedding for development ({self.provider})")
                    return [self._generate_mock_embedding(t) for t in texts]
                raise e

        # Mock deterministic embedding provider for testing and offline usage
        return [self._generate_mock_embedding(t) for t in texts]

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generates a normalized deterministic pseudo-embedding based on hash of text."""
        dim = self.vector_size
        seed_hash = hashlib.sha256(text.encode("utf-8")).digest()
        vector = []
        for i in range(dim):
            byte_val = seed_hash[i % len(seed_hash)]
            # Add trigonometric term based on position and character length
            val = math.sin(byte_val + i + len(text))
            vector.append(val)

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]
