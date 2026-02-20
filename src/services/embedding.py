"""
Embedding service with two providers:

- "local": sentence-transformers (no API calls, runs on CPU/GPU)
- "openrouter": OpenRouter API (same key as LLM, no local model needed)

Configured via EMBEDDING_PROVIDER in settings (.env overrides).
Both implement the same interface.
"""

import abc

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.config.settings import settings
from src.logger import get_logger

logger = get_logger(__name__)


class BaseEmbeddingService(abc.ABC):
    """Interface that any provider implements."""

    @abc.abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @abc.abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbeddingService(BaseEmbeddingService):
    """Embeddings via sentence-transformers."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model_local
        logger.info("loading_local_embedding_model", model=self.model_name)
        self.model = SentenceTransformer(self.model_name)

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        logger.info("embedding_batch_local", count=len(texts))
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True,
        ).tolist()


class OpenRouterEmbeddingService(BaseEmbeddingService):
    """Embeddings via OpenRouter (OpenAI-compatible API)."""

    def __init__(self):
        self.client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )
        self.model = settings.embedding_model_openrouter

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        logger.info("embedding_batch_openrouter", count=len(texts))
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings


def create_embedding_service() -> BaseEmbeddingService:
    """Return the configured provider."""
    provider = settings.embedding_provider.lower()

    if provider == "local":
        return LocalEmbeddingService()
    elif provider == "openrouter":
        return OpenRouterEmbeddingService()
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. Use 'local' or 'openrouter'."
        )
