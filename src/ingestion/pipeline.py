"""Ingestion pipeline: CSV -> embeddings -> Postgres."""

import pandas as pd
from pathlib import Path

from src.db.connection import get_session
from src.db.repository import ArticleRepository
from src.db.models import ArticleCreate
from src.services.embedding import BaseEmbeddingService, create_embedding_service
from src.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Ingestion pipeline for loading articles from CSV, generating embeddings, and storing in Postgres.
    """

    def __init__(self, embedding_service: BaseEmbeddingService | None = None):
        self.embedding_service = embedding_service or create_embedding_service()

    def ingest_csv(self, csv_path: str | Path) -> int:
        """Load CSV, embed, store. Skips if already ingested."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "text" not in df.columns:
            raise ValueError("CSV must contain a 'text' column")

        texts = df["text"].dropna().tolist()
        texts = [self._clean_text(t) for t in texts]
        logger.info("csv_loaded", count=len(texts))

        with get_session() as session:
            repo = ArticleRepository(session)
            existing = repo.count()
            if existing > 0:
                logger.info("already_ingested", count=existing)
                return existing

        embeddings = self.embedding_service.embed_batch(texts)
        articles = [
            ArticleCreate(text=t, embedding=e) for t, e in zip(texts, embeddings)
        ]

        with get_session() as session:
            repo = ArticleRepository(session)
            repo.insert_batch(articles)

        logger.info("ingestion_complete", count=len(articles))
        return len(articles)

    @staticmethod
    def _clean_text(text: str) -> str:
        import re

        text = text.strip().strip('"').strip()
        text = re.sub(r"\[\d+\]", "", text)  # [1], [2], etc.
        text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)  # **bold**, *italic*
        text = " ".join(text.split())
        return text
