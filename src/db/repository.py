"""Repository layer: all SQL operations isolated here"""

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db.models import ArticleCreate, ArticleRecord, DecisionCreate, DecisionRecord
from src.logger import get_logger

logger = get_logger(__name__)


class ArticleRepository:
    """
    Repository for managing article records in the database.
    """

    def __init__(self, session: Session):
        self.session = session

    def insert(self, article: ArticleCreate) -> int:
        result = self.session.execute(
            text(
                "INSERT INTO articles (text, embedding) "
                "VALUES (:text, :embedding) RETURNING id"
            ),
            {"text": article.text, "embedding": str(article.embedding)},
        )
        return result.scalar_one()

    def insert_batch(self, articles: list[ArticleCreate]) -> list[int]:
        ids = [self.insert(a) for a in articles]
        self.session.flush()
        logger.info("batch_inserted", count=len(ids))
        return ids

    def count(self) -> int:
        return self.session.execute(text("SELECT COUNT(*) FROM articles")).scalar_one()

    def search_semantic(
        self, embedding: list[float], limit: int
    ) -> list[ArticleRecord]:
        """Cosine similarity via pgvector."""
        result = self.session.execute(
            text(
                "SELECT id, text, 1 - (embedding <=> :embedding) AS score "
                "FROM articles ORDER BY embedding <=> :embedding LIMIT :limit"
            ),
            {"embedding": str(embedding), "limit": limit},
        )
        return [
            ArticleRecord(id=r.id, text=r.text, score=r.score)
            for r in result.fetchall()
        ]

    def search_keyword(self, query: str, limit: int) -> list[ArticleRecord]:
        """Full-text search via tsvector."""
        result = self.session.execute(
            text(
                "SELECT id, text, ts_rank(tsv, plainto_tsquery('english', :query)) AS score "
                "FROM articles WHERE tsv @@ plainto_tsquery('english', :query) "
                "ORDER BY score DESC LIMIT :limit"
            ),
            {"query": query, "limit": limit},
        )
        return [
            ArticleRecord(id=r.id, text=r.text, score=r.score)
            for r in result.fetchall()
        ]


class DecisionRepository:
    def __init__(self, session: Session):
        self.session = session

    def save(self, record: DecisionCreate) -> int:
        result = self.session.execute(
            text(
                "INSERT INTO decisions "
                "(incoming_text, decision, confidence, reasoning, top_match_id, top_match_similarity) "
                "VALUES (:incoming_text, :decision, :confidence, :reasoning, :top_match_id, :top_match_similarity) "
                "RETURNING id"
            ),
            record.model_dump(),
        )
        return result.scalar_one()

    def fetch_all(self) -> list[DecisionRecord]:
        """Retrieve all recorded decisions, most recent first."""
        result = self.session.execute(
            text(
                "SELECT id, incoming_text, decision, confidence, reasoning, "
                "top_match_id, top_match_similarity, created_at "
                "FROM decisions ORDER BY created_at DESC"
            )
        )
        return [
            DecisionRecord(
                id=r.id,
                incoming_text=r.incoming_text,
                decision=r.decision,
                confidence=r.confidence,
                reasoning=r.reasoning,
                top_match_id=r.top_match_id,
                top_match_similarity=r.top_match_similarity,
                created_at=r.created_at,
            )
            for r in result.fetchall()
        ]
