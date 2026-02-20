"""Database session management and schema initialization."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from src.config.settings import settings
from src.logger import get_logger


logger = get_logger(__name__)

engine = create_engine(settings.database_url, pool_pre_ping=True, pool_size=5)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    :return: Database session generator
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_schema() -> None:
    """
    Create extensions and tables.

    Vector dimension is read from settings. Make sure it matches the embedding provider's output.
    For future: index creation can be added here as well (in case we have more data).
    """
    dim = settings.embedding_dimension

    with get_session() as session:
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        session.execute(
            text(
                f"""
            CREATE TABLE IF NOT EXISTS articles (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                embedding vector({dim}),
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
            )
        """
            )
        )
        session.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS decisions (
                id SERIAL PRIMARY KEY,
                incoming_text TEXT NOT NULL,
                decision TEXT NOT NULL,
                confidence FLOAT NOT NULL,
                reasoning TEXT,
                top_match_id INT REFERENCES articles(id),
                top_match_similarity FLOAT,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """
            )
        )

    logger.info("schema_initialized", embedding_dimension=dim)


def check_connection() -> bool:
    try:
        with get_session() as session:
            session.execute(text("SELECT 1"))
            session.execute(text("SELECT vector '[1,2,3]'"))
        logger.info("database_ok")
        return True
    except Exception as e:
        logger.error("database_failed", error=str(e))
        return False
