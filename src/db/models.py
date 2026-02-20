"""Database domain models."""

from pydantic import BaseModel, Field
from datetime import datetime


class ArticleCreate(BaseModel):
    """Payload for inserting a new article."""

    text: str = Field(..., min_length=1)
    embedding: list[float] = Field(..., min_length=1)


class ArticleRecord(BaseModel):
    """An article row returned from the database."""

    id: int
    text: str
    score: float = 0.0

    model_config = {"frozen": True}


class DecisionCreate(BaseModel):
    """Payload for inserting a decision about whether to publish an article or not."""

    incoming_text: str
    decision: str
    confidence: float
    reasoning: str
    top_match_id: int | None = None
    top_match_similarity: float | None = None


class DecisionRecord(BaseModel):
    """A decision row returned from the database."""

    id: int
    incoming_text: str
    decision: str
    confidence: float
    reasoning: str
    top_match_id: int | None = None
    top_match_similarity: float | None = None
    created_at: datetime | None = None

    model_config = {"frozen": True}
