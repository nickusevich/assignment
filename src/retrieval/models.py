"""Retrieval domain models."""

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """Fused result after RRF of semantic + keyword search."""

    id: int
    text: str
    rrf_score: float
    semantic_rank: int | None = None
    keyword_rank: int | None = None
    semantic_score: float | None = None
    keyword_score: float | None = None

    model_config = {"frozen": True}


class RerankerResponse(BaseModel):
    """LLM reranker output."""

    relevance: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class RankedResult(BaseModel):
    """Final result after LLM reranking."""

    id: int
    text: str
    rrf_score: float
    relevance_score: float = Field(ge=0.0, le=1.0)
    relevance_reason: str = ""
    semantic_rank: int | None = None
    keyword_rank: int | None = None

    model_config = {"frozen": True}
