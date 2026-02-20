"""Novelty detection domain models."""

from enum import Enum
from pydantic import BaseModel, Field


class Decision(str, Enum):
    PUBLISH = "PUBLISH"
    SKIP = "SKIP"
    REVIEW = "REVIEW"


class NoveltyResponse(BaseModel):
    """LLM novelty assessor output."""

    decision: str = Field(..., pattern="^(PUBLISH|SKIP)$")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    new_information: list[str] = Field(default_factory=list)
    status_change_detected: bool = False


class NoveltyResult(BaseModel):
    """Complete pipeline result for a single incoming article."""

    incoming_text: str
    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    new_information: list[str] = Field(default_factory=list)
    status_change_detected: bool = False
    top_match_text: str | None = None
    top_match_similarity: float | None = None
    top_match_id: int | None = None
    relevant_matches_count: int = 0
