"""
LLM-based reranker.

Filters hybrid search candidates by asking the LLM whether each
candidate is genuinely about the same story as the query.
"""

from src.config.settings import settings
from src.retrieval.models import RetrievalResult, RankedResult, RerankerResponse
from src.retrieval.prompts import RERANKER_PROMPT
from src.services.llm import LLMClient
from src.logger import get_logger

logger = get_logger(__name__)


class LLMReranker:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()
        self.threshold = settings.reranker_relevance_threshold

    def rerank(
        self,
        query_text: str,
        candidates: list[RetrievalResult],
    ) -> list[RankedResult]:
        """Score relevance of each candidate. Returns those above threshold."""
        ranked: list[RankedResult] = []

        for candidate in candidates:
            prompt = RERANKER_PROMPT.format(
                query=query_text,
                candidate=candidate.text,
            )

            try:
                response = self.llm.call_structured(prompt, RerankerResponse)

                if response.relevance >= self.threshold:
                    ranked.append(
                        RankedResult(
                            id=candidate.id,
                            text=candidate.text,
                            rrf_score=candidate.rrf_score,
                            relevance_score=response.relevance,
                            relevance_reason=response.reason,
                            semantic_rank=candidate.semantic_rank,
                            keyword_rank=candidate.keyword_rank,
                        )
                    )

                logger.debug(
                    "rerank_result",
                    candidate_id=candidate.id,
                    relevance=response.relevance,
                )

            except Exception as e:
                logger.warning(
                    "reranker_fallback", candidate_id=candidate.id, error=str(e)
                )
                ranked.append(
                    RankedResult(
                        id=candidate.id,
                        text=candidate.text,
                        rrf_score=candidate.rrf_score,
                        relevance_score=candidate.rrf_score,
                        relevance_reason="reranker_fallback",
                        semantic_rank=candidate.semantic_rank,
                        keyword_rank=candidate.keyword_rank,
                    )
                )

        ranked.sort(key=lambda r: r.relevance_score, reverse=True)
        logger.info("reranking_done", input=len(candidates), output=len(ranked))
        return ranked
