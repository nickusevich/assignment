"""
Novelty detector. Task 2.

Pipeline:
    1. Hybrid retrieval (reuses Task 1)
    2. LLM reranking -> truly relevant matches. (Refer to retrieval/reranker.py and retrieval/prompts.py for implementation details)
    3. No relevant matches -> auto PUBLISH (new topic)
    4. Relevant matches -> LLM novelty assessment -> PUBLISH / SKIP / REVIEW (in case of low confidence or errors)
"""

from src.config.settings import settings
from src.novelty.models import Decision, NoveltyResponse, NoveltyResult
from src.novelty.prompts import NOVELTY_SYSTEM_PROMPT, NOVELTY_ASSESSMENT_PROMPT
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import LLMReranker
from src.retrieval.models import RankedResult
from src.services.llm import LLMClient
from src.logger import get_logger

logger = get_logger(__name__)


class NoveltyDetector:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: LLMReranker | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or LLMReranker()
        self.llm = llm_client or LLMClient()
        self.confidence_threshold = settings.novelty_confidence_threshold

    def assess(self, incoming_text: str) -> NoveltyResult:
        """Full pipeline: retrieve -> rerank -> assess novelty."""

        # Step 1: Hybrid retrieval
        candidates = self.retriever.retrieve(incoming_text)

        if not candidates:
            return NoveltyResult(
                incoming_text=incoming_text,
                decision=Decision.PUBLISH,
                confidence=0.95,
                reasoning="No existing articles in the database.",
            )

        # Step 2: LLM reranking: filter to truly relevant matches
        relevant = self.reranker.rerank(incoming_text, candidates)

        if not relevant:
            return NoveltyResult(
                incoming_text=incoming_text,
                decision=Decision.PUBLISH,
                confidence=0.90,
                reasoning="Retrieved articles are not about the same story. New topic.",
            )

        # Step 3: LLM novelty assessment against relevant matches
        return self._assess_novelty(incoming_text, relevant)

    def _assess_novelty(
        self,
        incoming_text: str,
        relevant_matches: list[RankedResult],
    ) -> NoveltyResult:
        """Ask LLM whether incoming article adds new info vs existing coverage."""

        existing_formatted = "\n\n---\n\n".join(
            f"[Article {i+1}]:\n{m.text}" for i, m in enumerate(relevant_matches)
        )

        prompt = NOVELTY_ASSESSMENT_PROMPT.format(
            existing_articles=existing_formatted,
            incoming_article=incoming_text,
        )

        try:
            response: NoveltyResponse = self.llm.call_structured(  # type: ignore[assignment]
                prompt, NoveltyResponse, system=NOVELTY_SYSTEM_PROMPT
            )

            # Low confidence -> REVIEW for human editor
            if response.confidence < self.confidence_threshold:
                decision = Decision.REVIEW
                reasoning = (
                    f"[LOW CONFIDENCE — FLAGGED FOR REVIEW] {response.reasoning}"
                )
            else:
                decision = Decision(response.decision)
                reasoning = response.reasoning

            top = relevant_matches[0]
            return NoveltyResult(
                incoming_text=incoming_text,
                decision=decision,
                confidence=response.confidence,
                reasoning=reasoning,
                new_information=response.new_information,
                status_change_detected=response.status_change_detected,
                top_match_text=top.text,
                top_match_similarity=top.relevance_score,
                top_match_id=top.id,
                relevant_matches_count=len(relevant_matches),
            )

        except Exception as e:
            logger.error("novelty_assessment_failed", error=str(e))
            return NoveltyResult(
                incoming_text=incoming_text,
                decision=Decision.REVIEW,
                confidence=0.0,
                reasoning=f"Assessment failed: {e}. Flagged for manual review.",
                top_match_text=relevant_matches[0].text if relevant_matches else None,
                top_match_id=relevant_matches[0].id if relevant_matches else None,
                relevant_matches_count=len(relevant_matches),
            )
