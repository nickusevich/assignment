"""
Hybrid retrieval with Reciprocal Rank Fusion (RRF).

Why hybrid? Semantic search misses exact entity names.
Keyword search misses paraphrases.
RRF combines both without needing the score normalization.

RRF: score(d) = Σ 1 / (k + rank_i(d))
"""

from collections import defaultdict

from src.config.settings import settings
from src.db.connection import get_session
from src.db.repository import ArticleRepository
from src.db.models import ArticleRecord
from src.retrieval.models import RetrievalResult
from src.services.embedding import BaseEmbeddingService, create_embedding_service
from src.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(self, embedding_service: BaseEmbeddingService | None = None):
        self.embedding_service = embedding_service or create_embedding_service()
        self.rrf_k = settings.rrf_k

    def retrieve(
        self, query_text: str, top_k: int | None = None
    ) -> list[RetrievalResult]:
        """
        Hybrid search: semantic + keyword + RRF fusion.

        then returns the top_k fused results.
        """
        top_k = top_k or settings.top_k

        query_embedding = self.embedding_service.embed(query_text)

        with get_session() as session:
            repo = ArticleRepository(session)
            semantic = repo.search_semantic(query_embedding, limit=top_k)
            keyword = repo.search_keyword(query_text, limit=top_k)

        fused = self._fuse_rrf(semantic, keyword, top_k)

        logger.info(
            "hybrid_retrieval",
            semantic_hits=len(semantic),
            keyword_hits=len(keyword),
            fused=len(fused),
        )
        return fused

    def _fuse_rrf(
        self,
        semantic: list[ArticleRecord],
        keyword: list[ArticleRecord],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Combine two ranked lists via RRF (Reciprocal Rank Fusion)."""
        scores: dict[int, float] = defaultdict(float)
        meta: dict[int, dict] = {}

        for rank, a in enumerate(semantic, start=1):
            scores[a.id] += 1.0 / (self.rrf_k + rank)
            meta.setdefault(a.id, {"text": a.text})
            meta[a.id]["semantic_rank"] = rank
            meta[a.id]["semantic_score"] = a.score

        for rank, a in enumerate(keyword, start=1):
            scores[a.id] += 1.0 / (self.rrf_k + rank)
            meta.setdefault(a.id, {"text": a.text})
            meta[a.id]["keyword_rank"] = rank
            meta[a.id]["keyword_score"] = a.score

        sorted_ids = sorted(scores.keys(), key=lambda id: scores[id], reverse=True)[
            :top_k
        ]

        return [
            RetrievalResult(
                id=did,
                text=meta[did]["text"],
                rrf_score=scores[did],
                semantic_rank=meta[did].get("semantic_rank"),
                keyword_rank=meta[did].get("keyword_rank"),
                semantic_score=meta[did].get("semantic_score"),
                keyword_score=meta[did].get("keyword_score"),
            )
            for did in sorted_ids
        ]
