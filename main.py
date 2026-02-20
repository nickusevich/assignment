"""
Football News Pipeline.

Usage:
    docker compose up                                                                          # Run all tasks
    docker compose run --rm app uv run python main.py --task 1 --query "Tottenham" --top-k 3   # Task 1: similar articles
    docker compose run --rm app uv run python main.py --task 2                                 # Task 2: publish/skip decisions
    docker compose run --rm app uv run python main.py --task 3                                 # Task 3: decision analysis
    docker compose run --rm app uv run python scripts/evaluate.py                              # Evaluation
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import settings
from src.logger import setup_logging, get_logger
from src.db.connection import check_connection, get_session, init_schema
from src.db.repository import ArticleRepository, DecisionRepository
from src.db.models import DecisionCreate
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import LLMReranker
from src.retrieval.models import RetrievalResult, RankedResult
from src.novelty.detector import NoveltyDetector
from src.novelty.models import Decision

setup_logging()
logger = get_logger("main")


@dataclass
class Dependencies:
    """Shared services."""
    retriever: HybridRetriever
    reranker: LLMReranker


# TASK 1
def get_top_k_similar(text: str, top_k: int = 5, retriever: HybridRetriever | None = None) -> list[RetrievalResult]:
    """
    Return the top-k most similar articles for a given input text.

    Pipeline:
        1. Semantic search (pgvector cosine similarity)
        2. Keyword search (Postgres tsvector)
        3. RRF fusion of both ranked lists
    """
    retriever = retriever or HybridRetriever()
    candidates = retriever.retrieve(text, top_k=top_k)
    return candidates[:top_k]


def run_task1(query: str, top_k: int, deps: Dependencies):
    results = get_top_k_similar(query, top_k, retriever=deps.retriever)

    print(f"\n{'='*70}")
    print(f" TASK 1: Top-{top_k} Similar Articles")
    print(f" Query: {query[:100]}...")
    print(f"{'='*70}\n")

    for i, r in enumerate(results, 1):
        print(f"  [{i}] RRF: {r.rrf_score:.4f}")
        print(f"      Semantic: rank {r.semantic_rank} | Keyword: rank {r.keyword_rank}")
        print(f"      Text: {r.text[:200]}...\n")

    return results


# TASK 2
# Builds on Task 1: same retriever + RERANKER, adds LLM novelty assessment.
def run_task2(incoming_path: Path, deps: Dependencies):
    with open(incoming_path) as f:
        articles = json.load(f).get("text_list", [])

    detector = NoveltyDetector(
        retriever=deps.retriever,
        reranker=deps.reranker,
    )

    print(f"\n{'='*70}")
    print(f" TASK 2: Publish/Skip Decisions ({len(articles)} articles)")
    print(f"{'='*70}\n")

    results = []
    for i, text in enumerate(articles, 1):
        result = detector.assess(text)
        results.append(result)

        with get_session() as session:
            DecisionRepository(session).save(DecisionCreate(
                incoming_text=result.incoming_text,
                decision=result.decision.value,
                confidence=result.confidence,
                reasoning=result.reasoning,
                top_match_id=result.top_match_id,
                top_match_similarity=result.top_match_similarity,
            ))

        print(f"  [{result.decision.value}] Article {i} (confidence: {result.confidence:.2f})")
        print(f"     {result.reasoning}")
        if result.new_information:
            print(f"     New info: {', '.join(result.new_information)}")
        print()

    pub = sum(1 for r in results if r.decision.value == "PUBLISH")
    skip = sum(1 for r in results if r.decision.value == "SKIP")
    rev = sum(1 for r in results if r.decision.value == "REVIEW")
    print(f"  Summary: {pub} publish, {skip} skip, {rev} review\n")

    # Save results
    output_path = Path(settings.output_dir) / "task2_decisions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2, default=str)
    logger.info("task2_saved", path=str(output_path))

    return results


# TASK 3
# Reads all decisions from Postgres and produces a detailed breakdown + stats.
def run_task3():
    with get_session() as session:
        records = DecisionRepository(session).fetch_all()

    if not records:
        print("\n  No decisions recorded yet. Run Task 2 first.\n")
        return []

    print(f"\n{'='*70}")
    print(f" TASK 3: Decision Analysis ({len(records)} decisions)")
    print(f"{'='*70}\n")

    # Per-article breakdown
    for i, r in enumerate(records, 1):
        print(f"  [{r.decision}] Decision #{r.id} (confidence: {r.confidence:.2f})")
        print(f"     Text: {r.incoming_text[:200]}...")
        print(f"     Reasoning: {r.reasoning}")
        if r.top_match_id:
            print(f"     Matched article #{r.top_match_id} (similarity: {r.top_match_similarity:.2f})")
        if r.created_at:
            print(f"     Recorded: {r.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    # Stats
    counts = Counter(r.decision for r in records)
    confidences = [r.confidence for r in records]
    avg_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)
    max_conf = max(confidences)

    matched = [r for r in records if r.top_match_similarity is not None]
    avg_sim = sum(r.top_match_similarity for r in matched) / len(matched) if matched else 0

    print(f"  {'─'*50}")
    print(f"  Decisions:  {counts.get('PUBLISH', 0)} publish, "
          f"{counts.get('SKIP', 0)} skip, "
          f"{counts.get('REVIEW', 0)} review")
    print(f"  Confidence: avg {avg_conf:.2f}, min {min_conf:.2f}, max {max_conf:.2f}")
    if matched:
        print(f"  Similarity: avg {avg_sim:.2f} (across {len(matched)} matched articles)")


    # Save report
    report = {
        "total_decisions": len(records),
        "breakdown": dict(counts),
        "confidence": {"avg": round(avg_conf, 3), "min": round(min_conf, 3), "max": round(max_conf, 3)},
        "avg_match_similarity": round(avg_sim, 3) if matched else None,
        "decisions": [r.model_dump() for r in records],
    }

    output_path = Path(settings.output_dir) / "task3_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("task3_saved", path=str(output_path))

    return records


# CLI
def main():
    parser = argparse.ArgumentParser(description="Football News Pipeline")
    parser.add_argument("--task", type=int, choices=[1, 2, 3])
    parser.add_argument("--query", type=str, default="Tottenham Lucas Bergvall, a high ankle sprain during the UEFA Champions League match against Borussia Dortmund last Tuesday, \nforcing him off after 63 minutes. Thomas Frank \nrevealed that the injury 'could be a longer one,' however further assessment \nwill take place in the next couple of days, in order to confirm an extended period on the sidelines. \nBergvall saw limited action against West Ham due to prior issues.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(settings.data_dir)

    init_schema()

    if not check_connection():
        logger.error("Database unavailable")
        sys.exit(1)

    count = IngestionPipeline().ingest_csv(data_dir / "news.csv")
    logger.info("ready", articles=count)

    deps = Dependencies(
        retriever=HybridRetriever(),
        reranker=LLMReranker(),
    )

    run_all = args.task is None
    if args.task == 1 or run_all:
        run_task1(args.query, args.top_k, deps)
    if args.task == 2 or run_all:
        run_task2(data_dir / "incoming_news.json", deps)
    if args.task == 3 or run_all:
        run_task3()


if __name__ == "__main__":
    main()