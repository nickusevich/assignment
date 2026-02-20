# Football News Pipeline

ML Engineering take-home assignment: Novelty detection for football news articles.

## Requirements

- Docker & Docker Compose
- OpenRouter API key
- Python 3.12+ & uv package manager (for local development)

## Quick Start
```bash
# 1. Set up environment
cp .env.example .env   
# Add your OpenRouter API key

# 2. Start services and run all tasks
docker compose up

# OR run tasks individually:
docker compose up -d db  # Start database
docker compose run --rm app uv run python main.py --task 1 --query "Tottenham" --top-k 3
docker compose run --rm app uv run python main.py --task 2
docker compose run --rm app uv run python main.py --task 3

# 3. Check results
ls outputs/
```


## Project Structure

```
├── data/
│   ├── news.csv              # Published articles database
│   ├── incoming_news.json    # Articles to evaluate
│   └── ground_truth.json     # Evaluation labels
├── outputs/
│   ├── task2_decisions.json  # Publish/skip decisions
│   └── task3_analysis.json   # Decision analytics
├── src/
│   ├── db/                   # Database layer (Postgres + pgvector)
│   ├── ingestion/            # CSV -> embeddings -> database
│   ├── retrieval/            # Hybrid search (semantic + keyword -> RRF)
│   ├── novelty/              # LLM-based novelty detection
│   └── services/             # LLM & embedding clients
└── main.py                   # Task orchestration
```

## Tasks

### Task 1: Similar Article Search

**Goal**: Find top-k most similar articles for a given text.

**Approach**: Hybrid retrieval system. Searches against articles from `data/news.csv` (ingested into Postgres on startup).
- Semantic search via pgvector (cosine similarity on embeddings)
- Keyword search via Postgres full-text search (tsvector)
- RRF (Reciprocal Rank Fusion) to merge rankings

**Run**:
```bash
docker compose run --rm app uv run python main.py --task 1 --query "your text" --top-k 5
```

### Task 2: Publish/Skip Decision

**Goal**: Determine if incoming articles add new information vs existing coverage.

Evaluates each article in `data/incoming_news.json` against the existing database (populated from `news.csv`) to decide PUBLISH, SKIP, or REVIEW. Decisions are saved to Postgres and exported to JSON.

**Approach**: 
1. **Hybrid Retrieval**: Find candidate similar articles (Task 1)
2. **LLM Reranking**: Filter to truly relevant matches (same story)
3. **Novelty Assessment**: LLM evaluates if incoming article has new info
   - PUBLISH: New facts, status changes, financial details
   - SKIP: Same info reworded
   - REVIEW: Low confidence cases (< 0.6) flagged for human review
4. **Decision tracking**: Save decision details to database

**Run**:
```bash
docker compose run --rm app uv run python main.py --task 2
```

**Output**: `outputs/task2_decisions.json`

### Task 3: Decision Analysis

**Goal**: Analyze and review decisions from Task 2.

Fetches all decisions saved by Task 2 from the database and produces a detailed breakdown with analytics.

**Implementation**: 
- Displays detailed breakdown of PUBLISH/SKIP/REVIEW decisions with full context (reasoning, confidence, matched articles)
- Shows analytics: decision distribution, confidence scores, similarity metrics

**Key Feature**: REVIEW workflow support
- Articles with low confidence (< 0.6) are automatically flagged as REVIEW
- Ready to integrate human review where reviewers can:
  - Review flagged articles with full context
  - Make final PUBLISH/SKIP decisions
  - Provide feedback to improve future decisions 

**Run**:
```bash
docker compose run --rm app uv run python main.py --task 3
```

**Output**: `outputs/task3_analysis.json`


## Evaluation

```bash
docker compose run --rm app uv run python scripts/evaluate.py
```

Compares decisions against `data/ground_truth.json`.


## Configuration

Edit `.env`:
```bash
# Embedding provider: "local" (sentence-transformers) or "openrouter" (API)
EMBEDDING_PROVIDER=openrouter

# LLM
LLM_MODEL=google/gemini-2.5-flash

# Retrieval
TOP_K=5
RRF_K=60

# Novelty thresholds
RERANKER_RELEVANCE_THRESHOLD=0.5
NOVELTY_CONFIDENCE_THRESHOLD=0.6
```


## Development Setup

1. **Set up environment variables**
```bash
cp .env.example .env  # Edit .env with your OpenRouter API key and other credentials
```

2. **Start infrastructure**
```bash
docker compose up -d db  # Start only the database
```

3. **Create virtual environment**
```bash
uv venv
```

4. **Install dependencies**
```bash
uv sync
```

5. **Run tasks**
```bash
uv run python main.py --task 1  # Default configuration
uv run python main.py --task 1 --query "clubs" --top-k 5  # Custom query
```

## Known Limitations & Future Work

- **Tests**: Unit and integration tests are not yet implemented. 
- **LLM latency**: Reranker calls the LLM sequentially per candidate (~6 calls per article). Async/concurrent processing could be implemented for higher throughput.
- **Database operations**: Batch inserts are sequential. For larger datasets, indexing would be needed.