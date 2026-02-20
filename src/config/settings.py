"""Settings. .env overrides some of these values."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://football:football@localhost:5432/football_news"

    # Embedding settings
    embedding_provider: str = (
        "openrouter"  # "local" or "openrouter" | for LLMs directly use openrouter
    )
    embedding_model_local: str = "all-mpnet-base-v2"
    embedding_model_openrouter: str = "openai/text-embedding-3-small"
    embedding_dimension: int = 1536 

    # LLM via OpenRouter // For future: can add local LLM settings here as well (Ollama)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "google/gemini-2.5-flash"

    # Retrieval
    top_k: int = 5
    rrf_k: int = 60

    # Reranker
    reranker_relevance_threshold: float = 0.5

    # Novelty
    novelty_confidence_threshold: float = 0.6

    # Paths
    data_dir: str = "data"
    output_dir: str = "outputs"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
