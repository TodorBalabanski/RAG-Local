from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Note: pydantic-settings maps env vars by field name (uppercased).
    # e.g. fetch_k <- FETCH_K, use_mmr <- USE_MMR
    # Generation
    llm_provider: Literal["anthropic", "openai"] = "anthropic"

    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Embeddings
    embedding_provider: Literal["sentence_transformers", "openai"] = (
        "sentence_transformers"
    )
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector store / retrieval
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "documents"

    # How many chunks to return to the answer chain (after reranking/diversification)
    top_k: int = 5

    # Fetch more candidates than we return, then rerank/diversify.
    fetch_k: int = 25  # env: FETCH_K

    # Optional quality knobs
    # - If set, drop candidates with cosine *distance* > max_distance (lower is better)
    max_distance: float | None = None

    # Enable MMR (diversity) before LLM reranking.
    use_mmr: bool = True
    mmr_lambda: float = 0.7  # 0..1, higher = more relevance, lower = more diversity

    model_config = {"env_file": ".env"}


settings = Settings()
