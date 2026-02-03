from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
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
    top_k: int = 5
    # Optional: fetch more candidates than you return, for reranking.
    fetch_k: int = 25

    model_config = {"env_file": ".env"}


settings = Settings()
