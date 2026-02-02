from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "documents"
    claude_model: str = "claude-sonnet-4-20250514"
    top_k: int = 5

    model_config = {"env_file": ".env"}


settings = Settings()
