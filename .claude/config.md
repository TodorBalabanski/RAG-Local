# Configuration

Settings are managed via `src/rag/config.py` using `pydantic_settings.BaseSettings`.
Loaded from `.env` file automatically. Singleton: `from rag.config import settings`.

## Environment Variables

| Variable | Type | Default | Notes |
|---|---|---|---|
| `LLM_PROVIDER` | `"anthropic"` or `"openai"` | `"anthropic"` | Selects which LLM backend to use |
| `ANTHROPIC_API_KEY` | str | `""` | Required when provider is anthropic |
| `CLAUDE_MODEL` | str | `"claude-sonnet-4-20250514"` | Anthropic model ID |
| `OPENAI_API_KEY` | str | `""` | Required when provider is openai |
| `OPENAI_MODEL` | str | `"gpt-4o"` | OpenAI model ID |
| `CHUNK_SIZE` | int | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | int | `200` | Overlap between consecutive chunks |
| `EMBEDDING_MODEL` | str | `"all-MiniLM-L6-v2"` | SentenceTransformer model name |
| `CHROMA_PERSIST_DIR` | str | `"./chroma_db"` | ChromaDB persistent storage path |
| `CHROMA_COLLECTION` | str | `"documents"` | ChromaDB collection name |
| `TOP_K` | int | `5` | Default number of documents retrieved per query |

## Dependencies (pyproject.toml)

- Python >= 3.11
- Build: hatchling
- Core: anthropic, openai, chromadb, sentence-transformers
- API: fastapi, uvicorn, httpx, python-multipart
- Docs: pypdf, beautifulsoup4
- Config: pydantic-settings
