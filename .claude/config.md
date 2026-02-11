# Configuration

Settings are managed via `src/rag/config.py` using `pydantic_settings.BaseSettings`.
Loaded from `.env` file automatically. Singleton: `from rag.config import settings`.

## Environment Variables

### LLM Generation

| Variable | Type | Default | Notes |
|---|---|---|---|
| `LLM_PROVIDER` | `"anthropic"` or `"openai"` | `"anthropic"` | Selects LLM backend for generation and reranking |
| `ANTHROPIC_API_KEY` | str | `""` | Required when provider is anthropic |
| `CLAUDE_MODEL` | str | `"claude-sonnet-4-20250514"` | Anthropic model ID |
| `OPENAI_API_KEY` | str | `""` | Required when provider is openai |
| `OPENAI_MODEL` | str | `"gpt-4o"` | OpenAI model ID |

### Embeddings

| Variable | Type | Default | Notes |
|---|---|---|---|
| `EMBEDDING_PROVIDER` | `"sentence_transformers"` or `"openai"` | `"sentence_transformers"` | Embedding backend |
| `EMBEDDING_MODEL` | str | `"all-MiniLM-L6-v2"` | Model name (SentenceTransformer or OpenAI) |

**Important**: Switching embedding providers requires deleting `chroma_db/` and re-ingesting documents due to dimension mismatch (384 vs 3072).

### Chunking

| Variable | Type | Default | Notes |
|---|---|---|---|
| `CHUNK_SIZE` | int | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | int | `200` | Overlap between consecutive chunks |

### Vector Store / Retrieval

| Variable | Type | Default | Notes |
|---|---|---|---|
| `CHROMA_PERSIST_DIR` | str | `"./chroma_db"` | ChromaDB persistent storage path |
| `CHROMA_COLLECTION` | str | `"documents"` | ChromaDB collection name |
| `TOP_K` | int | `5` | Number of documents returned per query |
| `FETCH_K` | int | `25` | Candidates fetched before LLM reranking |

## Dependencies (pyproject.toml)

- Python >= 3.11
- Build: hatchling
- LLM: anthropic, openai
- Vector DB: chromadb, sentence-transformers
- API: fastapi, uvicorn, httpx, python-multipart
- Document processing: pypdf, beautifulsoup4
- Config: pydantic-settings

## .env.example Template

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-sonnet-4-20250514

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

CHUNK_SIZE=1000
CHUNK_OVERLAP=200

CHROMA_PERSIST_DIR=./chroma_db
TOP_K=5
FETCH_K=25
```
