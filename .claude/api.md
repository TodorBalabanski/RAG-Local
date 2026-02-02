# API Reference

Entry point: `src/rag/api.py` -> `app = FastAPI(title="RAG API", version="0.1.0")`

Run with: `uvicorn rag.api:app --reload`

## Endpoints

### GET /health
Returns `{"status": "ok"}`.

### POST /ingest/files
- Accepts: `multipart/form-data` with field `files` (list of `UploadFile`)
- Supported types: PDF, TXT, MD, Markdown, RST, CSV
- Process: files -> temp file -> load -> chunk -> embed -> store in ChromaDB
- Returns: `IngestResponse` (`message`, `documents_ingested`, `chunks_created`)

### POST /ingest/urls
- Accepts: JSON `{"urls": ["https://..."]}`
- Request model: `IngestURLRequest`
- Process: fetch URL -> strip HTML tags -> chunk -> embed -> store
- Returns: `IngestResponse`

### POST /query
- Accepts: JSON `{"question": "...", "top_k": 5}` (top_k is optional)
- Request model: `QueryRequest`
- Process: embed question -> vector search -> build context -> LLM generates answer
- Returns: `QueryResponse` (`answer`, `sources[]` with `content` and `metadata`)

## Pydantic Models (src/rag/models.py)

- `QueryRequest`: `question: str`, `top_k: int | None`
- `QueryResponse`: `answer: str`, `sources: list[Source]`
- `Source`: `content: str`, `metadata: dict`
- `IngestResponse`: `message: str`, `documents_ingested: int`, `chunks_created: int`
- `IngestURLRequest`: `urls: list[str]`
