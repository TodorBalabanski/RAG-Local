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
- Process:
  1. Embed question
  2. Vector search for `fetch_k` candidates (default 25)
  3. LLM reranks to `top_k` best chunks
  4. LLM generates JSON answer with citations
- Returns: `QueryResponse` with `answer`, `citations[]`, and `sources[]`

## Pydantic Models (src/rag/models.py)

```python
class QueryRequest:
    question: str
    top_k: int | None = None

class Source:
    content: str
    metadata: dict

class Citation:
    source: str | None = None
    page: int | None = None
    chunk_index: int | None = None

class QueryResponse:
    answer: str
    citations: list[Citation]
    sources: list[Source]

class IngestResponse:
    message: str
    documents_ingested: int
    chunks_created: int

class IngestURLRequest:
    urls: list[str]
```

## Example Query Response

```json
{
  "answer": "The book discusses...",
  "citations": [
    {"source": "/path/to/file.pdf", "page": 42, "chunk_index": 3}
  ],
  "sources": [
    {"content": "chunk text...", "metadata": {"source": "...", "page": 42}}
  ]
}
```
