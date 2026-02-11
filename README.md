# RAG - Retrieval-Augmented Generation API

A REST API for ingesting documents (files and URLs), storing them in a vector database, and querying them with natural language using LLMs.

## Features

- **Document ingestion** from files (PDF, TXT, MD, CSV) and URLs
- **Vector search** using ChromaDB with sentence-transformer or OpenAI embeddings
- **Multi-provider LLM support** - Anthropic Claude and OpenAI
- **Source citations** returned with every answer

## Quick Start

### Prerequisites

- Python 3.11+

### Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Configuration

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` or `openai` | `anthropic` |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `CLAUDE_MODEL` | Claude model name | `claude-sonnet-4-20250514` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o` |
| `CHUNK_SIZE` | Text chunk size in characters | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `EMBEDDING_PROVIDER` | `sentence_transformers` or `openai` | `sentence_transformers` |
| `EMBEDDING_MODEL` | Embedding model (ST or OpenAI) | `all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | `./chroma_db` |
| `TOP_K` | Number of documents to retrieve | `5` |

### Running

```bash
uvicorn rag.api:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### `GET /health`

Health check. Returns `{"status": "ok"}`.

### `POST /ingest/files`

Upload files to ingest into the knowledge base.

```bash
curl -X POST http://localhost:8000/ingest/files \
  -F "files=@document.pdf"
```

### `POST /ingest/urls`

Ingest content from URLs.

```bash
curl -X POST http://localhost:8000/ingest/urls \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com/article"]}'
```

### `POST /query`

Query the knowledge base.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is...", "top_k": 5}'
```

Returns an answer with source citations.

### `POST /query/stream` (SSE)

Stream the answer as **Server-Sent Events**.

- Emits many `delta` events (raw text chunks from the model)
- Finishes with a single `result` event containing JSON `{answer, citations, sources}`

Example (prints the stream):

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is...", "top_k": 5}'
```

## Project Structure

```
src/rag/
  api.py             # FastAPI endpoints
  config.py          # Settings from environment variables
  models.py          # Request/response models
  chain.py           # RAG orchestration
  llm.py             # LLM provider abstraction
  retriever.py       # Document retrieval
  vector_store.py    # ChromaDB integration
  embeddings.py      # Sentence transformer embeddings
  chunker.py         # Document chunking
  document_loader.py # File and URL loading
```