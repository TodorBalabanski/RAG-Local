# Architecture

## Overview

RAG (Retrieval-Augmented Generation) API built with FastAPI. Ingests documents, stores embeddings in ChromaDB, and answers questions using retrieved context + LLM with structured JSON responses and citations.

## Data Flow

```
Ingest: File/URL -> document_loader.py -> chunker.py -> embeddings.py -> vector_store.py (ChromaDB)

Query:  Question -> retriever.py -> vector_store.py (fetch_k candidates)
                 -> reranker.py (LLM selects top_k)
                 -> chain.py (LLM generates JSON answer with citations)
                 -> Response with answer, citations, sources
```

## Module Dependency Graph

```
api.py
  ├── chain.py
  │     ├── llm.py (generate_json)
  │     └── retriever.py
  │           ├── embeddings.py (SentenceTransformer or OpenAI)
  │           ├── vector_store.py (ChromaDB)
  │           └── reranker.py
  │                 └── llm.py (generate_json)
  ├── chunker.py
  ├── document_loader.py
  └── vector_store.py
```

## Key Design Decisions

- **LLM reranking**: Retriever fetches `fetch_k` candidates (default 25), then `reranker.py` uses LLM to select the best `top_k` (default 5).
- **Structured JSON output**: Both `chain.py` and `reranker.py` use `generate_json()` to enforce JSON responses with citations/selections.
- **Dual embedding providers**: `embeddings.py` supports `sentence_transformers` (384 dims) or `openai` (3072 dims). Switching requires deleting `chroma_db/`.
- **Lazy imports**: `anthropic` and `openai` are imported inside functions so only the selected provider needs to be installed.
- **LRU caching**: Embedding model and ChromaDB client are cached with `@lru_cache`.
- **Recursive chunking** (`chunker.py`): Splits text using hierarchy of separators (`\n\n` > `\n` > `. ` > ` ` > `""`) with configurable overlap.
- **Cosine similarity**: ChromaDB collection uses `hnsw:space: cosine` for vector search.
- **Batch ingestion**: Documents are embedded and added to ChromaDB in batches of 100.

## Supported File Types

- PDF (`.pdf`) - extracted page-by-page via `pypdf`
- Text (`.txt`, `.md`, `.markdown`, `.rst`, `.csv`) - read as UTF-8
- URLs - fetched with `httpx`, HTML cleaned with BeautifulSoup (strips script, style, nav, footer, header)

## New Components

### reranker.py
LLM-based reranking of retrieved candidates. Formats candidates with truncated content, asks LLM to select best chunks via JSON response `{"selected": [1, 3, 5]}`.

### llm.py - generate_json()
Structured JSON generation:
- OpenAI: uses `response_format={"type": "json_object"}`
- Anthropic: appends "Return ONLY valid JSON" to system prompt, falls back gracefully on parse errors
