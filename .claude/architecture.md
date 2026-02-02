# Architecture

## Overview

RAG (Retrieval-Augmented Generation) API built with FastAPI. Ingests documents, stores embeddings in ChromaDB, and answers questions using retrieved context + LLM.

## Data Flow

```
Ingest: File/URL -> document_loader.py -> chunker.py -> embeddings.py -> vector_store.py (ChromaDB)
Query:  Question -> retriever.py -> vector_store.py -> chain.py -> llm.py -> Response with sources
```

## Module Dependency Graph

```
api.py
  ├── chain.py
  │     ├── llm.py (Anthropic / OpenAI)
  │     └── retriever.py
  │           ├── embeddings.py (SentenceTransformer)
  │           └── vector_store.py (ChromaDB)
  ├── chunker.py
  ├── document_loader.py
  └── vector_store.py
```

## Key Design Decisions

- **Lazy imports** in `llm.py`: `anthropic` and `openai` are imported inside functions so only the selected provider needs to be installed.
- **LRU caching**: Embedding model (`embeddings.py`) and ChromaDB client (`vector_store.py`) are cached with `@lru_cache` to avoid reloading.
- **Recursive chunking** (`chunker.py`): Splits text using a hierarchy of separators (`\n\n` > `\n` > `. ` > ` ` > `""`) with configurable overlap.
- **Cosine similarity**: ChromaDB collection uses `hnsw:space: cosine` for vector search.
- **Batch ingestion**: Documents are embedded and added to ChromaDB in batches of 100.

## Supported File Types

- PDF (`.pdf`) - extracted page-by-page via `pypdf`
- Text (`.txt`, `.md`, `.markdown`, `.rst`, `.csv`) - read as UTF-8
- URLs - fetched with `httpx`, HTML cleaned with BeautifulSoup (strips script, style, nav, footer, header)
