import json
import tempfile
from pathlib import Path

import asyncio

import anyio
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from rag.chain import answer, answer_stream
from rag.chunker import chunk_documents
from rag.document_loader import Document, load_file, load_url
from rag.models import (
    Citation,
    IngestResponse,
    IngestURLRequest,
    QueryRequest,
    QueryResponse,
    Source,
)
from rag.vector_store import add_documents

app = FastAPI(title="RAG API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(files: list[UploadFile]):
    all_docs: list[Document] = []

    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            docs = load_file(Path(tmp.name))
            all_docs.extend(docs)

    chunks = chunk_documents(all_docs)
    added = add_documents(chunks)

    return IngestResponse(
        message="Documents ingested successfully",
        documents_ingested=len(all_docs),
        chunks_created=added,
    )


@app.post("/ingest/urls", response_model=IngestResponse)
def ingest_urls(request: IngestURLRequest):
    all_docs: list[Document] = []

    for url in request.urls:
        try:
            docs = load_url(url)
            all_docs.extend(docs)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch {url}: {e}")

    chunks = chunk_documents(all_docs)
    added = add_documents(chunks)

    return IngestResponse(
        message="URLs ingested successfully",
        documents_ingested=len(all_docs),
        chunks_created=added,
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    response_text, citation_idxs, source_docs = answer(
        request.question, top_k=request.top_k
    )

    sources = [Source(content=doc.content, metadata=doc.metadata) for doc in source_docs]

    citations: list[Citation] = []
    for idx in citation_idxs:
        doc = source_docs[idx - 1]
        citations.append(
            Citation(
                source=doc.metadata.get("source"),
                page=doc.metadata.get("page"),
                chunk_index=doc.metadata.get("chunk_index"),
            )
        )

    return QueryResponse(answer=response_text, citations=citations, sources=sources)


@app.post("/query/stream")
def query_documents_stream(payload: QueryRequest, request: Request):
    """Server-Sent Events (SSE) streaming endpoint.

    Events:
      - event: delta   data: <text chunk>
      - event: result  data: <final JSON result>

    Notes:
    - We handle client disconnects gracefully (common cause of noisy ASGI ExceptionGroups).
    - The model is instructed to output JSON; we stream raw deltas and parse at the end.
    """

    async def sse():
        buffer = ""
        last_docs = []
        try:
            for delta, docs in answer_stream(payload.question, top_k=payload.top_k):
                if await request.is_disconnected():
                    return
                last_docs = docs
                buffer += delta
                yield f"event: delta\ndata: {json.dumps(delta)}\n\n"
                # give the event loop a chance to flush
                await asyncio.sleep(0)

            # best-effort parse
            try:
                data = json.loads(buffer)
            except Exception:
                data = {"answer": buffer.strip(), "citations": []}

            citations_raw = data.get("citations", [])
            citation_idxs = [c for c in citations_raw if isinstance(c, int)]

            sources = [
                Source(content=doc.content, metadata=doc.metadata).model_dump()
                for doc in last_docs
            ]
            citations: list[dict] = []
            for idx in citation_idxs:
                if 1 <= idx <= len(last_docs):
                    doc = last_docs[idx - 1]
                    citations.append(
                        Citation(
                            source=doc.metadata.get("source"),
                            page=doc.metadata.get("page"),
                            chunk_index=doc.metadata.get("chunk_index"),
                        ).model_dump()
                    )

            result = {
                "answer": str(data.get("answer", "")).strip(),
                "citations": citations,
                "sources": sources,
            }
            yield f"event: result\ndata: {json.dumps(result)}\n\n"

        except (asyncio.CancelledError, anyio.get_cancelled_exc_class()):
            # client disconnected / request cancelled
            return

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
