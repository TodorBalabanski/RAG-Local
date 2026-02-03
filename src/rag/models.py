from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class Source(BaseModel):
    content: str
    metadata: dict


class Citation(BaseModel):
    source: str | None = None
    page: int | None = None
    chunk_index: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sources: list[Source]


class IngestResponse(BaseModel):
    message: str
    documents_ingested: int
    chunks_created: int


class IngestURLRequest(BaseModel):
    urls: list[str]
