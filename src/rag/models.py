from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class Source(BaseModel):
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    message: str
    documents_ingested: int
    chunks_created: int


class IngestURLRequest(BaseModel):
    urls: list[str]
