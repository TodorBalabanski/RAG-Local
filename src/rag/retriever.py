from rag.config import settings
from rag.document_loader import Document
from rag.embeddings import embed_query
from rag.vector_store import query


def retrieve(question: str, top_k: int | None = None) -> list[Document]:
    k = top_k or settings.top_k
    embedding = embed_query(question)
    return query(embedding, top_k=k)
