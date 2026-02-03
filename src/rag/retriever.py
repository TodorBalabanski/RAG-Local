from rag.config import settings
from rag.document_loader import Document
from rag.embeddings import embed_query
from rag.reranker import rerank
from rag.vector_store import query


def retrieve(question: str, top_k: int | None = None) -> list[Document]:
    # Fetch more than we return, then rerank via LLM.
    k = top_k or settings.top_k
    fetch_k = max(k, settings.fetch_k)

    embedding = embed_query(question)
    candidates = query(embedding, top_k=fetch_k)

    return rerank(question, candidates, k)
