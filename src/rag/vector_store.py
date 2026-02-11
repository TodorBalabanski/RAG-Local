import uuid
from dataclasses import dataclass
from functools import lru_cache

import chromadb

from rag.config import settings
from rag.document_loader import Document
from rag.embeddings import embed_texts


@dataclass
class SearchResult:
    document: Document
    distance: float | None = None  # cosine distance (lower is better)
    embedding: list[float] | None = None


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection() -> chromadb.Collection:
    client = get_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(documents: list[Document]) -> int:
    if not documents:
        return 0

    collection = get_collection()
    texts = [doc.content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    ids = [str(uuid.uuid4()) for _ in documents]

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_metadatas = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_embeddings = embed_texts(batch_texts)

        collection.add(
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids,
        )

    return len(documents)


def query(query_embedding: list[float], top_k: int = settings.top_k) -> list[Document]:
    """Backward-compatible query (documents only)."""
    return [r.document for r in query_with_scores(query_embedding, top_k=top_k)]


def query_with_scores(
    query_embedding: list[float], top_k: int = settings.top_k
) -> list[SearchResult]:
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    embs = results.get("embeddings", [[]])[0]

    out: list[SearchResult] = []
    for i in range(min(len(docs), len(metas))):
        dist = dists[i] if i < len(dists) else None
        emb = embs[i] if i < len(embs) else None
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        out.append(SearchResult(Document(content=docs[i], metadata=metas[i]), dist, emb))
    return out
