import uuid
from functools import lru_cache

import chromadb

from rag.config import settings
from rag.document_loader import Document
from rag.embeddings import embed_texts


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
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        documents.append(Document(content=doc, metadata=meta))
    return documents
