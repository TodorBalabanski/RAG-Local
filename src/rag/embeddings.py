from __future__ import annotations

from functools import lru_cache

from rag.config import settings


@lru_cache(maxsize=1)
def _get_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def _embed_sentence_transformers(texts: list[str]) -> list[list[float]]:
    model = _get_sentence_transformer()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.embeddings.create(model=settings.embedding_model, input=texts)
    # OpenAI returns items in the same order as input
    return [item.embedding for item in resp.data]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    if settings.embedding_provider == "openai":
        return _embed_openai(texts)

    return _embed_sentence_transformers(texts)


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
