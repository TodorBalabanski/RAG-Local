from rag.config import settings
from rag.document_loader import Document
from rag.embeddings import embed_query
from rag.mmr import mmr_select
from rag.reranker import rerank
from rag.vector_store import query_with_scores


def _dedup(docs: list[Document]) -> list[Document]:
    seen: set[tuple] = set()
    out: list[Document] = []
    for d in docs:
        key = (
            d.metadata.get("source"),
            d.metadata.get("page"),
            d.metadata.get("chunk_index"),
            (d.content or "")[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def retrieve(question: str, top_k: int | None = None) -> list[Document]:
    """Retrieve documents for a question.

    Pipeline:
      1) Embed query
      2) Vector search (fetch_k)
      3) Optional distance threshold
      4) Optional MMR diversification
      5) LLM rerank down to k
    """

    k = top_k or settings.top_k
    fetch_k = max(k, settings.fetch_k)

    q_emb = embed_query(question)
    results = query_with_scores(q_emb, top_k=fetch_k)

    # attach distances into metadata (useful for debugging / responses)
    for r in results:
        if r.distance is not None:
            r.document.metadata = {**r.document.metadata, "distance": float(r.distance)}

    # optional distance filter
    if settings.max_distance is not None:
        results = [r for r in results if r.distance is None or r.distance <= settings.max_distance]

    # MMR diversify BEFORE LLM rerank to reduce near-duplicates
    candidates = results
    if settings.use_mmr and len(results) > k:
        idxs = mmr_select(
            q_emb,
            [r.embedding for r in results],
            k=min(fetch_k, len(results)),
            lambda_mult=float(settings.mmr_lambda),
        )
        candidates = [results[i] for i in idxs]

    docs = _dedup([r.document for r in candidates])

    # Final LLM rerank to k
    return rerank(question, docs, k)
