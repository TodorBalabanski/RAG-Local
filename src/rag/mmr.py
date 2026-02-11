from __future__ import annotations

import math


def _to_list(x):
    # Handle numpy arrays, lists, etc.
    if x is None:
        return None
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def _cosine_similarity(a, b) -> float:
    a = _to_list(a)
    b = _to_list(b)
    if a is None or b is None:
        return 0.0
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def mmr_select(
    query_emb: list[float],
    candidate_embs: list[list[float] | None],
    k: int,
    lambda_mult: float = 0.7,
) -> list[int]:
    """Return indices (0-based) selected by Maximal Marginal Relevance.

    Assumes higher cosine similarity = better.

    If embeddings are missing, falls back to [0..k-1].
    """

    n = len(candidate_embs)
    if k <= 0 or n == 0:
        return []

    # If any required embedding is missing, don't pretend — just take the first k.
    if not query_emb or any(e is None for e in candidate_embs):
        return list(range(min(k, n)))

    sims_to_query = [
        _cosine_similarity(query_emb, e if e is not None else []) for e in candidate_embs
    ]

    selected: list[int] = []
    candidates = set(range(n))

    # start with the most relevant
    first = max(candidates, key=lambda i: sims_to_query[i])
    selected.append(first)
    candidates.remove(first)

    while candidates and len(selected) < k:
        def score(i: int) -> float:
            rel = sims_to_query[i]
            div = max(
                _cosine_similarity(
                    candidate_embs[i] if candidate_embs[i] is not None else [],
                    candidate_embs[j] if candidate_embs[j] is not None else [],
                )
                for j in selected
            )
            return lambda_mult * rel - (1.0 - lambda_mult) * div

        best = max(candidates, key=score)
        selected.append(best)
        candidates.remove(best)

    return selected
