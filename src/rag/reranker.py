from __future__ import annotations

from dataclasses import dataclass

from rag.llm import generate_json
from rag.document_loader import Document


@dataclass
class RerankResult:
    indices: list[int]  # 0-based indices into the candidate list


def _format_candidates(docs: list[Document], max_chars: int = 900) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        page_txt = f", page {page}" if page else ""
        text = (d.content or "").strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        parts.append(f"[{i}] (source: {source}{page_txt}) {text}")
    return "\n".join(parts)


SYSTEM_PROMPT = """You are a retrieval reranking model.

You will be given:
- a user question
- a list of candidate context chunks

Select the chunks that are most likely to contain information needed to answer the question.
Prefer chunks that directly answer the question with specific details.

Return ONLY valid JSON with this shape:
{
  "selected": number[]
}

Rules:
- "selected" must contain unique integers.
- Indices are 1-based, referring to the candidate list.
- Return at most K indices.
"""


def rerank(question: str, candidates: list[Document], k: int) -> list[Document]:
    if k <= 0 or not candidates:
        return []
    if len(candidates) <= k:
        return candidates

    user_message = (
        f"Question: {question}\n\n"
        f"K: {k}\n\n"
        f"Candidates:\n{_format_candidates(candidates)}\n"
    )

    data = generate_json(SYSTEM_PROMPT, user_message)
    selected_raw = data.get("selected", [])

    selected: list[int] = []
    if isinstance(selected_raw, list):
        for x in selected_raw:
            if isinstance(x, int) and 1 <= x <= len(candidates):
                selected.append(x)

    # de-dup while preserving order
    seen = set()
    selected = [x for x in selected if not (x in seen or seen.add(x))]

    # enforce max k
    selected = selected[:k]

    # If model returned nothing usable, fall back to the original order.
    if not selected:
        return candidates[:k]

    # Convert to 0-based indices and return docs
    out = [candidates[i - 1] for i in selected]

    # If fewer than k were selected, fill remaining with original order (no duplicates)
    if len(out) < k:
        used_ids = {id(d) for d in out}
        for d in candidates:
            if id(d) in used_ids:
                continue
            out.append(d)
            if len(out) >= k:
                break

    return out
