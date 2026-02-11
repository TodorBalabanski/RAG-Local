from __future__ import annotations

from rag.document_loader import Document
from rag.llm import generate_json
from rag.retriever import retrieve

SYSTEM_PROMPT = """You are a rigorous RAG assistant.

Rules:
- Answer ONLY using the provided context chunks.
- If the context does not contain enough information, say so.
- You MUST provide citations as an array of integers referencing the context chunk numbers you used.
- If you cannot answer, return citations: []

Return ONLY JSON with this shape:
{
  "answer": string,
  "citations": number[]
}
"""

USER_TEMPLATE = """Context chunks:
{context}

Question: {question}
"""


def build_context(documents: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_txt = f", page {page}" if page else ""
        parts.append(f"[{i}] (source: {source}{page_txt})\n{doc.content}")
    return "\n\n".join(parts)


def answer(question: str, top_k: int | None = None) -> tuple[str, list[int], list[Document]]:
    documents = retrieve(question, top_k=top_k)
    context = build_context(documents)
    user_message = USER_TEMPLATE.format(context=context, question=question)

    data = generate_json(SYSTEM_PROMPT, user_message)
    response_text = str(data.get("answer", "")).strip()
    citations_raw = data.get("citations", [])

    citations: list[int] = []
    if isinstance(citations_raw, list):
        for c in citations_raw:
            if isinstance(c, int) and 1 <= c <= len(documents):
                citations.append(c)

    # de-dup while preserving order
    seen = set()
    citations = [c for c in citations if not (c in seen or seen.add(c))]

    return response_text, citations, documents


def answer_stream(question: str, top_k: int | None = None):
    """Yield (delta, documents) pairs while the model is generating.

    Notes:
    - The streamed deltas are raw text (the model is still asked to output JSON).
    - Caller should buffer deltas and parse JSON at the end.
    """

    from rag.llm import stream as llm_stream

    documents = retrieve(question, top_k=top_k)
    context = build_context(documents)
    user_message = USER_TEMPLATE.format(context=context, question=question)

    for delta in llm_stream(SYSTEM_PROMPT, user_message):
        yield delta, documents
