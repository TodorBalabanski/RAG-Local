from rag.document_loader import Document
from rag.llm import generate
from rag.retriever import retrieve

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the context below to answer. If the context does not contain enough information, say so.
Always cite which source(s) you used."""

USER_TEMPLATE = """Context:
{context}

Question: {question}"""


def build_context(documents: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.content}")
    return "\n\n".join(parts)


def answer(question: str, top_k: int | None = None) -> tuple[str, list[Document]]:
    documents = retrieve(question, top_k=top_k)
    context = build_context(documents)
    user_message = USER_TEMPLATE.format(context=context, question=question)
    response = generate(SYSTEM_PROMPT, user_message)
    return response, documents
