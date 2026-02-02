from rag.config import settings
from rag.document_loader import Document


def chunk_text(
    text: str,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[str]:
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: list[str] = []
    _recursive_split(text, separators, chunk_size, chunk_overlap, chunks)
    return chunks


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
    chunks: list[str],
) -> None:
    if len(text) <= chunk_size:
        if text.strip():
            chunks.append(text.strip())
        return

    separator = separators[0]
    remaining_separators = separators[1:] if len(separators) > 1 else separators

    parts = text.split(separator) if separator else list(text)
    current = ""

    for part in parts:
        candidate = f"{current}{separator}{part}" if current else part
        if len(candidate) > chunk_size and current:
            if len(current) > chunk_size and remaining_separators != separators:
                _recursive_split(
                    current, remaining_separators, chunk_size, chunk_overlap, chunks
                )
            elif current.strip():
                chunks.append(current.strip())
            overlap_text = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{overlap_text}{separator}{part}" if overlap_text else part
        else:
            current = candidate

    if current.strip():
        if len(current) > chunk_size and remaining_separators != separators:
            _recursive_split(
                current, remaining_separators, chunk_size, chunk_overlap, chunks
            )
        else:
            chunks.append(current.strip())


def chunk_documents(documents: list[Document]) -> list[Document]:
    chunked = []
    for doc in documents:
        texts = chunk_text(doc.content)
        for i, text in enumerate(texts):
            chunked.append(
                Document(
                    content=text,
                    metadata={**doc.metadata, "chunk_index": i},
                )
            )
    return chunked
