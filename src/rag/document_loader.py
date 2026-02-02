from dataclasses import dataclass, field
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)


def load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(path)
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(content=text, metadata={"source": str(path), "page": i + 1})
            )
    return documents


def load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    return [Document(content=text, metadata={"source": str(path)})]


def load_url(url: str) -> list[Document]:
    response = httpx.get(url, follow_redirects=True, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return [Document(content=text, metadata={"source": url})]


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in (".txt", ".md", ".markdown", ".rst", ".csv"):
        return load_text(path)
    raise ValueError(f"Unsupported file type: {suffix}")
