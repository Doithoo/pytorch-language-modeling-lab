"""Read UTF-8 text and JSONL corpora into document records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class CorpusError(ValueError):
    """Raised when source corpus files are missing or malformed."""


@dataclass(frozen=True)
class Document:
    document_id: str
    source: str
    text: str


def _read_text(path: Path, root: Path) -> list[Document]:
    content = path.read_text(encoding="utf-8")
    paragraphs = [part.strip() for part in content.replace("\r\n", "\n").split("\n\n") if part.strip()]
    if len(paragraphs) < 3:
        paragraphs = [line.strip() for line in content.splitlines() if line.strip()]
    relative = path.relative_to(root).as_posix()
    return [Document(f"{relative}#{index}", relative, text) for index, text in enumerate(paragraphs)]


def _read_jsonl(path: Path, root: Path) -> list[Document]:
    documents: list[Document] = []
    relative = path.relative_to(root).as_posix()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"invalid JSON at {relative}:{line_number}: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise CorpusError(f"{relative}:{line_number} must be an object with a string 'text' field")
        document_id = str(row.get("id", f"{relative}#{line_number}"))
        documents.append(Document(document_id, relative, row["text"]))
    return documents


def read_documents(corpus_dir: str | Path) -> tuple[list[Document], list[Path]]:
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    files = sorted([*root.rglob("*.txt"), *root.rglob("*.jsonl")])
    if not files:
        raise CorpusError(f"no .txt or .jsonl corpus files found in {root}")
    documents: list[Document] = []
    seen_ids: set[str] = set()
    for path in files:
        loaded = _read_text(path, root) if path.suffix == ".txt" else _read_jsonl(path, root)
        for document in loaded:
            if document.document_id in seen_ids:
                raise CorpusError(f"duplicate document id: {document.document_id}")
            seen_ids.add(document.document_id)
            documents.append(document)
    if len(documents) < 3:
        raise CorpusError("at least three non-empty documents are required for train/valid/test")
    return documents, files
