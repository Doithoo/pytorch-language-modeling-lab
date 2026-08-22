"""Conservative, deterministic corpus normalization."""

from __future__ import annotations

import re
import unicodedata

from .sources import Document

_HORIZONTAL_SPACE = re.compile(r"[\t\f\v ]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [_HORIZONTAL_SPACE.sub(" ", line).strip() for line in value.split("\n")]
    return _EXCESS_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def clean_documents(documents: list[Document]) -> tuple[list[Document], dict[str, int]]:
    cleaned: list[Document] = []
    dropped_empty = 0
    duplicate_count = 0
    seen_text: set[str] = set()
    for document in documents:
        text = normalize_text(document.text)
        if not text:
            dropped_empty += 1
            continue
        if text in seen_text:
            duplicate_count += 1
            continue
        seen_text.add(text)
        cleaned.append(Document(document.document_id, document.source, text))
    if len(cleaned) < 3:
        raise ValueError("normalization left fewer than three unique documents")
    return cleaned, {
        "input_documents": len(documents),
        "output_documents": len(cleaned),
        "dropped_empty": dropped_empty,
        "dropped_duplicates": duplicate_count,
    }
