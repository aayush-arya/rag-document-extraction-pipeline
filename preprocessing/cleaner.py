"""Cleaning that respects structure.

* table elements are passed through untouched - their cells were already
  normalised by the loader, and collapsing spaces/blank lines would destroy them
* page headers/footers are dropped (flagged by the PDF loader, plus a generic
  "same line on most pages" detector for files where the loader could not tell)
* text gets unicode / hyphenation / whitespace normalisation
"""

from __future__ import annotations

import re
import unicodedata

from langchain_core.documents import Document

_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s*)?\d+(?:\s*(?:/|of)\s*\d+)?\s*$", re.I)


def _signature(line: str) -> str:
    """'Report ... Page 12' -> 'report ... page #' so page numbers do not matter."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _repeated_lines(documents: list[Document]) -> set[str]:
    pages_by_signature: dict[str, set] = {}
    all_pages = set()
    for doc in documents:
        if doc.metadata.get("element_type") == "table":
            continue
        page = doc.metadata.get("page")
        all_pages.add(page)
        for line in doc.page_content.splitlines():
            if line.strip():
                pages_by_signature.setdefault(_signature(line), set()).add(page)
    if len(all_pages) < 3:
        return set()
    threshold = max(3, int(0.6 * len(all_pages)))
    return {sig for sig, pages in pages_by_signature.items() if len(pages) >= threshold}


def _clean_text(text: str, repeated: set[str], preserve_numeric_lines: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)          # de-hyphenate wrapped words
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ ]{2,}", " ", line).strip()
        if (not preserve_numeric_lines and _PAGE_NUMBER_LINE.match(line)) or (
            line and _signature(line) in repeated
        ):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_documents(documents: list[Document]) -> list[Document]:
    repeated = _repeated_lines(documents)
    cleaned: list[Document] = []

    for doc in documents:
        meta = doc.metadata

        if meta.get("element_type") == "table":
            cleaned.append(Document(page_content=doc.page_content, metadata=dict(meta)))
            continue

        if meta.get("region") in ("header", "footer"):
            continue

        spreadsheet = meta.get("file_type") in {"xlsx", "xlsm"}
        text = _clean_text(
            doc.page_content,
            set() if spreadsheet else repeated,
            preserve_numeric_lines=spreadsheet,
        )
        if text:
            cleaned.append(Document(page_content=text, metadata=dict(meta)))

    return cleaned
