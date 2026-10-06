"""Token-aware, structure-aware chunking.

Problems with the old character splitter and how this fixes them
-----------------------------------------------------------------
* It cut tables in the middle of a row.
    -> a table is chunked ONLY on row boundaries, and every chunk repeats the
       table title + column header, so each chunk is self-contained.
* A 1,200-character chunk of IDs/numbers can exceed the embedding model's
  input window and get silently truncated.
    -> chunk size is measured in tokens of the real embedding model and is
       capped below its window (see ``rag/embeddings.py``).
* A chunk gave no hint where it came from.
    -> every chunk starts with a context header (table title / section / pages),
       which both the embedding and the keyword index can use ("contextual
       chunk headers").
* Consecutive small text blocks (e.g. one line per field note) became tiny
  chunks with no context.
    -> blocks of the same section are packed together up to the token budget.

Tables that were split across pages are already one element by the time they
get here (the loaders stitch them), so a "continued" table is chunked as one.
"""

from __future__ import annotations

import os
import re

from langchain_core.documents import Document

from ingestion.table_utils import render_row, table_header_lines
from rag.embeddings import get_max_tokens, get_token_counter


def default_budget() -> int:
    override = os.getenv("CHUNK_TOKENS")
    if override:
        return int(override)
    return max(64, min(int(get_max_tokens() * 0.75), 384))


# --------------------------------------------------------------------------
# splitting helpers
# --------------------------------------------------------------------------

def _split_text(text: str, budget: int, count) -> list[str]:
    """Split one long text on sentence (then word) boundaries to fit ``budget``."""
    pieces = [p for p in re.split(r"(?<=[.!?;])\s+|\n+", text) if p.strip()]
    out, current, used = [], [], 0
    for piece in pieces:
        for part in _fit_words(piece, budget, count):
            tokens = count(part)
            if current and used + tokens > budget:
                out.append(" ".join(current))
                current, used = [], 0
            current.append(part)
            used += tokens
    if current:
        out.append(" ".join(current))
    return out


def _fit_words(piece: str, budget: int, count) -> list[str]:
    if count(piece) <= budget:
        return [piece]
    parts, current = [], []
    for word in piece.split():
        trial = " ".join(current + [word])
        if current and count(trial) > budget:
            parts.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        parts.append(" ".join(current))
    return parts


def _base_meta(meta: dict) -> dict:
    keep = ("source", "file_type", "sheet", "section")
    return {k: meta[k] for k in keep if k in meta}


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def _pack(costs: list[int], limit: int) -> list[tuple[int, int]]:
    """Greedy grouping of consecutive rows into (first, last) ranges within ``limit``."""
    groups, first, used = [], 0, 0
    for index, cost in enumerate(costs):
        if index > first and used + cost > limit:
            groups.append((first, index - 1))
            first, used = index, 0
        used += cost
    groups.append((first, len(costs) - 1))
    return groups


def _balanced_groups(costs: list[int], limit: int) -> list[tuple[int, int]]:
    """Like ``_pack`` but evens out chunk sizes (no 1-row tail chunk)."""
    if not costs:
        return []
    groups = _pack(costs, limit)
    if len(groups) > 1:
        base = -(-sum(costs) // len(groups))
        for extra in (0.25, 0.5, 0.75, 1.0):          # smallest even target that keeps the count
            balanced = _pack(costs, min(limit, base + int(extra * max(costs))))
            if len(balanced) <= len(groups):
                return balanced
    return groups


def _chunk_table(doc: Document, budget: int, count) -> list[Document]:
    meta = doc.metadata
    pages = meta.get("pages") or [meta.get("page", 1)]
    header = "\n".join(table_header_lines(
        meta.get("title", "Table"), meta.get("section"),
        pages if meta.get("file_type") == "pdf" else [],
        meta.get("columns", []), meta.get("has_header", False), meta.get("sheet"),
    ))
    available = budget - count(header) - 1
    rows = meta.get("rows", [])
    lines = [render_row(row) for row in rows]
    costs = [count(line) + 1 for line in lines]

    pieces: list[tuple[str, int, int]] = []        # (text, first_row, last_row) 0-based
    run_start = 0
    for index in range(len(rows) + 1):
        oversized = index < len(rows) and costs[index] > available
        if index < len(rows) and not oversized:
            continue
        if index > run_start:                      # normal rows run_start .. index-1
            for a, b in _balanced_groups(costs[run_start:index], available):
                a, b = a + run_start, b + run_start
                pieces.append((header + "\n" + "\n".join(lines[a:b + 1]), a, b))
        if oversized:                              # one huge row: split its text
            for part_no, part in enumerate(_split_text(lines[index], max(available - 8, 16), count), 1):
                pieces.append((f"{header}\n(row {index + 1}, part {part_no}) {part}", index, index))
        run_start = index + 1
    if not pieces:
        pieces.append((header, 0, -1))

    result = []
    for number, (text, first_row, last_row) in enumerate(pieces):
        metadata = {k: v for k, v in meta.items() if k not in ("rows", "part_titles")}
        metadata.update({
            "element_type": "table",
            "row_start": first_row + 1, "row_end": last_row + 1,
            "table_chunk": number, "table_chunks": len(pieces),
            "columns": meta.get("columns", []),
        })
        result.append(Document(page_content=text, metadata=metadata))
    return result


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------

def _chunk_text_group(units: list[dict], budget: int, count) -> list[Document]:
    """Pack consecutive text blocks (same section) into budget-sized chunks."""
    if not units:
        return []
    section = units[0]["meta"].get("section")
    element_type = units[0]["meta"].get("element_type", "text")
    overlap_budget = max(24, budget // 7)

    def header_for(pages: list) -> str:
        lines = []
        if section:
            lines.append(f"Section: {section}")
        if units[0]["meta"].get("file_type") == "pdf" and pages:
            lines.append("Pages: " + (f"{min(pages)}-{max(pages)}" if len(set(pages)) > 1 else str(pages[0])))
        elif units[0]["meta"].get("sheet"):
            lines.append(f"Sheet: {units[0]['meta']['sheet']}")
        if element_type == "ocr_text":
            lines.append("(text recognised from an image)")
        return "\n".join(lines)

    # expand over-long units first
    flat = []
    for unit in units:
        if count(unit["text"]) > budget - 40:
            for piece in _split_text(unit["text"], budget - 60, count):
                flat.append({**unit, "text": piece, "heading": False})
        else:
            flat.append(unit)

    chunks: list[list[dict]] = []
    current: list[dict] = []
    used = 0
    for unit in flat:
        header_cost = count(header_for(unit["pages"])) + 1
        cost = count(unit["text"]) + 1
        if current and used + cost > budget:
            carry_heading = current[-1] if current[-1]["heading"] and len(current) > 1 else None
            if carry_heading:
                current = current[:-1]
            chunks.append(current)
            tail = current[-1] if current else None
            current = []
            if carry_heading:
                current.append(carry_heading)
            elif tail and not tail["heading"] and count(tail["text"]) <= overlap_budget:
                current.append(tail)                       # small overlap for continuity
            used = header_cost + sum(count(u["text"]) + 1 for u in current)
        if not current:
            used = header_cost
        current.append(unit)
        used += cost
    if current:
        chunks.append(current)

    result = []
    for group in chunks:
        pages = sorted({p for u in group for p in u["pages"]})
        body = [u["text"] for u in group if not (u["heading"] and u["text"] == section)]
        text = "\n".join(filter(None, [header_for(pages), *body]))
        metadata = {**_base_meta(units[0]["meta"]),
                    "element_type": element_type,
                    "title": section, "page": pages[0] if pages else 1, "pages": pages,
                    "has_heading": any(u["heading"] for u in group)}
        result.append(Document(page_content=text, metadata=metadata))
    return result


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def chunk_documents(documents: list[Document], max_tokens: int | None = None,
                    token_counter=None) -> list[Document]:
    budget = max_tokens or default_budget()
    count = token_counter or get_token_counter()

    chunks: list[Document] = []
    group: list[dict] = []
    group_key = None

    def flush():
        nonlocal group, group_key
        chunks.extend(_chunk_text_group(group, budget, count))
        group, group_key = [], None

    for doc in documents:
        meta = doc.metadata
        if meta.get("element_type") == "table":
            flush()
            chunks.extend(_chunk_table(doc, budget, count))
            continue
        key = (meta.get("source"), meta.get("section"), meta.get("element_type"), meta.get("sheet"))
        if group and key != group_key:
            flush()
        group_key = key
        group.append({
            "text": doc.page_content, "heading": bool(meta.get("is_heading")),
            "pages": list(meta.get("pages") or [meta.get("page", 1)]), "meta": meta,
        })
    flush()

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"c{index:04d}"
        chunk.metadata["chunk_index"] = index
    return chunks
