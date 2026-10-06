"""Section-specific context assembly (replaces the single generic query).

Each thing the extractor needs gets its OWN retrieval:
  * several phrasings of the information need  -> fused (``invoke_many``)
  * a metadata filter (text only, first pages, ...)
  * structural lookups (a section by heading) that need no similarity at all
  * neighbouring chunks for continuity
The result is a short, relevant, de-duplicated context per section, in
document order, with a hard size cap so the LLM prompt stays small.
"""

from __future__ import annotations

import os

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

SECTION_SPECS = {
    "document_info": {
        "queries": [
            "document title, type, revision or version and the period it covers",
            "name of the person or organization, department and job designation",
            "contact email address and phone number",
            "document date, prepared by, issued on",
        ],
        "where": lambda m: m.get("element_type") in ("text", "ocr_text"),
        "k": 6,
        "first_pages": 1,           # the front page always carries the identity of the document
        "neighbors": 0,
    },
    "field_notes": {
        "queries": [
            "field notes raw observations dated entries",
            "note says discrepancy mismatch duplicate unclear recheck not copied",
            "remarks comments issues observed log",
        ],
        "where": lambda m: m.get("element_type") in ("text", "ocr_text"),
        "section_pattern": r"field notes|observations|remarks|log|notes",
        "k": 4,
        "first_pages": 0,
        "neighbors": 1,
    },
}


def _ordered(chunks):
    seen, result = set(), []
    for chunk in sorted(chunks, key=lambda c: c.metadata.get("chunk_index", 0)):
        key = chunk.metadata.get("chunk_id") or id(chunk)
        if key not in seen:
            seen.add(key)
            result.append(chunk)
    return result


def _strip_header(text: str) -> str:
    """Drop the 'Section:/Pages:' preamble lines the chunker adds."""
    lines = text.splitlines()
    while lines and lines[0].startswith(("Section:", "Pages:", "Sheet:", "(text recognised")):
        lines.pop(0)
    return "\n".join(lines)


def build_contexts(retriever, specs: dict | None = None) -> tuple[dict[str, str], list]:
    specs = specs or SECTION_SPECS
    contexts: dict[str, str] = {}
    trace: list[dict] = []
    all_chunks: list = []

    for name, spec in specs.items():
        picked = list(retriever.invoke_many(spec["queries"], k=spec.get("k", 6), where=spec.get("where")))

        if spec.get("section_pattern"):
            picked.extend(retriever.by_section(spec["section_pattern"], where=spec.get("where")))
        if spec.get("first_pages"):
            limit = spec["first_pages"]
            picked.extend(c for c in retriever.store.chunks
                          if spec["where"](c.metadata) and (c.metadata.get("page") or 99) <= limit)
        for chunk in list(picked):
            if spec.get("neighbors"):
                picked.extend(n for n in retriever.neighbors(chunk, spec["neighbors"])
                              if spec["where"](n.metadata))

        ordered = _ordered(picked)
        text, used = [], 0
        for chunk in ordered:
            body = _strip_header(chunk.page_content)
            if used + len(body) > MAX_CONTEXT_CHARS:
                break
            text.append(body)
            used += len(body)
        contexts[name] = "\n\n".join(text)
        all_chunks.extend(ordered)
        trace.append({"section": name, "chunks": [c.metadata.get("chunk_id") for c in ordered],
                      "chars": used})
    return contexts, _ordered(all_chunks), trace
