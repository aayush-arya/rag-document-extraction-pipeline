"""Shared table helpers used by every loader (PDF, XLSX, DOCX).

A loader turns a raw grid of cells into a ``RawTable``.  ``stitch_tables``
re-joins tables that were split across pages, and ``table_to_document`` turns
the final table into a LangChain ``Document`` that keeps BOTH:

* ``page_content``  - readable pipe-separated text (title + header + rows)
* ``metadata["rows"]`` / ``metadata["columns"]`` - the structured grid, so later
  stages (table parser, chunker) never have to re-parse text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

NULLISH = {"", "null", "none", "n/a", "na", "?", "-", "\u2014", "\u2013", "nan"}

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?|[-+]?\.\d+%?")


# --------------------------------------------------------------------------
# cell / header helpers
# --------------------------------------------------------------------------

def clean_cell(value) -> str:
    """Cell -> single-line, trimmed string (merged-cell ``None`` becomes '')."""
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ")
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def norm_name(value: str) -> str:
    """'Entity / Site' -> 'entity site' (used to compare headers)."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def is_number(value: str) -> bool:
    return bool(_NUMBER_RE.fullmatch(value.strip()))


def looks_like_header(row: list[str]) -> bool:
    """Heuristic: a header row is all short, non-empty, non-numeric labels."""
    cells = [clean_cell(c) for c in row]
    if len(cells) < 2:
        return False
    if any(c == "" or c.lower() in NULLISH for c in cells):
        return False
    if any(is_number(c) for c in cells):
        return False
    if any(len(c) > 40 for c in cells):
        return False
    return True


def make_unique_columns(names: list[str]) -> list[str]:
    """Blank -> col_N, repeated -> name_2, name_3 (source fragments repeat names)."""
    seen: dict[str, int] = {}
    result = []
    for index, name in enumerate(names):
        name = clean_cell(name) or f"col_{index + 1}"
        key = name.lower()
        seen[key] = seen.get(key, 0) + 1
        result.append(name if seen[key] == 1 else f"{name}_{seen[key]}")
    return result


def split_table(raw_rows) -> Optional[dict]:
    """Raw grid -> {title, band, columns, has_header, rows} (or None if empty).

    * multi-column table whose first row has ONE filled cell  -> title band
    * single-column table with >= 3 rows                      -> first row is the title
    * header row detected heuristically; otherwise columns are col_1..col_n
    """
    rows = [[clean_cell(c) for c in row] for row in raw_rows]
    rows = [row for row in rows if any(row)]
    if not rows:
        return None
    ncols = max(len(row) for row in rows)
    rows = [row + [""] * (ncols - len(row)) for row in rows]

    title, band = None, False
    if ncols > 1 and len(rows) >= 2 and sum(1 for c in rows[0] if c) == 1:
        title = next(c for c in rows[0] if c)
        band = True
        rows = rows[1:]
    elif ncols == 1 and len(rows) >= 3:
        title = rows[0][0]
        rows = rows[1:]

    has_header = ncols > 1 and len(rows) >= 2 and looks_like_header(rows[0])
    header_row = rows[0] if has_header else None
    if has_header:
        columns = make_unique_columns(rows[0])
        rows = rows[1:]
    else:
        columns = [f"col_{i + 1}" for i in range(ncols)]
    return {"title": title, "band": band, "columns": columns,
            "has_header": has_header, "rows": rows, "header_row": header_row}


# --------------------------------------------------------------------------
# RawTable + continuation stitching
# --------------------------------------------------------------------------

@dataclass
class RawTable:
    rows: list[list[str]]
    columns: list[str]
    has_header: bool
    title: Optional[str] = None      # own title (title band / first row)
    heading: Optional[str] = None    # heading printed directly above the table
    section: Optional[str] = None    # section the table belongs to
    pages: list[int] = field(default_factory=lambda: [1])
    kind: str = "body"               # "body" | "side" (side = titled box beside the text)
    x0: float = 0.0
    x1: float = 0.0
    top: float = 0.0
    bottom: float = 0.0
    page_height: float = 0.0
    sheet: Optional[str] = None
    header_row: Optional[list[str]] = None   # raw header cells (to undo a false header)
    paged: bool = True                       # False for xlsx/docx (page numbers meaningless)
    merged_parts: int = 1
    part_titles: list[str] = field(default_factory=list)

    @property
    def display_title(self) -> str:
        return self.title or self.heading or self.section or "Untitled table"


def _same_header(a: RawTable, b: RawTable) -> bool:
    return (
        a.has_header and b.has_header
        and [norm_name(c) for c in a.columns] == [norm_name(c) for c in b.columns]
    )


def is_continuation(prev: Optional[RawTable], cur: RawTable) -> bool:
    """Is ``cur`` the next part of ``prev`` (table split across a page break)?

    Rules - all need: both body tables, same column count, directly adjacent
    pages and (if geometry is known) the same horizontal span.  Then either
      a) identical header AND (cur has no heading of its own OR its heading says
         "continued/continuation"), or
      b) cur has no header/heading, and prev ends at the page bottom while cur
         starts at the next page top (a header-less table broken by the page).
    Titled side boxes that merely share a header (e.g. 'RELEASE CHECKPOINT |
    P03 TABLE 03' and 'P04 TABLE 01') are never merged.
    """
    if prev is None or prev.kind != "body" or cur.kind != "body":
        return False
    if len(prev.columns) != len(cur.columns):
        return False
    if cur.pages[0] - prev.pages[-1] != 1:
        return False
    if prev.x1 and cur.x1 and (abs(prev.x0 - cur.x0) > 20 or abs(prev.x1 - cur.x1) > 20):
        return False

    label = f"{cur.title or ''} {cur.heading or ''}"
    says_continued = bool(re.search(r"continu", label, re.I))
    no_own_heading = cur.title is None and cur.heading is None

    if _same_header(prev, cur):
        return says_continued or no_own_heading

    # Header-less table broken by the page.  The header heuristic may also have
    # mistaken the first data row of the continuation for a header, so we allow
    # `cur.has_header` when `prev` has none (merge_into puts that row back).
    if no_own_heading and prev.page_height and (not cur.has_header or not prev.has_header):
        near_bottom = prev.bottom >= prev.page_height - 70
        near_top = cur.top <= 70
        return near_bottom and near_top
    return False


def merge_into(prev: RawTable, cur: RawTable) -> None:
    if cur.has_header and not prev.has_header and cur.header_row:
        prev.rows.append(list(cur.header_row))   # it was really a data row
    prev.rows.extend(cur.rows)
    prev.pages = sorted(set(prev.pages + cur.pages))
    prev.bottom = cur.bottom          # so a 3rd page can still chain on
    prev.merged_parts += 1
    if cur.heading or cur.title:
        prev.part_titles.append(cur.title or cur.heading or "")


def stitch_tables(elements: list) -> list:
    """Walk a mixed list of Documents / RawTables in reading order and merge
    split tables.  Everything that is not a RawTable is passed through."""
    result, last_body = [], None
    for element in elements:
        if isinstance(element, RawTable) and element.kind == "body":
            if is_continuation(last_body, element):
                merge_into(last_body, element)
                continue
            last_body = element
        result.append(element)
    return result


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render_row(row: list[str]) -> str:
    return " | ".join(row)


def table_header_lines(title: str, section: Optional[str], pages, columns, has_header,
                       sheet: Optional[str] = None) -> list[str]:
    lines = [f"Table: {title}"]
    if section and section != title and section != sheet:
        lines.append(f"Section: {section}")
    if sheet:
        lines.append(f"Sheet: {sheet}")
    elif pages:
        lines.append("Pages: " + (f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0])))
    if has_header:
        lines.append("Columns: " + render_row(columns))
    return lines


def table_to_document(table: RawTable, source: str, file_type: str, table_id: str) -> Document:
    lines = table_header_lines(table.display_title, table.section,
                               table.pages if table.paged else [], table.columns,
                               table.has_header, table.sheet)
    lines.extend(render_row(row) for row in table.rows)
    metadata = {
        "source": source,
        "file_type": file_type,
        "element_type": "table",
        "table_id": table_id,
        "title": table.display_title,
        "own_title": table.title,
        "heading": table.heading,
        "section": table.section,
        "page": table.pages[0],
        "pages": table.pages,
        "columns": table.columns,
        "headers": (table.header_row if table.has_header and table.header_row
                    else table.columns),
        "has_header": table.has_header,
        "n_rows": len(table.rows),
        "kind": table.kind,
        "merged_parts": table.merged_parts,
        "part_titles": table.part_titles,
        "rows": table.rows,
    }
    if table.sheet:
        metadata["sheet"] = table.sheet
    return Document(page_content="\n".join(lines), metadata=metadata)
