"""Excel (.xlsx / .xlsm) loader.

* every sheet is scanned for *blocks* of data separated by blank rows / blank
  columns, so several tables on one sheet become several table elements
* each block keeps its structure (header, rows) and is rendered as
  pipe-separated rows - no pandas ``to_string`` padding to collapse later
* title bands, repeated/blank headers and header-less blocks are handled by the
  same helpers as the PDF loader
* a block that is a single column of free text becomes a text element
"""

from __future__ import annotations

import datetime as dt

import openpyxl
from langchain_core.documents import Document

from .table_utils import RawTable, split_table, table_to_document


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dt.datetime):
        return value.date().isoformat() if value.time() == dt.time(0) else value.isoformat(sep=" ")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, dt.time):
        return value.isoformat()
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.10g}"
    return str(value)


def _row_blocks(grid: list[list[str]]) -> list[list[list[str]]]:
    blocks, current = [], []
    for row in grid:
        if any(cell != "" for cell in row):
            current.append(row)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _column_blocks(block: list[list[str]]) -> list[list[list[str]]]:
    """Split a row-block into side-by-side blocks at fully empty columns."""
    width = max(len(r) for r in block)
    block = [r + [""] * (width - len(r)) for r in block]
    filled = [any(r[c] != "" for r in block) for c in range(width)]
    groups, start = [], None
    for col, has_data in enumerate(filled + [False]):
        if has_data and start is None:
            start = col
        elif not has_data and start is not None:
            groups.append((start, col))
            start = None
    return [[r[a:b] for r in block] for a, b in groups]


def _is_label_value_list(block) -> bool:
    """'Title' followed by 'label: value' lines (dashboard / snapshot boxes)."""
    lines = [r[0] for r in block if r[0]]
    return len(lines) >= 3 and all(":" in line for line in lines[1:])


def load_excel(file_path: str) -> list[Document]:
    workbook = openpyxl.load_workbook(file_path, data_only=True)
    documents: list[Document] = []
    table_count = 0

    for sheet_index, sheet in enumerate(workbook.worksheets, start=1):
        grid = [[_fmt(v) for v in row] for row in sheet.iter_rows(values_only=True)]
        for row_block in _row_blocks(grid):
            for block in _column_blocks(row_block):
                if max(len(r) for r in block) == 1 and not _is_label_value_list(block):
                    text = "\n".join(r[0] for r in block if r[0])
                    documents.append(Document(page_content=text, metadata={
                        "source": file_path, "file_type": "xlsx", "element_type": "text",
                        "page": sheet_index, "pages": [sheet_index], "sheet": sheet.title,
                        "section": sheet.title, "is_heading": False, "region": "body",
                    }))
                    continue
                parsed = split_table(block)
                if parsed is None:
                    continue
                table_count += 1
                table = RawTable(
                    rows=parsed["rows"], columns=parsed["columns"],
                    has_header=parsed["has_header"], title=parsed["title"],
                    section=sheet.title, pages=[sheet_index], sheet=sheet.title, paged=False,
                )
                documents.append(table_to_document(table, file_path, "xlsx", f"t{table_count:03d}"))
    return documents
