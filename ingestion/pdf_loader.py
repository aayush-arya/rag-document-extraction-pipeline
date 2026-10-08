"""Layout- and table-aware PDF loader (pdfplumber).

What it does per page
---------------------
1. Detects ruled tables (``find_tables``) and keeps each one as a structured
   table element - rows/columns are preserved, nothing is flattened to text.
   Borderless tables are tried as a conservative fallback.
2. Extracts the remaining text *outside* the table boxes, groups lines into
   blocks, detects headings (bold lines) and keeps true reading order, so side
   tables never interleave with the central text.
3. Gives every table a title: its own title band, or the heading printed
   directly above it, plus the current section.
4. Falls back to OCR / vision when a page has (almost) no text layer, and also
   OCRs large embedded images (charts, scanned inserts).

After all pages are read, tables that were split across a page break are
stitched back together (see ``table_utils.is_continuation``).
"""

from __future__ import annotations

import logging
import math
import statistics
import os
import re

import pdfplumber
from langchain_core.documents import Document

from .ocr import OCR_DPI, ocr_image
from .table_utils import RawTable, split_table, stitch_tables, table_to_document

logger = logging.getLogger(__name__)

LINE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "intersection_tolerance": 3,
}

OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", "40"))
OCR_IMAGES = os.getenv("PDF_OCR_IMAGES", "1") == "1"
TEXT_TABLES = os.getenv("PDF_TEXT_TABLES", "1") == "1"

HEADER_ZONE = 0.04      # top 4 % of the page
FOOTER_ZONE = 0.94      # bottom 6 % of the page
HEADING_MAX_CHARS = 120
HEADING_ABOVE_GAP = 28  # max distance (pt) between a heading and its table


# --------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------

def _fix_text(text: str) -> str:
    # pdfminer emits "(cid:127)" for glyphs it cannot map (bullets, symbols)
    return re.sub(r"\(cid:(\d+)\)", lambda m: "\u2022" if m.group(1) == "127" else "", text)


def _is_bold(fontname: str) -> bool:
    return bool(re.search(r"bold|black|heavy|semibold", fontname or "", re.I))


def _margin_rotation_boxes(page) -> list[tuple[float, float, float, float]]:
    """Find compact rotated text near page edges (watermarks and margin stamps)."""
    width = float(page.width)
    edge_chars = [
        char for char in page.chars
        if not char.get("upright", True)
        and (((char["x0"] + char["x1"]) / 2) >= 0.78 * width
             or ((char["x0"] + char["x1"]) / 2) <= 0.04 * width)
    ]
    if not edge_chars:
        return []

    # Merge neighboring vertical character columns only when their vertical
    # spans overlap; unrelated rotated objects remain separate candidates.
    columns: list[list[dict]] = []
    for char in sorted(edge_chars, key=lambda item: (item["x0"], item["top"])):
        if columns:
            prev = columns[-1]
            prev_x = statistics.median((item["x0"] + item["x1"]) / 2 for item in prev)
            overlaps_y = min(item["top"] for item in prev) <= char["bottom"] + 24 and (
                max(item["bottom"] for item in prev) >= char["top"] - 24
            )
            if abs((char["x0"] + char["x1"]) / 2 - prev_x) <= 24 and overlaps_y:
                prev.append(char)
                continue
        columns.append([char])

    boxes = []
    for group in columns:
        x0, x1 = min(c["x0"] for c in group), max(c["x1"] for c in group)
        top, bottom = min(c["top"] for c in group), max(c["bottom"] for c in group)
        compact_edge_item = (
            x1 - x0 <= 0.18 * width
            and len(group) >= 4
        )
        if compact_edge_item:
            boxes.append((x0 - 2, top - 2, x1 + 2, bottom + 2))
    return boxes


def _inside_boxes(char: dict, boxes) -> bool:
    cx = (char["x0"] + char["x1"]) / 2
    cy = (char["top"] + char["bottom"]) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)


def _is_diagonal(char: dict) -> bool:
    matrix = char.get("matrix")
    if not matrix:
        return False
    angle = abs(math.degrees(math.atan2(matrix[1], matrix[0]))) % 180
    return 12 < angle < 78 or 102 < angle < 168


def _table_rows(table, excluded_boxes) -> list[list[str | None]]:
    """Extract cells normally, preserving source character order in rotated cells."""
    rows = table.extract()
    page_chars = table.page.chars

    for row_index, row in enumerate(table.rows):
        for column_index, cell in enumerate(row.cells):
            if cell is None:
                continue
            x0, top, x1, bottom = cell
            cell_chars = [
                char for char in page_chars
                if x0 <= (char["x0"] + char["x1"]) / 2 < x1
                and top <= (char["top"] + char["bottom"]) / 2 < bottom
                and not _inside_boxes(char, excluded_boxes)
                and not _is_diagonal(char)
            ]
            rotated = [char for char in cell_chars if not char.get("upright", True)]
            if rotated and len(rotated) / len(cell_chars) >= 0.5:
                rows[row_index][column_index] = "".join(char["text"] for char in cell_chars).strip()
    return rows


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def _is_real_table(rows) -> bool:
    cells = [c for row in rows for c in row if c is not None and str(c).strip()]
    return bool(cells)


def _word_lines(page) -> list[list[dict]]:
    words = page.extract_words(keep_blank_chars=False, x_tolerance=1.5, y_tolerance=2,
                               extra_attrs=["size"])
    words.sort(key=lambda w: (w["top"], w["x0"]))
    lines: list[list[dict]] = []
    for word in words:
        if lines and abs(word["top"] - lines[-1][0]["top"]) <= 0.5 * word["size"]:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [sorted(line, key=lambda w: w["x0"]) for line in lines]


def _segments(line: list[dict]) -> list[dict]:
    """Split a text line into cells wherever there is a wide horizontal gap."""
    cells = [{"x0": line[0]["x0"], "x1": line[0]["x1"], "text": line[0]["text"]}]
    for word in line[1:]:
        gap = word["x0"] - cells[-1]["x1"]
        if gap > max(1.6 * word["size"], 8):
            cells.append({"x0": word["x0"], "x1": word["x1"], "text": word["text"]})
        else:
            cells[-1]["text"] += " " + word["text"]
            cells[-1]["x1"] = word["x1"]
    return cells


def _detect_text_tables(page, excluded_boxes=None) -> list:
    """Borderless tables: runs of >= 3 text lines that split into the same >= 3
    aligned cells.  Prose, headings and TOC lines never qualify."""
    found, run = [], []

    def close_run():
        if len(run) >= 3:
            ncols = max(set(len(r["cells"]) for r in run),
                        key=lambda n: sum(1 for r in run if len(r["cells"]) == n))
            full = [r for r in run if len(r["cells"]) == ncols]
            if ncols >= 2 and len(full) / len(run) >= 0.6:
                anchors = [sorted(r["cells"][c]["x0"] for r in full)[len(full) // 2]
                           for c in range(ncols)]
                rows = []
                for r in run:
                    row = [""] * ncols
                    for cell in r["cells"]:
                        col = min(range(ncols), key=lambda c: abs(anchors[c] - cell["x0"]))
                        row[col] = (row[col] + " " + cell["text"]).strip()
                    rows.append(row)
                x0 = min(c["x0"] for r in run for c in r["cells"])
                x1 = max(c["x1"] for r in run for c in r["cells"])
                found.append(((x0, run[0]["top"], x1, run[-1]["bottom"]), rows))
        run.clear()

    excluded_boxes = excluded_boxes or []
    text_page = page.filter(
        lambda obj: obj.get("object_type") != "char"
        or (obj.get("upright", True) and not _inside_boxes(obj, excluded_boxes)
            and not _is_diagonal(obj))
    )
    for line in _word_lines(text_page):
        cells = _segments(line)
        top, bottom = min(w["top"] for w in line), max(w["bottom"] for w in line)
        if len(cells) >= 3 and (not run or top - run[-1]["bottom"] <= 2.2 * (bottom - top)):
            run.append({"cells": cells, "top": top, "bottom": bottom})
        else:
            close_run()
            if len(cells) >= 3:
                run.append({"cells": cells, "top": top, "bottom": bottom})
    close_run()
    return found


def _boxes_overlap(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _detect_tables(page, excluded_boxes=None) -> list:
    excluded_boxes = excluded_boxes or []
    found = []
    for table in page.find_tables(table_settings=LINE_SETTINGS):
        intersects_stamp = any(_boxes_overlap(table.bbox, box) for box in excluded_boxes)
        if intersects_stamp and len(table.rows) <= 2:
            continue
        rows = _table_rows(table, [] if intersects_stamp else excluded_boxes)
        if _is_real_table(rows):
            found.append((table.bbox, rows))
    if TEXT_TABLES:   # borderless tables, ignoring anything inside a ruled table
        for bbox, rows in _detect_text_tables(page, excluded_boxes):
            if not any(_boxes_overlap(bbox, existing) for existing, _ in found):
                found.append((bbox, rows))
    return found


# --------------------------------------------------------------------------
# text lines -> blocks (reading order, headings)
# --------------------------------------------------------------------------

def _split_wide_line(raw: dict) -> list[dict]:
    """Split distant horizontal text groups so adjacent captions stay distinct."""
    chars = [char for char in raw.get("chars", []) if char.get("upright", True)]
    if len(chars) < 2:
        return [raw]
    chars.sort(key=lambda char: char["x0"])
    size = statistics.median(char.get("size", 0) for char in chars) or 8
    threshold = max(18.0, 2.5 * size)
    groups, current = [], [chars[0]]
    for char in chars[1:]:
        if char["x0"] - current[-1]["x1"] > threshold:
            groups.append(current)
            current = [char]
        else:
            current.append(char)
    groups.append(current)
    if len(groups) == 1:
        return [raw]

    split = []
    for group in groups:
        text = "".join(char["text"] for char in group).strip()
        if text:
            split.append({
                **raw, "text": text, "chars": group,
                "x0": min(char["x0"] for char in group),
                "x1": max(char["x1"] for char in group),
                "top": min(char["top"] for char in group),
                "bottom": max(char["bottom"] for char in group),
            })
    return split or [raw]


def _text_lines(page, bboxes, excluded_boxes=None) -> list[dict]:
    excluded_boxes = excluded_boxes or []

    def keep(obj):
        if obj.get("object_type") != "char":
            return True
        if _inside_boxes(obj, excluded_boxes) or _is_diagonal(obj):
            return False
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        return not any(b[0] - 1 <= cx <= b[2] + 1 and b[1] - 1 <= cy <= b[3] + 1 for b in bboxes)

    region = page.filter(keep)
    height = float(page.height)
    lines = []
    extraction_options = {"line_dir_rotated": "ltr", "char_dir_rotated": "btt"}
    for raw in region.extract_text_lines(return_chars=True, strip=True, **extraction_options):
        for part in _split_wide_line(raw):
            text = _fix_text(part["text"]).strip()
            if not text:
                continue
            chars = [c for c in part.get("chars", []) if c["text"].strip()]
            bold = (sum(1 for c in chars if _is_bold(c.get("fontname", ""))) / len(chars)) if chars else 0.0
            if part["bottom"] > FOOTER_ZONE * height:
                zone = "footer"
            elif part["top"] < HEADER_ZONE * height and len(text) < 100 and bold < 0.9:
                zone = "header"
            else:
                zone = "body"
            lines.append({
                "text": text, "x0": part["x0"], "x1": part["x1"],
                "top": part["top"], "bottom": part["bottom"],
                "height": max(part["bottom"] - part["top"], 1.0),
                "bold": bold, "region": zone,
                "heading": (zone == "body" and bold >= 0.9 and len(text) <= HEADING_MAX_CHARS
                            and sum(char.isalpha() for char in text) >= 3),
            })
    return lines


def _column_anchors(lines: list[dict], page_width: float) -> list[float]:
    """x-positions of text columns; [] means a single-column page."""
    xs = sorted(l["x0"] for l in lines if l["region"] == "body")
    if len(xs) < 6:
        return []
    clusters = [[xs[0]]]
    for x in xs[1:]:
        if x - clusters[-1][-1] > 0.2 * page_width:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    big = [c for c in clusters if len(c) >= 3]
    return [sum(c) / len(c) for c in big] if len(big) >= 2 else []


def _col_of(x0: float, anchors: list[float]) -> int:
    if not anchors:
        return 0
    return min(range(len(anchors)), key=lambda i: abs(anchors[i] - x0))


def _group_blocks(lines: list[dict], anchors: list[float]) -> list[dict]:
    ordered = sorted(lines, key=lambda l: (_col_of(l["x0"], anchors), l["top"], l["x0"]))
    blocks: list[dict] = []
    for line in ordered:
        col = _col_of(line["x0"], anchors)
        prev = blocks[-1] if blocks else None
        if (prev and prev["heading"] == line["heading"] and prev["region"] == line["region"]
                and prev["col"] == col
                and _overlaps(prev["x0"], prev["x1"], line["x0"] - 4, line["x1"] + 4)
                and line["top"] - prev["bottom"] <= 0.6 * line["height"]):
            prev["lines"].append(line["text"])
            prev["bottom"] = line["bottom"]
            prev["x1"] = max(prev["x1"], line["x1"])
        else:
            blocks.append({
                "lines": [line["text"]], "heading": line["heading"], "region": line["region"],
                "col": col, "x0": line["x0"], "x1": line["x1"],
                "top": line["top"], "bottom": line["bottom"],
            })
    for block in blocks:
        block["text"] = " ".join(block["lines"])
    return blocks


# --------------------------------------------------------------------------
# one page
# --------------------------------------------------------------------------

def _overlaps(a0, a1, b0, b1) -> bool:
    return a0 < b1 and a1 > b0


def _process_page(page, page_no: int, source: str, section):
    elements: list = []
    excluded_boxes = _margin_rotation_boxes(page)
    try:
        detected = _detect_tables(page, excluded_boxes)
    except Exception as error:  # a broken page must not kill the whole file
        logger.warning("Table detection failed on page %s: %s", page_no, error)
        detected = []

    tables = []
    for bbox, rows in detected:
        parsed = split_table(rows)
        if parsed is not None:
            tables.append((bbox, parsed))

    lines = _text_lines(page, [bbox for bbox, _ in tables], excluded_boxes)
    anchors = _column_anchors(lines, float(page.width))
    blocks = _group_blocks(lines, anchors)

    # ---- items in reading order (side tables are handled separately) -------
    items = []
    for block in blocks:
        kind = "heading" if block["heading"] else ("text" if block["region"] == "body" else block["region"])
        items.append({"kind": kind, "top": block["top"], "bottom": block["bottom"],
                      "x0": block["x0"], "x1": block["x1"], "text": block["text"],
                      "col": block["col"], "consumed": False})
    side = []
    for bbox, parsed in tables:
        item = {"kind": "side" if parsed["band"] else "table", "parsed": parsed,
                "x0": bbox[0], "top": bbox[1], "x1": bbox[2], "bottom": bbox[3],
                "col": _col_of(bbox[0], anchors), "consumed": False}
        (side if parsed["band"] else items).append(item)

    body = sorted((i for i in items if i["kind"] in ("heading", "text", "table")),
                  key=lambda i: (i["col"], i["top"]))
    for index, item in enumerate(body):          # heading printed right above a table
        if item["kind"] != "table" or index == 0:
            continue
        prev = body[index - 1]
        if (prev["kind"] == "heading" and prev["bottom"] <= item["top"] + 2
                and item["top"] - prev["bottom"] <= HEADING_ABOVE_GAP
                and _overlaps(prev["x0"], prev["x1"], item["x0"], item["x1"])):
            item["heading"] = prev["text"]
            prev["consumed"] = True

    def make_table(item) -> RawTable:
        parsed = item["parsed"]
        return RawTable(
            rows=parsed["rows"], columns=parsed["columns"], has_header=parsed["has_header"],
            title=parsed["title"], heading=item.get("heading"), section=section,
            header_row=parsed.get("header_row"),
            pages=[page_no], kind="side" if parsed["band"] else "body",
            x0=item["x0"], x1=item["x1"], top=item["top"], bottom=item["bottom"],
            page_height=float(page.height),
        )

    for item in body:
        if item["kind"] == "heading":
            section = item["text"]
            if not item["consumed"]:
                elements.append(_text_doc(item["text"], source, page_no, section, heading=True))
        elif item["kind"] == "text":
            elements.append(_text_doc(item["text"], source, page_no, section))
        else:
            elements.append(make_table(item))

    for item in sorted(side, key=lambda i: (round(i["x0"] / 100), i["top"])):
        elements.append(make_table(item))

    for item in items:                           # header / footer text, flagged for the cleaner
        if item["kind"] in ("header", "footer"):
            elements.append(_text_doc(item["text"], source, page_no, None, region=item["kind"]))

    # ---- OCR / vision fallback --------------------------------------------
    n_chars = sum(len(i["text"]) for i in items if i["kind"] in ("heading", "text"))
    n_chars += sum(len(c) for _, p in tables for r in p["rows"] for c in r)
    image_coverage = max(
        ((image["x1"] - image["x0"]) * (image["bottom"] - image["top"])
         for image in page.images),
        default=0,
    ) / max(float(page.width) * float(page.height), 1)
    if n_chars < OCR_MIN_CHARS or (image_coverage >= 0.45 and n_chars < 300):
        text = _ocr_region(page, None)
        if text:
            elements.append(_text_doc(text, source, page_no, section, element_type="ocr_text"))
    elif OCR_IMAGES:
        elements.extend(_ocr_page_images(page, page_no, source, section))
    return elements, section


def _text_doc(text, source, page_no, section, heading=False, region="body",
              element_type="text") -> Document:
    return Document(page_content=text, metadata={
        "source": source, "file_type": "pdf", "element_type": element_type,
        "page": page_no, "pages": [page_no], "section": section,
        "is_heading": heading, "region": region,
        "ocr": element_type == "ocr_text",
    })


def _ocr_region(page, bbox) -> str:
    try:
        target = page if bbox is None else page.crop(bbox)
        image = target.to_image(resolution=OCR_DPI).original
        return ocr_image(image)
    except Exception as error:
        logger.warning("Could not rasterise page %s for OCR: %s", page.page_number, error)
        return ""


def _ocr_page_images(page, page_no, source, section) -> list[Document]:
    docs, width, height = [], float(page.width), float(page.height)
    for image in page.images:
        w, h = image["x1"] - image["x0"], image["bottom"] - image["top"]
        if w * h < 0.06 * width * height:        # ignore logos / icons
            continue
        bbox = (max(0, image["x0"]), max(0, image["top"]),
                min(width, image["x1"]), min(height, image["bottom"]))
        text = _ocr_region(page, bbox)
        if len(text) >= 20:
            doc = _text_doc(text, source, page_no, section, element_type="ocr_text")
            doc.metadata["figure"] = True
            docs.append(doc)
    return docs


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def load_pdf(file_path: str) -> list[Document]:
    elements: list = []
    section = None
    with pdfplumber.open(file_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            page_elements, section = _process_page(page, page_no, file_path, section)
            elements.extend(page_elements)

    documents: list[Document] = []
    table_count = 0
    for element in stitch_tables(elements):
        if isinstance(element, RawTable):
            table_count += 1
            documents.append(table_to_document(element, file_path, "pdf", f"t{table_count:03d}"))
        else:
            documents.append(element)
    return documents
