"""OCR / vision fallback for image-only pages and embedded charts or scans.

Backend is chosen with the OCR_BACKEND env var:

    auto       (default) Tesseract if installed, otherwise RapidOCR if installed,
               otherwise Gemini vision if a GOOGLE_API_KEY is set, otherwise OCR
               is skipped with a warning
    tesseract  pytesseract + the Tesseract binary
    rapidocr   local ONNX OCR (pip install rapidocr-onnxruntime), no binary and
               no API quota; rebuilds table rows from box positions
    gemini     send the image to the Gemini model (reads tables / charts well)
    none       never OCR

Everything is imported lazily, so the pipeline runs without these extras.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import shutil
import statistics

logger = logging.getLogger(__name__)

OCR_DPI = int(os.getenv("OCR_DPI", "300"))
_warned = False
_rapid_engine = None

VISION_PROMPT = (
    "Transcribe ALL text in this image exactly as written. Keep tables as "
    "pipe-separated rows (one row per line, header first). For charts, list "
    "every label with its value. Do not summarise, translate or add commentary."
)


# --------------------------------------------------------------------------
# Availability checks
# --------------------------------------------------------------------------
def _tesseract_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
        return True
    except ImportError:
        return False


def _rapidocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_backend() -> str:
    backend = os.getenv("OCR_BACKEND", "auto").lower()
    if backend != "auto":
        return backend
    if _tesseract_available():
        return "tesseract"
    if _rapidocr_available():
        return "rapidocr"
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "none"


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
def _tesseract(image) -> str:
    import pytesseract

    return pytesseract.image_to_string(image, config="--psm 6")


def _rows_from_boxes(items) -> str:
    """Group RapidOCR detections into lines by vertical position.

    items: list of [box, text, score] where box is 4 (x, y) corner points.
    Words on the same line are joined with ' | ' when the horizontal gap is
    wide (looks like a table cell boundary) and with a space otherwise.
    """
    boxes = []
    for box, text, _score in items:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        boxes.append({
            "text": str(text).strip(),
            "x0": min(xs), "x1": max(xs),
            "yc": (min(ys) + max(ys)) / 2,
            "h": max(ys) - min(ys),
        })
    boxes = [b for b in boxes if b["text"]]
    if not boxes:
        return ""

    median_h = statistics.median(b["h"] for b in boxes) or 10
    boxes.sort(key=lambda b: b["yc"])

    rows, current = [], [boxes[0]]
    for b in boxes[1:]:
        row_yc = sum(x["yc"] for x in current) / len(current)
        if abs(b["yc"] - row_yc) <= median_h * 0.6:
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    rows.append(current)

    lines = []
    for row in rows:
        row.sort(key=lambda b: b["x0"])
        parts = [row[0]["text"]]
        for prev, cur in zip(row, row[1:]):
            gap = cur["x0"] - prev["x1"]
            parts.append(" | " if gap > median_h * 1.5 else " ")
            parts.append(cur["text"])
        lines.append("".join(parts))
    return "\n".join(lines)


def _rapidocr(image) -> str:
    global _rapid_engine
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    if _rapid_engine is None:  # load the model once and reuse across pages
        _rapid_engine = RapidOCR()
    result, _ = _rapid_engine(np.array(image.convert("RGB")))
    return _rows_from_boxes(result or [])


def _gemini(image) -> str:
    from langchain_core.messages import HumanMessage

    from extraction.llm import get_llm

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    message = HumanMessage(content=[
        {"type": "text", "text": VISION_PROMPT},
        {"type": "image_url", "image_url": f"data:image/png;base64,{encoded}"},
    ])
    return str(get_llm().invoke([message]).content)


_BACKENDS = {
    "tesseract": _tesseract,
    "rapidocr": _rapidocr,
    "gemini": _gemini,
}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def _tidy(text: str) -> str:
    lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def ocr_image(image) -> str:
    """PIL image -> text ('' when no backend is available or OCR fails)."""
    global _warned
    backend = _resolve_backend()
    if backend == "none":
        if not _warned:
            logger.warning("OCR needed but no backend available (install Tesseract, "
                           "`pip install rapidocr-onnxruntime`, or set GOOGLE_API_KEY / "
                           "OCR_BACKEND=gemini). Skipping.")
            _warned = True
        return ""
    engine = _BACKENDS.get(backend)
    if engine is None:
        logger.warning("Unknown OCR_BACKEND %r (use auto, tesseract, rapidocr, gemini, none).",
                       backend)
        return ""
    try:
        text = engine(image)
    except Exception as error:  # OCR must never crash the pipeline
        logger.warning("OCR (%s) failed: %s", backend, error)
        return ""
    return _tidy(text)