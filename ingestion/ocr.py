"""OCR / vision fallback for image-only pages and embedded charts or scans.

Backend is chosen with the OCR_BACKEND env var:

    auto       (default) Tesseract if installed, otherwise Gemini vision if a
               GOOGLE_API_KEY is set, otherwise OCR is skipped with a warning
    tesseract  pytesseract + the Tesseract binary
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

logger = logging.getLogger(__name__)

OCR_DPI = int(os.getenv("OCR_DPI", "300"))
_warned = False

VISION_PROMPT = (
    "Transcribe ALL text in this image exactly as written. Keep tables as "
    "pipe-separated rows (one row per line, header first). For charts, list "
    "every label with its value. Do not summarise, translate or add commentary."
)


def _tesseract_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_backend() -> str:
    backend = os.getenv("OCR_BACKEND", "auto").lower()
    if backend != "auto":
        return backend
    if _tesseract_available():
        return "tesseract"
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "none"


def _tesseract(image) -> str:
    import pytesseract

    return pytesseract.image_to_string(image, config="--psm 6")


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


def _tidy(text: str) -> str:
    lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def ocr_image(image) -> str:
    """PIL image -> text ('' when no backend is available or OCR fails)."""
    global _warned
    backend = _resolve_backend()
    if backend == "none":
        if not _warned:
            logger.warning("OCR needed but no backend available (install Tesseract "
                           "or set GOOGLE_API_KEY / OCR_BACKEND=gemini). Skipping.")
            _warned = True
        return ""
    try:
        text = _tesseract(image) if backend == "tesseract" else _gemini(image)
    except Exception as error:  # OCR must never crash the pipeline
        logger.warning("OCR (%s) failed: %s", backend, error)
        return ""
    return _tidy(text)
