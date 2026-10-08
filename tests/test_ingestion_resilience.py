"""Targeted regressions for rotated PDF content, OCR, and quota-safe caching."""

import importlib.util
from pathlib import Path

import pytest

from extraction.extractor import _retry, create_extractor
from extraction.llm import _FallbackChatModel
from extraction.schema import DocumentInfo, FieldNotesResult
from ingestion.pdf_loader import _split_wide_line, load_pdf
from output.txt_writer import write_txt
from preprocessing.chunker import chunk_documents
from preprocessing.cleaner import clean_documents

SAMPLE3 = Path(__file__).resolve().parents[1] / "data" / "input" / "sample3.pdf"
HAS_RAPIDOCR = importlib.util.find_spec("rapidocr_onnxruntime") is not None


def test_adjacent_caption_segments_stay_separate():
    chars = []
    for text, start in (("OWNER ALLOCATION", 10), ("RELEASE CHECKPOINT", 250)):
        x = start
        for char in text:
            chars.append({
                "text": char, "x0": x, "x1": x + 5, "top": 20, "bottom": 30,
                "size": 10, "upright": True, "fontname": "Helvetica-Bold",
            })
            x += 6
    raw = {
        "text": "OWNER ALLOCATION RELEASE CHECKPOINT", "chars": chars,
        "x0": 10, "x1": 360, "top": 20, "bottom": 30,
    }
    assert [line["text"] for line in _split_wide_line(raw)] == [
        "OWNER ALLOCATION", "RELEASE CHECKPOINT",
    ]


def test_quota_errors_stop_retry_without_backoff(monkeypatch):
    import extraction.extractor as extractor_module

    calls = []

    def fail():
        calls.append(1)
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(extractor_module.time, "sleep",
                        lambda *_: pytest.fail("quota errors must not sleep/retry"))
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        _retry(fail)
    assert len(calls) == 1


def test_extraction_skips_remaining_llm_sections_after_quota():
    class Structured:
        def __init__(self, owner):
            self.owner = owner

        def invoke(self, _messages):
            self.owner.calls += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    class QuotaLLM:
        calls = 0

        def with_structured_output(self, _schema):
            return Structured(self)

    llm = QuotaLLM()
    result = create_extractor(llm)(
        [], {"document_info": "document metadata", "field_notes": "unstructured notes"}
    )
    assert llm.calls == 1
    assert any("field_notes extraction skipped" in warning for warning in result.warnings)


def test_structured_llm_calls_are_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_CACHE_ENABLED", "1")

    class Structured:
        def __init__(self, schema, owner):
            self.schema, self.owner = schema, owner

        def invoke(self, _messages):
            self.owner.calls += 1
            return self.schema()

    class FakeLLM:
        calls = 0

        def with_structured_output(self, schema):
            return Structured(schema, self)

    llm = FakeLLM()
    extract = create_extractor(llm)
    contexts = {"document_info": "metadata context", "field_notes": "unstructured notes"}
    first = extract([], contexts)
    second = extract([], contexts)
    assert llm.calls == 2
    assert first.document_info == second.document_info == DocumentInfo()
    assert first.field_notes == second.field_notes == []


def test_primary_quota_uses_configured_fallback():
    class FakeModel:
        def __init__(self, result=None, error=None):
            self.result, self.error = result, error

        def with_structured_output(self, _schema):
            return self

        def invoke(self, _input):
            if self.error:
                raise self.error
            return self.result

    fallback = FakeModel(result="fallback response")
    model = _FallbackChatModel(
        FakeModel(error=RuntimeError("429 RESOURCE_EXHAUSTED")),
        fallback_factory=lambda: fallback,
    )
    assert model.with_structured_output(DocumentInfo).invoke([]) == "fallback response"


@pytest.mark.skipif(not SAMPLE3.exists() or not HAS_RAPIDOCR,
                    reason="sample3.pdf and RapidOCR are required")
def test_sample3_rotated_tables_and_scanned_ledger_survive_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_BACKEND", "rapidocr")
    monkeypatch.setenv("PIPELINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PIPELINE_CACHE_ENABLED", "1")
    documents = clean_documents(load_pdf(str(SAMPLE3)))

    rotated = [
        doc for doc in documents
        if doc.metadata.get("element_type") == "table" and doc.metadata.get("page") == 6
    ]
    assert len(rotated) == 1
    rotated_text = "\n".join(" | ".join(row) for row in rotated[0].metadata["rows"])
    assert "Idle d" in rotated_text
    assert "Avg age" in rotated_text
    assert "Fuel (kL)" in rotated_text
    assert "d eldI" not in rotated_text
    assert "ega gvA" not in rotated_text and ")Lk( leuF" not in rotated_text

    page_11_tables = [
        doc for doc in documents
        if doc.metadata.get("element_type") == "table" and doc.metadata.get("page") == 11
    ]
    assert len(page_11_tables) == 2
    assert all(doc.metadata["n_rows"] == 6 for doc in page_11_tables)
    assert page_11_tables[0].metadata["columns"] == ["ID", "Client", "Sector", "Since"]
    assert page_11_tables[1].metadata["columns"] == [
        "Annual value (cr)", "% of rev", "Renewal", "Tenor",
    ]

    ledger = [
        doc for doc in documents
        if doc.metadata.get("element_type") == "ocr_text" and doc.metadata.get("page") == 5
    ]
    assert ledger and any("MM/4471" in doc.page_content and "Tarang" in doc.page_content
                          for doc in ledger)
    chunks = chunk_documents(documents, token_counter=lambda text: len(text.split()))
    assert any("MM/4471" in chunk.page_content and "Tarang" in chunk.page_content
               for chunk in chunks)

    extracted = create_extractor()(documents, {})
    output = tmp_path / "sample3.txt"
    write_txt(extracted, output)
    report = output.read_text(encoding="utf-8")
    assert "OCR TRANSCRIPTS" in report
    assert "MM/4471" in report and "Tarang" in report


def test_ocr_image_result_cache(tmp_path, monkeypatch):
    from PIL import Image
    import ingestion.ocr as ocr

    monkeypatch.setenv("OCR_BACKEND", "rapidocr")
    monkeypatch.setenv("PIPELINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_CACHE_ENABLED", "1")
    calls = []
    monkeypatch.setattr(ocr, "_rapidocr", lambda _image: calls.append(1) or "cached text")
    monkeypatch.setitem(ocr._BACKENDS, "rapidocr",
                        lambda _image: calls.append(1) or "cached text")
    image = Image.new("RGB", (40, 20), "white")
    assert ocr.ocr_image(image) == "cached text"
    assert ocr.ocr_image(image.copy()) == "cached text"
    assert calls == [1]
