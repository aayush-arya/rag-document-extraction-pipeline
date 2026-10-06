"""Full graph with a fake LLM (no network)."""
from pathlib import Path

import pytest

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "input" / "sample.pdf"
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample.pdf missing")


class _Structured:
    def __init__(self, schema, llm):
        self.schema, self.llm = schema, llm

    def invoke(self, messages):
        from extraction.schema import DocumentInfo, FieldNotesResult
        self.llm.calls.append(self.schema.__name__)
        if self.schema is DocumentInfo:
            return DocumentInfo(title="OPERATIONS & DATA ARCHIVE — CONSOLIDATED RECORD",
                                name="Somebody Invented")
        return FieldNotesResult(notes=[])


class FakeLLM:
    def __init__(self):
        self.calls = []

    def with_structured_output(self, schema):
        return _Structured(schema, self)


def test_graph_runs_and_grounds_llm_output(tmp_path, monkeypatch):
    from extraction.extractor import create_extractor
    from graph import nodes
    from graph.workflow import create_extraction_graph

    monkeypatch.setattr(nodes, "OUTPUT_DIR", str(tmp_path))
    nodes.set_extractor(create_extractor(FakeLLM()))
    result = create_extraction_graph().invoke({"file_path": str(SAMPLE)})

    data = result["extracted_data"]
    assert len(data.master_records) == 64 and len(data.monthly_performance) == 12
    assert len(data.field_notes) == 7 and all(n.verified for n in data.field_notes)
    assert data.document_info.name is None                      # invented value dropped
    assert any("not found in source" in w for w in data.warnings)
    assert Path(result["output_path"]).exists() and Path(result["json_path"]).exists()


def test_missing_file_ends_gracefully():
    from graph.workflow import create_extraction_graph
    result = create_extraction_graph().invoke({"file_path": "nope.pdf"})
    assert "ingestion failed" in result["error"] and "extracted_data" not in result
