import os

from ingestion import load_document

from preprocessing.cleaner import clean_documents
from preprocessing.chunker import chunk_documents

from rag.vectorstore import create_vectorstore
from rag.retriever import get_retriever
from rag.context_builder import build_contexts

from extraction.extractor import create_extractor

from output.json_writer import write_json
from output.report_writer import write_txt

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/output")
MAX_ATTEMPTS = int(os.getenv("MAX_EXTRACTION_ATTEMPTS", "2"))

_extractor = None


def _get_extractor():
    """Built once; tests can replace it with ``set_extractor``."""
    global _extractor
    if _extractor is None:
        _extractor = create_extractor()
    return _extractor


def set_extractor(extract):
    global _extractor
    _extractor = extract


def ingest_node(state):
    try:
        documents = load_document(state["file_path"])
    except Exception as error:
        return {"documents": [], "error": f"ingestion failed: {error}"}
    if not documents:
        return {"documents": [], "error": "no content could be read from the file"}
    return {"documents": documents}


def clean_node(state):
    return {"cleaned_documents": clean_documents(state["documents"])}


def chunk_node(state):
    return {"chunks": chunk_documents(state["cleaned_documents"])}


def vectorstore_node(state):
    return {"vectorstore": create_vectorstore(state["chunks"])}


def retrieval_node(state):
    """Targeted, per-section retrieval (hybrid dense + BM25 + exact id match)."""
    attempts = state.get("attempts", 0)
    # on a retry, look at more chunks per section
    retriever = get_retriever(state["vectorstore"], k=8 + 6 * attempts)
    contexts, documents, trace = build_contexts(retriever)
    return {
        "retriever": retriever,
        "contexts": contexts,
        "retrieved_documents": documents,
        "retrieval_trace": trace,
    }


def extraction_node(state):
    result = _get_extractor()(
        state["cleaned_documents"], state["contexts"], source_file=state["file_path"],
    )
    return {
        "extracted_data": result,
        "warnings": list(result.warnings),
        "attempts": state.get("attempts", 0) + 1,
    }


def validate_node(state):
    """Flag LLM-section failures so the router can retry them once."""
    return {"warnings": state.get("warnings", [])}


def output_node(state):
    stem = os.path.splitext(os.path.basename(state["file_path"]))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{stem}_extracted.txt")
    json_path = os.path.join(OUTPUT_DIR, f"{stem}_extracted.json")
    write_txt(state["extracted_data"], output_path)
    write_json(state["extracted_data"], json_path)
    return {"output_path": output_path, "json_path": json_path}


# ----------------------------------------------------------------- routers

def route_after_ingest(state):
    return "end" if state.get("error") else "clean"


def route_after_validate(state):
    warnings = state.get("warnings", [])
    if any("RESOURCE_EXHAUSTED" in w for w in warnings):
        return "output"        # retrying cannot help
    failed = [w for w in warnings if "extraction failed" in w or "no context" in w]
    if failed and state.get("attempts", 0) < MAX_ATTEMPTS:
        return "retry"
    return "output"
