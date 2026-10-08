"""Hybrid extractor.

* Table data  -> ``table_parser`` (deterministic, lossless, no LLM)
* Free text   -> LLM with structured output, fed with the *section-specific*
                 context the retrieval node assembled
* Verification -> every LLM-produced string is checked against the source text;
                  ungrounded values are dropped and reported as warnings
"""

from __future__ import annotations

import logging
import hashlib
import json
import re
import time
from typing import Any, Callable, Optional

from langchain_core.prompts import ChatPromptTemplate

from pipeline_cache import cache_get, cache_put

from .llm import get_llm, is_quota_exhausted, llm_cache_identity
from .schema import (
    DocumentInfo, ExtractedDocument, FieldNote, FieldNoteDraft, FieldNotesResult,
    OCRPageText,
)
from .table_parser import parse_tables

logger = logging.getLogger(__name__)

RULES = """Rules:
1. Use ONLY information present in the context. Never invent or "fix" anything.
2. If a field is not present, return null.
3. Copy names, ids, numbers and dates exactly as printed (keep original date formats).
4. Return the information according to the required schema."""

INFO_PROMPT = ChatPromptTemplate.from_template(
    "You are a document extraction system.\n"
    "Extract the document-level metadata (type, title, revision, covered period, date, "
    "the main person/organization, department, designation, email, phone) and a short summary.\n\n"
    + RULES + "\n\nDocument context:\n\n{context}"
)

NOTES_PROMPT = ChatPromptTemplate.from_template(
    "You are a document extraction system.\n"
    "The context contains field notes / raw observations. Return ONE entry per distinct note: "
    "its date token exactly as printed, its topic label, and its text copied verbatim.\n\n"
    + RULES + "\n\nContext:\n\n{context}"
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9@+.]+", " ", text.lower()).strip()


def grounded(value: Optional[str], haystack: str) -> bool:
    """Is ``value`` really in the source?  Whitespace/punctuation-insensitive."""
    if not value:
        return True
    return _norm(value) in _norm(haystack)


def parse_pipe_notes(text: str) -> list[FieldNoteDraft]:
    """'date | area | text' lines -> notes, no LLM needed (returns [] if the shape is absent)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    notes = []
    for line in lines:
        parts = [p.strip() for p in line.split(" | ")]
        if len(parts) >= 3:
            notes.append(FieldNoteDraft(date_raw=parts[0], area=parts[1], text=" | ".join(parts[2:])))
    return notes if len(notes) >= max(2, int(0.5 * len(lines))) else []


def _retry(call: Callable[[], Any], attempts: int = 3):
    delay = 1.5
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as error:                      # network / quota / parse errors
            if is_quota_exhausted(error):
                raise
            logger.warning("LLM call failed (%s), retry %s/%s", error, attempt, attempts - 1)
            time.sleep(delay)
            delay *= 2


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------

def create_extractor(llm=None):
    """Return ``extract(documents, contexts, source_file=None) -> ExtractedDocument``.

    ``llm`` can be injected (tests / other providers); default is Gemini.
    """
    state = {"llm": llm}
    injected_llm = llm is not None

    def structured(schema):
        if state["llm"] is None:
            state["llm"] = get_llm()
        return state["llm"].with_structured_output(schema)

    def run(schema, prompt, context):
        messages = prompt.format_messages(context=context)
        identity = (
            f"injected:{type(state['llm']).__module__}.{type(state['llm']).__qualname__}:{id(state['llm'])}"
            if injected_llm else llm_cache_identity()
        )
        cache_payload = {
            "identity": identity,
            "schema": schema.model_json_schema(),
            "messages": [{"type": type(message).__name__, "content": message.content}
                         for message in messages],
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        cached = cache_get("llm", cache_key)
        if isinstance(cached, dict):
            return schema.model_validate(cached)
        try:
            result = _retry(lambda: structured(schema).invoke(messages))
        except Exception as error:
            if is_quota_exhausted(error):
                state["quota_exhausted"] = True
            raise
        if hasattr(result, "model_dump"):
            cache_put("llm", cache_key, result.model_dump(mode="json"))
        elif isinstance(result, dict):
            cache_put("llm", cache_key, result)
        return result

    def extract(documents, contexts: dict[str, str], source_file: Optional[str] = None) -> ExtractedDocument:
        warnings: list[str] = []
        parsed = parse_tables(documents)
        source_text = "\n".join(d.page_content for d in documents)
        ocr_text = [
            OCRPageText(page_number=doc.metadata.get("page"), text=doc.page_content)
            for doc in documents if doc.metadata.get("element_type") == "ocr_text"
        ]

        # ---- document info (LLM, grounded) ---------------------------------
        info = DocumentInfo()
        info_context = contexts.get("document_info", "")
        if info_context.strip():
            try:
                info = run(DocumentInfo, INFO_PROMPT, info_context)
                for field in ("name", "organization", "department", "designation",
                              "email", "phone", "title", "revision", "date", "date_range"):
                    value = getattr(info, field, None)
                    if value and not grounded(value, source_text):
                        warnings.append(f"document_info.{field} dropped: '{value}' not found in source")
                        setattr(info, field, None)
            except Exception as error:
                warnings.append(f"document_info extraction failed: {error}")
        else:
            warnings.append("document_info: no context retrieved")

        # ---- field notes (pattern first, LLM fallback, verified) -----------
        notes: list[FieldNote] = []
        notes_context = contexts.get("field_notes", "")
        if notes_context.strip():
            drafts = parse_pipe_notes(notes_context)
            if not drafts and state.get("quota_exhausted"):
                warnings.append("field_notes extraction skipped: LLM quota exhausted")
            elif not drafts:
                try:
                    drafts = run(FieldNotesResult, NOTES_PROMPT, notes_context).notes
                except Exception as error:
                    warnings.append(f"field_notes extraction failed: {error}")
            for draft in drafts:
                ok = grounded(draft.text, source_text)
                notes.append(FieldNote(**draft.model_dump(), verified=ok))
                if not ok:
                    warnings.append(f"field note not found verbatim in source: '{draft.text[:60]}'")

        # ---- assemble + sanity checks --------------------------------------
        master = parsed["master_records"]
        expected = sum(d.metadata.get("n_rows", 0) for d in documents
                       if d.metadata.get("table_id") in parsed["table_ids"].get("master_records", []))
        if master and expected and len(master) != expected:
            warnings.append(f"master_records: parsed {len(master)} rows but tables hold {expected}")

        report = {
            "tables_found": sum(1 for d in documents if d.metadata.get("element_type") == "table"),
            "master_records": len(master),
            "monthly_performance": len(parsed["monthly_performance"]),
            "value_groups": len(parsed["value_groups"]),
            "data_quality_flags": len(parsed["data_quality_flags"]),
            "field_notes": len(notes),
            "ancillary_tables": len(parsed["ancillary_tables"]),
            "stitched_tables": sum(1 for d in documents if d.metadata.get("merged_parts", 1) > 1),
            "duplicate_ids": sorted({r.record_id for r in master if "duplicate_id" in r.flags}),
            "ambiguous_dates": sum(1 for r in master if "date_ambiguous" in r.flags),
        }

        return ExtractedDocument(
            source_file=source_file,
            document_info=info,
            master_records=master,
            monthly_performance=parsed["monthly_performance"],
            value_groups=parsed["value_groups"],
            data_quality_flags=parsed["data_quality_flags"],
            field_notes=notes,
            other_tables=parsed["other_tables"],
            ancillary_tables=parsed["ancillary_tables"],
            ocr_text=ocr_text,
            warnings=warnings,
            report=report,
        )

    return extract
