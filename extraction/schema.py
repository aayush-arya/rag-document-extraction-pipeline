"""Output schema.

The old schema held ONE name / date / email per document.  A real messy file
holds *lists* of things, so every repeating thing is a list of typed records.

Two kinds of models live here:
* final models (``ExtractedDocument`` and its parts)
* ``...Draft`` / ``...Result`` models - the small, flat shapes the LLM is asked
  to fill (kept simple so Gemini structured output stays reliable)

Raw text is always preserved (``date_raw`` ...); normalised values (``date_iso``)
are only filled when the raw value is unambiguous.
"""

import re
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, model_validator

Number = Union[int, float]


# ----------------------------------------------------------- LLM-filled ----

class DocumentInfo(BaseModel):
    document_type: Optional[str] = Field(default=None, description="Kind of document, e.g. resume, invoice, archive, report")
    title: Optional[str] = Field(default=None, description="Document title exactly as printed")
    revision: Optional[str] = Field(default=None, description="Revision / version label if present")
    date_range: Optional[str] = Field(default=None, description="Period the document covers, as printed")
    date: Optional[str] = Field(default=None, description="Main document date as printed")
    name: Optional[str] = Field(default=None, description="Person or organization the document is about")
    organization: Optional[str] = Field(default=None, description="Organization or company name")
    department: Optional[str] = Field(default=None, description="Department")
    designation: Optional[str] = Field(default=None, description="Job title or designation")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    summary: Optional[str] = Field(default=None, description="Two or three sentence summary of the document")


class FieldNoteDraft(BaseModel):
    date_raw: Optional[str] = Field(default=None, description="Date/time token exactly as printed, un-normalised")
    area: Optional[str] = Field(default=None, description="Topic or category label of the note, e.g. Billing")
    text: str = Field(description="The observation text, copied verbatim")


class FieldNotesResult(BaseModel):
    notes: list[FieldNoteDraft] = Field(default_factory=list, description="One entry per distinct note, in document order")


# --------------------------------------------------------- table-parsed ----

class MasterRecord(BaseModel):
    record_id: Optional[str] = None
    name: Optional[str] = None
    date_raw: Optional[str] = None
    date_iso: Optional[str] = Field(default=None, description="ISO date, only when the raw date is unambiguous")
    entity_site: Optional[str] = None
    location: Optional[str] = None
    department: Optional[str] = None
    category: Optional[str] = None
    amount: Optional[Number] = None
    quantity: Optional[Number] = None
    unit_price: Optional[Number] = None
    discount: Optional[Number] = None
    priority: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    product: Optional[str] = None
    code: Optional[str] = None
    score: Optional[Number] = None
    last_updated_raw: Optional[str] = None
    metric_a: Optional[Number] = None
    metric_b: Optional[Number] = None
    status: Optional[str] = None
    owner: Optional[str] = None
    manager: Optional[str] = None
    notes: Optional[str] = None
    source_fields: dict[str, str] = Field(default_factory=dict)
    source_file: Optional[str] = None
    sheet_name: Optional[str] = None
    block_range: Optional[str] = None
    source_pages: list[int] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_source_fields(cls, values):
        """Promote aliased and legacy semicolon key/value attributes to typed fields."""
        if not isinstance(values, dict):
            return values

        alias_targets = {
            "id": "record_id", "record id": "record_id", "record": "record_id",
            "ticket id": "record_id", "ref": "record_id",
            "name": "name", "full name": "name", "employee name": "name",
            "department": "department", "dept": "department", "business unit": "department",
            "location": "location", "office": "location", "city": "location",
            "date": "date_raw", "record date": "date_raw", "created date": "date_raw",
            "status": "status", "record status": "status",
            "amount": "amount", "total": "amount", "value": "amount",
            "priority": "priority", "urgency": "priority",
        }

        def normalized_key(key):
            return re.sub(r"[^a-z0-9]+", " ", str(key).lower()).strip()

        source = values.get("source_fields")
        if isinstance(source, str):
            source_pairs = {}
            for pair in source.split(";"):
                key, separator, value = pair.partition("=")
                if separator and key.strip():
                    source_pairs[key.strip()] = value.strip()
        elif isinstance(source, dict):
            source_pairs = dict(source)
        else:
            source_pairs = {}

        # Also accept raw source column names supplied directly to model_validate.
        inputs = dict(source_pairs)
        for key, value in values.items():
            target = alias_targets.get(normalized_key(key))
            if target and key != target:
                inputs.setdefault(key, value)

        residual = {}
        for key, value in inputs.items():
            target = alias_targets.get(normalized_key(key))
            if not target:
                residual[str(key)] = "" if value is None else str(value)
                continue
            if values.get(target) in (None, ""):
                values[target] = value
            if target == "location" and values.get("entity_site") in (None, ""):
                values["entity_site"] = value

        values["source_fields"] = residual
        if values.get("location") and not values.get("entity_site"):
            values["entity_site"] = values["location"]
        return values


class MonthlyPerformance(BaseModel):
    month: Optional[str] = None
    requests: Optional[Number] = None
    resolved: Optional[Number] = None
    avg_hrs: Optional[Number] = None
    p95_hrs: Optional[Number] = None
    csat: Optional[Number] = None
    variance_pct: Optional[Number] = None
    comment: Optional[str] = None
    flags: list[str] = Field(default_factory=list)


class ValueItem(BaseModel):
    label: str
    value_raw: str
    value_number: Optional[Number] = None
    unit: Optional[str] = None


class ValueGroup(BaseModel):
    """A titled box of 'label: value' lines (system snapshot, dashboard captures)."""
    title: Optional[str] = None
    section: Optional[str] = None
    pages: list[int] = Field(default_factory=list)
    items: list[ValueItem] = Field(default_factory=list)


class DataQualityFlag(BaseModel):
    flag_id: Optional[str] = None
    area: Optional[str] = None
    severity: Optional[str] = None
    description: Optional[str] = None


class FieldNote(FieldNoteDraft):
    verified: Optional[bool] = Field(default=None, description="True when the note text was found verbatim in the source")


class GenericTable(BaseModel):
    """Any table that is not one of the typed record tables above (kept lossless)."""
    table_id: Optional[str] = None
    title: Optional[str] = None
    section: Optional[str] = None
    pages: list[int] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    has_header: bool = True
    n_rows: int = 0
    rows: list[dict[str, str]] = Field(default_factory=list)


class StructuredTable(BaseModel):
    table_name: str
    page_number: Optional[int] = None
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    file_name: Optional[str] = None
    sheet_name: Optional[str] = None
    sheet_index: Optional[int] = None
    block_range: Optional[str] = None


class OCRPageText(BaseModel):
    page_number: Optional[int] = None
    text: str


class ExtractedDocument(BaseModel):
    source_file: Optional[str] = None
    document_info: DocumentInfo = Field(default_factory=DocumentInfo)
    master_records: list[MasterRecord] = Field(default_factory=list)
    monthly_performance: list[MonthlyPerformance] = Field(default_factory=list)
    value_groups: list[ValueGroup] = Field(default_factory=list)
    data_quality_flags: list[DataQualityFlag] = Field(default_factory=list)
    field_notes: list[FieldNote] = Field(default_factory=list)
    other_tables: list[GenericTable] = Field(default_factory=list)
    ancillary_tables: List[StructuredTable] = Field(default_factory=list)
    ocr_text: list[OCRPageText] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    report: dict[str, Any] = Field(default_factory=dict)
