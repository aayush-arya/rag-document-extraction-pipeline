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

from typing import Any, Optional, Union

from pydantic import BaseModel, Field

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
    date_raw: Optional[str] = None
    date_iso: Optional[str] = Field(default=None, description="ISO date, only when the raw date is unambiguous")
    entity_site: Optional[str] = None
    category: Optional[str] = None
    metric_a: Optional[Number] = None
    metric_b: Optional[Number] = None
    status: Optional[str] = None
    owner: Optional[str] = None
    notes: Optional[str] = None
    source_pages: list[int] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


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


class ExtractedDocument(BaseModel):
    source_file: Optional[str] = None
    document_info: DocumentInfo = Field(default_factory=DocumentInfo)
    master_records: list[MasterRecord] = Field(default_factory=list)
    monthly_performance: list[MonthlyPerformance] = Field(default_factory=list)
    value_groups: list[ValueGroup] = Field(default_factory=list)
    data_quality_flags: list[DataQualityFlag] = Field(default_factory=list)
    field_notes: list[FieldNote] = Field(default_factory=list)
    other_tables: list[GenericTable] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    report: dict[str, Any] = Field(default_factory=dict)
