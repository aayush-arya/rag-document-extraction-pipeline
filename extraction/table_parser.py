"""Deterministic table -> typed-record parser.

An LLM is the wrong tool to copy 64 rows x 9 columns: it drops rows, shortens
values and "fixes" numbers.  Rows that already sit in a table are therefore
parsed directly (header-name matching, then number / date normalisation) and
the LLM is only used for what has no table structure.

Every table that is not recognised as a typed table is still kept, losslessly,
as a ``GenericTable`` - nothing in the file is silently dropped.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
from typing import Optional

from ingestion.table_utils import NULLISH, norm_name

from .schema import (
    DataQualityFlag, GenericTable, MasterRecord, MonthlyPerformance,
    StructuredTable, ValueGroup, ValueItem,
)

MASTER_ALIASES = {
    "record_id": ["id", "record id", "record", "ticket id", "ref", "reference", "case id", "transaction id"],
    "name": ["name", "full name", "customer", "customer name", "employee", "employee name"],
    "date_raw": ["date", "record date", "event date", "timestamp", "created date", "updated date", "last updated", "order date", "transaction date"],
    "entity_site": ["entity site", "entity", "site", "region", "branch"],
    "location": ["location", "office", "city"],
    "department": ["department", "dept", "business unit", "division", "team"],
    "category": ["category", "type", "class", "classification"],
    "amount": ["amount", "total amount", "sales", "revenue", "total"],
    "quantity": ["quantity", "qty", "units", "count"],
    "unit_price": ["unit price", "price per unit", "unit cost"],
    "discount": ["discount", "discount pct", "discount percent", "discount amount"],
    "priority": ["priority", "urgency"],
    "email": ["email", "email address"],
    "phone": ["phone", "phone number", "telephone"],
    "product": ["product", "item", "service"],
    "code": ["code", "product code", "sku"],
    "score": ["score", "rating"],
    "last_updated_raw": ["last updated", "updated at", "modified date"],
    "metric_a": ["metric a", "amount", "value", "total"],
    "metric_b": ["metric b", "measure b"],
    "status": ["status", "record status", "state"],
    "owner": ["owner", "assignee", "assigned to", "assigned group"],
    "manager": ["manager", "supervisor", "lead"],
    "notes": ["notes", "note", "remarks", "comment", "comments"],
}
MONTHLY_ALIASES = {
    "month": ["month", "period"],
    "requests": ["requests", "volume", "request count", "total requests"],
    "resolved": ["resolved", "closed", "completed", "resolved count"],
    "avg_hrs": ["avg hrs", "avg hours", "average hours"],
    "p95_hrs": ["p95 hrs", "p95 hours", "p95"],
    "csat": ["csat", "satisfaction", "customer satisfaction"],
    "variance_pct": ["variance", "variance pct", "variance percent", "variance %"],
    "comment": ["comment", "comments", "notes"],
}
DQ_ALIASES = {
    "flag_id": ["flag", "flag id", "id"],
    "area": ["area"],
    "severity": ["severity", "priority"],
    "description": ["description", "details"],
}

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


# --------------------------------------------------------------------------
# value helpers
# --------------------------------------------------------------------------

def is_null(value: Optional[str]) -> bool:
    return value is None or value.strip().lower() in NULLISH


def text_or_none(value: Optional[str]) -> Optional[str]:
    return None if is_null(value) else value.strip()


def to_number(value: Optional[str]):
    """'1,131' -> 1131, '24.6' -> 24.6, '+12.2%' -> 12.2, '—' -> None."""
    if is_null(value):
        return None
    cleaned = value.strip().replace(",", "").rstrip("%").lstrip("+")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return int(number) if number.is_integer() and "." not in cleaned else number


def parse_date(raw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """-> (iso_date | None, flag | None).  Only unambiguous dates get an ISO value."""
    if is_null(raw):
        return None, None
    text = raw.strip()
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", text)
    if m:
        return _safe(int(m[1]), int(m[2]), int(m[3]))
    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})", text)
    if m and m[2][:3].lower() in MONTHS:
        return _safe(int(m[3]), MONTHS[m[2][:3].lower()], int(m[1]))
    m = re.fullmatch(r"([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})", text)
    if m and m[1][:3].lower() in MONTHS:
        return _safe(int(m[3]), MONTHS[m[1][:3].lower()], int(m[2]))
    if re.fullmatch(r"[A-Za-z]{3,9}\.?\s+\d{1,2}", text) and text.split()[0][:3].lower() in MONTHS:
        return None, "date_missing_year"
    m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})", text)
    if m:
        a, b, year = int(m[1]), int(m[2]), int(m[3])
        year += 2000 if year < 100 else 0
        if a > 12 and b <= 12:
            return _safe(year, b, a)           # DD-MM-YYYY, unambiguous
        if b > 12 and a <= 12:
            return _safe(year, a, b)           # MM-DD-YYYY, unambiguous
        if a == b:
            return _safe(year, a, b)
        return None, "date_ambiguous"
    return None, "date_unparsed"


def _safe(year: int, month: int, day: int) -> tuple[Optional[str], Optional[str]]:
    try:
        return date(year, month, day).isoformat(), None
    except ValueError:
        return None, "date_invalid"


# --------------------------------------------------------------------------
# column matching
# --------------------------------------------------------------------------

def match_columns(columns: list[str], aliases: dict[str, list[str]]) -> dict[str, int]:
    """field -> column index (first column whose normalised name equals an alias)."""
    names = [norm_name(c) for c in columns]
    mapping: dict[str, int] = {}
    used: set[int] = set()
    for field, options in aliases.items():
        for option in options:
            index = next((i for i, n in enumerate(names) if n == option and i not in used), None)
            if index is not None:
                mapping[field] = index
                used.add(index)
                break
    return mapping


def _cell(row: list[str], mapping: dict[str, int], field: str) -> Optional[str]:
    index = mapping.get(field)
    return row[index] if index is not None and index < len(row) else None


# --------------------------------------------------------------------------
# typed tables
# --------------------------------------------------------------------------

def parse_master(meta: dict, mapping: dict[str, int]) -> list[MasterRecord]:
    records = []
    pages = meta.get("pages") or [meta.get("page", 1)]
    headers = list(meta.get("headers") or meta.get("columns", []))
    for row in meta["rows"]:
        raw_date = text_or_none(_cell(row, mapping, "date_raw"))
        iso, date_flag = parse_date(raw_date)
        notes = text_or_none(_cell(row, mapping, "notes"))
        owner = text_or_none(_cell(row, mapping, "owner"))
        flags = [date_flag] if date_flag else []
        if owner is None:
            flags.append("missing_owner")
        if notes and "duplicate" in notes.lower():
            flags.append("possible_duplicate")
        records.append(MasterRecord(
            record_id=text_or_none(_cell(row, mapping, "record_id")),
            name=text_or_none(_cell(row, mapping, "name")),
            date_raw=raw_date, date_iso=iso,
            entity_site=text_or_none(_cell(row, mapping, "entity_site")),
            location=text_or_none(_cell(row, mapping, "location"))
            or text_or_none(_cell(row, mapping, "entity_site")),
            department=text_or_none(_cell(row, mapping, "department")),
            category=text_or_none(_cell(row, mapping, "category")),
            amount=to_number(_cell(row, mapping, "amount")),
            quantity=to_number(_cell(row, mapping, "quantity")),
            unit_price=to_number(_cell(row, mapping, "unit_price")),
            discount=to_number(_cell(row, mapping, "discount")),
            priority=text_or_none(_cell(row, mapping, "priority")),
            email=text_or_none(_cell(row, mapping, "email")),
            phone=text_or_none(_cell(row, mapping, "phone")),
            product=text_or_none(_cell(row, mapping, "product")),
            code=text_or_none(_cell(row, mapping, "code")),
            score=to_number(_cell(row, mapping, "score")),
            last_updated_raw=text_or_none(_cell(row, mapping, "last_updated_raw")),
            metric_a=to_number(_cell(row, mapping, "metric_a")),
            metric_b=to_number(_cell(row, mapping, "metric_b")),
            status=text_or_none(_cell(row, mapping, "status")),
            owner=owner, notes=notes, source_pages=list(pages), flags=flags,
            manager=text_or_none(_cell(row, mapping, "manager")),
            source_fields={
                header: row[index] if index < len(row) else ""
                for index, header in enumerate(headers)
                if index not in set(mapping.values())
                and index < len(row)
                and row[index] not in (None, "")
            },
            source_file=meta.get("file_name") or meta.get("source"),
            sheet_name=meta.get("sheet_name") or meta.get("sheet"),
            block_range=meta.get("block_range"),
        ))
    ids = Counter(r.record_id for r in records if r.record_id)
    for record in records:
        if record.record_id and ids[record.record_id] > 1:
            record.flags.append("duplicate_id")
    return records


def parse_monthly(meta: dict, mapping: dict[str, int]) -> list[MonthlyPerformance]:
    rows = []
    for row in meta["rows"]:
        comment = text_or_none(_cell(row, mapping, "comment"))
        rows.append(MonthlyPerformance(
            month=text_or_none(_cell(row, mapping, "month")),
            requests=to_number(_cell(row, mapping, "requests")),
            resolved=to_number(_cell(row, mapping, "resolved")),
            avg_hrs=to_number(_cell(row, mapping, "avg_hrs")),
            p95_hrs=to_number(_cell(row, mapping, "p95_hrs")),
            csat=to_number(_cell(row, mapping, "csat")),
            variance_pct=to_number(_cell(row, mapping, "variance_pct")),
            comment=comment,
            flags=["data_gap"] if comment and "data gap" in comment.lower() else [],
        ))
    return rows


def parse_dq(meta: dict, mapping: dict[str, int]) -> list[DataQualityFlag]:
    return [DataQualityFlag(
        flag_id=text_or_none(_cell(row, mapping, "flag_id")),
        area=text_or_none(_cell(row, mapping, "area")),
        severity=text_or_none(_cell(row, mapping, "severity")),
        description=text_or_none(_cell(row, mapping, "description")),
    ) for row in meta["rows"]]


_VALUE = re.compile(r"^\s*(.+?)\s*[:=]\s*(.+?)\s*$")


def parse_value_group(meta: dict) -> Optional[ValueGroup]:
    """Single-column box of 'label: value' lines -> ValueGroup."""
    rows = [r[0] for r in meta["rows"] if r and r[0]]
    items = []
    for line in rows:
        m = _VALUE.match(line)
        if not m:
            return None
        raw = m.group(2)
        number = to_number(raw)
        unit = "%" if raw.strip().endswith("%") else None
        items.append(ValueItem(label=m.group(1), value_raw=raw, value_number=number, unit=unit))
    if not items:
        return None
    return ValueGroup(title=meta.get("title"), section=meta.get("section"),
                      pages=list(meta.get("pages") or []), items=items)


def to_generic(meta: dict) -> GenericTable:
    columns = meta.get("columns", [])
    rows = [{columns[i] if i < len(columns) else f"col_{i + 1}": v for i, v in enumerate(row)}
            for row in meta.get("rows", [])]
    return GenericTable(
        table_id=meta.get("table_id"), title=meta.get("title"), section=meta.get("section"),
        pages=list(meta.get("pages") or []), columns=list(columns),
        has_header=bool(meta.get("has_header")), n_rows=len(rows), rows=rows,
    )


def to_structured(meta: dict) -> StructuredTable:
    pages = meta.get("pages") or [meta.get("page")]
    return StructuredTable(
        table_name=meta.get("title") or meta.get("heading") or "Untitled table",
        page_number=pages[0] if pages else None,
        headers=list(meta.get("headers") or meta.get("columns", [])),
        rows=[list(row) for row in meta.get("rows", [])],
        file_name=meta.get("file_name"),
        sheet_name=meta.get("sheet_name") or meta.get("sheet"),
        sheet_index=meta.get("sheet_index"),
        block_range=meta.get("block_range"),
    )


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def parse_tables(documents) -> dict:
    """All table elements -> typed lists.  ``documents`` = cleaned loader output."""
    out = {"master_records": [], "monthly_performance": [], "value_groups": [],
           "data_quality_flags": [], "other_tables": [], "ancillary_tables": [],
           "table_ids": {}}

    for doc in documents:
        meta = doc.metadata
        if meta.get("element_type") != "table" or "rows" not in meta:
            continue
        columns = meta.get("columns", [])

        if meta.get("file_type") in {"xlsx", "xlsm"} and meta.get("secondary_sheet"):
            out["other_tables"].append(to_generic(meta))
            out["ancillary_tables"].append(to_structured(meta))
            continue

        if meta.get("has_header"):
            master = match_columns(columns, MASTER_ALIASES)
            identity_fields = {"record_id", "date_raw", "status", "entity_site", "department", "amount"}
            has_record_identity = (
                {"record_id", "date_raw", "status"} <= master.keys()
                or (len(master) >= 5 and bool({"record_id", "date_raw"} & master.keys()))
                or len(identity_fields.intersection(master)) >= 4
            )
            if has_record_identity:
                out["master_records"].extend(parse_master(meta, master))
                out["table_ids"].setdefault("master_records", []).append(meta.get("table_id"))
                continue
            monthly = match_columns(columns, MONTHLY_ALIASES)
            if {"month", "requests", "resolved"} <= monthly.keys() and len(monthly) >= 5:
                out["monthly_performance"].extend(parse_monthly(meta, monthly))
                out["table_ids"].setdefault("monthly_performance", []).append(meta.get("table_id"))
                continue
            dq = match_columns(columns, DQ_ALIASES)
            if {"flag_id", "description"} <= dq.keys() and len(dq) >= 3:
                out["data_quality_flags"].extend(parse_dq(meta, dq))
                out["table_ids"].setdefault("data_quality_flags", []).append(meta.get("table_id"))
                continue

        if len(columns) == 1:
            group = parse_value_group(meta)
            if group:
                out["value_groups"].append(group)
                continue

        out["other_tables"].append(to_generic(meta))
        out["ancillary_tables"].append(to_structured(meta))
    return out
