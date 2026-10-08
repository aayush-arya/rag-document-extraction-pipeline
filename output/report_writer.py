"""Complete, record-streaming plain-text extraction report writer."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _fmt(value) -> str:
    return "Not Found" if value in (None, "", []) else str(value)


def _cell(value) -> str:
    if value in (None, "", []):
        return "-"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{key}={item}" for key, item in value.items())
    return str(value)


def _record_value(row, key):
    value = _get(row, key)
    return _cell(value)


def _table(file, title, rows, columns, labels=None):
    file.write(f"\n{title} ({len(rows)})\n{'-' * (len(title) + 6)}\n")
    if not rows:
        file.write("None found\n")
        return
    labels = labels or {}
    file.write(" | ".join(labels.get(column, column) for column in columns) + "\n")
    for row in rows:
        file.write(" | ".join(_record_value(row, column) for column in columns) + "\n")


def write_txt(extracted_data, output_path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sections = (
        ("master_records", "Master Records"),
        ("monthly_performance", "Monthly Performance"),
        ("value_groups", "Value Groups"),
        ("data_quality_flags", "Data Quality Flags"),
        ("field_notes", "Field Notes"),
        ("ancillary_tables", "Ancillary Tables"),
    )
    report = dict(_get(extracted_data, "report", {}) or {})
    for key, label in sections:
        report[key] = len(_get(extracted_data, key, []) or [])

    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("STRUCTURED DOCUMENT EXTRACTION\n")
        file.write("================================\n")
        file.write(f"Source: {_fmt(_get(extracted_data, 'source_file'))}\n\n")
        file.write("EXTRACTION SUMMARY\n------------------\n")
        for key, label in sections:
            file.write(f"{label}: {report[key]}\n")

        file.write("\nDOCUMENT INFO\n-------------\n")
        info = _get(extracted_data, "document_info", {})
        info_fields = type(info).model_fields if isinstance(info, BaseModel) else info.keys()
        for key in info_fields:
            file.write(f"{key.replace('_', ' ').title()}: {_fmt(_get(info, key))}\n")

        file.write("\nEXTRACTION REPORT\n-----------------\n")
        for key, value in report.items():
            file.write(f"{key.replace('_', ' ').title()}: {_fmt(value)}\n")

        _table(
            file, "MASTER RECORDS", _get(extracted_data, "master_records", []),
            ["record_id", "name", "department", "location", "date_raw", "date_iso",
             "status", "amount", "priority", "entity_site", "category", "quantity",
             "unit_price", "discount", "email", "phone", "product", "code", "score",
             "last_updated_raw", "metric_a", "metric_b", "owner", "manager", "notes", "flags"],
            labels={
                "record_id": "ID", "name": "Name", "department": "Department",
                "location": "Location", "date_raw": "Date", "date_iso": "Date (ISO)",
                "status": "Status", "amount": "Amount", "priority": "Priority",
                "entity_site": "Entity / Site",
            },
        )
        _table(
            file, "MONTHLY PERFORMANCE", _get(extracted_data, "monthly_performance", []),
            ["month", "requests", "resolved", "avg_hrs", "p95_hrs", "csat", "variance_pct", "comment"],
        )

        value_groups = _get(extracted_data, "value_groups", [])
        file.write(f"\nVALUE GROUPS ({len(value_groups)})\n---------------\n")
        for group in value_groups:
            file.write(f"[{_fmt(_get(group, 'title'))}]\n")
            for item in _get(group, "items", []):
                file.write(f"  {_get(item, 'label')}: {_get(item, 'value_raw')}\n")

        _table(
            file, "DATA QUALITY FLAGS", _get(extracted_data, "data_quality_flags", []),
            ["flag_id", "area", "severity", "description"],
        )
        _table(file, "FIELD NOTES", _get(extracted_data, "field_notes", []),
               ["date_raw", "area", "text", "verified"])

        ocr_text = _get(extracted_data, "ocr_text", [])
        if ocr_text:
            file.write(f"\nOCR TRANSCRIPTS ({len(ocr_text)})\n---------------\n")
            for page_text in ocr_text:
                file.write(f"\nPage {_fmt(_get(page_text, 'page_number'))}\n")
                file.write("-" * 20 + "\n")
                file.write(_get(page_text, "text", "") + "\n")

        ancillary_tables = _get(extracted_data, "ancillary_tables", [])
        file.write(f"\nANCILLARY TABLES ({len(ancillary_tables)})\n-----------------\n")
        for table in ancillary_tables:
            name = _get(table, "table_name", "Untitled table")
            sheet_name = _get(table, "sheet_name")
            block_range = _get(table, "block_range")
            context = (
                f"Sheet {sheet_name}" + (f", {block_range}" if block_range else "")
                if sheet_name else f"Page {_fmt(_get(table, 'page_number'))}"
            )
            file.write(f"\n{name} ({context})\n")
            file.write("-" * (len(name) + 16) + "\n")
            headers = _get(table, "headers", [])
            file.write("Headers: " + (" | ".join(_cell(header) for header in headers)
                                      if headers else "(none)") + "\n")
            rows = _get(table, "rows", [])
            for row in rows:
                file.write(" | ".join(_cell(cell) for cell in row) + "\n")
            if not rows:
                file.write("None found\n")

        warnings = _get(extracted_data, "warnings", [])
        if warnings:
            file.write("\nWARNINGS\n--------\n")
            for warning in warnings:
                file.write(f"- {warning}\n")
    return str(path)
