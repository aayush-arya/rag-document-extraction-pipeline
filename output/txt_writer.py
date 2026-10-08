import json
from pathlib import Path


def _dump(extracted_data) -> dict:
    return extracted_data.model_dump()


def _fmt(value) -> str:
    return "Not Found" if value in (None, "", []) else str(value)


def _cell(value) -> str:
    if value in (None, "", []):
        return "-"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def write_json(extracted_data, output_path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(_dump(extracted_data), file, indent=2, ensure_ascii=False, default=str)
    return str(path)


def _table(file, title, rows, columns):
    file.write(f"\n{title} ({len(rows)})\n{'-' * (len(title) + 6)}\n")
    if not rows:
        file.write("None found\n")
        return
    file.write(" | ".join(columns) + "\n")
    for row in rows:
        file.write(" | ".join(_cell(row.get(c)) for c in columns) + "\n")


def write_txt(extracted_data, output_path) -> str:
    data = _dump(extracted_data)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write("STRUCTURED DOCUMENT EXTRACTION\n")
        file.write("================================\n")
        file.write(f"Source: {_fmt(data.get('source_file'))}\n\n")

        file.write("DOCUMENT INFO\n-------------\n")
        for key, value in data["document_info"].items():
            file.write(f"{key.replace('_', ' ').title()}: {_fmt(value)}\n")

        file.write("\nEXTRACTION REPORT\n-----------------\n")
        for key, value in data["report"].items():
            file.write(f"{key.replace('_', ' ').title()}: {_fmt(value)}\n")

        _table(file, "MASTER RECORDS", data["master_records"],
               ["record_id", "date_raw", "date_iso", "entity_site", "category",
                "metric_a", "metric_b", "status", "owner", "notes", "flags"])
        _table(file, "MONTHLY PERFORMANCE", data["monthly_performance"],
               ["month", "requests", "resolved", "avg_hrs", "p95_hrs", "csat", "variance_pct", "comment"])

        file.write(f"\nVALUE GROUPS ({len(data['value_groups'])})\n---------------\n")
        for group in data["value_groups"]:
            file.write(f"[{_fmt(group['title'])}]\n")
            for item in group["items"]:
                file.write(f"  {item['label']}: {item['value_raw']}\n")

        _table(file, "DATA QUALITY FLAGS", data["data_quality_flags"],
               ["flag_id", "area", "severity", "description"])
        _table(file, "FIELD NOTES", data["field_notes"], ["date_raw", "area", "text", "verified"])

        file.write(f"\nANCILLARY TABLES ({len(data['ancillary_tables'])})\n-----------------\n")
        for table in data["ancillary_tables"]:
            file.write(f"\n{table['table_name']} (Page {_fmt(table['page_number'])})\n")
            file.write("-" * (len(table["table_name"]) + 16) + "\n")
            file.write("Headers: " + (" | ".join(_cell(header) for header in table["headers"])
                                      if table["headers"] else "(none)") + "\n")
            for row in table["rows"]:
                file.write(" | ".join(_cell(cell) for cell in row) + "\n")
            if not table["rows"]:
                file.write("None found\n")

        if data["warnings"]:
            file.write("\nWARNINGS\n--------\n")
            for warning in data["warnings"]:
                file.write(f"- {warning}\n")

    return str(path)
