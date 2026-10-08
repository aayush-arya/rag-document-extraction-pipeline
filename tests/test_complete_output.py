import json

from langchain_core.documents import Document

from extraction.schema import ExtractedDocument, MasterRecord
from extraction.table_parser import parse_tables
from output.json_writer import write_json
from output.report_writer import write_txt
from preprocessing.chunker import chunk_documents


def test_report_and_json_write_every_record_beyond_previous_preview_size(tmp_path):
    records = [
        MasterRecord(record_id=f"REC-{index:05d}", status="Closed")
        for index in range(1, 1306)
    ]
    data = ExtractedDocument(
        source_file="large.xlsx",
        master_records=records,
        report={"master_records": len(records)},
    )

    text_path = tmp_path / "all-records.txt"
    json_path = tmp_path / "all-records.json"
    write_txt(data, text_path)
    write_json(data, json_path)

    text = text_path.read_text(encoding="utf-8")
    assert "Master Records: 1305" in text
    assert "MASTER RECORDS (1305)" in text
    assert "REC-00001" in text
    assert "REC-01200" in text
    assert "REC-01305" in text

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["master_records"]) == 1305
    assert payload["master_records"][0]["record_id"] == "REC-00001"
    assert payload["master_records"][-1]["record_id"] == "REC-01305"


def test_aliases_classify_wide_sales_records_into_typed_fields():
    headers = [
        "ID", "Name", "Department", "Location", "Date", "Status", "Amount",
        "Quantity", "Unit Price", "Discount",
    ]
    row = ["REC-1", "Ari", "Finance", "Pune", "2026-10-08", "Closed",
           "1250.75", "3", "500", "10%"]
    document = Document(page_content=" | ".join(row), metadata={
        "element_type": "table",
        "table_id": "t001",
        "has_header": True,
        "columns": headers,
        "headers": headers,
        "rows": [row],
    })

    parsed = parse_tables([document])
    assert len(parsed["master_records"]) == 1
    assert not parsed["ancillary_tables"]
    record = parsed["master_records"][0]
    assert record.record_id == "REC-1"
    assert record.department == "Finance"
    assert record.location == "Pune"
    assert record.entity_site == "Pune"
    assert record.amount == 1250.75
    assert record.quantity == 3
    assert record.unit_price == 500
    assert record.discount == 10
    assert record.source_fields == {}


def test_legacy_semicolon_key_value_fields_are_promoted():
    record = MasterRecord.model_validate({
        "source_fields": (
            "ID=REC-00001; Name=Ari Sharma; Department=Finance; Location=Mumbai; "
            "Date=2026-10-08; Status=Closed; Amount=1250.50; Priority=High; "
            "Custom Tag=legacy"
        )
    })

    assert record.record_id == "REC-00001"
    assert record.name == "Ari Sharma"
    assert record.department == "Finance"
    assert record.location == "Mumbai"
    assert record.entity_site == "Mumbai"
    assert record.date_raw == "2026-10-08"
    assert record.status == "Closed"
    assert record.amount == 1250.5
    assert record.priority == "High"
    assert record.source_fields == {"Custom Tag": "legacy"}


def test_report_writes_primary_fields_as_separate_columns(tmp_path):
    data = ExtractedDocument(master_records=[
        MasterRecord(
            record_id="REC-00001", name="Ari Sharma", department="Finance",
            location="Mumbai", date_raw="2026-10-08", status="Closed",
            amount=1250.5, priority="High",
            source_fields={"Custom Tag": "legacy"},
        )
    ])
    output = tmp_path / "typed-fields.txt"
    write_txt(data, output)
    report = output.read_text(encoding="utf-8")
    assert "ID | Name | Department | Location | Date | Date (ISO) | Status | Amount | Priority" in report
    assert "REC-00001 | Ari Sharma | Finance | Mumbai | 2026-10-08" in report
    assert "ID=REC-00001;" not in report
    assert "Custom Tag=legacy" not in report


def test_secondary_sheet_records_remain_lossless_ancillary_tables():
    headers = ["ID", "Date", "Status", "Department", "Amount"]
    row = ["R-2", "2026-10-08", "Open", "Operations", "7.25"]
    document = Document(page_content=" | ".join(row), metadata={
        "element_type": "table",
        "table_id": "t002",
        "file_type": "xlsx",
        "secondary_sheet": True,
        "sheet_name": "Transactions",
        "block_range": "Transactions!A1:E2",
        "has_header": True,
        "columns": headers,
        "headers": headers,
        "rows": [row],
    })

    parsed = parse_tables([document])
    assert not parsed["master_records"]
    assert len(parsed["ancillary_tables"]) == 1
    assert parsed["ancillary_tables"][0].sheet_name == "Transactions"
    assert parsed["ancillary_tables"][0].block_range == "Transactions!A1:E2"
    assert parsed["ancillary_tables"][0].rows == [row]


def test_table_chunks_cover_all_rows_without_dropping_the_tail():
    row_count = 1305
    rows = [[f"REC-{index:05d}", "Closed"] for index in range(1, row_count + 1)]
    document = Document(
        page_content="",
        metadata={
            "element_type": "table",
            "title": "Large table",
            "columns": ["ID", "Status"],
            "has_header": True,
            "rows": rows,
            "n_rows": row_count,
        },
    )

    chunks = chunk_documents([document], max_tokens=120, token_counter=len)
    table_chunks = [chunk for chunk in chunks if chunk.metadata.get("element_type") == "table"]
    assert sum(chunk.metadata["row_end"] - chunk.metadata["row_start"] + 1
               for chunk in table_chunks) == row_count
    assert min(chunk.metadata["row_start"] for chunk in table_chunks) == 1
    assert max(chunk.metadata["row_end"] for chunk in table_chunks) == row_count
    rendered = "\n".join(chunk.page_content for chunk in table_chunks)
    assert "REC-00001 | Closed" in rendered
    assert "REC-01200 | Closed" in rendered
    assert "REC-01305 | Closed" in rendered
