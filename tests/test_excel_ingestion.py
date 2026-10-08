from datetime import date
from pathlib import Path

import openpyxl

from extraction.table_parser import parse_tables
from extraction.schema import ExtractedDocument
from ingestion import load_document
from output.report_writer import write_txt
from preprocessing.chunker import chunk_documents
from preprocessing.cleaner import clean_documents


def _workbook(path: Path):
    workbook = openpyxl.Workbook()
    primary = workbook.active
    primary.title = "Primary"
    primary.merge_cells("A1:I1")
    primary["A1"] = "Operations Register"
    primary.append(["ID", "Date", "Location", "Category", "Metric A", "Metric B", "Status", "Owner", "Notes"])
    primary["A2"] = "ID"
    primary["B2"] = "Date"
    primary["C2"] = "Location"
    primary["D2"] = "Category"
    primary["E2"] = "Metric A"
    primary["F2"] = "Metric B"
    primary["G2"] = "Status"
    primary["H2"] = "Owner"
    primary["I2"] = "Notes"
    primary.append(["R-1", 46000, "North", "Sales", 0.1256, 2, "Open", "Ari", "keep"])
    primary["E3"].number_format = "0.00%"

    primary["A5"] = "Formula"
    primary["B5"] = "Result"
    primary["A6"] = "Total"
    primary["B6"] = "=SUM(1,2)"
    primary["A8"] = "2026"
    primary["A9"] = 123

    secondary = workbook.create_sheet("Raw Imports")
    secondary.append(["Key", "Value"])
    secondary.append(["MM/4471", 123456789.12345678])
    secondary.append(["MM/4502", 3.25])
    workbook.save(path)


def test_excel_workbook_preserves_blocks_types_and_sheet_context(tmp_path):
    path = tmp_path / "multi.xlsx"
    _workbook(path)

    loaded = load_document(str(path))
    tables = [doc for doc in loaded if doc.metadata.get("element_type") == "table"]
    text = [doc for doc in loaded if doc.metadata.get("element_type") == "text"]
    assert len(tables) == 3

    master_table = next(doc for doc in tables if doc.metadata["sheet_name"] == "Primary"
                        and doc.metadata["block_range"] == "Primary!A1:I3")
    assert master_table.metadata["file_name"] == "multi.xlsx"
    assert master_table.metadata["headers"] == [
        "ID", "Date", "Location", "Category", "Metric A", "Metric B", "Status", "Owner", "Notes"
    ]
    assert master_table.metadata["rows"] == [
        ["R-1", "2025-12-09", "North", "Sales", "12.56%", "2", "Open", "Ari", "keep"]
    ]

    formula_table = next(doc for doc in tables if doc.metadata["block_range"] == "Primary!A5:B6")
    assert formula_table.metadata["rows"] == [["Total", "=SUM(1,2)"]]
    imported = next(doc for doc in tables if doc.metadata["sheet_name"] == "Raw Imports")
    assert imported.metadata["sheet_index"] == 2
    assert imported.metadata["block_range"] == "'Raw Imports'!A1:B3"

    cleaned_text = clean_documents(text)
    assert any(doc.page_content == "2026\n123" for doc in cleaned_text)
    chunks = chunk_documents(clean_documents(loaded), max_tokens=120, token_counter=len)
    master_chunks = [chunk for chunk in chunks if chunk.metadata.get("block_range") == "Primary!A1:I3"]
    assert master_chunks
    assert all(chunk.metadata["file_name"] == "multi.xlsx" for chunk in master_chunks)
    assert all(chunk.metadata["sheet_name"] == "Primary" for chunk in master_chunks)
    assert all(chunk.metadata["sheet_index"] == 1 for chunk in master_chunks)

    parsed = parse_tables(clean_documents(loaded))
    assert len(parsed["master_records"]) == 1
    assert parsed["master_records"][0].date_iso == "2025-12-09"
    assert parsed["master_records"][0].metric_a == 12.56
    ancillary = parsed["ancillary_tables"]
    assert len(ancillary) == 3
    raw_import = next(table for table in ancillary if table.sheet_name == "Raw Imports")
    assert raw_import.file_name == "multi.xlsx"
    assert raw_import.sheet_index == 2
    assert raw_import.block_range == "'Raw Imports'!A1:B3"
    # Preserve the exact numeric literal stored in the XLSX XML (rather than
    # rounding it again through Python's binary float conversion).
    assert raw_import.rows[0] == ["MM/4471", "123456789.1234568"]
    assert any(table.block_range == "Primary!A8:A9" for table in ancillary)


def test_xlsm_uses_excel_loader(tmp_path):
    path = tmp_path / "macro-enabled.xlsm"
    _workbook(path)
    docs = load_document(str(path))
    assert docs
    assert all(doc.metadata["file_type"] == "xlsm" for doc in docs)


def test_excel_numeric_date_format_is_normalized_to_day(tmp_path):
    path = tmp_path / "date.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["Timestamp", "Value"])
    sheet.append([date(2025, 1, 2), 1])
    workbook.save(path)

    table = next(doc for doc in load_document(str(path)) if doc.metadata["element_type"] == "table")
    assert table.metadata["rows"] == [["2025-01-02", "1"]]


def test_merged_data_cells_are_propagated_across_rows(tmp_path):
    path = tmp_path / "merged-data.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["Category", "Amount"])
    sheet.append(["Shared", 10])
    sheet.append([None, 20])
    sheet.merge_cells("A2:A3")
    workbook.save(path)

    table = next(doc for doc in load_document(str(path)) if doc.metadata["element_type"] == "table")
    assert table.metadata["rows"] == [["Shared", "10"], ["Shared", "20"]]


def test_sparse_multirow_merged_headers_are_combined(tmp_path):
    path = tmp_path / "sparse-headers.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"] = "ID"
    sheet.merge_cells("B1:C1")
    sheet["B1"] = "Financial"
    sheet["B2"] = "Budget"
    sheet["C2"] = "Actual"
    sheet.append([])
    sheet["A3"] = "R-1"
    sheet["B3"] = 10
    sheet["C3"] = 12
    workbook.save(path)

    table = next(doc for doc in load_document(str(path)) if doc.metadata["element_type"] == "table")
    assert table.metadata["headers"] == ["ID", "Financial / Budget", "Financial / Actual"]
    assert table.metadata["rows"] == [["R-1", "10", "12"]]


def test_far_right_side_table_survives_empty_columns_and_exports_z111(tmp_path):
    path = tmp_path / "wide-side-block.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Export"

    # Left-side table establishes a distant main grid. The right block is in
    # Y:AD with four completely empty spacer columns U:X.
    for row, values in enumerate(
        [["ID", "Name"], ["REC-1", "Main row"]], start=1
    ):
        for column, value in enumerate(values, start=1):
            sheet.cell(row=row, column=column, value=value)

    sheet.merge_cells("Y93:AD93")
    sheet["Y93"] = "CUSTOMER EXPORT - PART 2 / CONTINUATION"
    for column, value in zip(range(25, 31),
                             ["Category", "Code", "Department", "Score", "Qty", "Status"]):
        sheet.cell(row=95, column=column, value=value)
    for row in range(96, 112):
        sheet.cell(row=row, column=25, value="Customer")
        sheet.cell(row=row, column=26, value=f"VAL-{3700 + row}")
        sheet.cell(row=row, column=27, value="Sales")
        sheet.cell(row=row, column=28, value=80 + row / 100)
        sheet.cell(row=row, column=29, value=row - 95)
        sheet.cell(row=row, column=30, value="Active")
    sheet["Z111"] = "VAL-3778"
    workbook.save(path)

    documents = load_document(str(path))
    side_table = next(
        doc for doc in documents
        if doc.metadata.get("element_type") == "table"
        and doc.metadata.get("block_range") == "Export!Y95:AD111"
    )
    assert side_table.metadata["title"] == "CUSTOMER EXPORT - PART 2 / CONTINUATION"
    assert side_table.metadata["headers"] == [
        "Category", "Code", "Department", "Score", "Qty", "Status"
    ]
    assert side_table.metadata["rows"][-1][1] == "VAL-3778"

    parsed = parse_tables(clean_documents(documents))
    assert any(
        table.table_name == "CUSTOMER EXPORT - PART 2 / CONTINUATION"
        and table.rows[-1][1] == "VAL-3778"
        for table in parsed["ancillary_tables"]
    )
    output = tmp_path / "wide-side-block.txt"
    write_txt(ExtractedDocument(
        source_file=str(path),
        ancillary_tables=parsed["ancillary_tables"],
        other_tables=parsed["other_tables"],
    ), output)
    assert "VAL-3778" in output.read_text(encoding="utf-8")


def test_lower_right_table_after_large_blank_gap_exports_ai340(tmp_path):
    path = tmp_path / "deep-lower-right.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Ledger"
    sheet["A1"] = "ID"
    sheet["B1"] = "Name"
    for row in range(2, 302):
        sheet.cell(row=row, column=1, value=f"REC-{row - 1:05d}")
        sheet.cell(row=row, column=2, value=f"Person {row}")

    sheet.merge_cells("AA330:AJ330")
    sheet["AA330"] = "LEGACY DATA / DO NOT SORT"
    headers = [
        "Location", "Unit Price", "Manager", "Departmer", "Category",
        "Score", "Status", "Owner", "Code", "Notes",
    ]
    for column, header in enumerate(headers, start=27):
        sheet.cell(row=331, column=column, value=header)
    for row in range(332, 341):
        for column, value in enumerate(
            ["Mumbai", 125.75, "A. Singh", "Operations", "Legacy",
             88.5, "Active", "A. Singh", f"VAL-{5200 + row - 332}", "keep"],
            start=27,
        ):
            sheet.cell(row=row, column=column, value=value)
    sheet["AI340"] = "VAL-5262"
    sheet["AA1144"] = "FRAGMENT 6 - SOURCE UNKNOWN"
    sheet["AA1145"] = "VAL-6662"
    workbook.save(path)

    documents = load_document(str(path))
    table = next(
        doc for doc in documents
        if doc.metadata.get("element_type") == "table"
        and doc.metadata.get("block_range") == "Ledger!AA330:AJ340"
    )
    assert table.metadata["title"] == "LEGACY DATA / DO NOT SORT"
    assert len(table.metadata["rows"]) == 9
    assert table.metadata["rows"][-1][8] == "VAL-5262"

    parsed = parse_tables(clean_documents(documents))
    assert any(
        ancillary.table_name == "LEGACY DATA / DO NOT SORT"
        and ancillary.rows[-1][8] == "VAL-5262"
        for ancillary in parsed["ancillary_tables"]
    )
    fragment = next(
        ancillary for ancillary in parsed["ancillary_tables"]
        if ancillary.block_range == "Ledger!AA1144:AA1145"
    )
    assert fragment.rows == [["FRAGMENT 6 - SOURCE UNKNOWN"], ["VAL-6662"]]
    output = tmp_path / "deep-lower-right.txt"
    write_txt(ExtractedDocument(
        source_file=str(path),
        ancillary_tables=parsed["ancillary_tables"],
        other_tables=parsed["other_tables"],
    ), output)
    report = output.read_text(encoding="utf-8")
    assert "VAL-5262" in report
    assert "VAL-6662" in report


def test_unmapped_columns_from_typed_excel_tables_are_reported(tmp_path):
    path = tmp_path / "typed-with-side-values.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Messy_Data"
    sheet.append([
        "ID", "Name", "Date", "Status", "Department", "Location",
        "Amount", "Priority", "Legacy Payload",
    ])
    sheet.append([
        "REC-01144", "Diya Malhotra", "2024-04-22", "In Review",
        "Support", "Bengaluru", 279259.39, "Low", "VAL-6662",
    ])
    workbook.save(path)

    documents = load_document(str(path))
    parsed = parse_tables(clean_documents(documents))
    residual = next(
        table for table in parsed["ancillary_tables"]
        if table.table_name.endswith("additional columns")
    )
    assert residual.headers == ["Legacy Payload"]
    assert residual.rows == [["VAL-6662"]]

    output = tmp_path / "typed-with-side-values.txt"
    write_txt(ExtractedDocument(
        source_file=str(path),
        master_records=parsed["master_records"],
        ancillary_tables=parsed["ancillary_tables"],
    ), output)
    assert "VAL-6662" in output.read_text(encoding="utf-8")
