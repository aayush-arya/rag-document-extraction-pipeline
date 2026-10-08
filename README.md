# RAG Document Extraction Pipeline

> An intelligent multi-format document extraction pipeline that transforms messy and unstructured PDF, Excel, and Word documents into structured text using Retrieval-Augmented Generation (RAG), LangChain, LangGraph, embeddings, vector search, and LLM-based extraction.

---

## 📌 Overview

The **RAG Document Extraction Pipeline** is designed to extract meaningful and structured information from messy, semi-structured, and unstructured documents.

The system accepts documents such as:

- PDF (`.pdf`)
- Excel (`.xlsx`, `.xlsm`)
- Word (`.doc`, `.docx`)

and processes them through a multi-stage pipeline involving:

**Document Ingestion → Cleaning → Chunking → Embeddings → Vector Search → RAG Retrieval → LLM Extraction → Validation → Structured TXT Output**

The goal is to make information buried inside large and poorly formatted documents easier to retrieve and convert into a consistent structured format.

---

## 🎯 Problem Statement

Real-world documents are rarely clean and consistently structured.

Information may be distributed across:

- Different pages
- Tables
- Paragraphs
- Broken rows and columns
- Repeated headers
- Irregular formatting
- Large documents
- Multiple file formats

Traditional rule-based extraction approaches often require a separate parser and extraction logic for every document format and structure.

This project addresses the problem by combining traditional document processing with modern LLM-based retrieval and extraction.

---

## 🚀 Key Features

- 📄 Multi-format document ingestion
- 📊 Excel and spreadsheet processing
- 📝 PDF text extraction
- 📃 Word document processing
- 🧹 Document cleaning and normalization
- ✂️ Intelligent document chunking
- 🧠 Semantic embeddings
- 🔎 Vector similarity search
- 📚 Retrieval-Augmented Generation (RAG)
- 🤖 LLM-powered information extraction
- 🧩 Structured output validation
- 🔄 LangGraph-based workflow orchestration
- 📦 Structured `.txt` output generation
- 🔐 Environment-based API key configuration
- 🧱 Modular and extensible architecture

### Excel workbook handling

Excel ingestion inspects every worksheet in `.xlsx` and `.xlsm` workbooks,
splits independent blocks separated by empty rows or columns, expands merged
cells, and prefers calculated formula values while retaining formulas when no
cached result exists. Multi-column blocks remain structured tables with their
headers, rows, sheet index/name, source filename, and A1 block range. One-column
notes remain document text. Recognized tables on the primary sheet enter the
typed table parser; untyped blocks and tables on secondary sheets are preserved
as ancillary tables.

Reports serialize every extracted record and table row; output generation has no
preview cap. Structured table parsing runs over the complete ingested document
elements, independently of the bounded retrieval contexts used for LLM-only
document metadata and field-note extraction. Report headers show the actual
record counts, and typed transaction records retain their original source
attributes in dedicated schema fields. Recognized aliases such as Name,
Department, Location, Date, Status, Amount, and Priority are mapped directly;
legacy semicolon-delimited key/value data is promoted during schema validation.
Only unmatched, non-empty metadata is retained in `source_fields`, which is
excluded from the human-readable master-record report.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │     Input Files     │
                    │                     │
                    │ PDF / XLSX / DOCX   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   File Detection    │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       Document Ingestion       │
              │                                │
              │ PyMuPDF | Pandas | python-docx│
              └───────────────┬────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Cleaning &          │
                    │ Normalization       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Chunking       │
                    │                     │
                    │ LangChain Splitter  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Embeddings      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FAISS Vector    │
                    │       Store         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    RAG Retriever    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     LangGraph       │
                    │     Workflow        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   LLM Extraction    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Validation      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Structured TXT File │
                    └─────────────────────┘