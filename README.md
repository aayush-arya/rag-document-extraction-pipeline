# Messy-document extraction (LangGraph + hybrid RAG)

Supports `.pdf`, `.xlsx`, `.docx`.

    pip install -r requirements.txt
    cp .env.example .env            # add GOOGLE_API_KEY
    pytest -q                       # offline tests (no key, no model download)
    python -m tests.run_graph data/input/sample.pdf
    python -m tests.run_rag_chain "What does the Aug 18 note say about GGN?"

Scanned pages / images: install the Tesseract binary (OCR_BACKEND=tesseract) or
set OCR_BACKEND=gemini to use Gemini vision.

## Flow
ingest -> clean -> chunk -> vectorstore -> retrieve -> extract -> validate -> output
(ingest error -> END; failed LLM section -> back to retrieve with wider k, max 2 attempts)

## Removed dependencies
pymupdf, faiss-cpu, pandas, langchain-community, langchain-text-splitters
(replaced by pdfplumber, an in-memory numpy+BM25 hybrid index, openpyxl, a custom chunker).
