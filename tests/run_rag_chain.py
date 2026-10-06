"""Ask questions (needs GOOGLE_API_KEY):  python -m tests.run_rag_chain "your question" """
import sys

from ingestion import load_document
from preprocessing.cleaner import clean_documents
from preprocessing.chunker import chunk_documents
from rag.vectorstore import create_vectorstore
from rag.retriever import get_retriever
from rag.rag_chain import create_rag_chain

file_path = "data/input/sample.pdf"
chunks = chunk_documents(clean_documents(load_document(file_path)))
rag = create_rag_chain(get_retriever(create_vectorstore(chunks)))

for query in (sys.argv[1:] or ["What does the Aug 18 delivery note say about site code GGN?",
                                "Invoice batch 77B: how many records, and what does the reconciliation sheet say?"]):
    result = rag(query)
    print(f"\nQ: {query}\nA: {result['answer']}\nSources:")
    for d in result["documents"][:4]:
        print("  -", d.metadata.get("title") or d.metadata.get("section"), "p", d.metadata.get("pages"))
