from typing import Any, TypedDict


class ExtractionState(TypedDict, total=False):

    file_path: str

    documents: list[Any]            # structured elements from the loader
    cleaned_documents: list[Any]
    chunks: list[Any]

    vectorstore: Any
    retriever: Any
    retrieved_documents: list[Any]
    contexts: dict[str, str]        # section name -> retrieved context
    retrieval_trace: list[dict]

    extracted_data: Any
    warnings: list[str]
    attempts: int

    output_path: str
    json_path: str
    error: str
