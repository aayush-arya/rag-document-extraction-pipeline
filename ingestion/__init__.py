from .file_detector import detect_file_type
from .pdf_loader import load_pdf
from .excel_loader import load_excel
from .docx_loader import load_docx


def load_document(file_path: str):
    """Load a .pdf / .xlsx / .docx into a list of structured elements.

    Every element is a LangChain ``Document`` with ``metadata["element_type"]``
    of ``"table"`` (structured rows kept in metadata), ``"text"`` or
    ``"ocr_text"``.
    """
    file_type = detect_file_type(file_path)

    if file_type == "pdf":
        return load_pdf(file_path)

    if file_type == "xlsx":
        return load_excel(file_path)

    if file_type == "docx":
        return load_docx(file_path)

    if file_type == "doc":
        raise ValueError(
            "Old .doc files are not supported. Please convert the file to .docx first."
        )

    if file_type == "xls":
        raise ValueError(
            "Old .xls files are not supported. Please re-save the file as .xlsx first."
        )

    raise ValueError(f"Unsupported file type: {file_type}")
