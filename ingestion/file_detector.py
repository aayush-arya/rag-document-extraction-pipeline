from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".docx": "docx",
    ".doc": "doc",
    ".xls": "xls",
}


def detect_file_type(file_path: str) -> str:
    extension = Path(file_path).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension or '(no extension)'}. "
            f"Supported: .pdf, .xlsx, .docx"
        )

    return SUPPORTED_EXTENSIONS[extension]
