"""Parse PDF and DOCX files into structured text with metadata."""

import os
from typing import Dict, List


def parse_file(file_path: str) -> List[Dict]:
    """Parse a PDF or DOCX file and return list of text chunks with metadata.

    Args:
        file_path: Absolute or relative path to the file to parse.

    Returns:
        List of dicts with keys "text" and "metadata".
        metadata contains "source" (filename) and "page" (page number).
    """
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)

    if ext == ".pdf":
        return _parse_pdf(file_path, filename)
    elif ext == ".docx":
        return _parse_docx(file_path, filename)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: .pdf, .docx")


def _parse_pdf(file_path: str, filename: str) -> List[Dict]:
    """Parse a PDF file using PyMuPDF.

    Args:
        file_path: Path to the PDF file.
        filename: Name of the file for metadata.

    Returns:
        List of dicts with text and metadata per page.
    """
    try:
        import fitz  # PyMuPDF

        documents = []
        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text()
                if text.strip():
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": filename,
                            "page": page_num + 1,
                        },
                    })
        return documents
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF file '{file_path}': {e}") from e


def _parse_docx(file_path: str, filename: str) -> List[Dict]:
    """Parse a DOCX file using python-docx.

    Args:
        file_path: Path to the DOCX file.
        filename: Name of the file for metadata.

    Returns:
        List of dicts with text and metadata per paragraph.
    """
    try:
        from docx import Document

        documents = []
        doc = Document(file_path)
        for para_num, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text.strip()
            if text:
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "page": para_num + 1,
                    },
                })
        return documents
    except Exception as e:
        raise RuntimeError(f"Failed to parse DOCX file '{file_path}': {e}") from e
