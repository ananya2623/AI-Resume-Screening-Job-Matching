"""
PDF text extraction utilities.

Primary extractor: pdfplumber (better layout handling).
Fallback: PyPDF2 (in case pdfplumber fails on a malformed PDF).
"""

import pdfplumber
from PyPDF2 import PdfReader


def extract_text_from_pdf(file_obj) -> str:
    """
    Extract text from an uploaded PDF file-like object.

    Tries pdfplumber first; falls back to PyPDF2 if pdfplumber
    returns no usable text or raises an error.
    """
    text = _extract_with_pdfplumber(file_obj)
    if text.strip():
        return text

    file_obj.seek(0)
    return _extract_with_pypdf2(file_obj)


def _extract_with_pdfplumber(file_obj) -> str:
    try:
        file_obj.seek(0)
        chunks = []
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    chunks.append(page_text)
        return "\n".join(chunks)
    except Exception:
        return ""


def _extract_with_pypdf2(file_obj) -> str:
    try:
        reader = PdfReader(file_obj)
        chunks = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(chunks)
    except Exception:
        return ""
