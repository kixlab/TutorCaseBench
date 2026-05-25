import os
from PyPDF2 import PdfReader
from typing import List


# =============================================================================
# utils/analyze_paper_utils.py — PDF Text Extractor
# =============================================================================
#
# Extracts text from PDF files using PyPDF2.
# Inserts #PAGE{n}# markers per page and supports excluded_pages.
#
# =============================================================================


def extract_text_from_pdf(path: str, excluded_pages: list = None) -> str:
    """
    Extract all text from a PDF file.
    
    Iterates through all pages in the PDF and extracts text content.
    Handles extraction errors gracefully by returning empty strings for problematic pages.
    
    Args:
        path: Full file path to the PDF document
        
    Returns:
        str: Concatenated text from all pages, separated by newlines
        
    Raises:
        FileNotFoundError: If the PDF file path does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    
    reader = PdfReader(path)
    pages: List[str] = []
    page_idx = 0
    for page in reader.pages:
        page_idx += 1
        if excluded_pages and page_idx in excluded_pages:
            pages.append("")
            continue
        try:
            pages.append("#PAGE{}#".format(page_idx) + (page.extract_text() or ""))
        except Exception:
            pages.append("")
    return "\n".join(pages)