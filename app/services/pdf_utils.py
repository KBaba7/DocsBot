from pathlib import Path
from io import BytesIO

from pypdf import PdfReader


def extract_pdf_pages(file_path: Path, max_pages: int | None = None) -> tuple[list[tuple[int, str]], int]:
    reader = PdfReader(str(file_path))
    total_pages = len(reader.pages)
    page_limit = min(total_pages, max_pages) if max_pages else total_pages
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages[:page_limit], start=1):
        pages.append((index, (page.extract_text() or "").strip()))
    return pages, total_pages


def extract_pdf_text(file_path: Path, max_pages: int | None = None) -> tuple[str, int]:
    pages, total_pages = extract_pdf_pages(file_path, max_pages=max_pages)
    text = "\n\n".join(page_text for _, page_text in pages).strip()
    return text, total_pages


def extract_pdf_pages_from_bytes(content: bytes, max_pages: int | None = None) -> tuple[list[tuple[int, str]], int]:
    reader = PdfReader(BytesIO(content))
    total_pages = len(reader.pages)
    page_limit = min(total_pages, max_pages) if max_pages else total_pages
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages[:page_limit], start=1):
        pages.append((index, (page.extract_text() or "").strip()))
    return pages, total_pages


def extract_pdf_text_from_bytes(content: bytes, max_pages: int | None = None) -> tuple[str, int]:
    pages, total_pages = extract_pdf_pages_from_bytes(content, max_pages=max_pages)
    text = "\n\n".join(page_text for _, page_text in pages).strip()
    return text, total_pages


def count_pdf_pages_from_bytes(content: bytes) -> int:
    reader = PdfReader(BytesIO(content))
    return len(reader.pages)
