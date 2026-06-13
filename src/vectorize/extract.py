"""Extract text from the constitution PDF."""

from __future__ import annotations

from pathlib import Path

import pymupdf


def extract_text(pdf_path: str | Path) -> str:
    """Return the full text of a PDF, one page per chunk joined by newlines."""
    path = Path(pdf_path)
    if not path.exists():
        msg = f"PDF not found: {path}"
        raise FileNotFoundError(msg)

    doc = pymupdf.open(path)
    try:
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text())
    finally:
        doc.close()

    return "\n".join(pages)
