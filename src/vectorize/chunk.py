"""Parse the constitution text into article-level chunks with hierarchical metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Article:
    article_type: str  # "permanent" or "transitory"
    article_number: str
    title: str | None
    title_name: str | None
    chapter: str | None
    chapter_name: str | None
    text: str
    source: str


_TITLE_RE = re.compile(r"^(TITULO\s+[IVX]+)\s*$", re.IGNORECASE)
_TITLE_NAMED_RE = re.compile(r"^(TITULO\s+[IVX]+)\s+(.+)$", re.IGNORECASE)
_CHAPTER_RE = re.compile(r"^(CAPITULO\s+\d+)\s*$", re.IGNORECASE)
_CHAPTER_NAMED_RE = re.compile(r"^(CAPITULO\s+\d+)\s+(.+)$", re.IGNORECASE)
_TRANSITORY_RE = re.compile(
    r"^Art[íi]culo\s+transitorio\s+(\d+)\.\s*(.*)$",
    re.IGNORECASE,
)
_PERMANENT_RE = re.compile(
    r"^Art[íi]culo\s+(\d+)\.\s*(.*)$",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_articles(text: str, source: str = "constitucion-politica-colombia-1991.pdf") -> list[Article]:
    """Walk the document line by line and emit one Article per article boundary."""
    lines = text.splitlines()

    current_title: str | None = None
    current_title_name: str | None = None
    current_chapter: str | None = None
    current_chapter_name: str | None = None

    articles: list[Article] = []
    current: dict | None = None

    def flush() -> None:
        if current is None:
            return
        current["text"] = _clean(current["text"])
        if not current["text"]:
            return
        articles.append(
            Article(
                article_type=current["article_type"],
                article_number=current["article_number"],
                title=current["title"],
                title_name=current["title_name"],
                chapter=current["chapter"],
                chapter_name=current["chapter_name"],
                text=current["text"],
                source=source,
            )
        )

    def _is_name_line(candidate: str) -> bool:
        if not candidate:
            return False
        if _TITLE_RE.match(candidate) or _CHAPTER_RE.match(candidate):
            return False
        if _PERMANENT_RE.match(candidate) or _TRANSITORY_RE.match(candidate):
            return False
        if candidate.startswith("NOTA ") or candidate.startswith("Nota:"):
            return False
        return not ("Artículo" in candidate or "PARAGRAFO" in candidate.upper())

    pending = None  # ("title", "TITULO II") or ("chapter", "CAPITULO 1")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        if pending is not None:
            kind, code = pending
            if _is_name_line(line):
                if kind == "title":
                    current_title_name = line
                else:
                    current_chapter_name = line
                pending = None
                i += 1
                continue
            pending = None

        named_title = _TITLE_NAMED_RE.match(line)
        if named_title:
            flush()
            current = None
            current_title = named_title.group(1).upper()
            current_title_name = named_title.group(2).strip()
            current_chapter = None
            current_chapter_name = None
            pending = None
            i += 1
            continue

        bare_title = _TITLE_RE.match(line)
        if bare_title:
            flush()
            current = None
            current_title = bare_title.group(1).upper()
            current_title_name = None
            current_chapter = None
            current_chapter_name = None
            pending = ("title", current_title)
            i += 1
            continue

        named_chapter = _CHAPTER_NAMED_RE.match(line)
        if named_chapter:
            flush()
            current = None
            current_chapter = named_chapter.group(1).upper()
            current_chapter_name = named_chapter.group(2).strip()
            pending = None
            i += 1
            continue

        bare_chapter = _CHAPTER_RE.match(line)
        if bare_chapter:
            flush()
            current = None
            current_chapter = bare_chapter.group(1).upper()
            current_chapter_name = None
            pending = ("chapter", current_chapter)
            i += 1
            continue

        transitory = _TRANSITORY_RE.match(line)
        if transitory:
            flush()
            current = {
                "article_type": "transitory",
                "article_number": transitory.group(1),
                "title": current_title,
                "title_name": current_title_name,
                "chapter": current_chapter,
                "chapter_name": current_chapter_name,
                "text": transitory.group(2),
            }
            i += 1
            continue

        permanent = _PERMANENT_RE.match(line)
        if permanent:
            flush()
            current = {
                "article_type": "permanent",
                "article_number": permanent.group(1),
                "title": current_title,
                "title_name": current_title_name,
                "chapter": current_chapter,
                "chapter_name": current_chapter_name,
                "text": permanent.group(2),
            }
            i += 1
            continue

        if current is not None:
            current["text"] += " " + line
        i += 1

    flush()
    return articles
