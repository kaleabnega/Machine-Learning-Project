"""Chunking utilities and metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import re

from .scraper import Course


@dataclass
class Chunk:
    chunk_id: str
    url: str
    code: str
    title: str
    section: str
    text: str


def normalize_course_code(code: str) -> str:
    """Return a normalized 'ABCD 123' variant for matching."""
    if not code:
        return ""
    code = code.upper().strip()
    match = re.match(r"([A-Z]{2,5})\s*-?\s*(\d{3})", code)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return code


def course_code_aliases(code: str) -> List[str]:
    """Return compact/dashed aliases so retrieval can match bare codes."""
    norm = normalize_course_code(code)
    if not norm:
        return []
    compact = norm.replace(" ", "")
    dashed = norm.replace(" ", "-")
    return sorted({norm, compact, dashed})


def _chunk_text(course: Course, section: str, body: str) -> str:
    """Prefix chunk body with contextual headers and alias lines."""
    parts = [part for part in (course.code, course.title) if part]
    header = " | ".join(parts)
    if section:
        header = " - ".join([value for value in (header, section.title()) if value])
    alias_line = ""
    if course.code:
        aliases = course_code_aliases(course.code)
        if aliases:
            alias_line = "Aliases: " + " | ".join(aliases)
    text_parts = [value for value in (header, body, alias_line) if value]
    return "\n".join(text_parts)


def make_chunks(course: Course, max_chars: int = 1100, overlap: int = 120) -> List[Chunk]:
    """Create citable chunks for each Course record."""
    base = (course.code or course.title or "course").replace(" ", "_")

    chunks: List[Chunk] = []
    if course.description:
        text = _chunk_text(course, "description", course.description)
        chunks.append(
            Chunk(
                chunk_id=f"{base}::description",
                url=course.url,
                code=course.code,
                title=course.title or course.code,
                section="description",
                text=text,
            )
        )
    if course.prerequisites:
        text = _chunk_text(course, "prerequisites", course.prerequisites)
        chunks.append(
            Chunk(
                chunk_id=f"{base}::prerequisites",
                url=course.url,
                code=course.code,
                title=course.title or course.code,
                section="prerequisites",
                text=text,
            )
        )
    if course.credits:
        text = _chunk_text(course, "credits", course.credits)
        chunks.append(
            Chunk(
                chunk_id=f"{base}::credits",
                url=course.url,
                code=course.code,
                title=course.title or course.code,
                section="credits",
                text=text,
            )
        )

    if not chunks and course.full_text:
        text = course.full_text
        start = 0
        count = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            piece = text[start:end]
            cid = f"{base}::full_{count:02d}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    url=course.url,
                    code=course.code,
                    title=course.title or course.code,
                    section="full",
                    text=piece,
                )
            )
            if end >= len(text):
                break
            start = end - overlap
            count += 1

    return chunks


__all__ = ["Chunk", "make_chunks", "normalize_course_code", "course_code_aliases"]

