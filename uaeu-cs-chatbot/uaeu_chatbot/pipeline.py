"""High-level orchestration helpers to build or load the UAEU CS catalog index."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import faiss

from .chunking import Chunk, make_chunks
from .config import CATALOG_INDEX_URL, DEFAULT_REBUILD, INDEX_ROOT
from .indexing import build_faiss_index, load_index, persist_index
from .scraper import Course, build_courses, discover_course_urls


def build_corpus_from_catalog(index_url: str = CATALOG_INDEX_URL) -> List[Chunk]:
    """Discover, harvest, extract, and chunk all courses from the catalog index."""
    course_urls = discover_course_urls(index_url)
    records: List[Course] = build_courses(course_urls)
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(make_chunks(record))
    return chunks


def build_and_persist() -> Tuple[faiss.IndexFlatIP, List[Chunk]]:
    """Rebuild the entire RAG stack and persist vectors + metadata."""
    chunks = build_corpus_from_catalog()
    index, chunks = build_faiss_index(chunks)
    persist_index(index, chunks)
    return index, chunks


def load_or_build(rebuild: bool | None = None) -> Tuple[faiss.IndexFlatIP, List[Chunk]]:
    """Load cached artifacts, rebuilding them when requested or missing."""
    if rebuild is None:
        rebuild = DEFAULT_REBUILD
    faiss_path = INDEX_ROOT / "vectors.faiss"
    meta_path = INDEX_ROOT / "metadata.jsonl"
    if rebuild or not (faiss_path.exists() and meta_path.exists()):
        return build_and_persist()
    return load_index()


__all__ = ["build_corpus_from_catalog", "build_and_persist", "load_or_build"]

