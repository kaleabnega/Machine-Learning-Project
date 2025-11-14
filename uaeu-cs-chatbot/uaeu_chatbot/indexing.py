"""FAISS indexing, persistence, and corpus orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .chunking import Chunk
from .config import INDEX_ROOT
from .hf import hf_embed, normalize_rows


def build_faiss_index(chunks: List[Chunk]) -> Tuple[faiss.IndexFlatIP, List[Chunk]]:
    """Embed chunks and build an IndexFlatIP."""
    texts = [chunk.text for chunk in chunks]
    embeddings = hf_embed(texts, mode="document")
    embeddings = normalize_rows(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks


def persist_index(index: faiss.Index, chunks: List[Chunk]) -> None:
    """Write FAISS and metadata artifacts to disk."""
    faiss_path = INDEX_ROOT / "vectors.faiss"
    meta_path = INDEX_ROOT / "metadata.jsonl"
    faiss.write_index(index, str(faiss_path))
    with meta_path.open("w", encoding="utf-8") as handle:
        for row_id, chunk in enumerate(chunks):
            rec = {**asdict(chunk), "row_id": row_id}
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_index() -> Tuple[faiss.Index, List[Chunk]]:
    """Load FAISS and metadata artifacts from disk."""
    faiss_path = INDEX_ROOT / "vectors.faiss"
    meta_path = INDEX_ROOT / "metadata.jsonl"
    if not faiss_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index artifacts not found. Rebuild the catalog first.")
    index = faiss.read_index(str(faiss_path))
    chunks: List[Chunk] = []
    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=rec["chunk_id"],
                    url=rec["url"],
                    code=rec["code"],
                    title=rec["title"],
                    section=rec["section"],
                    text=rec["text"],
                )
            )
    return index, chunks


__all__ = ["build_faiss_index", "persist_index", "load_index"]

