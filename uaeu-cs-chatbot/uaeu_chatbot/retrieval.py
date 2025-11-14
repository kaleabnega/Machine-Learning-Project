"""Retrieval helpers and answer generation pipeline."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import re

import faiss

from .chunking import Chunk, normalize_course_code
from .config import TOP_K_RETRIEVAL
from .hf import hf_chat, hf_embed, normalize_rows


CODE_PATTERN = re.compile(r"\b([A-Z]{2,5})[-\s]?(\d{3})\b")

SYSTEM_PROMPT = (
    "You are a precise Student Assistant for the United Arab Emirates University (UAEU) Computer Science program. "
    "Answer ONLY using the provided UAEU sources (domain must be uaeu.ac.ae). "
    "If the sources are insufficient, say: 'I do not have sufficient UAEU catalog information to answer this question.' "
    "Do NOT mention or recommend any non-UAEU institutions or contacts. "
    "Use formal language, correct punctuation, and cite like [code • section • chunk_id] for every factual claim."
)


def detect_course_codes(text: str) -> List[str]:
    """Extract normalized course codes from free-form text."""
    found = []
    for match in CODE_PATTERN.finditer(text.upper()):
        found.append(f"{match.group(1)} {match.group(2)}")
    seen = set()
    ordered: List[str] = []
    for code in found:
        if code not in seen:
            ordered.append(code)
            seen.add(code)
    return ordered


def retrieve(query: str, index: faiss.Index, chunks: List[Chunk], top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[Chunk, float]]:
    """Combine exact course-code hits with semantic similarity search."""
    query_codes = detect_course_codes(query)
    code_set = set(query_codes)
    code_rank = {code: rank for rank, code in enumerate(query_codes)}
    exact_hits: List[Tuple[int, int, Chunk]] = []
    seen_indices = set()
    if code_set:
        for idx, chunk in enumerate(chunks):
            normalized = normalize_course_code(chunk.code or "")
            if normalized and normalized in code_set:
                rank = code_rank.get(normalized, len(code_rank))
                exact_hits.append((rank, idx, chunk))
                seen_indices.add(idx)
    augmented_query = query.strip()
    if code_set:
        aliases = ", ".join(sorted(code_set))
        augmented_query = f"{augmented_query} (course codes: {aliases})"
    qvec = hf_embed([augmented_query], mode="query")
    qvec = normalize_rows(qvec)
    scores, ids = index.search(qvec, top_k)
    results: List[Tuple[Chunk, float]] = []
    for _, idx, chunk in sorted(exact_hits, key=lambda item: (item[0], item[1])):
        results.append((chunk, 1.1))
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx == -1 or idx in seen_indices:
            continue
        seen_indices.add(idx)
        results.append((chunks[idx], float(score)))
        if len(results) >= top_k:
            break
    return results[:top_k]


def format_sources(evidence: Sequence[Tuple[Chunk, float]]) -> str:
    """Format retrieval results for prompting."""
    blocks = []
    for rank, (chunk, score) in enumerate(evidence, start=1):
        header = f"[{rank}] {chunk.code or chunk.title} • {chunk.section} • {chunk.chunk_id} • {chunk.url} • sim={score:.3f}"
        blocks.append(header + "\n" + chunk.text)
    return "\n\n---\n\n".join(blocks) if blocks else "(no sources)"


def answer_query(query: str, index: faiss.Index, chunks: List[Chunk], top_k: int = TOP_K_RETRIEVAL) -> str:
    """Retrieve supporting chunks and craft a citation-rich answer."""
    evidence = retrieve(query, index, chunks, top_k=top_k)
    context = format_sources(evidence)
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Sources:\n{context}\n\n"
        "Instructions:\n"
        "- Answer using ONLY the information in the Sources above.\n"
        "- Provide inline citations for every factual statement using the pattern [code • section • chunk_id].\n"
        "- If the question asks for information that the Sources do not state, say so clearly and suggest the official contact.\n"
        "- If multiple sources conflict, prefer registrar/official catalog pages and note the discrepancy."
    )
    return hf_chat(SYSTEM_PROMPT, user_prompt)


__all__ = ["answer_query", "retrieve", "format_sources", "detect_course_codes"]

