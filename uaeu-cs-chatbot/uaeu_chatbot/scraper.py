"""Scrape UAEU catalog pages and convert them into structured Course records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
from urllib.parse import parse_qs, urljoin, urlparse

import re
import requests
from bs4 import BeautifulSoup

from .config import (
    ALLOWED_HOST,
    CATALOG_INDEX_URL,
    HEADINGS_TO_STOP,
    REQUEST_TIMEOUT,
    TARGET_SECTIONS,
    USER_AGENT,
)

# ------------------------------- Data models ------------------------------- #


@dataclass
class Course:
    url: str
    code: str
    title: str
    credits: str
    prerequisites: str
    description: str
    full_text: str


# ----------------------------- Helper functions ---------------------------- #

COURSE_PATH_RE = re.compile(r"/catalog/courses/course_\d+\.shtml", re.I)
RE_CODE = re.compile(r"\b[A-Z]{2,5}\s?-?\s?\d{3}\b")
RE_CREDITS_TRAILING = re.compile(r"\b(\d+)\s*(credit(?:s)?|credit hours)\b", re.I)
RE_CREDITS_LEADING = re.compile(r"credit(?:s)?(?: hours)?\s*:?\s*(\d+)", re.I)
RE_PREREQ_LINE = re.compile(r"pre[\-\s]?requisite", re.I)
RE_COREQ_LINE = re.compile(r"co[\-\s]?requisite", re.I)


def fetch(url: str) -> str:
    """Fetch a URL with a descriptive User-Agent."""
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.text


def normalize_label(text: str) -> str:
    """Lowercase label text and collapse whitespace."""
    return " ".join(text.lower().split())


def discover_course_urls(root_url: str = CATALOG_INDEX_URL) -> List[str]:
    """Return course URLs restricted to the target catalog sections."""
    html = fetch(root_url)
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []

    for heading in soup.find_all(["h3", "h4", "h5"]):
        label = normalize_label(heading.get_text(" ", strip=True))
        if label not in TARGET_SECTIONS:
            continue

        node = heading
        while node:
            node = node.find_next_sibling()
            if node is None:
                break
            name = getattr(node, "name", None)
            if name and name.lower() in HEADINGS_TO_STOP:
                break
            if name is None:
                continue
            for anchor in node.find_all("a", href=True):
                abs_url = urljoin(root_url, anchor["href"])
                parsed = urlparse(abs_url)
                if parsed.netloc.lower().endswith(ALLOWED_HOST) and COURSE_PATH_RE.search(parsed.path):
                    urls.append(abs_url)

    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url not in seen:
            unique.append(url)
            seen.add(url)
    return unique


def text_clean(value: str) -> str:
    """Normalize whitespace for consistent downstream parsing."""
    value = value.replace("\u00a0", " ")
    value = value.replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def course_content_root(soup: BeautifulSoup, url: str):
    """Return the div whose id matches the course id=... query parameter."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    raw_id = ""
    for key in ("id", "course"):
        val = query.get(key)
        if val:
            raw_id = val[0].strip()
            break
    if raw_id:
        for candidate in (raw_id, raw_id.upper(), raw_id.lower()):
            if not candidate:
                continue
            found = soup.find(id=candidate)
            if found:
                return found, raw_id
    return soup, raw_id


def iter_labeled_segments(root) -> List[tuple[str | None, str]]:
    """Yield (heading_label, text) pairs within the course div."""
    segments: list[tuple[str | None, str]] = []
    current_label: str | None = None
    for child in root.children:
        if getattr(child, "name", None):
            text = child.get_text(" ", strip=True)
            clean = text_clean(text)
            if not clean:
                continue
            name = child.name.lower()
            if name in {"h2", "h3", "h4", "h5", "strong"}:
                current_label = normalize_label(clean)
            segments.append((current_label, clean))
        else:
            clean = text_clean(str(child))
            if clean:
                segments.append((current_label, clean))
    return segments


def find_section_text(segments: Sequence[tuple[str | None, str]], predicate) -> str:
    """Return text belonging to the first heading that satisfies the predicate."""
    for idx, (label, text) in enumerate(segments):
        target_label = label or ""
        if predicate(target_label) or predicate(text):
            if predicate(text) and len(text.split()) > 4:
                return text
            collected: list[str] = []
            base = label
            j = idx + 1
            while j < len(segments):
                next_label, next_text = segments[j]
                if next_label != base:
                    break
                collected.append(next_text)
                j += 1
            if collected:
                return " ".join(collected)
            return text
    return ""


def detect_credits(segments: Sequence[tuple[str | None, str]], text_blob: str) -> str:
    """Find the credit hours string either in segments or the raw blob."""

    def from_text(value: str) -> str:
        match = RE_CREDITS_TRAILING.search(value)
        if match:
            return f"{match.group(1)} credits"
        match = RE_CREDITS_LEADING.search(value)
        if match:
            return f"{match.group(1)} credits"
        return ""

    for _, text in segments:
        found = from_text(text)
        if found:
            return found
    return from_text(text_blob)


def extract_code_from_url(url: str, raw_id: str) -> str:
    """Attempt to get the course code from the URL query, falling back to the raw id."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    candidate = (query.get("id") or query.get("course") or [])
    if candidate:
        raw = candidate[0].strip().upper()
        match = re.match(r"([A-Z]{2,5})\s?-?\s?(\d{3})", raw)
        if match:
            return f"{match.group(1)} {match.group(2)}"
        return raw
    return raw_id.upper() if raw_id else ""


def extract_course_fields(url: str) -> Course:
    """Fetch a course page and fill Course fields using heuristics."""
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")
    content_root, raw_id = course_content_root(soup, url)

    h1 = content_root.find("h1") or soup.find("h1")
    title = text_clean(h1.get_text(" ", strip=True)) if h1 else ""

    paragraph_nodes = content_root.find_all(["p", "li"])
    paragraphs = [text_clean(p.get_text(" ", strip=True)) for p in paragraph_nodes if p.get_text(strip=True)]

    segments = iter_labeled_segments(content_root)
    text_blob = text_clean(content_root.get_text("\n", strip=True))

    code = extract_code_from_url(url, raw_id)
    if not code:
        for text in paragraphs:
            match = RE_CODE.search(text)
            if match:
                code = re.sub(r"[\s-]+", " ", match.group(0).upper()).strip()
                break

    credits = detect_credits(segments, text_blob)

    prereq = find_section_text(segments, lambda value: bool(RE_PREREQ_LINE.search(value or "")))
    if not prereq:
        coreq = find_section_text(segments, lambda value: bool(RE_COREQ_LINE.search(value or "")))
        if coreq:
            prereq = f"Corequisite: {coreq}"

    desc_candidates = [t for t in paragraphs if len(t.split()) > 8 and not RE_PREREQ_LINE.search(t)]
    description = max(desc_candidates, key=lambda s: len(s), default="")

    full_parts = [code, title, credits, prereq, description]
    full_text = text_clean("\n\n".join([part for part in full_parts if part]))

    return Course(
        url=url,
        code=code,
        title=title,
        credits=credits,
        prerequisites=prereq,
        description=description,
        full_text=full_text,
    )


def build_courses(urls: Iterable[str]) -> List[Course]:
    """Fetch and parse all course URLs."""
    records: list[Course] = []
    for idx, url in enumerate(urls, start=1):
        try:
            records.append(extract_course_fields(url))
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[WARN] Failed to parse {url}: {exc}")
    return records


__all__ = [
    "Course",
    "build_courses",
    "discover_course_urls",
    "extract_course_fields",
    "normalize_label",
]

