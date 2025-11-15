"""Shared configuration for the UAEU CS chatbot stack."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Set

from dotenv import load_dotenv

# Resolve important directories relative to the repo root.
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_ROOT: Final[Path] = BASE_DIR / "web_corpus"
INDEX_ROOT: Final[Path] = BASE_DIR / "rag_index"
TEMPLATES_DIR: Final[Path] = Path(__file__).resolve().parent / "templates"
STATIC_DIR: Final[Path] = Path(__file__).resolve().parent / "static"

DATA_ROOT.mkdir(parents=True, exist_ok=True)
INDEX_ROOT.mkdir(parents=True, exist_ok=True)

# Load .env variables if present (mirrors the notebook workflow).
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

# Catalog scraping configuration.
CATALOG_INDEX_URL: Final[str] = os.getenv(
    "CATALOG_INDEX_URL",
    "https://www.uaeu.ac.ae/en/catalog/undergraduate/programs/bachelor-of-science-in-computer-science.shtml",
)
ALLOWED_HOST: Final[str] = os.getenv("CATALOG_ALLOWED_HOST", "uaeu.ac.ae")
TARGET_SECTIONS: Final[Set[str]] = {
    "major requirements (req. ch: 40)",
    "major electives (req. ch: 12)",
}
HEADINGS_TO_STOP: Final[Set[str]] = {"h2", "h3", "h4", "h5"}

USER_AGENT: Final[str] = os.getenv(
    "CRAWLER_USER_AGENT",
    "StudentAssistantBot/1.0 (+https://example.com; contact@example.com)",
)
REQUEST_TIMEOUT: Final[int] = int(os.getenv("REQUEST_TIMEOUT", "30"))

# Hugging Face inference configuration.
HF_API_TOKEN: Final[str | None] = os.getenv("HF_API_TOKEN")
HF_ROUTER_BASE: Final[str] = os.getenv("HF_ROUTER_BASE", "https://router.huggingface.co/v1")
HF_CHAT_MODEL: Final[str] = os.getenv(
    "HF_CHAT_MODEL",
    "meta-llama/Llama-3.1-8B-Instruct:novita",
)
HF_EMBED_MODEL: Final[str] = os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en")

GEN_TEMPERATURE: Final[float] = float(os.getenv("GEN_TEMPERATURE", "0.2"))
GEN_TOP_P: Final[float] = float(os.getenv("GEN_TOP_P", "0.95"))
GEN_MAX_TOKENS: Final[int] = int(os.getenv("GEN_MAX_TOKENS", "600"))
TOP_K_RETRIEVAL: Final[int] = int(os.getenv("TOP_K_RETRIEVAL", "5"))


# Toggle to force a rebuild when the web app starts.
DEFAULT_REBUILD: Final[bool] = os.getenv("CHATBOT_REBUILD", "false").lower() == "true"

