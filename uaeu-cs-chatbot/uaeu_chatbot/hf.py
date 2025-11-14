"""Hugging Face client helpers for embedding + chat."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from huggingface_hub import InferenceClient
from openai import OpenAI

from .config import (
    GEN_MAX_TOKENS,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    HF_API_TOKEN,
    HF_CHAT_MODEL,
    HF_EMBED_MODEL,
    HF_ROUTER_BASE,
)

_CHAT_CLIENT: Optional[OpenAI] = None
_EMBED_CLIENT: Optional[InferenceClient] = None


def get_chat_client() -> OpenAI:
    """Return a cached OpenAI client pointed at the HF router."""
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN is not configured.")
    global _CHAT_CLIENT
    if _CHAT_CLIENT is None:
        _CHAT_CLIENT = OpenAI(api_key=HF_API_TOKEN, base_url=HF_ROUTER_BASE)
    return _CHAT_CLIENT


def get_embed_client() -> InferenceClient:
    """Return a cached HF inference client for embeddings."""
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN is not configured.")
    global _EMBED_CLIENT
    if _EMBED_CLIENT is None:
        _EMBED_CLIENT = InferenceClient(provider="hf-inference", api_key=HF_API_TOKEN)
    return _EMBED_CLIENT


def hf_embed(texts: List[str], mode: str = "document") -> np.ndarray:
    """Embed texts using the configured HF feature-extraction model."""
    client = get_embed_client()
    prefix = "query: " if mode == "query" else "passage: "
    prefixed = [prefix + text.replace("\n", " ") for text in texts]
    data = client.feature_extraction(prefixed, model=HF_EMBED_MODEL)
    if data is None:
        raise RuntimeError("Received no embeddings from Hugging Face inference API.")
    if isinstance(data, np.ndarray):
        return data.astype("float32")
    if isinstance(data, list):
        if not data:
            raise RuntimeError("Received empty embedding list from Hugging Face inference API.")
        if isinstance(data[0], list):
            vectors = data
        else:
            vectors = [data]
        return np.array(vectors, dtype="float32")
    raise RuntimeError(f"Unexpected embedding response type: {type(data)}")


def hf_chat(system_prompt: str, user_prompt: str) -> str:
    """Call the chat model hosted on the HF router."""
    client = get_chat_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=HF_CHAT_MODEL,
        messages=messages,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        max_tokens=GEN_MAX_TOKENS,
    )
    choice = response.choices[0]
    return (choice.message.content or "").strip()


def normalize_rows(v: np.ndarray) -> np.ndarray:
    """L2-normalize embedding rows."""
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


__all__ = ["hf_embed", "hf_chat", "normalize_rows"]

