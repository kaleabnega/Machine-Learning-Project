"""FastAPI application exposing the UAEU CS chatbot as a web experience."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import faiss

from .pipeline import load_or_build
from .retrieval import answer_query


class RAGState:
    """Holds the in-memory FAISS index and chunk metadata."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.chunks = []

    def ensure_ready(self) -> tuple[faiss.Index, list]:
        if self.index is None or not self.chunks:
            self.index, self.chunks = load_or_build()
        return self.index, self.chunks


APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"

state = RAGState()
app = FastAPI(title="UAEU CS Chatbot", version="1.0.0")

app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://uaeu-chatbot.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _warm_cache() -> None:
    """Warm the FAISS index at startup to avoid a slow first request."""
    state.ensure_ready()


# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request) -> HTMLResponse:
#     """Render the chat UI."""
#     return templates.TemplateResponse("index.html", {"request": request})


@app.get("/", response_class=HTMLResponse)
def home():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/ask")
async def ask(payload: Dict[str, Any]) -> JSONResponse:
    """Answer a user question via RAG."""
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    index, chunks = state.ensure_ready()
    response = answer_query(question, index, chunks)
    return JSONResponse({"answer": response})


__all__ = ["app"]
