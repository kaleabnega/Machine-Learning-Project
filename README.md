UAEU CS Chatbot
================

An end-to-end Retrieval Augmented Generation (RAG) notebook that powers a question-answering chatbot for the United Arab Emirates University (UAEU) Computer Science program. The notebook scrapes the official catalog, extracts course metadata, builds a FAISS index with Hugging Face embeddings, and serves answers via a hosted Meta-llama model behind the Hugging Face router.

Key Features
------------
- Automated catalog harvesting – discovers course detail pages starting from the CS program root URL and follows only “Major Requirements” and “Major Electives” links.
- Robust field extraction – normalizes whitespace, infers course codes, credits, prerequisites, and descriptions even when DOM structure varies.
- Chunked metadata with aliases – each chunk carries code/title headers plus alias lines (`ITBP301`, `ITBP 301`, `ITBP-301`) so code-only queries retrieve the correct evidence.
- Hosted embeddings & chat – uses Hugging Face Inference (`BAAI/bge-small-en` feature extraction) for vectors and the Hugging Face router for `meta-llama/Llama-3.1-8B-Instruct:novita` responses; no local model downloads required.
- FAISS retrieval with hybrid matching – combines exact course-code hits with semantic similarity to ensure reliable coverage while remaining lightweight.

Requirements
------------
- Python 3.10+ (tested inside the project’s virtual environment).
- Packages installed via `pip install -r requirements.txt` (notebook expects at least: `beautifulsoup4`, `requests`, `huggingface_hub`, `openai`, `faiss-cpu`, `numpy`, `pandas`, `tqdm`).
- Hugging Face API token stored in `.env`:

```
HF_API_TOKEN=hf_xxx_your_token_here
```

Make sure the token has inference access to the selected embedding and chat models.

Notebook Workflow
-----------------
1. **Configuration cell** – sets catalog URLs, model IDs, retrieval hyper-parameters, and paths under `uaeu-cs-chatbot/rag_index/`.
2. **Discovery & extraction** – fetches the CS program page, filters course links from the allowed categories, retrieves each course page, and populates a `Course` dataclass.
3. **Chunking** – `make_chunks` writes description/prerequisite/credit chunks (or overlapping fallbacks) with alias metadata for better lexical coverage.
4. **Embedding & FAISS** – `hf_embed` sends prefixed text (`passage:` / `query:`) to Hugging Face Inference, normalizes vectors, and builds a `faiss.IndexFlatIP`.
5. **Persistence** – `persist_index` saves `vectors.faiss` and `metadata.jsonl`; `load_index` restores them when `REBUILD = False`.
6. **Retrieval & generation** – `retrieve` first grabs exact course-code matches, then runs semantic search; `hf_chat` crafts a response constrained to cited UAEU sources.

Running the Notebook
--------------------
1. Activate the project’s virtual environment and ensure dependencies are installed.
2. Populate `.env` with `HF_API_TOKEN` (and restart the kernel so the notebook picks it up).
3. Open `uaeu-cs-chatbot/chatbot.ipynb` and set the `REBUILD` flag:
   - `True` (default) to rescrape and rebuild `metadata.jsonl`/`vectors.faiss`.
   - Switch to `False` after a successful run to reuse the saved artifacts.
4. Execute the notebook top-to-bottom (`Run All`). The first run contacts the catalog and Hugging Face endpoints, so expect a few minutes for scraping + embedding.
5. Use the final “sample question” cell (or your own) to interact with the chatbot. Queries can reference course codes (`ITBP301`) or names (“Security Principles & Practice”).

Running the Web App
-------------------
1. Ensure dependencies are installed (`pip install -r requirements.txt`) and `.env` contains a valid `HF_API_TOKEN`.
2. From `uaeu-cs-chatbot/`, start the FastAPI server:

   ```
   uvicorn uaeu_chatbot.webapp:app --reload
   ```

3. Visit `http://127.0.0.1:8000` to use the styled UI. The first request may take a bit while the catalog index is rebuilt or loaded from `rag_index/`.

Troubleshooting
---------------
- **“HF_API_TOKEN is not configured”** – add the token to `.env` or export it before running the notebook/web app.
- **Hugging Face 4xx/5xx errors** – rerun the failing cell/request; the free inference tier occasionally queues requests.
- **Course reported as unknown** – rebuild the index so the latest chunks (with alias lines) are embedded, and double-check that the course lives under the targeted catalog sections.
- **Slow responses** – hosted inference latency depends on the HF plan. Consider reducing `TOP_K_RETRIEVAL`/`GEN_MAX_TOKENS`, or upgrade your HF account.

Repository Layout
-----------------
- `uaeu-cs-chatbot/chatbot.ipynb` – main notebook (scraping, indexing, retrieval, chat).
- `uaeu-cs-chatbot/rag_index/metadata.jsonl` – JSONL metadata produced by the notebook.
- `uaeu-cs-chatbot/rag_index/vectors.faiss` – FAISS index saved after embedding.
- `uaeu-cs-chatbot/uaeu_chatbot/` – reusable Python package + FastAPI app (`webapp.py`, templates, static assets).

Feel free to adapt the notebook for other UAEU programs by changing the root catalog URL and category filters, or to swap models by updating the config cell.
