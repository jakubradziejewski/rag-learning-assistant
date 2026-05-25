# RAG Learning Assistant

A local AI tool to upload PDFs and ask questions about them. Everything runs on your machine — no cloud APIs, no API keys.

## How it works

1. Upload a PDF → Docling parses it (text, tables, OCR)
2. Content is split into structure-based chunks (headings, sections, paragraphs)
3. Chunks are embedded with `mxbai-embed-large` and stored in ChromaDB
4. You ask a question → relevant chunks retrieved → `llama3.2` answers with sources

## Architecture

```
Docker Desktop
├── backend container (FastAPI + Docling)  :8000
└── chromadb container (vector store)      :8001

Docker Model Runner (host, not container)
├── ai/mxbai-embed-large  (embeddings)
└── ai/llama3.2           (LLM)
```

## Prerequisites

**Docker Desktop 4.42+** — [docker.com/products/docker-desktop](https://docker.com/products/docker-desktop)

After installing, run:
```bash
docker desktop enable model-runner --tcp 12434
```

**uv**
```bash
pip install uv
```

**Pull models (once)**
Important: for pulling models do not use eduroam, as it seems to be blocking cloudfare. 
```bash
docker model pull ai/mxbai-embed-large
docker model pull ai/llama3.2
```

## Running

```bash
git clone https://github.com/your-username/rag-learning-assistant
cd rag-learning-assistant
uv lock
docker compose up --build
```

Open **http://localhost:8000/docs**

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload a PDF |
| `POST` | `/documents/query` | Ask a question |

Query body:
```json
{
  "question": "What is gradient descent?",
  "n_results": 5,
  "temperature": 0.0
}
```

`temperature: 0.0` = strict context only. Higher = model uses own knowledge too (clearly stated).

## Common commands

```bash
docker compose up               # start
docker compose up --build       # rebuild (after changing Dockerfile or pyproject.toml)
docker compose down             # stop
docker compose down -v          # stop + delete all stored vectors
docker system prune             # clean up unused images and cache
```

## Notes

PyTorch is CPU-only inside the container — Docling's models are small enough that CPU is fine.
However, upload take considerable amount of time, switching to GPU may be beneficial, as well as, looking at other parsers - Dockling is particulary computationally explensive. Apart from it DMR (Docker Model Runner) handles all GPU-heavy work (embeddings, LLM) natively on the host. What was needed to make this work:

- **CPU PyTorch** — added `torch` and `torchvision` as direct dependencies pointing at the CPU index, otherwise uv pulls CUDA via Docling's transitive deps
- **Volume mount** — set `ENV UV_PROJECT_ENVIRONMENT=/usr/local` so packages install outside `/app`, preventing the volume mount from overwriting them at runtime
- **ChromaDB healthcheck** — the image has no `curl` or `wget`; removed healthcheck entirely for dev
- **Dimension mismatch** — if you switch embedding models, run `docker compose down -v` to reset the collection
- **DMR TCP** — must explicitly run `docker desktop enable model-runner --tcp 12434` or containers can't reach it

## Stack

| | |
|---|---|
| Backend | FastAPI + uv |
| PDF parsing | Docling + HierarchicalChunker |
| Embeddings | mxbai-embed-large (DMR) |
| Vector store | ChromaDB |
| LLM | llama3.2 (DMR) |

## What's next

- Hybrid search (BM25 + vector)
- Filter noisy short chunks
- Conversation history (PostgreSQL)
- Spaced repetition scheduler (SM-2/FSRS)
- LLM teaching agent
- Frontend