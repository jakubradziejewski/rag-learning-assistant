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

> Can't enable Model Runner (common on **Docker Desktop for Linux**, where the runner won't start)? Use the [Ollama path](#alternative-ollama-instead-of-docker-model-runner) instead.

**uv**
```bash
pip install uv
```

**Pull models (once)**
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
Open **http://localhost:8501** for the Streamlit UI.

## Alternative: Ollama instead of Docker Model Runner

Use this path if Docker Model Runner isn't available — e.g. **Docker Desktop on Linux**, where the DMR runner can't start (`/dev/dri` missing) and the VM has no GPU access. Ollama runs the models **natively on the host**, so it can also drive your NVIDIA GPU directly.

> **Note:** the committed `docker-compose.yml` is already wired for this path — `DMR_BASE_URL` points at `http://host.docker.internal:11434/v1` for both `backend` and `streamlit`. To switch back to Docker Model Runner, set those to `http://model-runner.docker.internal:12434/engines/llama.cpp/v1`.

**1. Install Ollama** and expose it to containers:
```bash
curl -fsSL https://ollama.com/install.sh | sh

# bind to 0.0.0.0 so containers can reach it via host.docker.internal
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0:11434"\n' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

**2. Pull the models and alias them** to the `ai/*` names the code expects:
```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
ollama cp llama3.2 ai/llama3.2
ollama cp mxbai-embed-large ai/mxbai-embed-large
```

The `ollama cp` aliases mean **no code changes** — `EMBED_MODEL` / `LLM_MODEL` (the `ai/...` names) resolve straight to Ollama.

**3. Run** as usual:
```bash
docker compose up --build
```

Verify: `curl http://localhost:11434/v1/models` should list `ai/llama3.2` and `ai/mxbai-embed-large`, and `ollama ps` shows them loaded on the GPU.

Binding Ollama to `0.0.0.0` exposes the API on your LAN — fine on a trusted network; revert to `127.0.0.1` if you roam untrusted networks.

## Streamlit (spaced repetition MVP)

This UI lets you upload PDFs, generate questions/flashcards per chunk, and run a daily FSRS review session stored in JSON.

With `docker compose up`, Streamlit runs automatically at **http://localhost:8501**.

If you want to run Streamlit on the host (outside Docker), set:
```bash
export DMR_BASE_URL="http://localhost:12434/engines/llama.cpp/v1"
export CHROMADB_HOST="localhost"
export CHROMADB_PORT="8001"
```

Then run:
```bash
uv run streamlit run streamlit_app.py
```

Study data is stored at `data/srs_state.json` (gitignored).

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