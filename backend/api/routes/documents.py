import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.core.rag.parser import parse_pdf
from backend.core.rag.embedder import embed_text
from backend.core.rag.llm import ask
from backend.core.storage.vector_store import store_chunks, search

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    include_chunks: bool = False,
    max_chunks: int | None = None,
):
    logger.info("Upload request received: filename=%s", file.filename)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{doc_id}.pdf"

    logger.info("Saving uploaded PDF: doc_id=%s path=%s", doc_id, dest)

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info("Starting PDF parsing: doc_id=%s", doc_id)
    chunks = parse_pdf(dest)
    logger.info("PDF parsing finished: doc_id=%s chunks=%s", doc_id, len(chunks))

    logger.info("Starting embeddings: doc_id=%s chunks=%s", doc_id, len(chunks))

    embeddings = [embed_text(chunk["text"]) for chunk in chunks]
    logger.info("Embeddings finished: doc_id=%s chunks=%s", doc_id, len(embeddings))
    logger.info("Storing chunks: doc_id=%s", doc_id)
    stored = store_chunks(doc_id, chunks, embeddings)
    logger.info("Upload complete: doc_id=%s stored_chunks=%s", doc_id, stored)

    response = {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_stored": stored,
    }

    if include_chunks:
        limited = chunks if max_chunks is None else chunks[: max_chunks]
        response["chunks"] = limited
        response["chunks_returned"] = len(limited)

    return response


class QueryRequest(BaseModel):
    question: str
    n_results: int = 5
    temperature: float = 0.0


@router.post("/query")
def query(req: QueryRequest):
    logger.info("Query request received: n_results=%s temperature=%s", req.n_results, req.temperature)

    logger.info("Generating query embedding")
    query_embedding = embed_text(req.question)
    logger.info("Searching vector store: n_results=%s", req.n_results)
    results = search(query_embedding, n_results=req.n_results)

    context_chunks = [r["text"] for r in results]
    logger.info("Calling LLM with context_chunks=%s", len(context_chunks))
    answer = ask(req.question, context_chunks, temperature=req.temperature)
    logger.info("Query complete: sources=%s", len(results))

    return {
        "question": req.question,
        "answer": answer,
        "sources": [
            {
                "text": r["text"],
                "section": r["metadata"].get("section_path", ""),
                "pages": r["metadata"].get("page_numbers", ""),
                "relevance_score": round(1 - r["distance"], 3),
            }
            for r in results
        ],
    }