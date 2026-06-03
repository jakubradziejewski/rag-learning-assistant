import shutil
import uuid
from pathlib import Path
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.core.rag.parser import parse_pdf
from backend.core.rag.embedder import embed_text
from backend.core.rag.llm import ask
from backend.core.storage.vector_store import get_all_chunks, search, store_chunks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    include_chunks: bool = False,
    include_ocr: bool = False,
    include_table_structure: bool = False,
    max_chunks: int | None = None,
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{doc_id}.pdf"

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = parse_pdf(dest, do_ocr=include_ocr, do_table_structure=include_table_structure)

    embeddings = [embed_text(chunk["text"]) for chunk in chunks]
    stored = store_chunks(doc_id, chunks, embeddings)

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
    query_embedding = embed_text(req.question)
    results = search(query_embedding, n_results=req.n_results)

    context_chunks = [r["text"] for r in results]
    answer = ask(req.question, context_chunks, temperature=req.temperature)

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


@router.get("/chunks")
def list_chunks():
    chunks = get_all_chunks()
    return {
        "count": len(chunks),
        "chunks": chunks,
    }