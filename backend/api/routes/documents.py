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

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{doc_id}.pdf"

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = parse_pdf(dest)

    embeddings = [embed_text(chunk["text"]) for chunk in chunks]
    stored = store_chunks(doc_id, chunks, embeddings)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_stored": stored,
    }


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