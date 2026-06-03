import shutil
import uuid
from pathlib import Path
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.core.rag.parser import parse_pdf
from backend.core.rag.embedder import embed_text
from backend.core.rag.llm import ask
from backend.core.storage.vector_store import get_all_chunks, search, store_chunks, hybrid_search

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
    use_hybrid: bool = True
    bm25_weight: float = 0.3


@router.post("/query")
def query(req: QueryRequest):
    query_embedding = embed_text(req.question)
    
    if req.use_hybrid:
        results = hybrid_search(
            query_text=req.question,
            query_embedding=query_embedding,
            n_results=req.n_results,
            bm25_weight=req.bm25_weight,
            vector_weight=1.0 - req.bm25_weight,
        )
    else:
        results = search(query_embedding, n_results=req.n_results)

    context_chunks = [r["text"] for r in results]
    answer = ask(req.question, context_chunks, temperature=req.temperature)

    return {
        "question": req.question,
        "answer": answer,
        "search_method": "hybrid" if req.use_hybrid else "vector",
        "sources": [
            {
                "text": r["text"],
                "section": r["metadata"].get("section_path", ""),
                "pages": r["metadata"].get("page_numbers", ""),
                "relevance_score": round(1 - r["distance"], 3),
                "bm25_score": round(r.get("bm25_score", 0.0), 3) if req.use_hybrid else None,
                "vector_score": round(r.get("vector_score", 0.0), 3) if req.use_hybrid else None,
                "hybrid_score": round(r.get("hybrid_score", 0.0), 3) if req.use_hybrid else None,
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

class SuggestTopicsRequest(BaseModel):
    context: str

@router.post("/suggest_topics")
def suggest_topics(req: SuggestTopicsRequest):
    from backend.core.rag.llm import suggest_topics_from_context
    topics = suggest_topics_from_context(req.context)
    return {"topics": topics}


class SearchChunksRequest(BaseModel):
    query: str
    n_results: int = 20

@router.post("/search")
def search_chunks(req: SearchChunksRequest):
    query_embedding = embed_text(req.query)
    results = search(query_embedding, n_results=req.n_results)
    chunks = [
        {
            "doc_id": r["metadata"].get("doc_id", ""),
            "chunk_index": r["metadata"].get("chunk_index", 0),
            "text": r["text"],
            "section_path": r["metadata"].get("section_path", ""),
            "page_numbers": r["metadata"].get("page_numbers", []),
        }
        for r in results
    ]
    return {"chunks": chunks}