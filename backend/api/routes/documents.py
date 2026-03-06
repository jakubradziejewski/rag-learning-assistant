import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.core.rag.parser import parse_pdf

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

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_parsed": len(chunks),
        "chunks": chunks,
    }