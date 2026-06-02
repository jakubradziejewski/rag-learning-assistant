import ast
import logging
import os

import chromadb
from chromadb.config import Settings

COLLECTION_NAME = "documents"
logger = logging.getLogger(__name__)


def get_client() -> chromadb.HttpClient:
    host = os.getenv("CHROMADB_HOST", "chromadb")
    port = int(os.getenv("CHROMADB_PORT", "8000"))
    return chromadb.HttpClient(
        host=host,
        port=port,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client: chromadb.HttpClient):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(doc_id: str, chunks: list[dict], embeddings: list[list[float]]) -> int:
    logger.info("Opening vector store client: doc_id=%s chunks=%s", doc_id, len(chunks))
    client = get_client()
    collection = get_collection(client)

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = f"{doc_id}_chunk_{chunk['chunk_index']}"
        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append({
            "doc_id": doc_id,
            "page_numbers": str(chunk["page_numbers"]),
            "section_path": chunk["section_path"],
            "chunk_index": chunk["chunk_index"],
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info("Chunks stored: doc_id=%s stored=%s", doc_id, len(ids))

    return len(ids)


def search(query_embedding: list[float], n_results: int = 5) -> list[dict]:
    logger.info("Searching vector store: n_results=%s", n_results)
    client = get_client()
    collection = get_collection(client)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({"text": text, "metadata": meta, "distance": dist})

    logger.info("Vector search finished: results=%s", len(output))

    return output


def _parse_page_numbers(value) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value if str(item).isdigit()]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return [int(item) for item in parsed if str(item).isdigit()]
    return []


def list_chunks() -> list[dict]:
    logger.info("Listing all chunks from vector store")
    client = get_client()
    collection = get_collection(client)

    results = collection.get(include=["documents", "metadatas"])

    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    output = []
    for chunk_id, text, meta in zip(ids, documents, metadatas):
        meta = meta or {}
        chunk_index = meta.get("chunk_index")
        doc_id = meta.get("doc_id", "")
        try:
            chunk_index = int(chunk_index) if chunk_index is not None else None
        except (TypeError, ValueError):
            chunk_index = None

        if not doc_id and isinstance(chunk_id, str) and "_chunk_" in chunk_id:
            doc_id = chunk_id.split("_chunk_", 1)[0]

        if chunk_index is None and isinstance(chunk_id, str) and "_chunk_" in chunk_id:
            chunk_suffix = chunk_id.rsplit("_chunk_", 1)[-1]
            if chunk_suffix.isdigit():
                chunk_index = int(chunk_suffix)

        output.append(
            {
                "id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "text": text,
                "section_path": meta.get("section_path", ""),
                "page_numbers": _parse_page_numbers(meta.get("page_numbers")),
            }
        )

    logger.info("Chunk listing complete: chunks=%s", len(output))
    return output