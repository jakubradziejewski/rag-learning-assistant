import json
import os

import chromadb
from chromadb.config import Settings

COLLECTION_NAME = "documents"


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


def clear_documents() -> None:
    try:
        get_client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass


def store_chunks(doc_id: str, chunks: list[dict], embeddings: list[list[float]], filename: str = "") -> int:
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
            "filename": filename,
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

    return len(ids)


def search(query_embedding: list[float], n_results: int = 5) -> list[dict]:
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

    return output


def _parse_page_numbers(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value if isinstance(v, (int, float, str))]
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [int(v) for v in parsed if isinstance(v, (int, float, str))]
    return []


def get_all_chunks() -> list[dict]:
    client = get_client()
    collection = get_collection(client)

    results = collection.get(include=["documents", "metadatas"])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    ids = results.get("ids", [])

    chunks: list[dict] = []
    for text, meta, chunk_id in zip(documents, metadatas, ids):
        meta = meta or {}
        chunk_index = meta.get("chunk_index", 0)
        try:
            chunk_index = int(chunk_index)
        except (TypeError, ValueError):
            chunk_index = 0

        chunks.append(
            {
                "id": chunk_id,
                "doc_id": meta.get("doc_id", ""),
                "filename": meta.get("filename", ""),
                "text": text or "",
                "page_numbers": _parse_page_numbers(meta.get("page_numbers")),
                "section_path": meta.get("section_path", ""),
                "chunk_index": chunk_index,
            }
        )

    return chunks