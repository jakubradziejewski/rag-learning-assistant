import logging

import chromadb
from chromadb.config import Settings

COLLECTION_NAME = "documents"
logger = logging.getLogger(__name__)


def get_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(
        host="localhost", #this is as localhost because we have networkmode
        port=8001,
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