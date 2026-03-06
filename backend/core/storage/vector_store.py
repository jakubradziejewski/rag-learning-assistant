import chromadb
from chromadb.config import Settings


COLLECTION_NAME = "documents"


def get_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(
        host="chromadb",
        port=8000,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client: chromadb.HttpClient):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(doc_id: str, chunks: list[dict]) -> int:
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
        documents=documents,
        metadatas=metadatas,
    )

    return len(ids)