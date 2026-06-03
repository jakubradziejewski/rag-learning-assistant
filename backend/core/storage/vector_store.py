import json
import os
from typing import Optional

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

COLLECTION_NAME = "documents"

# Global BM25 index (rebuilds on each app restart or when docs are added)
_bm25_index: Optional[BM25Okapi] = None
_bm25_docs: list[str] = []
_bm25_doc_ids: list[str] = []


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

    # Rebuild BM25 index after adding new chunks
    _rebuild_bm25_index()

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


def _rebuild_bm25_index() -> None:
    """Rebuild the BM25 index from all stored chunks."""
    global _bm25_index, _bm25_docs, _bm25_doc_ids
    
    client = get_client()
    collection = get_collection(client)
    
    results = collection.get(include=["documents"])
    docs = results.get("documents", [])
    ids = results.get("ids", [])
    
    # Tokenize documents (simple whitespace split)
    tokenized_docs = []
    for doc in docs:
        tokens = doc.lower().split()
        tokenized_docs.append(tokens)
    
    _bm25_docs = docs
    _bm25_doc_ids = ids
    _bm25_index = BM25Okapi(tokenized_docs) if tokenized_docs else None


def _normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1] range."""
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [0.5] * len(scores)
    
    return [(s - min_score) / (max_score - min_score) for s in scores]


def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    n_results: int = 5,
    bm25_weight: float = 0.3,
    vector_weight: float = 0.7,
) -> list[dict]:
    """
    Hybrid search combining BM25 (keyword) and vector (semantic) search.
    
    Args:
        query_text: Raw query text for BM25
        query_embedding: Embedding vector for semantic search
        n_results: Number of results to return
        bm25_weight: Weight for BM25 scores (0-1)
        vector_weight: Weight for vector scores (0-1)
    
    Returns:
        List of results sorted by hybrid score
    """
    global _bm25_index, _bm25_docs, _bm25_doc_ids
    
    # Ensure BM25 index is initialized
    if _bm25_index is None:
        _rebuild_bm25_index()
    
    # Vector search
    vector_results = search(query_embedding, n_results=n_results * 2)
    
    # BM25 search
    bm25_scores = []
    bm25_results = []
    if _bm25_index is not None:
        query_tokens = query_text.lower().split()
        bm25_scores = _bm25_index.get_scores(query_tokens)
        
        # Get top BM25 results
        scored_docs = [(i, score) for i, score in enumerate(bm25_scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        for idx, score in scored_docs[:n_results * 2]:
            if idx < len(_bm25_docs):
                bm25_results.append({
                    "text": _bm25_docs[idx],
                    "id": _bm25_doc_ids[idx],
                    "bm25_score": score,
                })
    
    # Normalize scores
    vector_scores = _normalize_scores([1 - r["distance"] for r in vector_results])
    bm25_score_values = _normalize_scores([r["bm25_score"] for r in bm25_results]) if bm25_results else []
    
    # Combine results by ID
    combined: dict[str, dict] = {}
    
    # Add vector search results
    for result, norm_score in zip(vector_results, vector_scores):
        result_id = result["metadata"].get("chunk_index", "")
        combined[result_id] = {
            "text": result["text"],
            "metadata": result["metadata"],
            "vector_score": norm_score,
            "bm25_score": 0.0,
            "hybrid_score": 0.0,
        }
    
    # Add/update with BM25 results
    for result, norm_score in zip(bm25_results, bm25_score_values):
        result_id = result["id"]
        if result_id not in combined:
            combined[result_id] = {
                "text": result["text"],
                "metadata": {},
                "vector_score": 0.0,
                "bm25_score": norm_score,
                "hybrid_score": 0.0,
            }
        else:
            combined[result_id]["bm25_score"] = norm_score
    
    # Calculate hybrid scores
    for entry in combined.values():
        entry["hybrid_score"] = (
            entry["bm25_score"] * bm25_weight + 
            entry["vector_score"] * vector_weight
        )
    
    # Sort by hybrid score and return top N
    sorted_results = sorted(
        [
            {
                "text": entry["text"],
                "metadata": entry["metadata"],
                "distance": 1 - entry["vector_score"],  # Convert back to distance
                "bm25_score": entry["bm25_score"],
                "vector_score": entry["vector_score"],
                "hybrid_score": entry["hybrid_score"],
            }
            for entry in combined.values()
        ],
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )
    
    return sorted_results[:n_results]


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