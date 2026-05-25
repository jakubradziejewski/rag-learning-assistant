import os

from openai import OpenAI

DMR_BASE_URL = os.getenv(
    "DMR_BASE_URL",
    "http://model-runner.docker.internal:12434/engines/llama.cpp/v1",
)

client = OpenAI(
    base_url=DMR_BASE_URL,
    api_key="ignored",
)

EMBED_MODEL = "ai/mxbai-embed-large"
MAX_EMBED_CHARS = int(os.getenv("MAX_EMBED_CHARS", "1200"))
EMBED_CHUNK_OVERLAP = int(os.getenv("EMBED_CHUNK_OVERLAP", "200"))
MAX_EMBED_CHUNKS = int(os.getenv("MAX_EMBED_CHUNKS", "8"))


def _split_text(text: str) -> list[str]:
    if len(text) <= MAX_EMBED_CHARS:
        return [text]

    overlap = min(max(0, EMBED_CHUNK_OVERLAP), MAX_EMBED_CHARS - 1)
    chunks: list[str] = []
    start = 0
    while start < len(text) and len(chunks) < MAX_EMBED_CHUNKS:
        end = start + MAX_EMBED_CHARS
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def _average_embeddings(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []

    length = len(embeddings[0])
    totals = [0.0] * length
    counted = 0
    for emb in embeddings:
        if len(emb) != length:
            continue
        for idx, value in enumerate(emb):
            totals[idx] += value
        counted += 1

    count = max(1, counted)
    return [value / count for value in totals]


def _is_oversize_error(err: Exception) -> bool:
    message = str(err).lower()
    return "too large" in message and "batch size" in message


def _embed_chunk(text: str, depth: int = 0) -> list[float]:
    try:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=text,
        )
        return response.data[0].embedding
    except Exception as exc:  # pragma: no cover - depends on model error shape
        if not _is_oversize_error(exc):
            raise

        # Fallback: split and average if the embedding server rejects the size.
        if depth >= 6 or len(text) < 2:
            raise

        midpoint = len(text) // 2
        left = text[:midpoint].strip()
        right = text[midpoint:].strip()
        if not left or not right:
            raise

        left_emb = _embed_chunk(left, depth + 1)
        right_emb = _embed_chunk(right, depth + 1)
        return _average_embeddings([left_emb, right_emb])


def embed_text(text: str) -> list[float]:
    trimmed = text.strip()
    if not trimmed:
        trimmed = " "

    chunks = _split_text(trimmed)
    embeddings: list[list[float]] = []
    for chunk in chunks:
        embeddings.append(_embed_chunk(chunk))

    if len(embeddings) == 1:
        return embeddings[0]

    # Average chunk embeddings into a single vector for storage/search.
    return _average_embeddings(embeddings)