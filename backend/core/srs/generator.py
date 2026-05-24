from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.core.rag.llm import generate_study_items
from backend.core.srs.fsrs_scheduler import new_card_dict


def _fallback_items(chunk_text: str) -> dict[str, Any]:
    short = chunk_text.strip()
    if len(short) > 400:
        short = f"{short[:400]}..."

    return {
        "flashcard_front": "Summarize the key concept.",
        "flashcard_back": short,
        "key_points": [],
    }


def _normalize_payload(payload: dict[str, Any], chunk_text: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _fallback_items(chunk_text)

    required = {"flashcard_front", "flashcard_back"}
    if not required.issubset(payload):
        return _fallback_items(chunk_text)

    return payload


def generate_items_for_chunk(doc_id: str, chunk: dict[str, Any]) -> list[dict[str, Any]]:
    payload = generate_study_items(
        chunk_text=chunk["text"],
        section_path=chunk.get("section_path", ""),
        page_numbers=chunk.get("page_numbers", []),
    )

    payload = _normalize_payload(payload, chunk["text"])
    timestamp = datetime.now(timezone.utc).isoformat()

    items: list[dict[str, Any]] = []

    flash_front = str(payload.get("flashcard_front", "")).strip()
    flash_back = str(payload.get("flashcard_back", "")).strip()
    if flash_front and flash_back:
        items.append(
            {
                "id": f"{doc_id}_c{chunk['chunk_index']}_f",
                "doc_id": doc_id,
                "chunk_index": chunk["chunk_index"],
                "type": "flashcard",
                "prompt": flash_front,
                "answer": flash_back,
                "metadata": {
                    "section_path": chunk.get("section_path", ""),
                    "page_numbers": chunk.get("page_numbers", []),
                },
                "source_text": chunk["text"],
                "card": new_card_dict(),
                "created_at": timestamp,
                "last_review": None,
            }
        )

    return items


def generate_items_for_chunks(doc_id: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for chunk in chunks:
        items.extend(generate_items_for_chunk(doc_id, chunk))
    return items
