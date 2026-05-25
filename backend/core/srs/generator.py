from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import re
from typing import Any

from backend.core.rag.llm import generate_study_items
from backend.core.srs.fsrs_scheduler import new_card_dict

_GENERIC_PROMPTS = {
    "summarize the key concept",
    "summarize the key concept.",
    "what is the main idea of this chunk",
    "what is the main idea of this chunk?",
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def _is_too_generic_prompt(prompt: str) -> bool:
    normalized = " ".join(prompt.lower().strip().split())
    return normalized in _GENERIC_PROMPTS


def _is_too_generic_answer(answer: str) -> bool:
    normalized = " ".join(answer.lower().strip().split())
    if len(normalized) < 20:
        return True
    return normalized in {
        "intrinsic vs extrinsic",
        "intrinsic and extrinsic",
        "research and development in computer science",
    }


def _extract_keywords(text: str, limit: int = 8) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", text.lower())
    filtered = [token for token in tokens if token not in _STOPWORDS]
    if not filtered:
        return []

    counts = Counter(filtered)
    return [token for token, _ in counts.most_common(limit)]


def _is_answer_grounded(answer: str, chunk_text: str, section_path: str) -> bool:
    keywords = _extract_keywords(chunk_text)
    keywords.extend(_extract_keywords(section_path, limit=4))
    if not keywords:
        return True

    answer_lower = answer.lower()
    return any(re.search(rf"\b{re.escape(keyword)}\b", answer_lower) for keyword in keywords)


def _derive_topic(chunk_text: str) -> str:
    for line in chunk_text.splitlines():
        cleaned = " ".join(line.strip().split())
        if cleaned:
            topic = cleaned.strip().strip("\"'")
            if len(topic) > 80:
                topic = f"{topic[:80].rstrip()}..."
            return topic.rstrip(".:;!?")
    return "the topic"


def _fallback_items(chunk_text: str, section_path: str) -> dict[str, Any]:
    short = chunk_text.strip()
    if len(short) > 800:
        short = f"{short[:800]}..."

    topic = section_path.strip() or _derive_topic(chunk_text)
    prompt = f"Explain the concept: {topic}."

    return {
        "flashcard_front": prompt,
        "flashcard_back": short,
        "key_points": [],
    }


def _normalize_payload(payload: dict[str, Any], chunk_text: str, section_path: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _fallback_items(chunk_text, section_path)

    required = {"flashcard_front", "flashcard_back"}
    if not required.issubset(payload):
        return _fallback_items(chunk_text, section_path)

    flashcard_front = str(payload.get("flashcard_front", ""))
    flashcard_back = str(payload.get("flashcard_back", ""))

    if _is_too_generic_prompt(flashcard_front):
        return _fallback_items(chunk_text, section_path)

    if _is_too_generic_answer(flashcard_back):
        return _fallback_items(chunk_text, section_path)

    if not _is_answer_grounded(flashcard_back, chunk_text, section_path):
        return _fallback_items(chunk_text, section_path)

    return payload


def generate_items_for_chunk(doc_id: str, chunk: dict[str, Any]) -> list[dict[str, Any]]:
    payload = generate_study_items(
        chunk_text=chunk["text"],
        section_path=chunk.get("section_path", ""),
        page_numbers=chunk.get("page_numbers", []),
    )

    payload = _normalize_payload(payload, chunk["text"], chunk.get("section_path", ""))
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
    seen_prompts: set[str] = set()
    for chunk in chunks:
        for item in generate_items_for_chunk(doc_id, chunk):
            prompt = " ".join(str(item.get("prompt", "")).lower().split())
            if not prompt or prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)
            items.append(item)
    return items
