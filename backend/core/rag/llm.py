import json
import os
import re
import logging

from openai import OpenAI

import platform

if platform.system() == "Linux":
    base_url = "http://172.17.0.1:12434"
else:
    base_url = "http://model-runner.docker.internal:12434"

client = OpenAI(
    base_url=base_url + "/engines/llama.cpp/v1",
    api_key="ignored",
)

LLM_MODEL = "ai/llama3.2"
logger = logging.getLogger(__name__)


def ask(question: str, context_chunks: list[str], temperature: float = 0.0) -> str:
    context = "\n\n".join(context_chunks)

    logger.info(
        "Generating answer: question_chars=%s context_chunks=%s temperature=%s",
        len(question),
        len(context_chunks),
        temperature,
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful study assistant. "
                    "When context is provided, prioritize it in your answer. "
                    "If the answer is not in the context, you may use your own knowledge "
                    "But you must clearly state which parts of the answer are based on the provided context and which are not. "
                    "Try to separate them into two paragraphs, and start the one based on the context with 'Based on the provided context, ...'"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    
    logger.info("Answer generated")
    return response.choices[0].message.content


def generate_study_items(
    chunk_text: str,
    section_path: str = "",
    page_numbers: list[int] | None = None,
) -> dict:
    print(f"Generating study items for chunk (section: '{section_path}', pages: {page_numbers})")
    pages = ", ".join(str(p) for p in (page_numbers or []))

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate study materials. "
                    "Return JSON only, no markdown. "
                    "Keys: question, answer, flashcard_front, flashcard_back, key_points. "
                    "Write self-contained questions and answers based on the chunk and related real-world knowledge. "
                    "Do not introduce unrelated topics. "
                    "Make flashcard_front specific and unambiguous (include scope or qualifiers from the chunk). "
                    "Make flashcard_back directly answer the prompt and include 2-3 concrete terms from the chunk. "
                    "Avoid meta-questions like 'requires answering the following questions' and do not list questions in the answer. "
                    "Do not reference slides/pages or use deictic phrases like 'previous slide', 'this page', or 'as shown'. "
                    "Do not ask about the lecturer, institute, location, emails, or copyright notices. "
                    "Avoid vague one-word answers like 'unknown' or 'not specified'. "
                    "If the chunk is only metadata (names, affiliations, contact info, copyright), set flashcard_front and flashcard_back to empty strings. "
                    "If the chunk does not define a term, do not invent a definition; instead return empty strings. "
                    "Avoid subjective or opinion questions. "
                    "Make questions specific (definition, mechanism, comparison, or consequence). "
                    "Answers can be 2-5 sentences when needed, but remain factual. "
                    "Avoid questions about dates, places or people. "
                    "Examples: "
                    "Bad: flashcard_front='Explain the concept: Specification of a supervised learning problem requires answering the following questions.' "
                    "Bad: flashcard_back='- What kind of training data is offered? ...' "
                    "Bad: flashcard_front='Institution of Computing Science' | Good: flashcard_front='' and flashcard_back='' "
                    "Bad: flashcard_front='Institution of Computing Science?' (entity name only) | Good: flashcard_front='' and flashcard_back='' "
                ),
            },
            {
                "role": "user",
                "content": (
                    "Chunk metadata:\n"
                    f"Section: {section_path or 'N/A'}\n"
                    f"Pages: {pages or 'N/A'}\n\n"
                    "Chunk:\n"
                    f"{chunk_text}\n\n"
                    "Return JSON only."
                ),
            },
        ],
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        fallback_topic = section_path.strip() or _derive_topic(chunk_text)
        return {
            "question": f"Explain the concept: {fallback_topic}.",
            "answer": chunk_text[:800],
            "flashcard_front": f"Explain the concept: {fallback_topic}.",
            "flashcard_back": chunk_text[:800],
            "key_points": [],
        }


def _derive_topic(chunk_text: str) -> str:
    for line in chunk_text.splitlines():
        cleaned = " ".join(line.strip().split())
        if cleaned:
            topic = _clean_topic_line(cleaned)
            if len(topic) > 80:
                topic = f"{topic[:80].rstrip()}..."
            return topic.rstrip(".:;!?")
    return "the topic"


def _clean_topic_line(text: str) -> str:
    topic = text.strip().strip("\"'")
    topic = re.sub(r"^\[\d+\]\s*", "", topic)
    topic = re.sub(r"^\d+[\.)]\s*", "", topic)
    return topic
