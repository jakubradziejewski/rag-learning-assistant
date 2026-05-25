import json
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

LLM_MODEL = "ai/llama3.2"


def ask(question: str, context_chunks: list[str], temperature: float = 0.0) -> str:
    context = "\n\n".join(context_chunks)

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

    return response.choices[0].message.content


def generate_study_items(
    chunk_text: str,
    section_path: str = "",
    page_numbers: list[int] | None = None,
) -> dict:
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
                    "Write self-contained questions and answers grounded only in the provided chunk. "
                    "Do not use outside knowledge or introduce unrelated topics. "
                    "Make flashcard_front specific and unambiguous (include scope or qualifiers from the chunk). "
                    "Ensure flashcard_back includes 2-3 concrete details or terms found in the chunk. "
                    "Do not reference the source text or lecture context with pronouns or deictic phrases "
                    "(no 'she', 'he', 'they', 'this', 'that', 'the slide', 'the lecture', 'the example'). "
                    "Avoid subjective or opinion questions. "
                    "Make questions specific (definition, mechanism, comparison, or consequence). "
                    "Answers can be 2-4 sentences when needed, but remain factual."
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
        fallback_topic = section_path.strip() or "the topic in this section"
        return {
            "question": f"Explain the concept: {fallback_topic}.",
            "answer": chunk_text[:800],
            "flashcard_front": f"Explain the concept: {fallback_topic}.",
            "flashcard_back": chunk_text[:800],
            "key_points": [],
        }