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
                    "Keep answers short and factual."
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
        return {
            "question": "What is the main idea of this chunk?",
            "answer": chunk_text[:400],
            "flashcard_front": "Summarize the key concept.",
            "flashcard_back": chunk_text[:400],
            "key_points": [],
        }