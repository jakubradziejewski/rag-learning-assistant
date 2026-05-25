import logging

from openai import OpenAI

import platform

if platform.system() == "Windows":
    print("running on Windows")
elif platform.system() == "Linux":
    print("running on Linux")

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