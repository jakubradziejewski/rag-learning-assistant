from openai import OpenAI

client = OpenAI(
    base_url="http://model-runner.docker.internal:12434/engines/llama.cpp/v1",
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