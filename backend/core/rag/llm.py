from openai import OpenAI

client = OpenAI(
    base_url="http://model-runner.docker.internal:12434/engines/llama.cpp/v1",
    api_key="ignored",
)

LLM_MODEL = "ai/llama3.2"


def ask(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful study assistant. Answer the question using only the provided context. If the answer is not in the context, say so.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )

    return response.choices[0].message.content