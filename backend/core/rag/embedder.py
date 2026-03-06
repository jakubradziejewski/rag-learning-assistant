from openai import OpenAI

client = OpenAI(
    base_url="http://model-runner.docker.internal:12434/engines/llama.cpp/v1",
    api_key="ignored",
)

EMBED_MODEL = "ai/mxbai-embed-large"


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return response.data[0].embedding