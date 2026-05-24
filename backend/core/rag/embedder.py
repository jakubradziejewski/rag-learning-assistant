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


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return response.data[0].embedding