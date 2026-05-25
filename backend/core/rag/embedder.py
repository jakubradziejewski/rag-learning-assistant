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

EMBED_MODEL = "ai/mxbai-embed-large"
logger = logging.getLogger(__name__)


def embed_text(text: str) -> list[float]:
    logger.info("Generating embedding: chars=%s", len(text))
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    logger.info("Embedding generated")
    return response.data[0].embedding