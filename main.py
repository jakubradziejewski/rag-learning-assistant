import logging
import os

from fastapi import FastAPI
from backend.api.routes.documents import router as documents_router

def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    else:
        logging.getLogger().setLevel(level)


configure_logging()

app = FastAPI()

app.include_router(documents_router)

@app.get("/health")
def health():
    return {"status": "ok"}