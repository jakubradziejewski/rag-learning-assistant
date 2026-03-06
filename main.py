from fastapi import FastAPI
from backend.api.routes.documents import router as documents_router

app = FastAPI()

app.include_router(documents_router)

@app.get("/health")
def health():
    return {"status": "ok"}