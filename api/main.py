"""
main.py - FastAPI app serving the GDPR RAG pipeline.

Endpoints:
  GET  /health  - liveness check (no model calls, just confirms the app is up)
  POST /ask     - main endpoint: question in, grounded answer + citations out

Imports embeddings.py, retrieval.py, generation.py, config.py from src/ directly -
SAME code as Colab, no duplication (Phase 5 requirement).
"""

import os
import sys
import time
import traceback

# Make src/ importable - works both in Colab (path below) and on Render
# (where SRC_DIR will be set via env var or a relative path, confirmed in a later step)
SRC_DIR = os.environ.get("SRC_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config
import embeddings
import retrieval
import generation


app = FastAPI(title=config.API_TITLE, version=config.API_VERSION)


# ============================================================
# Pydantic v2 models - request/response validation
# ============================================================

class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="The GDPR question to ask")
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=20, description="Number of chunks to retrieve")


class RetrievedChunk(BaseModel):
    chunk_id: str
    article: str | None = None
    article_title: str | None = None
    score: float


class AskResponse(BaseModel):
    query: str
    answer: str
    cited_articles: list[str]
    retrieved_chunks: list[RetrievedChunk]
    latency_seconds: float


class HealthResponse(BaseModel):
    status: str
    embedding_backend: str
    strategy: str


class ErrorResponse(BaseModel):
    error: str
    detail: str


# ============================================================
# Structured error handling - ONE consistent error shape,
# not FastAPI's default inconsistent validation-vs-HTTPException shapes
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=type(exc).__name__,
            detail=str(exc),
        ).model_dump(),
    )


# ============================================================
# Startup: validate config loudly BEFORE accepting any requests
# ============================================================

@app.on_event("startup")
def startup_validation():
    config.validate_config()
    print("Startup validation passed. Backend=" + config.EMBEDDING_BACKEND + ", Strategy=" + config.DEFAULT_STRATEGY)


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        embedding_backend=config.EMBEDDING_BACKEND,
        strategy=config.DEFAULT_STRATEGY,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    start_time = time.time()

    query_vector = embeddings.embed_query(request.query)
    retrieved_chunks = retrieval.retrieve(
        query_vector,
        strategy=config.DEFAULT_STRATEGY,
        top_k=request.top_k,
    )
    result = generation.generate_answer(request.query, retrieved_chunks)

    latency = time.time() - start_time

    return AskResponse(
        query=result["query"],
        answer=result["answer"],
        cited_articles=result["cited_articles"],
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id=c["chunk_id"],
                article=c.get("article"),
                article_title=c.get("article_title"),
                score=c["score"],
            )
            for c in retrieved_chunks
        ],
        latency_seconds=round(latency, 2),
    )
