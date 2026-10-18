"""
main.py - FastAPI app serving the GDPR RAG pipeline.

Endpoints:
  GET  /health  - liveness check (no model calls, just confirms the app is up)
  POST /ask     - main endpoint: question in, grounded answer + citations out

Imports embeddings.py, retrieval.py, generation.py, config.py, cache.py from src/
directly - SAME code as Colab, no duplication (Phase 5 requirement).
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config
import embeddings
import retrieval
import generation
import cache
import tracking


app = FastAPI(title=config.API_TITLE, version=config.API_VERSION)

# CORS: allow browser-based frontends (e.g. Streamlit Community Cloud app) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ASSUMPTION: open for now since this is a public read-only Q&A API; tighten later if needed
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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
    cache_hit: bool = False


class HealthResponse(BaseModel):
    status: str
    embedding_backend: str
    strategy: str
    cache_enabled: bool


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
    print(
        "Startup validation passed. Backend=" + config.EMBEDDING_BACKEND
        + ", Strategy=" + config.DEFAULT_STRATEGY
        + ", CacheEnabled=" + str(config.CACHE_ENABLED)
        + ", TrackingEnabled=" + str(config.TRACKING_ENABLED)
    )


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        embedding_backend=config.EMBEDDING_BACKEND,
        strategy=config.DEFAULT_STRATEGY,
        cache_enabled=config.CACHE_ENABLED,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    start_time = time.time()

    # --- Cache check first: skip the ~17-25s Gemini call entirely on a hit ---
    cache_check_start = time.time()
    cached = cache.get_cached_answer(request.query, top_k=request.top_k)
    cache_check_ms = (time.time() - cache_check_start) * 1000

    if cached is not None:
        latency = time.time() - start_time
        try:
            tracking.record_query(
                cache_hit=True,
                timings_ms={
                    "cache_check_ms": round(cache_check_ms, 2),
                    "total_ms": round(latency * 1000, 2),
                },
            )
        except Exception as e:
            print(f"⚠️ Tracking call raised unexpectedly (ignored, response unaffected): {e}")
        return AskResponse(
            query=cached["query"],
            answer=cached["answer"],
            cited_articles=cached["cited_articles"],
            retrieved_chunks=[RetrievedChunk(**c) for c in cached["retrieved_chunks"]],
            latency_seconds=round(latency, 2),
            cache_hit=True,
        )

    # --- Cache miss: run the full pipeline as before ---
    embedding_start = time.time()
    query_vector = embeddings.embed_query(request.query)
    embedding_ms = (time.time() - embedding_start) * 1000

    retrieval_start = time.time()
    retrieved_chunks = retrieval.retrieve(
        query_vector,
        strategy=config.DEFAULT_STRATEGY,
        top_k=request.top_k,
    )
    retrieval_ms = (time.time() - retrieval_start) * 1000

    generation_start = time.time()
    result = generation.generate_answer(request.query, retrieved_chunks)
    generation_ms = (time.time() - generation_start) * 1000

    latency = time.time() - start_time

    response = AskResponse(
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
        cache_hit=False,
    )

    # --- Write to cache for next time (failure here must not affect the response) ---
    cache.set_cached_answer(
        request.query,
        top_k=request.top_k,
        response=response.model_dump(exclude={"latency_seconds", "cache_hit"}),
    )

    # --- Record cost/latency for this query (failure here must not affect the response) ---
    try:
        tracking.record_query(
            cache_hit=False,
            timings_ms={
                "cache_check_ms": round(cache_check_ms, 2),
                "embedding_ms": round(embedding_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "total_ms": round(latency * 1000, 2),
            },
            prompt_tokens=result["prompt_tokens"],
            output_tokens=result["output_tokens"],
        )
    except Exception as e:
        print(f"⚠️ Tracking call raised unexpectedly (ignored, response unaffected): {e}")

    return response
