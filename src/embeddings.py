"""
embeddings.py — Embedding query/document text.

Backend is selected via the EMBEDDING_BACKEND env var:
  - "sentence_transformers" (default) — PyTorch, used in Colab (already installed from Phase 2/3)
  - "fastembed"              — ONNX runtime, lightweight (~90MB), used in deployed FastAPI app

Both backends load the SAME model (BAAI/bge-small-en-v1.5) and were numerically verified
to produce near-identical output (cosine similarity > 0.999 on test sentences) before this
module was written — see Phase 5 Step 2 verification in the Colab notebook.

IMPORTANT: model name is loaded from embedding_config.json (Phase 2 output), not hardcoded,
so Phase 5 always matches whatever model built the FAISS indices.
"""

import os
import json
import numpy as np

_EMBEDDING_CONFIG_PATH = os.environ.get(
    "EMBEDDING_CONFIG_PATH",
    "/content/drive/MyDrive/rag_project/indices/embedding_config.json"
)

_model = None  # lazy-loaded singleton, loaded once per process


def _load_model_name() -> str:
    """Reads the model name Phase 2 actually used — never hardcode this."""
    with open(_EMBEDDING_CONFIG_PATH, "r") as f:
        config = json.load(f)
    return config["model_name"]


def get_embedding_backend() -> str:
    return os.environ.get("EMBEDDING_BACKEND", "sentence_transformers")


def _get_model():
    """Lazy-load the embedding model once, using whichever backend is configured."""
    global _model
    if _model is not None:
        return _model

    model_name = _load_model_name()
    backend = get_embedding_backend()

    if backend == "fastembed":
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=model_name)
    elif backend == "sentence_transformers":
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
    else:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND='{backend}'. "
            f"Must be 'sentence_transformers' or 'fastembed'."
        )

    return _model


def embed_query(query: str) -> np.ndarray:
    """
    Embeds a single query string. Returns a normalized 1D np.ndarray (384-dim for BGE-small).
    Same function signature as Phase 3's embed_query() — drop-in replacement.
    """
    model = _get_model()
    backend = get_embedding_backend()

    if backend == "fastembed":
        # fastembed .embed() returns a generator of np.ndarray, already normalized by default
        vec = next(model.embed([query]))
        return np.array(vec)
    else:
        # sentence-transformers: must explicitly request normalization (matches Phase 2 config)
        vec = model.encode([query], normalize_embeddings=True)
        return vec[0]


def embed_documents(texts: list) -> np.ndarray:
    """Batch-embeds multiple texts. Returns 2D np.ndarray (n_texts, 384)."""
    model = _get_model()
    backend = get_embedding_backend()

    if backend == "fastembed":
        return np.array(list(model.embed(texts)))
    else:
        return model.encode(texts, normalize_embeddings=True)
