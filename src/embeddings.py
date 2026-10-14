"""
embeddings.py - Embedding query/document text.

Backend selected via EMBEDDING_BACKEND env var (see config.py).
EMBEDDING_CONFIG_PATH is imported from config.py - single source of truth for paths,
not duplicated here (this bug caused a Render FileNotFoundError: embeddings.py had its
own hardcoded Colab Drive path default, separate from config.py's already-fixed one).
"""

import os
import json
import numpy as np

import config  # single source of truth for paths and backend selection

_model = None


def _load_model_name() -> str:
    with open(config.EMBEDDING_CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    return cfg["model_name"]


def get_embedding_backend() -> str:
    return config.EMBEDDING_BACKEND


def _get_model():
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
            "Unknown EMBEDDING_BACKEND='" + str(backend) + "'. "
            "Must be 'sentence_transformers' or 'fastembed'."
        )

    return _model


def embed_query(query: str) -> np.ndarray:
    model = _get_model()
    backend = get_embedding_backend()

    if backend == "fastembed":
        vec = next(model.embed([query]))
        return np.array(vec)
    else:
        vec = model.encode([query], normalize_embeddings=True)
        return vec[0]


def embed_documents(texts: list) -> np.ndarray:
    model = _get_model()
    backend = get_embedding_backend()

    if backend == "fastembed":
        return np.array(list(model.embed(texts)))
    else:
        return model.encode(texts, normalize_embeddings=True)
