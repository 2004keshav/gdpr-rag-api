"""
retrieval.py - Loads the FAISS index + metadata map and performs similarity search.

INDICES_DIR is imported from config.py - single source of truth for paths, not
duplicated here (same class of bug as the embeddings.py fix: this file had its own
hardcoded Colab Drive path default, separate from config.py's already-fixed one).

Deployed API uses ONLY the "structure" strategy. No reranker (dropped for RAM savings).
"""

import os
import json
import faiss
import numpy as np

import config  # single source of truth for paths

_STRATEGY_FILES = {
    "structure": ("faiss_structure.index", "structure_metadata_map.json"),
    "fixed": ("faiss_fixed.index", "fixed_metadata_map.json"),
    "recursive": ("faiss_recursive.index", "recursive_metadata_map.json"),
}

_loaded_indices = {}
_loaded_metadata = {}


def _load_strategy(strategy: str):
    if strategy not in _STRATEGY_FILES:
        raise ValueError(
            "Unknown strategy '" + strategy + "'. Must be one of: " + str(list(_STRATEGY_FILES.keys()))
        )

    if strategy in _loaded_indices:
        return _loaded_indices[strategy], _loaded_metadata[strategy]

    index_fname, metadata_fname = _STRATEGY_FILES[strategy]
    index_path = os.path.join(config.INDICES_DIR, index_fname)
    metadata_path = os.path.join(config.INDICES_DIR, metadata_fname)

    if not os.path.exists(index_path):
        raise FileNotFoundError("FAISS index not found: " + index_path)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Metadata map not found: " + metadata_path)

    index = faiss.read_index(index_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    _loaded_indices[strategy] = index
    _loaded_metadata[strategy] = metadata

    return index, metadata


def retrieve(query_vector: np.ndarray, strategy: str = "structure", top_k: int = 5) -> list:
    index, metadata = _load_strategy(strategy)

    query_2d = np.array([query_vector], dtype=np.float32)
    scores, indices = index.search(query_2d, top_k)
    scores = scores[0]
    indices = indices[0]

    results = []
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue

        meta = metadata[str(idx)]

        results.append({
            "chunk_id": meta.get("chunk_id"),
            "text": meta.get("text"),
            "score": float(score),
            "article": meta.get("article"),
            "source_articles": meta.get("source_articles"),
            "chapter": meta.get("chapter"),
            "article_title": meta.get("article_title"),
            "parent_chunk_id": meta.get("parent_chunk_id"),
        })

    return results
