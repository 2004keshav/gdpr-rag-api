"""
retrieval.py — Loads the FAISS index + metadata map for a given chunking strategy
and performs similarity search.

Deployed API uses ONLY the "structure" strategy (Phase 4-confirmed winner, verified
against faiss_structure.index in Phase 5 Step 4b — built from structure_aware_fixed.json,
128 chunks). Other strategies (fixed/recursive) are NOT loaded in the deployed API to
save RAM — this module supports them for Colab use, but config.py (Phase 5) will only
ever request "structure" in the deployed API.

No reranker here — dropped for the deployed API per Phase 5 RAM analysis (Phase 4 showed
recall@5 tied with/without reranking for "structure" chunking; only MRR, a ranking-order
metric, changed).
"""

import os
import json
import faiss
import numpy as np

INDICES_DIR = os.environ.get(
    "INDICES_DIR",
    "/content/drive/MyDrive/rag_project/indices"
)

_STRATEGY_FILES = {
    "structure": ("faiss_structure.index", "structure_metadata_map.json"),
    "fixed": ("faiss_fixed.index", "fixed_metadata_map.json"),
    "recursive": ("faiss_recursive.index", "recursive_metadata_map.json"),
}

_loaded_indices = {}   # cache: strategy -> faiss index object
_loaded_metadata = {}  # cache: strategy -> metadata dict


def _load_strategy(strategy: str):
    """Loads (and caches) the FAISS index + metadata map for one strategy."""
    if strategy not in _STRATEGY_FILES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Must be one of: {list(_STRATEGY_FILES.keys())}"
        )

    if strategy in _loaded_indices:
        return _loaded_indices[strategy], _loaded_metadata[strategy]

    index_fname, metadata_fname = _STRATEGY_FILES[strategy]
    index_path = os.path.join(INDICES_DIR, index_fname)
    metadata_path = os.path.join(INDICES_DIR, metadata_fname)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata map not found: {metadata_path}")

    index = faiss.read_index(index_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    _loaded_indices[strategy] = index
    _loaded_metadata[strategy] = metadata

    return index, metadata


def retrieve(query_vector: np.ndarray, strategy: str = "structure", top_k: int = 5) -> list:
    """
    Searches the FAISS index for the given strategy and returns top_k results.

    Args:
        query_vector: normalized 1D np.ndarray (384-dim), from embeddings.embed_query()
        strategy: one of "structure", "fixed", "recursive"
        top_k: number of results to return

    Returns:
        list of dicts, each with keys matching the CONFIRMED metadata schema:
        {chunk_id, text, score, article, source_articles, chapter, article_title, parent_chunk_id}
    """
    index, metadata = _load_strategy(strategy)

    # FAISS expects 2D input (1, dim) and float32
    query_2d = np.array([query_vector], dtype=np.float32)

    scores, indices = index.search(query_2d, top_k)
    scores = scores[0]
    indices = indices[0]

    results = []
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue  # FAISS pads with -1 if fewer than top_k results exist

        # CONFIRMED (Step 4b): metadata_map keys are strings after JSON reload
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
