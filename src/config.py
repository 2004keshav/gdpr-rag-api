"""
config.py - Centralized configuration for the RAG API.

Paths default to RELATIVE paths based on the repo structure (data/ folder next to src/
and api/), which is what Render will use. Can still be overridden via env vars for
local/Colab use with different paths.
"""

import os

# --- Paths: default to repo-relative "data/" folder (works on Render as-is) ---
# On Render, the working directory is the repo root, so "data/" resolves correctly.
INDICES_DIR = os.environ.get("INDICES_DIR", "data")
EMBEDDING_CONFIG_PATH = os.environ.get(
    "EMBEDDING_CONFIG_PATH",
    os.path.join(INDICES_DIR, "embedding_config.json")
)

# --- Embedding backend ---
# Render MUST set EMBEDDING_BACKEND=fastembed via dashboard env vars (RAM constraint).
# Defaults to fastembed here since this is the deployment repo copy (opposite of the
# Colab default, which is sentence_transformers).
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "fastembed")

# --- Retrieval config ---
DEFAULT_STRATEGY = "structure"
DEFAULT_TOP_K = 5

# --- Generation config ---
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"
LLM_MAX_RETRIES = 4
LLM_TIMEOUT_MS = 30000

# --- API metadata ---
API_TITLE = "GDPR RAG API"
API_VERSION = "1.0.0"


def validate_config():
    """Fails loudly at startup if required config is missing."""
    errors = []

    if not os.environ.get("GEMINI_API_KEY"):
        errors.append("GEMINI_API_KEY environment variable is not set.")

    if EMBEDDING_BACKEND not in ("sentence_transformers", "fastembed"):
        errors.append(
            "EMBEDDING_BACKEND must be 'sentence_transformers' or 'fastembed', got: " + str(EMBEDDING_BACKEND)
        )

    if not os.path.exists(EMBEDDING_CONFIG_PATH):
        errors.append("embedding_config.json not found at: " + EMBEDDING_CONFIG_PATH)

    if errors:
        raise RuntimeError("Config validation FAILED:\n" + "\n".join("  - " + e for e in errors))

    return True
