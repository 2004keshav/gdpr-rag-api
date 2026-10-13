"""
config.py - Centralized configuration for the RAG API.

INDICES_DIR is computed relative to this file's location (repo root / data), NOT the
process's current working directory - this matters because Render's "Root Directory"
setting changes cwd to api/, which would otherwise break a plain relative "data" path.
"""

import os

# --- Compute repo root from this file's own location: src/config.py -> src/ -> repo root ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDICES_DIR = os.environ.get("INDICES_DIR", os.path.join(_REPO_ROOT, "data"))
EMBEDDING_CONFIG_PATH = os.environ.get(
    "EMBEDDING_CONFIG_PATH",
    os.path.join(INDICES_DIR, "embedding_config.json")
)

EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "fastembed")

DEFAULT_STRATEGY = "structure"
DEFAULT_TOP_K = 5

GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"
LLM_MAX_RETRIES = 4
LLM_TIMEOUT_MS = 30000

API_TITLE = "GDPR RAG API"
API_VERSION = "1.0.0"


def validate_config():
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
