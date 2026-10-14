"""
cache.py - Optional caching layer for /ask responses, backed by Upstash Redis.

Design principles:
  - FAIL-OPEN: any cache error (network, auth, malformed data) must NEVER crash
    the main pipeline. Worst case: cache is skipped, user still gets a fresh answer.
  - Reads ALL config from config.py - no independent hardcoded defaults
    (Phase 5 lesson: duplicate path/config logic caused two failed deploys).
  - Cache key includes query + top_k, because different top_k values can
    legitimately produce different retrieved chunks and thus different answers.
    Caching on query alone would risk serving a mismatched (wrong top_k) answer.
  - Exact-match only (normalized query string -> SHA-256 hash). No semantic/
    embedding-similarity caching - that would risk serving a cached answer to
    a subtly different question, undermining the Phase 4 grounding/citation
    guarantees. This was an explicit, confirmed decision, not an oversight.
"""

import hashlib
import json
import re

import config

_redis_client = None
_client_init_attempted = False


def _get_client():
    """
    Lazily creates and caches the Upstash Redis client.
    Returns None if caching is disabled or client creation fails - callers
    must handle None gracefully (that's the fail-open contract).
    """
    global _redis_client, _client_init_attempted

    if not config.CACHE_ENABLED:
        return None

    if _redis_client is not None:
        return _redis_client

    if _client_init_attempted:
        # Already tried once and failed this process lifetime - don't retry every request.
        return None

    _client_init_attempted = True
    try:
        from upstash_redis import Redis
        _redis_client = Redis(
            url=config.UPSTASH_REDIS_REST_URL,
            token=config.UPSTASH_REDIS_REST_TOKEN,
        )
        return _redis_client
    except Exception as e:
        print(f"⚠️ Cache client init failed (caching disabled for this process): {e}")
        return None


def normalize_query(query: str) -> str:
    """Lowercase + collapse whitespace. Does not alter meaning (safe for exact-match)."""
    normalized = query.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def make_cache_key(query: str, top_k: int) -> str:
    """
    Builds a namespaced, fixed-length cache key from the normalized query AND top_k.
    Including top_k prevents serving a cached answer retrieved with a different
    number of chunks than what was requested.
    """
    normalized = normalize_query(query)
    key_material = f"{normalized}|top_k={top_k}"
    hash_digest = hashlib.sha256(key_material.encode('utf-8')).hexdigest()
    return f"qa:{hash_digest}"


def get_cached_answer(query: str, top_k: int) -> dict | None:
    """
    Returns the cached response dict for this (query, top_k) pair, or None
    on a cache miss OR any error. Never raises.
    """
    client = _get_client()
    if client is None:
        return None

    key = make_cache_key(query, top_k)
    try:
        cached_value = client.get(key)
        if cached_value is None:
            return None
        return json.loads(cached_value)
    except Exception as e:
        print(f"⚠️ Cache read error (non-fatal, proceeding without cache): {e}")
        return None


def set_cached_answer(query: str, top_k: int, response: dict) -> bool:
    """
    Stores response dict in cache with TTL from config. Returns True/False
    for observability but failure is never fatal to the caller.
    """
    client = _get_client()
    if client is None:
        return False

    key = make_cache_key(query, top_k)
    try:
        value = json.dumps(response)
        client.set(key, value, ex=config.CACHE_TTL_SECONDS)
        return True
    except Exception as e:
        print(f"⚠️ Cache write error (non-fatal, answer still returned to user): {e}")
        return False
