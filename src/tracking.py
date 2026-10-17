"""
tracking.py - Per-query cost and latency tracking, backed by Upstash Redis.

Design principles (mirrors cache.py's contract):
  - FAIL-OPEN: any tracking error (network, auth, malformed data) must NEVER
    crash the main /ask pipeline. Worst case: this one query goes untracked.
  - Reads ALL config from config.py - no independent hardcoded defaults.
  - Deliberately does NOT share cache.py's Redis client object, to avoid
    modifying the already-verified, production-live Phase 6 cache.py file.
    This duplicates a small (~15 line) client-init snippet; both read the
    same config values, so there is no divergent "source of truth" risk.
  - Cost is computed from config.py pricing rates (gemini-3.1-flash-lite,
    paid-tier rates, verified against https://ai.google.dev/gemini-api/docs/pricing
    on 2026-10-16). On the free tier (config.GEMINI_PRICING_TIER == "free"),
    real spend is $0 regardless of this number - cost_usd is still computed
    and stored so the pipeline requires no code change the day billing is
    linked and the tier flips to "paid".
  - Server-side only: tracking data is never added to the public AskResponse
    (confirmed decision - avoids exposing cost/timing info to API callers).
"""

import json
import random
from datetime import datetime, timezone

import config

_redis_client = None
_client_init_attempted = False


def _get_client():
    """
    Lazily creates and caches this module's own Upstash Redis client.
    Returns None if tracking is disabled or client creation fails - callers
    must handle None gracefully (fail-open contract, same as cache.py).
    """
    global _redis_client, _client_init_attempted

    if not config.TRACKING_ENABLED:
        return None

    if _redis_client is not None:
        return _redis_client

    if _client_init_attempted:
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
        print(f"⚠️ Tracking client init failed (tracking disabled for this process): {e}")
        return None


def compute_cost_usd(prompt_tokens: int, output_tokens: int) -> float:
    """
    Computes estimated cost in USD from token counts using the rates in
    config.py. Returns 0.0 for a cache hit (0 tokens in, 0 tokens out).
    This is a cost ESTIMATE based on the configured per-token rates; see
    config.GEMINI_PRICING_TIER for whether this reflects real billed spend.
    """
    input_cost = (prompt_tokens / 1_000_000) * config.GEMINI_INPUT_PRICE_PER_1M_USD
    output_cost = (output_tokens / 1_000_000) * config.GEMINI_OUTPUT_PRICE_PER_1M_USD
    return round(input_cost + output_cost, 8)


def record_query(
    cache_hit: bool,
    timings_ms: dict,
    prompt_tokens: int = 0,
    output_tokens: int = 0,
) -> bool:
    """
    Records one /ask query's cost and latency breakdown to Upstash Redis.

    Args:
        cache_hit: whether this query was served from cache.
        timings_ms: dict of stage timings in milliseconds. Expected keys
            (any may be omitted/zero for a cache hit):
            "cache_check_ms", "embedding_ms", "retrieval_ms",
            "generation_ms", "total_ms".
        prompt_tokens / output_tokens: from response.usage_metadata on a
            cache miss; 0 on a cache hit (no Gemini call was made).

    Returns True if the record was written, False on any failure (fail-open -
    caller should never let this raise or block the /ask response).
    """
    client = _get_client()
    if client is None:
        return False

    try:
        cost_usd = compute_cost_usd(prompt_tokens, output_tokens)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache_hit": cache_hit,
            "pricing_tier": config.GEMINI_PRICING_TIER,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "cost_usd": cost_usd,
            "latency_ms": {
                "cache_check": timings_ms.get("cache_check_ms", 0),
                "embedding": timings_ms.get("embedding_ms", 0),
                "retrieval": timings_ms.get("retrieval_ms", 0),
                "generation": timings_ms.get("generation_ms", 0),
                "total": timings_ms.get("total_ms", 0),
            },
        }

        client.rpush(config.TRACKING_LIST_KEY, json.dumps(record))

        # Throttled trim: only attempt on ~1-in-N writes, to conserve the
        # Upstash free-tier 500K-commands/month quota. Probabilistic rather
        # than a counter, so it works correctly even if Render restarts the
        # process or runs multiple workers (no shared in-memory state needed).
        if random.randint(1, config.TRACKING_TRIM_EVERY_N) == 1:
            client.ltrim(config.TRACKING_LIST_KEY, -config.TRACKING_MAX_RECORDS, -1)

        return True
    except Exception as e:
        print(f"⚠️ Tracking write error (non-fatal, /ask response unaffected): {e}")
        return False


def get_recent_records(limit: int = 50) -> list[dict] | None:
    """
    Reads the most recent `limit` tracking records (newest last, same order
    as stored). Returns None on any error or if tracking is disabled.
    This is for manual verification/debugging - not wired into any API
    endpoint in this phase (aggregation/reporting endpoints are out of scope).
    """
    client = _get_client()
    if client is None:
        return None

    try:
        raw_records = client.lrange(config.TRACKING_LIST_KEY, -limit, -1)
        return [json.loads(r) for r in raw_records]
    except Exception as e:
        print(f"⚠️ Tracking read error: {e}")
        return None
