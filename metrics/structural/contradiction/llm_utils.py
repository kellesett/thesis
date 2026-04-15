# metrics/structural/contradiction/llm_utils.py
# Shared LLM call helper with diskcache + retry + None content check.

import hashlib
import sys
from pathlib import Path

from openai import OpenAI

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metrics.utils import TokenCounter, llm_json_call


# ── Cache key ─────────────────────────────────────────────────────────────────

def _cache_key(model: str, s1: str, s2: str, suffix: str) -> str:
    """SHA-256 key: invalidates when model or either sentence changes."""
    raw = f"{model}||{s1}||{s2}||{suffix}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ── LLM call ─────────────────────────────────────────────────────────────────

def llm_json_cached(
    client: OpenAI,
    model: str,
    prompt: str,
    s1: str,
    s2: str,
    suffix: str,
    cache,
    max_retries: int = 3,
    disable_reasoning: bool = False,
    provider: str | None = None,
    token_counter: TokenCounter | None = None,
) -> dict:
    """Call LLM, parse JSON, cache result. Returns {"status": "failed"} on persistent error.

    Token usage is reported to token_counter only for fresh API calls (not cache hits).

    Args:
        client: OpenAI client.
        model: Model name.
        prompt: Full prompt string (static prefix first for KV-cache reuse).
        s1: First sentence — used as part of cache key.
        s2: Second sentence — used as part of cache key.
        suffix: Distinguishes topic vs contradiction cache entries ("topic"|"contradiction").
        cache: diskcache.Cache instance, or None to skip caching.
        max_retries: Retry attempts with exponential backoff.
        disable_reasoning: Disable thinking tokens (Qwen3, DeepSeek-R1).
        provider: Optional OpenRouter provider override.
        token_counter: Optional thread-safe accumulator for tracking API token usage.
    """
    key = _cache_key(model, s1, s2, suffix)
    if cache is not None and key in cache:
        return cache[key]

    result = llm_json_call(
        client, model,
        messages=[{"role": "user", "content": prompt}],
        max_retries=max_retries,
        provider=provider,
        disable_reasoning=disable_reasoning,
        token_counter=token_counter,
        on_failure=lambda e: {"_error": str(e), "status": "failed"},
    )
    if cache is not None and result.get("status") != "failed":
        cache[key] = result
    return result
