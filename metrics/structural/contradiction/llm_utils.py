# metrics/structural/contradiction/llm_utils.py
# Shared LLM call helper with diskcache + retry + None content check.

import hashlib
import json
import re
import sys
import time

from openai import OpenAI


def _cache_key(model: str, s1: str, s2: str, suffix: str) -> str:
    """SHA-256 key: invalidates when model or either sentence changes."""
    raw = f"{model}||{s1}||{s2}||{suffix}"
    return hashlib.sha256(raw.encode()).hexdigest()


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
) -> dict:
    """Call LLM, parse JSON, cache result. Returns {"status": "failed"} on persistent error.

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
    """
    key = _cache_key(model, s1, s2, suffix)
    if cache is not None and key in cache:
        return cache[key]

    extra_body: dict = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
    if disable_reasoning:
        extra_body["reasoning"] = {"enabled": False}

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                extra_body=extra_body or None,
            )
            content = resp.choices[0].message.content
            if content is None:
                print(
                    f"[ERROR] OpenRouter None content "
                    f"(finish_reason={resp.choices[0].finish_reason}).\n"
                    f"Full response: {resp}",
                    file=sys.stderr,
                )
                raise RuntimeError(
                    f"Model returned None content "
                    f"(finish_reason={resp.choices[0].finish_reason})."
                )
            raw = content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
            if cache is not None:
                cache[key] = result
            return result
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return {"_error": str(e), "status": "failed"}

    return {"status": "failed"}
