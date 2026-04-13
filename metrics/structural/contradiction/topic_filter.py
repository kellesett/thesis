# metrics/structural/contradiction/topic_filter.py
# Stage 2 — topic filter: LLM decides whether each candidate pair discusses
# the same subject. Parallelised via ThreadPoolExecutor.

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from .llm_utils import TokenCounter, llm_json_cached
from .prompts import (
    TOPIC_FILTER_PROMPT,
    TOPIC_FILTER_SCHEMA_COMPACT,
    TOPIC_FILTER_SCHEMA_FULL,
)


def _filter_one(
    cand: dict,
    client: OpenAI,
    model: str,
    max_retries: int,
    disable_reasoning: bool,
    provider: str | None,
    cache,
    token_counter: TokenCounter | None,
    log_reasoning: bool,
) -> dict:
    schema = TOPIC_FILTER_SCHEMA_FULL if log_reasoning else TOPIC_FILTER_SCHEMA_COMPACT
    prompt = TOPIC_FILTER_PROMPT.format(schema=schema, s1=cand["s1"][:400], s2=cand["s2"][:400])
    result = llm_json_cached(
        client, model, prompt,
        cand["s1"], cand["s2"], "topic",
        cache, max_retries, disable_reasoning, provider,
        token_counter=token_counter,
    )
    return {
        **cand,
        "same_subject":    result.get("same_subject", False),
        "topic_reasoning": result.get("reasoning", ""),
        "status":          result.get("status", "ok"),
    }


def run_topic_filter(
    candidates: list[dict],
    client: OpenAI,
    cfg: dict,
    cache,
    pbar=None,
    token_counter: TokenCounter | None = None,
    log_reasoning: bool = True,
) -> list[dict]:
    """Run topic filter on all candidates in parallel.

    Args:
        candidates:    Output of generate_candidates().
        client:        OpenAI client.
        cfg:           Config dict with judge_model, concurrency, etc.
        cache:         diskcache.Cache instance.
        pbar:          Optional tqdm bar (must be reset to len(candidates) before call).
                       Updated once per completed pair; postfix shows live stats.
        token_counter: Optional shared TokenCounter for live token tracking.

    Returns:
        List of candidates enriched with same_subject and status fields.
        Order preserved (same indices as input).
    """
    model             = cfg["judge_model"]
    max_retries       = cfg.get("max_retries", 3)
    disable_reasoning = not cfg.get("judge_reasoning", True)
    provider          = cfg.get("judge_provider")
    concurrency       = cfg.get("judge_workers", 25)

    results: list[dict | None] = [None] * len(candidates)
    n_selected = 0

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(
                _filter_one, cand, client, model,
                max_retries, disable_reasoning, provider, cache,
                token_counter, log_reasoning,
            ): i
            for i, cand in enumerate(candidates)
        }
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            if results[i].get("same_subject") and results[i].get("status") != "failed":
                n_selected += 1
            if pbar is not None:
                pbar.update(1)
                postfix = f"{n_selected}/{pbar.n} sel"
                if token_counter is not None:
                    postfix += f" | {token_counter.fmt()}"
                pbar.set_postfix_str(postfix)

    return results
