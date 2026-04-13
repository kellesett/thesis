# metrics/structural/contradiction/topic_filter.py
# Stage 2 — topic filter: LLM decides whether each candidate pair discusses
# the same subject. Parallelised via ThreadPoolExecutor.

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from .llm_utils import llm_json_cached
from .prompts import TOPIC_FILTER_PROMPT


def _filter_one(
    cand: dict,
    client: OpenAI,
    model: str,
    max_retries: int,
    disable_reasoning: bool,
    provider: str | None,
    cache,
) -> dict:
    prompt = TOPIC_FILTER_PROMPT.format(s1=cand["s1"][:400], s2=cand["s2"][:400])
    result = llm_json_cached(
        client, model, prompt,
        cand["s1"], cand["s2"], "topic",
        cache, max_retries, disable_reasoning, provider,
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
) -> list[dict]:
    """Run topic filter on all candidates in parallel.

    Args:
        candidates: Output of generate_candidates().
        client: OpenAI client.
        cfg: Config dict with judge_model, concurrency, etc.
        cache: diskcache.Cache instance.

    Returns:
        List of candidates enriched with same_subject and status fields.
        Order preserved (same indices as input).
    """
    model             = cfg["judge_model"]
    max_retries       = cfg.get("max_retries", 3)
    disable_reasoning = not cfg.get("judge_reasoning", True)
    provider          = cfg.get("judge_provider")
    concurrency       = cfg.get("concurrency", 25)

    results: list[dict | None] = [None] * len(candidates)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(
                _filter_one, cand, client, model,
                max_retries, disable_reasoning, provider, cache,
            ): i
            for i, cand in enumerate(candidates)
        }
        with tqdm(total=len(candidates), desc="  2/5 topic filter", leave=False, unit="pair") as pbar:
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
                pbar.update(1)

    return results
