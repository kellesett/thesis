# metrics/structural/contradiction/check.py
# Stage 3 — contradiction check: LLM decides whether each topic-filtered pair
# is a genuine contradiction. Parallelised via ThreadPoolExecutor.

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from .llm_utils import llm_json_cached
from .prompts import CONTRADICTION_PROMPT


def _check_one(
    pair: dict,
    client: OpenAI,
    model: str,
    max_retries: int,
    disable_reasoning: bool,
    provider: str | None,
    cache,
) -> dict:
    prompt = CONTRADICTION_PROMPT.format(
        section_i=pair["t1"],
        s1=pair["s1"][:400],
        section_j=pair["t2"],
        s2=pair["s2"][:400],
    )
    result = llm_json_cached(
        client, model, prompt,
        pair["s1"], pair["s2"], "contradiction",
        cache, max_retries, disable_reasoning, provider,
    )
    return {
        **pair,
        "is_contradiction":   result.get("is_contradiction", False),
        "reasoning":          result.get("reasoning", ""),
        "contradiction_type": result.get("contradiction_type", "none"),
        "status":             result.get("status", "ok"),
    }


def run_contradiction_check(
    pairs: list[dict],
    client: OpenAI,
    cfg: dict,
    cache,
) -> list[dict]:
    """Run contradiction check on topic-filtered pairs in parallel.

    Args:
        pairs: Candidates with same_subject=True from run_topic_filter().
        client: OpenAI client.
        cfg: Config dict.
        cache: diskcache.Cache instance.

    Returns:
        List of pairs enriched with is_contradiction, reasoning, contradiction_type.
        Order preserved.
    """
    model             = cfg["judge_model"]
    max_retries       = cfg.get("max_retries", 3)
    disable_reasoning = not cfg.get("judge_reasoning", True)
    provider          = cfg.get("judge_provider")
    concurrency       = cfg.get("concurrency", 25)

    results: list[dict | None] = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(
                _check_one, pair, client, model,
                max_retries, disable_reasoning, provider, cache,
            ): i
            for i, pair in enumerate(pairs)
        }
        with tqdm(total=len(pairs), desc="  3/5 contradiction", leave=False, unit="pair") as pbar:
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
                pbar.update(1)

    return results
