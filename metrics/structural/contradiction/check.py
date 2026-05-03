# metrics/structural/contradiction/check.py
# Stage 3 — contradiction check: LLM decides whether each topic-filtered pair
# is a genuine contradiction. Parallelised via ThreadPoolExecutor.

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from .llm_utils import TokenCounter, llm_json_cached
from .prompts import (
    CONTRADICTION_PROMPT,
    CONTRADICTION_SCHEMA_COMPACT,
    CONTRADICTION_SCHEMA_FULL,
    PARAGRAPH_CONTRADICTION_PROMPT,
    PARAGRAPH_CONTRADICTION_SCHEMA_COMPACT,
    PARAGRAPH_CONTRADICTION_SCHEMA_FULL,
)


def _check_one(
    pair: dict,
    client: OpenAI,
    model: str,
    max_retries: int,
    disable_reasoning: bool,
    provider: str | None,
    cache,
    token_counter: TokenCounter | None,
    log_reasoning: bool,
    paragraph_max_chars: int,
) -> dict:
    is_paragraph = pair.get("unit") == "paragraph"
    if is_paragraph:
        schema = (
            PARAGRAPH_CONTRADICTION_SCHEMA_FULL
            if log_reasoning else
            PARAGRAPH_CONTRADICTION_SCHEMA_COMPACT
        )
        prompt = PARAGRAPH_CONTRADICTION_PROMPT.format(
            schema=schema,
            section_i=pair["t1"],
            s1=pair["s1"][:paragraph_max_chars],
            section_j=pair["t2"],
            s2=pair["s2"][:paragraph_max_chars],
        )
        suffix = "paragraph_contradiction"
    else:
        schema = CONTRADICTION_SCHEMA_FULL if log_reasoning else CONTRADICTION_SCHEMA_COMPACT
        prompt = CONTRADICTION_PROMPT.format(
            schema=schema,
            section_i=pair["t1"],
            s1=pair["s1"][:400],
            section_j=pair["t2"],
            s2=pair["s2"][:400],
        )
        suffix = "contradiction"

    result = llm_json_cached(
        client, model, prompt,
        pair["s1"], pair["s2"], suffix,
        cache, max_retries, disable_reasoning, provider,
        token_counter=token_counter,
    )
    if is_paragraph:
        contradictions = result.get("contradictions") or []
        has_contradiction = bool(result.get("has_contradiction", False) or contradictions)
        first_type = "none"
        if contradictions:
            first_type = contradictions[0].get("contradiction_type", "none")
        return {
            **pair,
            "is_contradiction": has_contradiction,
            "sentence_pairs": contradictions,
            "reasoning": result.get("reasoning", ""),
            "contradiction_type": first_type,
            "status": result.get("status", "ok"),
        }

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
    pbar=None,
    token_counter: TokenCounter | None = None,
    log_reasoning: bool = True,
) -> list[dict]:
    """Run contradiction check on topic-filtered pairs in parallel.

    Args:
        pairs:         Candidates with same_subject=True from run_topic_filter().
        client:        OpenAI client.
        cfg:           Config dict.
        cache:         diskcache.Cache instance.
        pbar:          Optional tqdm bar (must be reset to len(pairs) before call).
                       Updated once per completed pair; postfix shows live stats.
        token_counter: Optional shared TokenCounter for live token tracking.

    Returns:
        List of pairs enriched with is_contradiction, reasoning, contradiction_type.
        Order preserved.
    """
    model             = cfg["judge_model"]
    max_retries       = cfg.get("max_retries", 3)
    disable_reasoning = not cfg.get("judge_reasoning", True)
    provider          = cfg.get("judge_provider")
    concurrency       = cfg.get("judge_workers", 25)
    paragraph_max_chars = cfg.get("paragraph_max_chars", 1800)

    results: list[dict | None] = [None] * len(pairs)
    n_confirmed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(
                _check_one, pair, client, model,
                max_retries, disable_reasoning, provider, cache,
                token_counter, log_reasoning, paragraph_max_chars,
            ): i
            for i, pair in enumerate(pairs)
        }
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            if results[i].get("is_contradiction") and results[i].get("status") != "failed":
                n_confirmed += 1
            if pbar is not None:
                pbar.update(1)
                postfix = f"{n_confirmed}/{pbar.n} confirmed"
                if token_counter is not None:
                    postfix += f" | {token_counter.fmt()}"
                pbar.set_postfix_str(postfix)

    return results
