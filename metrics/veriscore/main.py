#!/usr/bin/env python3
"""
metrics/veriscore/main.py
Atomic claim extraction from generated surveys using VeriScore.

Implements Sun et al., ACL 2024 (arXiv:2406.19276), extraction stage only:
  Sentence split (spaCy) → LLM per sentence (non-QA) → deduplicated claims

Results saved to: results/scores/<dataset>_<model>_claims/<survey_id>.json
(same directory as claimify — alternative extractor, compatible format)

Format:
  {
    "survey_id":   "0",
    "dataset_id":  "SurGE",
    "model_id":    "perplexity_dr",
    "query":       "...",
    "n_sentences": 142,
    "n_claims":    87,
    "claims": [
      {
        "claim_id": 0,
        "claim":    "...",
        "sources": [                      // all sentences where this claim appeared
          {"sentence": "...", "sentence_idx": 3},
          {"sentence": "...", "sentence_idx": 47}
        ]
      }
    ],
    "judge_model": "...",
    "pipeline":    "veriscore",
    "timestamp":   "2026-..."
  }

Usage (inside Docker):
    python metrics/veriscore/main.py --dataset SurGE --model perplexity_dr
"""

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import spacy
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger(__name__)

ROOT   = Path(__file__).parent.parent.parent
REPO   = ROOT / "repos" / "VeriScore"
CONFIG = Path(__file__).parent / "config.yaml"
sys.path.insert(0, str(ROOT))

from src.log_setup import setup_logging
from metrics.utils import (
    TokenCounter, load_config as load_config_util, make_client,
    check_and_load_cache, load_generation_files,
)


def llm_call(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1000,
    disable_reasoning: bool = False,
    provider: str | None = None,
    reasoning_effort: str | None = None,
    token_counter: TokenCounter | None = None,
) -> str:
    """Call LLM for claim extraction.

    Args:
        client: OpenAI client instance.
        model: Model name to use.
        system: System prompt.
        user: User message with snippet and sentence.
        max_tokens: Maximum tokens in response.
        disable_reasoning: Whether to disable reasoning tokens.
        provider: Optional OpenRouter provider name (e.g. "alibaba").
        reasoning_effort: "low"|"medium"|"high" for reasoning-capable models (takes priority over disable_reasoning).

    Returns:
        Stripped response text from LLM.

    Raises:
        RuntimeError: If LLM returns None content.
    """
    extra_body: dict = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
    if reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort
    elif disable_reasoning:
        extra_body["reasoning"] = {"enabled": False}

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0,
        extra_body=extra_body or None,
    )
    content = response.choices[0].message.content
    if content is None:
        print(f"[ERROR] OpenRouter None content (finish_reason={response.choices[0].finish_reason}).\n"
              f"Full response: {response}", file=sys.stderr)
        raise RuntimeError(
            f"Model returned None content (finish_reason={response.choices[0].finish_reason}). "
            f"Try increasing judge_max_tokens or disabling reasoning."
        )
    if token_counter is not None and response.usage is not None:
        cost = (response.usage.model_extra or {}).get("cost") or 0.0
        token_counter.add(
            response.usage.prompt_tokens or 0,
            response.usage.completion_tokens or 0,
            float(cost),
        )
    return content.strip()


# ── Claim Extraction ──────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM = (
    "You are a helpful assistant who can extract verifiable atomic claims from "
    "a piece of text. Each atomic fact should be verifiable against reliable "
    "external world knowledge (e.g., via Wikipedia)"
)

_NON_QA_PROMPT_TEMPLATE: str | None = None  # Loaded in main()


def _load_template() -> str:
    """Load VeriScore extraction prompt template.

    Returns:
        Template string with {snippet} and {sentence} placeholders.

    Raises:
        FileNotFoundError: If template file does not exist.
    """
    global _NON_QA_PROMPT_TEMPLATE
    if _NON_QA_PROMPT_TEMPLATE is None:
        template_path = REPO / "prompt" / "extraction_non_qa_template.txt"
        if not template_path.exists():
            raise FileNotFoundError(
                f"VeriScore prompt template not found at {template_path}. "
                f"Ensure VeriScore repository is cloned at {REPO}"
            )
        _NON_QA_PROMPT_TEMPLATE = template_path.read_text()
    return _NON_QA_PROMPT_TEMPLATE


def _build_snippet(sentences: list[str], i: int) -> str:
    """Build context window snippet for sentence i (non-QA mode).

    Args:
        sentences: Full list of sentences from text.
        i: Index of target sentence.

    Returns:
        Context snippet with target sentence marked with <SOS> and <EOS>.
    """
    lead_sent = sentences[0]
    context1  = " ".join(sentences[max(0, i - 3):i])
    sentence  = f"<SOS>{sentences[i].strip()}<EOS>"
    context2  = " ".join(sentences[i + 1:i + 2])

    if len(sentences) <= 5:
        return f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()
    else:
        return f"{lead_sent.strip()} {context1.strip()} {sentence.strip()} {context2.strip()}".strip()


def _extract_sentence_claims(
    i: int,
    sentences: list[str],
    client: OpenAI,
    model: str,
    max_tokens: int,
    disable_reasoning: bool,
    template: str,
    provider: str | None = None,
    reasoning_effort: str | None = None,
    token_counter: TokenCounter | None = None,
) -> tuple[int, list[str]]:
    """Extract claims from a single sentence. Returns (index, claims).

    Keeping the index allows order-preserving reassembly after parallel execution.

    Raises:
        RuntimeError: If LLM call fails.
    """
    raw_sentence = sentences[i]
    snippet  = _build_snippet(sentences, i)
    prompt   = template.format(snippet=snippet, sentence=raw_sentence)

    try:
        response = llm_call(client, model, _EXTRACTION_SYSTEM, prompt,
                            max_tokens=max_tokens,
                            disable_reasoning=disable_reasoning,
                            provider=provider,
                            reasoning_effort=reasoning_effort,
                            token_counter=token_counter)
    except Exception as e:
        raise RuntimeError(
            f"LLM failed for sentence {i} ({raw_sentence[:60]!r}): {e}"
        ) from e

    if not response or "No verifiable claim." in response:
        return i, []

    claims: list[str] = []
    for line in response.split("\n"):
        line = line.strip().replace("- ", "")
        line = re.sub(r"^\d+\.?\s+", "", line)
        if not line or line.startswith("Note:"):
            continue
        claims.append(line)

    return i, claims


def extract_claims(
    text: str,
    nlp,
    client: OpenAI,
    model: str,
    max_tokens: int = 1000,
    disable_reasoning: bool = False,
    sents_bar: tqdm | None = None,
    workers: int = 1,
    provider: str | None = None,
    reasoning_effort: str | None = None,
    token_counter: TokenCounter | None = None,
    global_tokens: TokenCounter | None = None,
) -> list[tuple[str, list[tuple[str, int]]]]:
    """Extract deduplicated atomic claims from text with ALL source locations.

    Sentences are processed in parallel (workers LLM calls at a time).
    Results are reassembled in sentence order, then deduplicated by claim
    text — but crucially, **every occurrence is preserved as a source**.
    If the same atomic claim surfaces from multiple sentences (e.g. a fact
    repeated in intro + conclusion), each sentence contributes to the
    returned source list. This is essential for factchecking: citations in
    every paragraph where the claim appears must be unioned into the
    evidence pool — keeping only the first location would silently drop
    supporting refs that live near the later occurrence.

    Args:
        text: Survey text to extract claims from.
        nlp: spaCy NLP pipeline for sentence splitting.
        client: OpenAI client instance.
        model: Judge model name.
        max_tokens: Maximum tokens per LLM response.
        disable_reasoning: Whether to disable reasoning tokens.
        sents_bar: Optional tqdm progress bar.
        workers: Number of parallel LLM calls.
        provider: Optional OpenRouter provider name.

    Returns:
        List of ``(claim, sources)`` where ``sources`` is a non-empty list of
        ``(sentence_text, sentence_idx)`` tuples — one per occurrence in
        sentence-index order. ``sentence_idx`` is 0-based in the spaCy split
        of ``text`` and is a deterministic pointer when the same spaCy model
        is applied downstream.

    Raises:
        RuntimeError: On LLM failure for any sentence.
    """
    sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    template  = _load_template()

    # claims_by_sent[i] = list of raw claims from sentence i
    claims_by_sent: list[list[str]] = [[] for _ in sentences]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                _extract_sentence_claims,
                i, sentences, client, model, max_tokens, disable_reasoning, template, provider, reasoning_effort, token_counter,
            ): i
            for i in range(len(sentences))
        }
        for future in as_completed(futures):
            i, sent_claims = future.result()  # raises on LLM failure → propagates up
            claims_by_sent[i] = sent_claims
            if sents_bar is not None:
                # refresh=False on the postfix update — let ``update(1)``
                # below be the SINGLE redraw for this tick. Default
                # ``set_postfix_str`` forces a refresh which, combined with
                # update()'s own redraw, doubles the terminal-write rate
                # and bypasses mininterval-based throttling. At parallel=50
                # this floods docker's stdout forwarder and `\r`-based
                # overwrites collapse into appended lines.
                if token_counter is not None:
                    if global_tokens is not None:
                        sents_bar.set_postfix_str(
                            f"surv: {token_counter.fmt()}  total: {global_tokens.fmt()}",
                            refresh=False,
                        )
                    else:
                        sents_bar.set_postfix_str(token_counter.fmt(), refresh=False)
                sents_bar.update(1)

    # Dedup on claim TEXT but COLLECT every sentence where the claim occurred.
    # Python 3.7+ dict preserves insertion order, so ``occurrences.items()``
    # walks claims in the same order as the old ``list(dict.fromkeys(flat))``.
    # Each value is a list of (sentence_text, sentence_idx) — empty is
    # impossible by construction (we only append under an extant claim).
    occurrences: dict[str, list[tuple[str, int]]] = {}
    for sent_idx, sent_claims in enumerate(claims_by_sent):
        for claim in sent_claims:
            occurrences.setdefault(claim, []).append(
                (sentences[sent_idx], sent_idx)
            )
    return list(occurrences.items())


# ── Per-survey processing ─────────────────────────────────────────────────────

def process_survey(
    gen: dict,
    out_path: Path,
    cfg: dict,
    nlp,
    client: OpenAI,
    *,
    global_tokens: TokenCounter | None = None,
) -> dict | None:
    """Run claim extraction for one survey.

    A fresh inner tqdm bar is created for THIS survey with ``leave=False``
    (no explicit ``position=`` — see the inline comment below) and closed
    when the survey finishes. Simpler and more robust than:
      (a) a long-lived ``reset()``-reused bar — after the first survey's
          bar finishes at 100%, a second ``reset(total=...)`` on the same
          instance interacts badly with ANSI cursor positioning and tqdm
          starts appending lines instead of overwriting;
      (b) explicit ``position=1`` on the recreated bar — in Docker with -t
          the repeated create/close at a fixed position produced duplicate
          rendering on the second survey onwards. Auto-positioning (omit
          ``position=``) is the canonical nested-tqdm pattern and survives
          all the state transitions cleanly.

    ``global_tokens`` is an optional run-wide :class:`TokenCounter`; per-survey
    tokens are rolled up into it after extraction so the outer bar can
    display cumulative usage/cost.
    """
    survey_id = str(gen["id"])
    out_file  = out_path / f"{survey_id}.json"

    cached = check_and_load_cache(out_file, cfg, survey_id, required_keys=("survey_id", "claims"))
    if cached is not None:
        logger.info(
            "[SKIP] sid=%s — %d claims already saved",
            survey_id, cached["n_claims"],
        )
        return cached

    if not gen.get("success", False):
        logger.info("[SKIP] sid=%s — generation not successful", survey_id)
        return None

    text = gen.get("text", "").strip()
    if not text:
        logger.info("[SKIP] sid=%s — empty text", survey_id)
        return None

    question    = gen.get("query", "")
    n_sentences = len([s for s in nlp(text).sents if s.text.strip()])

    logger.info("[PROC] sid=%s | %s", survey_id, question[:70])

    disable_reasoning = not cfg.get("judge_reasoning", True)
    max_tokens        = cfg.get("judge_max_tokens", 1000)
    workers           = cfg.get("sent_workers", 1)
    provider          = cfg.get("judge_provider") or None
    reasoning_effort  = cfg.get("judge_reasoning_effort") or None

    # Per-survey counter; rolled up into global_tokens at the end so the outer
    # bar can show cumulative cost across the whole run.
    per_tokens = TokenCounter()

    # Fresh inner bar per survey (close+recreate, not reset+reuse — see
    # docstring). ``leave=False`` wipes it on close. IMPORTANTLY: no explicit
    # ``position=`` — tqdm auto-positions via its global ``_instances``
    # registry relative to other active bars. In Docker with -t the explicit
    # position=0/1 combo (outer long-lived + inner repeatedly recreated)
    # produced duplicate rendering on the second survey onwards, likely due
    # to mismatched cursor bookkeeping when the second inner bar's init
    # ANSI sequences landed on a row the outer bar had just refreshed.
    # Auto-positioning is the canonical nested-tqdm pattern and survives all
    # the state transitions cleanly.
    sents_bar = tqdm(
        total=max(n_sentences, 1),
        desc="  sents", unit="sent",
        leave=False, dynamic_ncols=True,
        mininterval=0.3,
    )
    _postfix = (
        f"surv: {per_tokens.fmt()}  total: {global_tokens.fmt()}"
        if global_tokens is not None else per_tokens.fmt()
    )
    sents_bar.set_postfix_str(_postfix)

    try:
        claims = extract_claims(text, nlp, client, cfg["judge_model"],
                                max_tokens=max_tokens,
                                disable_reasoning=disable_reasoning,
                                sents_bar=sents_bar,
                                workers=workers,
                                provider=provider,
                                reasoning_effort=reasoning_effort,
                                token_counter=per_tokens,
                                global_tokens=global_tokens)
    finally:
        sents_bar.close()

    # Roll per-survey usage up; do this AFTER extract returns so the inner
    # bar has shown per-survey running totals all the way through.
    if global_tokens is not None:
        global_tokens.add(
            per_tokens.in_tokens,
            per_tokens.out_tokens,
            per_tokens.cost_usd,
        )

    result = {
        "survey_id":   survey_id,
        "dataset_id":  gen["dataset_id"],
        "model_id":    gen["model_id"],
        "query":       question,
        "n_sentences": n_sentences,
        "n_claims":    len(claims),
        "claims": [
            {
                "claim_id": i,
                "claim":    c,
                # Every occurrence of this claim's text in the doc — not just
                # the first. Downstream factcheck unions citations across all
                # entries so multi-mention claims get their full evidence pool.
                "sources": [
                    {"sentence": s, "sentence_idx": si}
                    for s, si in occs
                ],
            }
            for i, (c, occs) in enumerate(claims)
        ],
        "judge_model": cfg["judge_model"],
        "pipeline":    "veriscore",
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "usage": {
            "in_tokens":  per_tokens.in_tokens,
            "out_tokens": per_tokens.out_tokens,
            "cost_usd":   round(per_tokens.cost_usd, 6),
        },
    }

    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    logger.info(
        "[DONE] sid=%s — %d claims  [%s]",
        survey_id, len(claims), per_tokens.fmt(),
    )
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging("veriscore")

    # Reserve stderr for tqdm — drop the console StreamHandler that
    # setup_logging attached so nothing stomps the live bars. All log records
    # keep flowing to results/logs/veriscore.log via FileHandler (DEBUG+).
    # FileHandler subclasses StreamHandler, so order the check by type.
    log_file: Path | None = None
    for h in list(logging.root.handlers):
        if isinstance(h, logging.FileHandler):
            log_file = Path(getattr(h, "baseFilename", "")) or log_file
        elif isinstance(h, logging.StreamHandler):
            logging.root.removeHandler(h)
    if log_file is not None:
        print(
            f"Logs → {log_file}  (tail -f to follow; stderr reserved for tqdm)",
            file=sys.stderr, flush=True,
        )

    parser = argparse.ArgumentParser(description="VeriScore claim extraction")
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. SurGE)")
    parser.add_argument("--model",   required=True, help="Model id (e.g. perplexity_dr)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only surveys with survey_id <= LIMIT "
                             "(inclusive, id-based — not positional; handles sparse sets "
                             "like SurGE_reference where ids may skip).")
    parser.add_argument("--id", type=int, default=None,
                        help="Process only this survey_id.")
    args = parser.parse_args()

    cfg    = load_config_util(CONFIG)
    client = make_client(cfg)

    # Check for template file early
    try:
        _load_template()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.error(
            "spaCy model 'en_core_web_sm' not found. "
            "Install with: python -m spacy download en_core_web_sm"
        )
        sys.exit(1)

    gen_dir = ROOT / "results" / "generations" / f"{args.dataset}_{args.model}"
    if not gen_dir.exists():
        print(f"[ERROR] Generation dir not found: {gen_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = ROOT / "results" / "scores" / f"{args.dataset}_{args.model}_claims"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_files = load_generation_files(gen_dir)
    # load_generation_files sorts lexically ("10.json" < "2.json"); re-sort
    # numerically so --limit / --id behave as "first N by survey_id".
    gen_files.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)

    if args.id is not None:
        gen_files = [f for f in gen_files if f.stem == str(args.id)]
    elif args.limit is not None:
        # id-based inclusive filter, not positional. For dense datasets this is
        # equivalent to "first N"; for sparse ones (e.g. SurGE_reference with
        # ids like 0,1,2,3,5,12,..) it keeps the semantic natural.
        gen_files = [
            f for f in gen_files
            if f.stem.isdigit() and int(f.stem) <= args.limit
        ]

    # Pre-run banner — goes to stdout (printed before any tqdm bar opens,
    # so no collision) and to the log file so post-mortems know what ran.
    banner = (
        f"[veriscore] {args.dataset} / {args.model}\n"
        f"            {len(gen_files)} surveys → {out_dir}\n"
        f"            model: {cfg['judge_model']}"
    )
    print("\n" + banner + "\n")
    for line in banner.splitlines():
        logger.info(line)

    n_ok, n_skip = 0, 0
    total_claims = 0

    # Run-wide token accumulator; the inner sents bar is created fresh in
    # process_survey per survey (see its docstring for why we don't reset()).
    global_tokens = TokenCounter()
    # Outer bar — no explicit position= either. Canonical nested pattern:
    # tqdm picks position=0 automatically for the only active bar at this
    # scope, and any inner bars created by process_survey get position=1.
    surveys_bar = tqdm(
        total=len(gen_files), desc="surveys", unit="survey",
        leave=True, dynamic_ncols=True,
    )

    try:
        for gf in gen_files:
            surveys_bar.set_postfix_str(f"{gf.stem}  {global_tokens.fmt()}")
            gen = json.loads(gf.read_text())

            result = process_survey(
                gen, out_dir, cfg, nlp, client,
                global_tokens=global_tokens,
            )
            if result is None:
                n_skip += 1
            else:
                n_ok         += 1
                total_claims += result["n_claims"]
                surveys_bar.set_postfix_str(
                    f"{gf.stem} → {result['n_claims']} claims  {global_tokens.fmt()}"
                )
            surveys_bar.update(1)
    finally:
        surveys_bar.close()

    # Post-run summary — same reasoning as the banner: bars are already
    # closed by the `finally:` above, so print() is safe. Duplicate to
    # logger.info so the file record is self-contained.
    summary_lines = [
        f"[veriscore] done — ok={n_ok} skip={n_skip}",
        f"            total claims: {total_claims}",
    ]
    if n_ok > 0:
        summary_lines.append(f"            avg per survey: {total_claims // n_ok}")
    summary_lines.append(
        f"            usage: in={global_tokens.in_tokens}  out={global_tokens.out_tokens}  "
        f"cost=${global_tokens.cost_usd:.6f}  ({global_tokens.fmt()})"
    )
    print("\n" + "\n".join(summary_lines))
    for line in summary_lines:
        logger.info(line)


if __name__ == "__main__":
    main()
