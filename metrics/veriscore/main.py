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
      {"claim_id": 0, "claim": "...", "source_sentence": ""}
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


# ── Config & client ───────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load YAML configuration from CONFIG path.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def make_client(cfg: dict) -> OpenAI:
    """Create OpenAI client from config.

    Args:
        cfg: Config dict with judge_api_key_env and judge_base_url.

    Returns:
        Initialized OpenAI client.

    Raises:
        RuntimeError: If API key environment variable not set.
    """
    api_key = os.environ.get(cfg["judge_api_key_env"], "")
    if not api_key:
        raise RuntimeError(f"API key not set: env var '{cfg['judge_api_key_env']}'")
    return OpenAI(api_key=api_key, base_url=cfg["judge_base_url"])


def llm_call(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1000,
    disable_reasoning: bool = False,
    provider: str | None = None,
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

    Returns:
        Stripped response text from LLM.

    Raises:
        RuntimeError: If LLM returns None content.
    """
    extra_body: dict = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
    if disable_reasoning:
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


def _make_bar(desc: str, total: int) -> tqdm:
    return tqdm(total=total, desc=desc, unit="sent", leave=True, dynamic_ncols=True)


def _extract_sentence_claims(
    i: int,
    sentences: list[str],
    client: OpenAI,
    model: str,
    max_tokens: int,
    disable_reasoning: bool,
    template: str,
    provider: str | None = None,
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
                            provider=provider)
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
) -> list[str]:
    """Extract deduplicated atomic claims from text.

    Sentences are processed in parallel (workers LLM calls at a time).
    Results are reassembled in sentence order, then deduplicated with
    dict.fromkeys() to preserve first-occurrence order.

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
        Deduplicated list of claim strings (order preserved by sentence index).

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
                i, sentences, client, model, max_tokens, disable_reasoning, template, provider,
            ): i
            for i in range(len(sentences))
        }
        for future in as_completed(futures):
            i, sent_claims = future.result()  # raises on LLM failure → propagates up
            claims_by_sent[i] = sent_claims
            if sents_bar is not None:
                sents_bar.update(1)

    # Flatten in sentence order, then deduplicate preserving first occurrence
    flat = [claim for sent_claims in claims_by_sent for claim in sent_claims]
    return list(dict.fromkeys(flat))


# ── Per-survey processing ─────────────────────────────────────────────────────

def process_survey(
    gen: dict,
    out_path: Path,
    cfg: dict,
    nlp,
    client: OpenAI,
) -> dict | None:
    survey_id = str(gen["id"])
    out_file  = out_path / f"{survey_id}.json"

    if cfg.get("resume") and out_file.exists():
        try:
            existing = json.loads(out_file.read_text())
            if existing.get("pipeline") == "veriscore":
                tqdm.write(
                    f"  [SKIP] {survey_id} — {existing['n_claims']} claims already saved",
                    file=sys.stderr,
                )
                return existing
        except Exception:
            pass
        tqdm.write(f"  [WARN] {survey_id} — will re-process", file=sys.stderr)

    if not gen.get("success", False):
        tqdm.write(f"  [SKIP] {survey_id} — generation not successful", file=sys.stderr)
        return None

    text = gen.get("text", "").strip()
    if not text:
        tqdm.write(f"  [SKIP] {survey_id} — empty text", file=sys.stderr)
        return None

    question    = gen.get("query", "")
    n_sentences = len([s for s in nlp(text).sents if s.text.strip()])

    tqdm.write(f"  [PROC] {survey_id} | {question[:70]}", file=sys.stderr)

    disable_reasoning = not cfg.get("judge_reasoning", True)
    max_tokens        = cfg.get("judge_max_tokens", 1000)
    workers           = cfg.get("sent_workers", 1)
    provider          = cfg.get("judge_provider")        # None → OpenRouter выбирает сам
    sents_bar = _make_bar("  sents", n_sentences)
    claims = extract_claims(text, nlp, client, cfg["judge_model"],
                            max_tokens=max_tokens,
                            disable_reasoning=disable_reasoning,
                            sents_bar=sents_bar,
                            workers=workers,
                            provider=provider)
    sents_bar.close()

    result = {
        "survey_id":   survey_id,
        "dataset_id":  gen["dataset_id"],
        "model_id":    gen["model_id"],
        "query":       question,
        "n_sentences": n_sentences,
        "n_claims":    len(claims),
        "claims": [
            {"claim_id": i, "claim": c, "source_sentence": ""}
            for i, c in enumerate(claims)
        ],
        "judge_model": cfg["judge_model"],
        "pipeline":    "veriscore",
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    tqdm.write(f"         → {len(claims)} claims", file=sys.stderr)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging("veriscore")
    parser = argparse.ArgumentParser(description="VeriScore claim extraction")
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. SurGE)")
    parser.add_argument("--model",   required=True, help="Model id (e.g. perplexity_dr)")
    args = parser.parse_args()

    cfg    = load_config()
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

    gen_files = sorted(gen_dir.glob("*.json"))
    gen_files = [f for f in gen_files if not re.search(r"_(raw|old)\.json$", f.name)]

    print(f"\n[veriscore] {args.dataset} / {args.model}")
    print(f"            {len(gen_files)} surveys → {out_dir}")
    print(f"            model: {cfg['judge_model']}\n")

    n_ok, n_skip = 0, 0
    total_claims = 0

    surveys_bar = tqdm(
        total=len(gen_files), desc="surveys", unit="survey",
        leave=True, dynamic_ncols=True,
    )

    for gf in gen_files:
        surveys_bar.set_postfix_str(gf.stem)
        gen = json.loads(gf.read_text())

        result = process_survey(gen, out_dir, cfg, nlp, client)
        if result is None:
            n_skip += 1
        else:
            n_ok         += 1
            total_claims += result["n_claims"]
            surveys_bar.set_postfix_str(f"{gf.stem} → {result['n_claims']} claims")
        surveys_bar.update(1)

    surveys_bar.close()

    print(f"\n[veriscore] done — ok={n_ok} skip={n_skip}")
    print(f"            total claims: {total_claims}")
    if n_ok > 0:
        print(f"            avg per survey: {total_claims // n_ok}")


if __name__ == "__main__":
    main()
