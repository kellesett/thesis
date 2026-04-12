#!/usr/bin/env python3
"""
metrics/veriscore/main.py
VeriScore factuality metric for generated surveys.

Implements Sun et al., ACL 2024 (arXiv:2406.19276):
  Sentence split → Claim extraction (LLM, non-QA) → Evidence retrieval (Serper)
  → Claim verification (LLM, few-shot trinary/binary)

Results saved to: results/scores/<dataset>_<model>_veriscore/<survey_id>.json

Format:
  {
    "survey_id": "0",
    "dataset_id": "SurGE",
    "model_id": "perplexity_dr",
    "query": "...",
    "n_sentences": 142,
    "n_claims": 87,
    "n_supported": 54,
    "veriscore_precision": 0.6207,
    "claims": [
      {
        "claim_id": 0,
        "claim": "...",
        "verification_result": "supported",
        "search_results_str": "..."
      }
    ],
    "judge_model": "...",
    "timestamp": "2026-..."
  }

Prompt templates and few-shot demos are read from repos/VeriScore/.

Usage (inside Docker):
    python metrics/veriscore/main.py --dataset SurGE --model perplexity_dr
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import spacy
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

ROOT   = Path(__file__).parent.parent.parent
REPO   = ROOT / "repos" / "VeriScore"
CONFIG = Path(__file__).parent / "config.yaml"


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG) as f:
        return yaml.safe_load(f)


# ── LLM helpers ───────────────────────────────────────────────────────────────

def make_client(cfg: dict) -> OpenAI:
    api_key = os.environ.get(cfg["judge_api_key_env"], "")
    if not api_key:
        raise RuntimeError(f"API key not set: env var '{cfg['judge_api_key_env']}'")
    return OpenAI(api_key=api_key, base_url=cfg["judge_base_url"])


def llm_call(client: OpenAI, model: str, system: str, user: str,
             max_tokens: int = 1000) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ── Claim Extraction ──────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM = (
    "You are a helpful assistant who can extract verifiable atomic claims from "
    "a piece of text. Each atomic fact should be verifiable against reliable "
    "external world knowledge (e.g., via Wikipedia)"
)

_NON_QA_PROMPT_TEMPLATE = (REPO / "prompt" / "extraction_non_qa_template.txt").read_text()


def _build_snippet(sentences: list[str], i: int) -> str:
    """Build context window snippet for sentence i (non-QA mode)."""
    lead_sent = sentences[0]
    context1  = " ".join(sentences[max(0, i - 3):i])
    sentence  = f"<SOS>{sentences[i].strip()}<EOS>"
    context2  = " ".join(sentences[i + 1:i + 2])

    if len(sentences) <= 5:
        return f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()
    else:
        # long text: prepend lead sentence to help with coreference
        return f"{lead_sent.strip()} {context1.strip()} {sentence.strip()} {context2.strip()}".strip()


def extract_claims(
    text: str,
    nlp,
    client: OpenAI,
    model: str,
) -> list[str]:
    """
    Extract deduplicated atomic claims from survey text using VeriScore's
    non-QA approach: sentence-by-sentence with context window.
    Raises RuntimeError on LLM failure.
    """
    sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    all_claims: list[str] = []

    for i, raw_sentence in enumerate(sentences):
        snippet = _build_snippet(sentences, i)
        prompt  = _NON_QA_PROMPT_TEMPLATE.format(snippet=snippet, sentence=raw_sentence)

        try:
            response = llm_call(client, model, _EXTRACTION_SYSTEM, prompt)
        except Exception as e:
            raise RuntimeError(
                f"Extraction LLM failed for sentence {i} "
                f"({raw_sentence[:60]!r}): {e}"
            ) from e

        if not response or "No verifiable claim." in response:
            continue

        for line in response.split("\n"):
            line = line.strip().replace("- ", "")
            line = re.sub(r"^\d+\.?\s+", "", line)
            if not line or line.startswith("Note:"):
                continue
            if line not in all_claims:
                all_claims.append(line)

    return all_claims


# ── Evidence Retrieval ────────────────────────────────────────────────────────

class SearchCache:
    """Persistent JSON cache for Serper API results."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self._data: dict = {}
        self._dirty = 0
        if cache_path.exists():
            try:
                self._data = json.loads(cache_path.read_text())
                tqdm.write(
                    f"           [cache] loaded {len(self._data)} search entries",
                    file=sys.stderr,
                )
            except Exception as e:
                tqdm.write(f"  [WARN] search cache corrupt, starting fresh: {e}",
                           file=sys.stderr)

    def get(self, query: str) -> dict | None:
        return self._data.get(query.strip())

    def set(self, query: str, result: dict) -> None:
        self._data[query.strip()] = result
        self._dirty += 1
        if self._dirty % 10 == 0:
            self.flush()

    def flush(self) -> None:
        self.cache_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2)
        )


def search_evidence(
    claims: list[str],
    serper_key: str,
    cache: SearchCache,
    search_res_num: int = 5,
) -> dict[str, list[dict]]:
    """
    Retrieve top search results from Serper Google Search for each claim.
    Returns: {claim: [{title, snippet, link}, ...]}
    Raises RuntimeError on API error.
    """
    url     = "https://google.serper.dev/search"
    headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
    results: dict[str, list[dict]] = {}

    for claim in claims:
        cached = cache.get(claim)
        if cached is not None:
            raw = cached
        else:
            payload  = json.dumps({"q": claim})
            response = requests.post(url, headers=headers, data=payload, timeout=30)
            response.raise_for_status()
            raw = response.json()
            cache.set(claim, raw)

        if "statusCode" in raw:
            raise RuntimeError(f"Serper API error: {raw.get('message', raw)}")

        organic = raw.get("organic", [])
        results[claim] = [
            {
                "title":   item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link":    item.get("link", ""),
            }
            for item in organic[:search_res_num]
        ]

    cache.flush()
    return results


# ── Claim Verification ────────────────────────────────────────────────────────

_VERIFICATION_SYSTEM_3 = (
    "You are a helpful assistant who can judge whether a claim is supported or "
    "contradicted by the search results, or whether there is no enough information "
    "to make a judgement."
)
_VERIFICATION_SYSTEM_2 = (
    "You are a helpful assistant who can judge whether a claim is supported by "
    "the search results or not."
)


def _load_verification_prompt(label_n: int, demo_path: Path) -> tuple[str, str]:
    """
    Build the few-shot verification prompt template and task suffix.
    Returns (prompt_initial, your_task_template).
    prompt_initial already has few-shot examples formatted in.
    your_task_template is a format string with {claim} and {search_results}.
    """
    instruction_file = (
        "verification_instruction_trinary.txt" if label_n == 3
        else "verification_instruction_binary.txt"
    )
    instruction_temp = (REPO / "prompt" / instruction_file).read_text()

    examples = [
        json.loads(line)
        for line in demo_path.read_text().splitlines()
        if line.strip()
    ]

    # VeriScore non-Claude format: Claim → search_results → decision
    elements: list[str] = []
    for ex in examples:
        label = ex["human_label"]
        if label_n == 2:
            label = "Supported." if label.lower().startswith("support") else "Unsupported."
        elements.extend([ex["claim"], ex["search_result"], label])

    prompt_initial = instruction_temp.format(*elements)
    your_task_template = (
        "Your task:\n\nClaim: {claim}\n\n{search_results}\n\nYour decision:"
    )
    return prompt_initial, your_task_template


def _format_search_results(snippets: list[dict]) -> str:
    parts: list[str] = []
    for i, item in enumerate(snippets, 1):
        parts.append(
            f"Search result {i}\n"
            f"Title: {item['title'].strip()}\n"
            f"Link: {item['link'].strip()}\n"
            f"Content: {item['snippet'].strip()}\n"
        )
    return "\n".join(parts)


def _parse_verification_label(response: str, label_n: int) -> str:
    """
    Parse model response into a normalized label string.
    VeriScore format: ###Supported.### / ###Contradicted.### / ###Inconclusive.###
    """
    parts = response.split("###")
    if len(parts) >= 3:
        raw = parts[1].strip().rstrip(".").lower()
        return raw  # "supported" / "contradicted" / "inconclusive" / "unsupported"

    # Fallback: keyword scan
    resp = response.lower()
    if label_n == 2:
        if "unsupported" in resp:
            return "unsupported"
        if "supported" in resp:
            return "supported"
        return "unsupported"
    else:
        if "contradicted" in resp:
            return "contradicted"
        if "inconclusive" in resp or "no enough" in resp:
            return "inconclusive"
        if "supported" in resp:
            return "supported"
        return "inconclusive"


def verify_claims(
    evidence: dict[str, list[dict]],
    client: OpenAI,
    model: str,
    label_n: int,
    prompt_initial: str,
    your_task_template: str,
) -> list[dict]:
    """
    Verify each claim against its search results.
    Returns list of {claim, search_results_str, verification_result}.
    Raises RuntimeError on LLM failure.
    """
    system = _VERIFICATION_SYSTEM_3 if label_n == 3 else _VERIFICATION_SYSTEM_2
    out: list[dict] = []

    for claim, snippets in evidence.items():
        search_res_str = _format_search_results(snippets)
        task_suffix = your_task_template.format(
            claim=claim,
            search_results=search_res_str.strip(),
        )
        prompt = f"{prompt_initial}\n\n{task_suffix}"

        try:
            response = llm_call(client, model, system, prompt, max_tokens=200)
        except Exception as e:
            raise RuntimeError(
                f"Verification LLM failed for claim ({claim[:60]!r}): {e}"
            ) from e

        label = _parse_verification_label(response, label_n)
        out.append({
            "claim":               claim,
            "search_results_str":  search_res_str,
            "verification_result": label,
        })

    return out


# ── Per-survey pipeline ───────────────────────────────────────────────────────

def process_survey(
    gen: dict,
    out_path: Path,
    cache_search: SearchCache,
    cfg: dict,
    nlp,
    client: OpenAI,
    prompt_initial: str,
    your_task_template: str,
) -> dict | None:
    survey_id = str(gen["id"])
    out_file  = out_path / f"{survey_id}.json"

    if cfg.get("resume") and out_file.exists():
        try:
            existing = json.loads(out_file.read_text())
            tqdm.write(
                f"  [SKIP] {survey_id} — "
                f"{existing['n_claims']} claims / "
                f"{existing.get('n_supported', '?')} supported already saved",
                file=sys.stderr,
            )
            return existing
        except Exception:
            tqdm.write(f"  [WARN] {survey_id} — corrupt cache, re-processing",
                       file=sys.stderr)

    if not gen.get("success", False):
        tqdm.write(f"  [SKIP] {survey_id} — generation not successful", file=sys.stderr)
        return None

    text = gen.get("text", "").strip()
    if not text:
        tqdm.write(f"  [SKIP] {survey_id} — empty text", file=sys.stderr)
        return None

    question = gen.get("query", "")
    tqdm.write(f"  [PROC] {survey_id} | {question[:70]}", file=sys.stderr)

    n_sentences = len([s for s in nlp(text).sents if s.text.strip()])

    # Stage 1: extract claims
    tqdm.write(f"         [1/3] extracting from {n_sentences} sentences ...",
               file=sys.stderr)
    claims = extract_claims(text, nlp, client, cfg["judge_model"])
    tqdm.write(f"               → {len(claims)} claims", file=sys.stderr)

    if not claims:
        result = _make_result(gen, survey_id, n_sentences, [], cfg)
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    # Stage 2: search evidence
    tqdm.write(f"         [2/3] searching evidence for {len(claims)} claims ...",
               file=sys.stderr)
    evidence = search_evidence(
        claims,
        serper_key=os.environ[cfg["serper_api_key_env"]],
        cache=cache_search,
        search_res_num=cfg.get("search_res_num", 5),
    )
    tqdm.write("               → done", file=sys.stderr)

    # Stage 3: verify claims
    tqdm.write("         [3/3] verifying ...", file=sys.stderr)
    verified = verify_claims(
        evidence,
        client,
        cfg["judge_model"],
        cfg.get("label_n", 3),
        prompt_initial,
        your_task_template,
    )

    result = _make_result(gen, survey_id, n_sentences, verified, cfg)
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    p = result["veriscore_precision"]
    tqdm.write(
        f"               → P={p:.3f} "
        f"({result['n_supported']}/{result['n_claims']} supported)",
        file=sys.stderr,
    )
    return result


def _make_result(
    gen: dict,
    survey_id: str,
    n_sentences: int,
    verified: list[dict],
    cfg: dict,
) -> dict:
    n_claims    = len(verified)
    n_supported = sum(1 for v in verified if v["verification_result"] == "supported")
    precision   = (n_supported / n_claims) if n_claims > 0 else None

    return {
        "survey_id":           survey_id,
        "dataset_id":          gen["dataset_id"],
        "model_id":            gen["model_id"],
        "query":               gen.get("query", ""),
        "n_sentences":         n_sentences,
        "n_claims":            n_claims,
        "n_supported":         n_supported,
        "veriscore_precision": round(precision, 4) if precision is not None else None,
        "claims": [
            {
                "claim_id":            i,
                "claim":               v["claim"],
                "verification_result": v["verification_result"],
                "search_results_str":  v["search_results_str"],
            }
            for i, v in enumerate(verified)
        ],
        "judge_model": cfg["judge_model"],
        "timestamp":           datetime.now(timezone.utc).isoformat(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VeriScore — factuality metric")
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. SurGE)")
    parser.add_argument("--model",   required=True, help="Model id (e.g. perplexity_dr)")
    args = parser.parse_args()

    cfg = load_config()

    client = make_client(cfg)

    nlp = spacy.load("en_core_web_sm")

    demo_path = REPO / "data" / "demos" / "few_shot_examples.jsonl"
    prompt_initial, your_task_template = _load_verification_prompt(
        cfg.get("label_n", 3), demo_path
    )

    gen_dir = ROOT / "results" / "generations" / f"{args.dataset}_{args.model}"
    if not gen_dir.exists():
        print(f"[ERROR] Generation dir not found: {gen_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = ROOT / "results" / "scores" / f"{args.dataset}_{args.model}_veriscore"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path("/tmp") / "veriscore" / f"{args.dataset}_{args.model}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_search = SearchCache(cache_dir / "search_cache.json")

    gen_files = sorted(gen_dir.glob("*.json"))
    gen_files = [f for f in gen_files if not re.search(r"_(raw|old)\.json$", f.name)]

    print(f"\n[veriscore] {args.dataset} / {args.model}")
    print(f"            {len(gen_files)} surveys → {out_dir}")
    print(f"            judge model:          {cfg['judge_model']}")
    print(f"            search results/claim: {cfg.get('search_res_num', 5)}")
    print(f"            label_n:              {cfg.get('label_n', 3)}")
    print(f"            cache:                {cache_dir}\n")

    n_ok, n_skip, n_err = 0, 0, 0
    total_claims    = 0
    total_supported = 0

    surveys_bar = tqdm(
        total=len(gen_files), desc="surveys", unit="survey",
        leave=True, dynamic_ncols=True,
    )

    for gf in gen_files:
        surveys_bar.set_postfix_str(gf.stem)
        try:
            gen = json.loads(gf.read_text())
        except Exception as e:
            tqdm.write(f"  [ERROR] reading {gf.name}: {e}", file=sys.stderr)
            n_err += 1
            surveys_bar.update(1)
            continue

        result = process_survey(
            gen, out_dir, cache_search, cfg, nlp,
            client, prompt_initial, your_task_template,
        )

        if result is None:
            n_skip += 1
        else:
            n_ok          += 1
            nc             = result["n_claims"]
            ns             = result.get("n_supported", 0)
            total_claims   += nc
            total_supported += ns
            p = result.get("veriscore_precision")
            surveys_bar.set_postfix_str(
                f"{gf.stem} → {nc} claims, P={p:.3f}" if p is not None
                else f"{gf.stem} → {nc} claims"
            )
        surveys_bar.update(1)

    surveys_bar.close()

    print(f"\n[veriscore] done — ok={n_ok} skip={n_skip} err={n_err}")
    print(f"            total claims:     {total_claims}")
    print(f"            total supported:  {total_supported}")
    if total_claims > 0:
        print(f"            macro precision:  {total_supported / total_claims:.3f}")


if __name__ == "__main__":
    main()
