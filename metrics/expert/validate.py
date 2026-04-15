#!/usr/bin/env python3
"""
metrics/expert/validate.py
Validation of expert LLM judge against labelled test sets.

Test files (in results/scores/):
  expert_classes_test.json   — multi-label: crit, comp_total, open
  expert_modalities_test.json — modality_level 1-5

Usage (inside Docker):
    python metrics/expert/validate.py
"""
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

ROOT   = Path(__file__).parent.parent.parent
CONFIG = Path(__file__).parent / "config.yaml"
sys.path.insert(0, str(ROOT))

from metrics.utils import make_client, load_config, TokenCounter

SCORES_DIR = ROOT / "results" / "scores"


# ── LLM call (same as main.py) ─────────────────────────────────────────────────

def llm_json(
    client: OpenAI, model: str, system: str, user: str, max_retries: int,
    provider: str | None = None, disable_reasoning: bool = False,
    token_counter: TokenCounter | None = None,
    reasoning_effort: str | None = None, max_tokens: int | None = None,
) -> dict:
    extra_body: dict = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
    if reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort
    elif disable_reasoning:
        extra_body["reasoning"] = {"enabled": False}

    for attempt in range(1, max_retries + 1):
        try:
            create_kwargs: dict = dict(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": user}],
                temperature=0,
                extra_body=extra_body or None,
            )
            if max_tokens is not None:
                create_kwargs["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**create_kwargs)
            content = resp.choices[0].message.content
            if content is None:
                raise RuntimeError(f"None content (finish_reason={resp.choices[0].finish_reason})")
            if token_counter is not None and resp.usage is not None:
                cost = (resp.usage.model_extra or {}).get("cost") or 0.0
                token_counter.add(resp.usage.prompt_tokens or 0,
                                  resp.usage.completion_tokens or 0, float(cost))
            raw = re.sub(r"^```(?:json)?\s*", "", content.strip())
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print(f"[validate] LLM failed after {max_retries} retries: {type(e).__name__}: {e}", flush=True)
                sys.exit(1)
    return {}


# ── Judge prompts (copied from main.py) ───────────────────────────────────────

_ALL_SYS = (
    "You are evaluating an atomic claim from a scientific survey along four independent dimensions. "
    "Analyse the claim carefully and answer all four parts in a single JSON object."
)

_ALL_USER = """\
The claim:
"{claim}"

Evaluate the claim on the four dimensions below and return a single JSON object with all keys.

## C.1 — Criticality
A claim is critical if it:
- Points out a limitation, weakness, or failure mode of a cited work
- Mentions a negative result or conditions under which a method fails
- Identifies a contradiction or tension between different works
- Questions the validity, generalizability, or scope of a claim
- Discusses trade-offs where one approach sacrifices something for another
A claim is NOT critical if it merely describes what a paper does, presents results neutrally, or states general facts.
Keys: "is_critical" (bool), "critical_type" ("limitation"|"negative_result"|"contradiction"|"trade_off"|"none"), "c1_reasoning" (brief)

## C.2 — Comparativeness
A claim is comparative if it explicitly compares two or more methods, models, approaches, or results using constructions like "outperforms", "in contrast to", "unlike", "better than", "whereas", or ranks/orders approaches.
A claim is NOT comparative if it only describes a single item or mentions multiple works without drawing contrasts.
Keys: "is_comparative" (bool), "compared_entities" (list of strings | null), "c2_reasoning" (brief)

## C.3 — Open question
A claim is an open question if it explicitly states something remains unknown/unresolved/disputed, points to gaps in understanding, identifies future research directions, or uses markers like "remains unclear", "open question", "further research is needed".
A claim is NOT an open question if it describes established results or uses hedging only for style.
Keys: "is_open_question" (bool), "question_type" ("generalization"|"scalability"|"mechanism"|"theoretical"|"empirical"|"none"), "c3_reasoning" (brief)

## C.4 — Epistemic modality
Classify the confidence level expressed by the claim's linguistic markers (not whether the claim is true):
1. Categorical — no hedging, uses "is", "does", "demonstrates"
2. Strong — mild qualifiers, uses "generally", "typically", "has been shown to"
3. Moderate — clear hedging, uses "often", "tends to", "can"
4. Weak — strong hedging, uses "may", "might", "could", "suggests", "appears to"
5. Explicit uncertainty — "remains unclear", "is debated", "open question"
Keys: "modality_level" (1–5), "hedging_markers" (list of strings), "c4_reasoning" (brief)

Respond with a single JSON object containing all keys listed above."""


def judge_claim(
    claim: str, client: OpenAI, model: str, max_retries: int,
    provider: str | None, disable_reasoning: bool,
    token_counter: TokenCounter,
    reasoning_effort: str | None, max_tokens: int | None,
) -> dict:
    return llm_json(
        client, model, _ALL_SYS,
        _ALL_USER.format(claim=claim[:600]),
        max_retries, provider, disable_reasoning,
        token_counter=token_counter,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
    )


# ── Metrics ────────────────────────────────────────────────────────────────────

def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 3), round(r, 3), round(f1, 3)


def print_table(title: str, rows: list[tuple]) -> None:
    """rows: (label, precision, recall, f1, support)"""
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")
    print(f"  {'Label':<22} {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Sup':>5}")
    print(f"  {'─' * 54}")
    for label, p, r, f1, sup in rows:
        print(f"  {label:<22} {p:>6.3f}  {r:>6.3f}  {f1:>6.3f}  {sup:>5}")
    print(f"{'━' * 60}")


def print_confusion_matrix(matrix: dict[int, dict[int, int]], levels: list[int], level_names: dict[int, str]) -> None:
    print(f"\n  Confusion matrix  (rows = true, cols = predicted)")
    header = "  " + " " * 16 + "".join(f"  {l:>2}" for l in levels)
    print(header)
    print("  " + "─" * (16 + 4 * len(levels)))
    for true_l in levels:
        row_label = f"{true_l} {level_names[true_l]}"
        row = "  " + f"{row_label:<16}" + "".join(f"  {matrix[true_l].get(pred_l, 0):>2}" for pred_l in levels)
        print(row)
    print()


# ── Validate classes ───────────────────────────────────────────────────────────

def validate_classes(cfg: dict, client: OpenAI) -> None:
    test_file = SCORES_DIR / "expert_classes_test.json"
    if not test_file.exists():
        print(f"[SKIP] {test_file} not found", flush=True)
        return

    with open(test_file) as f:
        items = json.load(f)

    model             = cfg["judge_model"]
    max_retries       = cfg.get("max_retries", 3)
    provider          = cfg.get("judge_provider")
    disable_reasoning = cfg.get("judge_disable_reasoning", False)
    reasoning_effort  = cfg.get("judge_reasoning_effort") or None
    max_tokens        = cfg.get("judge_max_tokens") or None
    workers           = cfg.get("judge_workers", 8)

    token_counter = TokenCounter()
    predictions: list[dict | None] = [None] * len(items)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                judge_claim, item["claim"], client, model, max_retries,
                provider, disable_reasoning, token_counter,
                reasoning_effort, max_tokens,
            ): i
            for i, item in enumerate(items)
        }
        with tqdm(total=len(items), desc="  classes", unit="claim") as bar:
            for future in as_completed(futures):
                i = futures[future]
                try:
                    predictions[i] = future.result()
                except Exception as e:
                    predictions[i] = {}
                bar.update(1)
                bar.set_postfix_str(token_counter.fmt())

    # Compute precision/recall for each binary label
    labels_map = {
        "crit":       "is_critical",
        "comp_total": "is_comparative",
        "open":       "is_open_question",
    }
    label_display = {
        "crit":       "C.1 Critical",
        "comp_total": "C.2 Comparative",
        "open":       "C.3 Open question",
    }

    rows = []
    for key, pred_key in labels_map.items():
        tp = fp = fn = 0
        for item, pred in zip(items, predictions):
            true_val = item["labels"][key]
            pred_val = (pred or {}).get(pred_key, False)
            if true_val and pred_val:
                tp += 1
            elif not true_val and pred_val:
                fp += 1
            elif true_val and not pred_val:
                fn += 1
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        support = sum(1 for item in items if item["labels"][key])
        rows.append((label_display[key], p, r, f1, support))

    print_table(
        f"expert_classes_test.json  ({len(items)} claims)  [{token_counter.fmt()}]",
        rows,
    )


# ── Validate modalities ────────────────────────────────────────────────────────

def validate_modalities(cfg: dict, client: OpenAI) -> None:
    test_file = SCORES_DIR / "expert_modalities_test.json"
    if not test_file.exists():
        print(f"[SKIP] {test_file} not found", flush=True)
        return

    with open(test_file) as f:
        items = json.load(f)

    model             = cfg["judge_model"]
    max_retries       = cfg.get("max_retries", 3)
    provider          = cfg.get("judge_provider")
    disable_reasoning = cfg.get("judge_disable_reasoning", False)
    reasoning_effort  = cfg.get("judge_reasoning_effort") or None
    max_tokens        = cfg.get("judge_max_tokens") or None
    workers           = cfg.get("judge_workers", 8)

    token_counter = TokenCounter()
    predictions: list[dict | None] = [None] * len(items)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                judge_claim, item["claim"], client, model, max_retries,
                provider, disable_reasoning, token_counter,
                reasoning_effort, max_tokens,
            ): i
            for i, item in enumerate(items)
        }
        with tqdm(total=len(items), desc="  modalities", unit="claim") as bar:
            for future in as_completed(futures):
                i = futures[future]
                try:
                    predictions[i] = future.result()
                except Exception as e:
                    predictions[i] = {}
                bar.update(1)
                bar.set_postfix_str(token_counter.fmt())

    levels = [1, 2, 3, 4, 5]
    level_names = {1: "Categorical", 2: "Strong", 3: "Moderate", 4: "Weak", 5: "Uncertain"}

    # Precision/recall per level
    rows = []
    for level in levels:
        tp = fp = fn = 0
        for item, pred in zip(items, predictions):
            true_l = item["modality_level"]
            pred_l = (pred or {}).get("modality_level", -1)
            if true_l == level and pred_l == level:
                tp += 1
            elif true_l != level and pred_l == level:
                fp += 1
            elif true_l == level and pred_l != level:
                fn += 1
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        support = sum(1 for item in items if item["modality_level"] == level)
        rows.append((f"{level} {level_names[level]}", p, r, f1, support))

    print_table(
        f"expert_modalities_test.json  ({len(items)} claims)  [{token_counter.fmt()}]",
        rows,
    )

    # Confusion matrix
    matrix: dict[int, dict[int, int]] = {l: defaultdict(int) for l in levels}
    for item, pred in zip(items, predictions):
        true_l = item["modality_level"]
        pred_l = (pred or {}).get("modality_level", -1)
        if pred_l in levels:
            matrix[true_l][pred_l] += 1

    print_confusion_matrix(matrix, levels, level_names)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg    = load_config(CONFIG)
    client = make_client(cfg)

    print(f"\n[validate:expert]  model={cfg['judge_model']}")

    validate_classes(cfg, client)
    validate_modalities(cfg, client)

    print("\n[validate:expert]  done\n")


if __name__ == "__main__":
    main()
