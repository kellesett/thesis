#!/usr/bin/env python3
"""
metrics/factuality/validate.py
Validation of factuality LLM classifier against labelled test set.

Test file (in results/scores/):
  factuality_classes_test.json  — category A / B / C / D

Usage (inside Docker):
    python metrics/factuality/validate.py
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


# ── Classifier prompts (copied from main.py) ───────────────────────────────────

_CATEGORY_SYSTEM = (
    "You are classifying atomic claims from a scientific survey into one of four "
    "categories based on what type of information they convey."
)

_CATEGORY_PROMPT = """\
The claim to classify:
"{claim}"

Source paper context (if available):
"{source_context}"

Categories:
A — General topical claims. Statements that can be made based on only reading the abstract. They describe what a paper is about, what problem it addresses, or its general contribution without specific methodological or quantitative details.
B — Methodological refinements. Statements containing specific details about how methods work, requiring information typically found in the Methods section.
C — Quantitative claims. Statements containing specific numerical values, typically from the Results section.
D — Critical or comparative claims. Statements making evaluative judgments, comparing approaches, discussing limitations, or pointing to contradictions — typically from Discussion or Related Work sections.

Respond with a JSON object:
{{"category": "A" | "B" | "C" | "D", "reasoning": "brief explanation", "confidence": "high" | "medium" | "low"}}"""


def classify_claim(
    claim: str, client: OpenAI, model: str, max_retries: int,
    disable_reasoning: bool = False, provider: str | None = None,
    token_counter: TokenCounter | None = None,
    reasoning_effort: str | None = None,
) -> str | None:
    """Returns predicted category letter (A/B/C/D), or None on failure."""
    prompt = _CATEGORY_PROMPT.format(claim=claim[:600], source_context="Not available")
    extra_body: dict = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
    if reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort
    elif disable_reasoning:
        extra_body["reasoning"] = {"enabled": False}

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": _CATEGORY_SYSTEM},
                          {"role": "user",   "content": prompt}],
                temperature=0,
                extra_body=extra_body or None,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise RuntimeError(f"None content (finish_reason={resp.choices[0].finish_reason})")
            if token_counter is not None and resp.usage is not None:
                cost = (resp.usage.model_extra or {}).get("cost") or 0.0
                token_counter.add(resp.usage.prompt_tokens or 0,
                                  resp.usage.completion_tokens or 0, float(cost))
            raw = re.sub(r"^```(?:json)?\s*", "", content.strip())
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            cat = parsed.get("category", "").upper()
            if cat in ("A", "B", "C", "D"):
                return cat
            raise ValueError(f"Unexpected category: {cat!r}")
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print(f"[validate] LLM failed after {max_retries} retries: {type(e).__name__}: {e}", flush=True)
                sys.exit(1)
    return None


# ── Metrics ────────────────────────────────────────────────────────────────────

def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 3), round(r, 3), round(f1, 3)


def print_table(title: str, rows: list[tuple]) -> None:
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")
    print(f"  {'Category':<22} {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Sup':>5}")
    print(f"  {'─' * 54}")
    for label, p, r, f1, sup in rows:
        print(f"  {label:<22} {p:>6.3f}  {r:>6.3f}  {f1:>6.3f}  {sup:>5}")
    print(f"{'━' * 60}")


def print_confusion_matrix(matrix: dict, categories: list[str], cat_names: dict[str, str]) -> None:
    print(f"\n  Confusion matrix  (rows = true, cols = predicted)")
    header = "  " + " " * 18 + "".join(f"  {c:>2}" for c in categories)
    print(header)
    print("  " + "─" * (18 + 4 * len(categories)))
    for true_c in categories:
        row_label = f"{true_c} {cat_names[true_c]}"
        row = "  " + f"{row_label:<18}" + "".join(
            f"  {matrix[true_c].get(pred_c, 0):>2}" for pred_c in categories
        )
        print(row)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg    = load_config(CONFIG)
    client = make_client(cfg)

    test_file = SCORES_DIR / "factuality_classes_test.json"
    if not test_file.exists():
        print(f"[ERROR] {test_file} not found", flush=True)
        sys.exit(1)

    with open(test_file) as f:
        items = json.load(f)

    model             = cfg["judge_model"]
    max_retries       = cfg.get("max_retries", 3)
    provider          = cfg.get("judge_provider")
    disable_reasoning = not cfg.get("judge_reasoning", True)
    reasoning_effort  = cfg.get("judge_reasoning_effort") or None
    workers           = cfg.get("judge_workers", 8)

    print(f"\n[validate:factuality]  model={model}")
    print(f"  {len(items)} claims  workers={workers}\n")

    token_counter = TokenCounter()
    predictions: list[str | None] = [None] * len(items)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                classify_claim, item["claim"], client, model, max_retries,
                disable_reasoning, provider, token_counter, reasoning_effort,
            ): i
            for i, item in enumerate(items)
        }
        with tqdm(total=len(items), desc="  classifying", unit="claim") as bar:
            for future in as_completed(futures):
                i = futures[future]
                try:
                    predictions[i] = future.result()
                except Exception as e:
                    predictions[i] = None
                bar.update(1)
                bar.set_postfix_str(token_counter.fmt())

    categories = ["A", "B", "C", "D"]
    cat_names  = {
        "A": "General",
        "B": "Methodological",
        "C": "Quantitative",
        "D": "Critical/Comp",
    }

    # Precision/recall per category
    rows = []
    for cat in categories:
        tp = fp = fn = 0
        for item, pred in zip(items, predictions):
            true_c = item["category"]
            if true_c == cat and pred == cat:
                tp += 1
            elif true_c != cat and pred == cat:
                fp += 1
            elif true_c == cat and pred != cat:
                fn += 1
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        support = sum(1 for item in items if item["category"] == cat)
        rows.append((f"{cat} {cat_names[cat]}", p, r, f1, support))

    print_table(
        f"factuality_classes_test.json  ({len(items)} claims)  [{token_counter.fmt()}]",
        rows,
    )

    # Confusion matrix
    matrix: dict[str, dict[str, int]] = {c: defaultdict(int) for c in categories}
    for item, pred in zip(items, predictions):
        true_c = item["category"]
        if pred in categories:
            matrix[true_c][pred] += 1

    print_confusion_matrix(matrix, categories, cat_names)

    print("[validate:factuality]  done\n")


if __name__ == "__main__":
    main()
