#!/usr/bin/env python3
"""Evaluate structural metrics on a gold-labeled synthetic dataset.

Input is the JSON array produced by the synthetic-survey prompt:
[
  {
    "id": "synthetic_01",
    "title": "...",
    "markdown_text": "...",
    "gold": {
      "contradictions": [{"s1_id": "...", "s2_id": "..."}],
      "repetitions": [{"s1_id": "...", "s2_id": "..."}],
      "hard_negatives": [{"s1_id": "...", "s2_id": "..."}]
    }
  }
]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import diskcache
from dotenv import load_dotenv
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).parent.parent.parent
CONFIG = Path(__file__).parent / "config.yaml"
sys.path.insert(0, str(ROOT))

from metrics.structural.contradiction.aggregate import compute_m_contr
from metrics.structural.main import (
    build_hyperparams_id,
    compute_m_rep,
    load_nli_model,
    load_specter,
)
from metrics.utils import load_config, make_client

load_dotenv()

HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
SENT_ID_RE = re.compile(r"\[([A-Za-z0-9_.:-]+\.S\d+)\]")
PARA_RE = re.compile(r"\n\s*\n+")


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def safe_div(num: int, den: int) -> float:
    return num / den if den else 0.0


def prf(pred: set[tuple[str, str]], gold: set[tuple[str, str]]) -> dict:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def prf_from_counts(tp: int, fp: int, fn: int) -> dict:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def extract_sent_id(sentence: str) -> str | None:
    match = SENT_ID_RE.search(sentence)
    return match.group(1) if match else None


def ids_to_pairs(records: Iterable[dict]) -> set[tuple[str, str]]:
    pairs = set()
    for rec in records:
        s1_id = rec.get("s1_id")
        s2_id = rec.get("s2_id")
        if s1_id and s2_id:
            pairs.add(pair_key(s1_id, s2_id))
    return pairs


def predicted_pairs(records: Iterable[dict], flag: str) -> set[tuple[str, str]]:
    pairs = set()
    for rec in records:
        if not rec.get(flag) or rec.get("status") == "failed":
            continue
        if rec.get("unit") == "paragraph":
            for pair in rec.get("sentence_pairs", []):
                s1_id = extract_sent_id(pair.get("statement_1", ""))
                s2_id = extract_sent_id(pair.get("statement_2", ""))
                if s1_id and s2_id:
                    pairs.add(pair_key(s1_id, s2_id))
            continue
        s1_id = extract_sent_id(rec.get("s1", ""))
        s2_id = extract_sent_id(rec.get("s2", ""))
        if s1_id and s2_id:
            pairs.add(pair_key(s1_id, s2_id))
    return pairs


def sentence_ids_from_record(rec: dict, prefix: str) -> list[str]:
    sent_key = "sentences1" if prefix == "s1" else "sentences2"
    ids = [sid for sent in rec.get(sent_key, []) for sid in [extract_sent_id(sent)] if sid]
    if ids:
        return ids
    sid = extract_sent_id(rec.get(prefix, ""))
    return [sid] if sid else []


def all_record_pairs(records: Iterable[dict]) -> set[tuple[str, str]]:
    pairs = set()
    for rec in records:
        s1_ids = sentence_ids_from_record(rec, "s1")
        s2_ids = sentence_ids_from_record(rec, "s2")
        for s1_id in s1_ids:
            for s2_id in s2_ids:
                pairs.add(pair_key(s1_id, s2_id))
    return pairs


def split_synthetic_sentences(text: str) -> list[str]:
    """Split by synthetic sentence ids, preserving the ids for gold matching."""
    matches = list(SENT_ID_RE.finditer(text))
    sentences = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sent = re.sub(r"\s+", " ", text[start:end]).strip()
        if sent:
            sentences.append(sent)
    return sentences


def split_synthetic_paragraphs(text: str) -> list[dict]:
    paragraphs = []
    for block in PARA_RE.split(text):
        sentences = split_synthetic_sentences(block)
        if sentences:
            paragraphs.append({"text": " ".join(sentences), "sentences": sentences})
    return paragraphs


def split_synthetic_sections(markdown_text: str) -> list[dict]:
    headings = list(HEADING_RE.finditer(markdown_text))
    sections = []
    for idx, match in enumerate(headings):
        title = match.group(1).strip()
        start = match.end()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(markdown_text)
        paragraphs = split_synthetic_paragraphs(markdown_text[start:end])
        sentences = [sent for par in paragraphs for sent in par["sentences"]]
        if sentences:
            sections.append({"title": title, "sentences": sentences, "paragraphs": paragraphs})
    if not sections:
        paragraphs = split_synthetic_paragraphs(markdown_text)
        sentences = [sent for par in paragraphs for sent in par["sentences"]]
        sections = [{"title": "Full text", "sentences": sentences, "paragraphs": paragraphs}]
    return sections


def read_json(path: Path, default=None) -> list | dict:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(path: Path) -> list[dict]:
    data = read_json(path)
    if isinstance(data, dict):
        data = data.get("items") or data.get("surveys") or data.get("data")
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array or object with items/surveys/data: {path}")
    return data


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def validate_gold(item: dict, sections: list[dict]) -> None:
    all_ids = {
        sent_id
        for sec in sections
        for sent in sec["sentences"]
        for sent_id in [extract_sent_id(sent)]
        if sent_id
    }
    if not all_ids:
        raise ValueError(f"{item.get('id')}: no synthetic sentence ids found")

    gold = item.get("gold", {})
    for group in ("contradictions", "repetitions", "hard_negatives"):
        for rec in gold.get(group, []):
            missing = [sid for sid in (rec.get("s1_id"), rec.get("s2_id")) if sid not in all_ids]
            if missing:
                raise ValueError(f"{item.get('id')}: unknown ids in gold.{group}: {missing}")


def recall_at_stage(gold: set[tuple[str, str]], stage_pairs: set[tuple[str, str]]) -> float:
    return round(safe_div(len(gold & stage_pairs), len(gold)), 4)


def evaluate_item(
    item: dict,
    cfg: dict,
    client,
    specter_model,
    nli_pipe,
    cache,
    stage_base: Path,
    out_dir: Path,
    hparams_id: str,
    specter_bar=None,
    topic_bar=None,
    contr_bar=None,
    rep_bar=None,
    nli_bar=None,
) -> dict:
    survey_id = item["id"]
    markdown_text = item.get("markdown_text", "").strip()
    if not markdown_text:
        raise ValueError(f"{survey_id}: empty markdown_text")

    t0 = time.time()
    sections = split_synthetic_sections(markdown_text)
    validate_gold(item, sections)

    run_m_contr = cfg.get("run_m_contr", True)
    run_m_rep = cfg.get("run_m_rep", True)

    gold = item.get("gold", {})
    gold_contr = ids_to_pairs(gold.get("contradictions", []))
    gold_rep = ids_to_pairs(gold.get("repetitions", []))
    hard_neg = ids_to_pairs(gold.get("hard_negatives", []))

    result = {
        "survey_id": survey_id,
        "title": item.get("title", ""),
        "topic": item.get("topic", ""),
        "n_sections": len(sections),
        "n_sentences": sum(len(sec["sentences"]) for sec in sections),
        "run_m_contr": run_m_contr,
        "run_m_rep": run_m_rep,
        "gold_contradictions": len(gold_contr),
        "gold_repetitions": len(gold_rep),
    }

    if run_m_contr:
        contr_stage_dir = stage_base / "contradiction" / hparams_id / survey_id
        contr = compute_m_contr(
            survey_id,
            sections,
            specter_model,
            client,
            cfg,
            cache,
            contr_stage_dir,
            specter_bar=specter_bar,
            topic_bar=topic_bar,
            contr_bar=contr_bar,
        )

        contr_candidates = read_json(contr_stage_dir / "candidates.json")
        topic_filtered = read_json(contr_stage_dir / "topic_filtered.json", default=[])
        checked_contr = read_json(contr_stage_dir / "contradictions.json", default=[])
        contr_stage1_pairs = all_record_pairs(contr_candidates)
        topic_pairs = all_record_pairs(
            rec
            for rec in topic_filtered
            if rec.get("same_subject") and rec.get("status") != "failed"
        )
        checked_contr_pairs = all_record_pairs(
            rec for rec in checked_contr if rec.get("status") != "failed"
        )
        pred_contr = predicted_pairs(checked_contr, "is_contradiction")
        contr_prf = prf(pred_contr, gold_contr)
        result.update({
            "contradiction_unit": contr.get("contradiction_unit", cfg.get("contradiction_unit", "sentence")),
            "pred_contradictions": len(pred_contr),
            "contr_tp": contr_prf["tp"],
            "contr_fp": contr_prf["fp"],
            "contr_fn": contr_prf["fn"],
            "contr_precision": contr_prf["precision"],
            "contr_recall": contr_prf["recall"],
            "contr_f1": contr_prf["f1"],
            "contr_stage1_recall": recall_at_stage(gold_contr, contr_stage1_pairs),
            "contr_topic_recall": recall_at_stage(gold_contr, topic_pairs),
            "contr_checked_recall": recall_at_stage(gold_contr, checked_contr_pairs),
            "contr_hard_negative_fp": len(pred_contr & hard_neg),
            "contr_n_candidates_stage1": contr.get("n_candidates_stage1", 0),
            "contr_n_after_topic_filter": contr.get("n_after_topic_filter", 0),
            "contr_n_checked": len(checked_contr),
        })

    if run_m_rep:
        rep_stage_dir = stage_base / "repetition" / hparams_id / survey_id
        rep = compute_m_rep(
            sections,
            specter_model,
            nli_pipe,
            cfg,
            rep_stage_dir,
            rep_bar=rep_bar,
            nli_bar=nli_bar,
        )

        rep_candidates = read_json(rep_stage_dir / "candidates.json")
        checked_rep = read_json(rep_stage_dir / "duplicates.json")
        rep_stage1_pairs = all_record_pairs(rep_candidates)
        checked_rep_pairs = all_record_pairs(checked_rep)
        pred_rep = predicted_pairs(checked_rep, "is_duplicate")
        rep_prf = prf(pred_rep, gold_rep)
        result.update({
            "pred_repetitions": len(pred_rep),
            "rep_tp": rep_prf["tp"],
            "rep_fp": rep_prf["fp"],
            "rep_fn": rep_prf["fn"],
            "rep_precision": rep_prf["precision"],
            "rep_recall": rep_prf["recall"],
            "rep_f1": rep_prf["f1"],
            "rep_stage1_recall": recall_at_stage(gold_rep, rep_stage1_pairs),
            "rep_checked_recall": recall_at_stage(gold_rep, checked_rep_pairs),
            "rep_n_candidates": rep.get("n_candidates", 0),
            "rep_n_checked": len(checked_rep),
        })

    result["latency_sec"] = round(time.time() - t0, 1)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    out_file = out_dir / f"{survey_id}.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def aggregate(results: list[dict]) -> dict:
    run_m_contr = any(result.get("run_m_contr", True) for result in results)
    run_m_rep = any(result.get("run_m_rep", True) for result in results)
    summary = {
        "n_surveys": len(results),
        "run_m_contr": run_m_contr,
        "run_m_rep": run_m_rep,
        "latency_sec": round(sum(float(r["latency_sec"]) for r in results), 1),
    }
    if run_m_contr:
        contr = prf_from_counts(
            sum(int(r.get("contr_tp", 0)) for r in results),
            sum(int(r.get("contr_fp", 0)) for r in results),
            sum(int(r.get("contr_fn", 0)) for r in results),
        )
        summary.update({
            "contr_precision": contr["precision"],
            "contr_recall": contr["recall"],
            "contr_f1": contr["f1"],
            "contr_hard_negative_fp": sum(int(r.get("contr_hard_negative_fp", 0)) for r in results),
            "contr_n_candidates_stage1": sum(int(r.get("contr_n_candidates_stage1", 0)) for r in results),
            "contr_n_after_topic_filter": sum(int(r.get("contr_n_after_topic_filter", 0)) for r in results),
            "contr_n_checked": sum(int(r.get("contr_n_checked", 0)) for r in results),
        })
    if run_m_rep:
        rep = prf_from_counts(
            sum(int(r.get("rep_tp", 0)) for r in results),
            sum(int(r.get("rep_fp", 0)) for r in results),
            sum(int(r.get("rep_fn", 0)) for r in results),
        )
        summary.update({
            "rep_precision": rep["precision"],
            "rep_recall": rep["recall"],
            "rep_f1": rep["f1"],
            "rep_n_candidates": sum(int(r.get("rep_n_candidates", 0)) for r in results),
            "rep_n_checked": sum(int(r.get("rep_n_checked", 0)) for r in results),
        })
    return summary


def write_csv(results: list[dict], out_dir: Path) -> None:
    if not results:
        return
    fields = list(results[0].keys())
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate structural metrics on synthetic gold data")
    parser.add_argument("--data", required=True, type=Path, help="Synthetic JSON array path")
    parser.add_argument("--config", default=CONFIG, type=Path)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--stage-dir", default=ROOT / "tmp" / "structural" / "synthetic_eval", type=Path)
    parser.add_argument("--cache-dir", default=ROOT / "tmp" / "structural" / "synthetic_cache", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="Process first N synthetic surveys")
    parser.add_argument("--contradiction-unit", choices=("sentence", "paragraph"), default=None)
    parser.add_argument("--top-k-per-sentence", type=int, default=None)
    parser.add_argument("--min-similarity-threshold", type=float, default=None)
    parser.add_argument("--paragraph-top-k", type=int, default=None)
    parser.add_argument("--paragraph-min-similarity-threshold", type=float, default=None)
    parser.add_argument("--rep-embedding-prefilter", type=float, default=None)
    parser.add_argument("--only-m-contr", action="store_true",
                        help="Evaluate only M_contr; skip M_rep and NLI model loading.")
    parser.add_argument("--only-m-rep", action="store_true",
                        help="Evaluate only M_rep; skip M_contr and LLM judge calls.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.only_m_contr and args.only_m_rep:
        parser.error("--only-m-contr and --only-m-rep are mutually exclusive")
    if args.only_m_contr:
        cfg["run_m_contr"] = True
        cfg["run_m_rep"] = False
    if args.only_m_rep:
        cfg["run_m_contr"] = False
        cfg["run_m_rep"] = True
    if args.contradiction_unit is not None:
        cfg["contradiction_unit"] = args.contradiction_unit
    if args.top_k_per_sentence is not None:
        cfg["top_k_per_sentence"] = args.top_k_per_sentence or None
    if args.min_similarity_threshold is not None:
        cfg["min_similarity_threshold"] = args.min_similarity_threshold
    if args.paragraph_top_k is not None:
        cfg["paragraph_top_k"] = args.paragraph_top_k or None
    if args.paragraph_min_similarity_threshold is not None:
        cfg["paragraph_min_similarity_threshold"] = args.paragraph_min_similarity_threshold
    if args.rep_embedding_prefilter is not None:
        cfg["rep_embedding_prefilter"] = args.rep_embedding_prefilter
    run_m_contr = cfg.get("run_m_contr", True)
    run_m_rep = cfg.get("run_m_rep", True)
    if not run_m_contr and not run_m_rep:
        parser.error("At least one of run_m_contr/run_m_rep must be enabled")

    items = load_dataset(args.data)
    if args.limit is not None:
        items = items[:args.limit]

    run_id = f"{cfg['judge_id']}_{cfg['judge_comment']}"
    data_id = file_hash(args.data)
    hparams_id = f"{build_hyperparams_id(cfg)}_data{data_id}"
    run_mode = "all" if run_m_contr and run_m_rep else "contr" if run_m_contr else "rep"
    run_suffix = "" if run_mode == "all" else f"_{run_mode}"
    out_dir = args.out or (ROOT / "results" / "scores" / f"structural_synthetic_{run_id}_{hparams_id}{run_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n[structural synthetic] Loading models...")
    client = make_client(cfg) if run_m_contr else None
    specter_model = load_specter(cfg["specter_model_path"])
    nli_pipe = load_nli_model(cfg["nli_model_path"]) if run_m_rep else None
    cache = diskcache.Cache(str(args.cache_dir))

    print(f"  data       : {args.data}")
    print(f"  surveys    : {len(items)}")
    print(f"  data_hash  : {data_id}")
    print(f"  hparams_id : {hparams_id}")
    print(f"  mode       : M_contr={'on' if run_m_contr else 'off'}  M_rep={'on' if run_m_rep else 'off'}")
    print(f"  stage_dir  : {args.stage_dir}")
    print(f"  cache_dir  : {args.cache_dir}")
    print(f"  out        : {out_dir}\n")

    survey_bar = tqdm(total=len(items), desc="Synthetic       ", position=0, leave=True, unit="survey")
    specter_bar = tqdm(total=1, desc="  1/5 SPECTER   ", position=1, leave=True, unit="pair") if run_m_contr else None
    topic_bar = tqdm(total=1, desc="  2/5 topic flt ", position=2, leave=True, unit="pair") if run_m_contr else None
    contr_bar = tqdm(total=1, desc="  3/5 contr chk ", position=3, leave=True, unit="pair") if run_m_contr else None
    rep_bar = tqdm(total=1, desc="  4/5 rep SPECTER", position=4, leave=True, unit="pair") if run_m_rep else None
    nli_bar = tqdm(total=1, desc="  5/5 NLI       ", position=5, leave=True, unit="pair") if run_m_rep else None
    inner_bars = [bar for bar in [specter_bar, topic_bar, contr_bar, rep_bar, nli_bar] if bar is not None]

    results = []
    try:
        for item in items:
            survey_bar.set_postfix_str(item.get("title", item["id"])[:55])
            result = evaluate_item(
                item,
                cfg,
                client,
                specter_model,
                nli_pipe,
                cache,
                args.stage_dir,
                out_dir,
                hparams_id,
                specter_bar=specter_bar,
                topic_bar=topic_bar,
                contr_bar=contr_bar,
                rep_bar=rep_bar,
                nli_bar=nli_bar,
            )
            results.append(result)
            survey_bar.update(1)
            parts = []
            if result.get("run_m_contr"):
                parts.append(
                    f"contr P/R/F1={result['contr_precision']:.4f}/"
                    f"{result['contr_recall']:.4f}/{result['contr_f1']:.4f} "
                    f"hardFP={result['contr_hard_negative_fp']}"
                )
            if result.get("run_m_rep"):
                parts.append(
                    f"rep P/R/F1={result['rep_precision']:.4f}/"
                    f"{result['rep_recall']:.4f}/{result['rep_f1']:.4f}"
                )
            tqdm.write(f"{result['survey_id']}: {' '.join(parts)}")

            for bar in inner_bars:
                bar.reset(total=1)
                bar.set_postfix_str("")
    finally:
        for bar in [nli_bar, rep_bar, contr_bar, topic_bar, specter_bar, survey_bar]:
            if bar is not None:
                bar.close()

    write_csv(results, out_dir)
    summary = aggregate(results)
    (out_dir / "aggregate.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[aggregate]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nScores -> {out_dir}")


if __name__ == "__main__":
    main()
