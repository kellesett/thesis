#!/usr/bin/env python3
"""
metrics/structural/main.py
Structural quality metrics for generated surveys (group A).

Computes three metrics per survey:

  M_contr  — cross-section contradiction rate (A.1)
             Two-stage: NLI entity-filtered candidates → LLM judge validation.
             Lower is better (0 = no contradictions).

  M_term   — terminological inconsistency rate (A.2, exploratory)
             NER → SPECTER embedding → HDBSCAN clusters → LLM judge.
             Lower is better.

  M_rep    — cross-section repetition rate (A.3)
             SPECTER pre-filter → bi-directional NLI entailment.
             Lower is better.

Output per survey: results/scores/<dataset_id>_<model_id>_structural/<survey_id>.json
Summary:           results/scores/<dataset_id>_<model_id>_structural/summary.csv

Usage (inside Docker):
    python metrics/structural/main.py --dataset SurGE --model perplexity_dr
"""
import argparse
import csv
import itertools
import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

ROOT   = Path(__file__).parent.parent.parent
CONFIG = Path(__file__).parent / "config.yaml"
sys.path.insert(0, str(ROOT))

from metrics.utils import make_client, load_config


# ── Model loading ─────────────────────────────────────────────────────────────

def load_ner_model(path: str):
    """Load ITER SciERC NER model as HuggingFace token-classification pipeline."""
    from transformers import pipeline
    print(f"  Loading NER: {path}")
    return pipeline(
        "token-classification",
        model=path,
        aggregation_strategy="max",
        device=-1,  # CPU
    )


def load_nli_model(path: str):
    """Load DeBERTa-v2-xlarge-mnli as a text-classification pipeline."""
    from transformers import pipeline
    print(f"  Loading NLI: {path}")
    return pipeline(
        "text-classification",
        model=path,
        device=-1,
        top_k=None,
    )


def load_specter(path: str):
    """Load SPECTER sentence-transformers model for embeddings."""
    from sentence_transformers import SentenceTransformer
    print(f"  Loading SPECTER: {path}")
    return SentenceTransformer(path, device="cpu")


# ── Text utilities ────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
_SENT_RE    = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_to_sentences(text: str) -> list[str]:
    """Naive sentence splitter — good enough for scientific English."""
    # First strip citation markers
    text = re.sub(r"\[[^\]]{1,40}\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENT_RE.split(text) if len(s.strip()) > 30]


def split_sections(text: str) -> list[dict]:
    """Return list of {"title": str, "sentences": list[str]}."""
    headings = list(_HEADING_RE.finditer(text))
    sections = []
    for i, m in enumerate(headings):
        title = m.group(1).strip()
        start = m.end()
        end   = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body  = text[start:end].strip()
        sents = split_to_sentences(body)
        if sents:
            sections.append({"title": title, "sentences": sents})
    if not sections:
        sents = split_to_sentences(text)
        sections = [{"title": "Full text", "sentences": sents}]
    return sections


# ── NLI helpers ───────────────────────────────────────────────────────────────

def nli_scores(nli_pipe, premise: str, hypothesis: str) -> dict:
    """
    Run NLI and return dict of label→score.
    DeBERTa-xlarge-mnli labels: ENTAILMENT, NEUTRAL, CONTRADICTION.
    """
    result = nli_pipe(f"{premise} [SEP] {hypothesis}", truncation=True, max_length=512)
    # result is a list of [{"label": ..., "score": ...}]
    if isinstance(result[0], list):
        result = result[0]
    return {r["label"].upper(): r["score"] for r in result}


# ── A.1 — Contradiction detection ────────────────────────────────────────────

_CONTR_PROMPT = """\
You are evaluating whether two statements from a scientific survey contradict each other.

Statement 1 (from section "{section_i}"):
"{statement_1}"

Statement 2 (from section "{section_j}"):
"{statement_2}"

These statements were flagged as potentially contradictory because they mention the same entities: {shared_entities}.

A contradiction exists when the two statements make incompatible claims about the same entity, method, result, or phenomenon. Note carefully:
- Different formulations of the same fact are NOT contradictions
- Statements about different aspects of the same entity are NOT contradictions
- Statements using different terminology but compatible meanings are NOT contradictions
- Temporal evolution ("earlier work said X, later work showed Y") is NOT a contradiction within the survey's own voice
- A genuine contradiction requires that both statements cannot simultaneously be true

Respond with a JSON object:
{{"is_contradiction": true | false, "reasoning": "brief explanation", "contradiction_type": "factual" | "methodological" | "quantitative" | "none"}}"""


def extract_entities(ner_pipe, sentences: list[str]) -> list[set[str]]:
    """Return per-sentence sets of entity surface forms (lowercased)."""
    result = []
    for sent in sentences:
        try:
            ents = ner_pipe(sent[:512])
            words = {e["word"].lower().strip("##") for e in ents if e.get("word")}
            result.append(words)
        except Exception:
            result.append(set())
    return result


def compute_m_contr(
    sections: list[dict],
    ner_pipe,
    nli_pipe,
    client: OpenAI,
    cfg: dict,
) -> dict:
    """Compute cross-section contradiction rate (A.1).

    Two-stage pipeline: NLI entity-filtered candidate pairs → LLM judge validation.

    Args:
        sections: List of sections with sentences.
        ner_pipe: NER pipeline for entity extraction.
        nli_pipe: NLI pipeline for entailment checking.
        client: OpenAI client for LLM judge.
        cfg: Config with thresholds and model name.

    Returns:
        Dict with m_contr (rate), n_candidates, n_confirmed.
    """
    nli_threshold  = cfg.get("contr_nli_threshold", 0.5)
    max_candidates = cfg.get("contr_max_candidates", 200)
    model          = cfg["judge_model"]

    # Flatten (sentence, section_title, entity_set) for all sections
    flat: list[tuple[str, str, set]] = []
    for sec in sections:
        ent_sets = extract_entities(ner_pipe, sec["sentences"])
        for sent, ents in zip(sec["sentences"], ent_sets):
            flat.append((sent, sec["title"], ents))

    # Cross-section pairs with shared entities
    candidates = []
    for i, (s1, t1, e1) in enumerate(flat):
        for j, (s2, t2, e2) in enumerate(flat):
            if j <= i:
                continue
            if t1 == t2:
                continue  # same section
            shared = e1 & e2
            if not shared:
                continue
            scores = nli_scores(nli_pipe, s1, s2)
            if scores.get("CONTRADICTION", 0) >= nli_threshold:
                candidates.append({
                    "s1": s1, "t1": t1,
                    "s2": s2, "t2": t2,
                    "shared_entities": list(shared)[:5],
                    "nli_contradiction": scores.get("CONTRADICTION", 0),
                })

    # Cap for cost control
    if len(candidates) > max_candidates:
        candidates = sorted(
            candidates, key=lambda x: x["nli_contradiction"], reverse=True
        )[:max_candidates]

    if not candidates:
        return {"m_contr": 0.0, "n_candidates": 0, "n_confirmed": 0}

    # LLM judge validation
    n_confirmed = 0
    for cand in candidates:
        prompt = _CONTR_PROMPT.format(
            section_i=cand["t1"],
            statement_1=cand["s1"][:400],
            section_j=cand["t2"],
            statement_2=cand["s2"][:400],
            shared_entities=", ".join(cand["shared_entities"]),
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = resp.choices[0].message.content
            if content is None:
                print(f"[ERROR] OpenRouter None content (finish_reason={resp.choices[0].finish_reason}).\n"
                      f"Full response: {resp}", file=sys.stderr)
                raise RuntimeError(
                    f"OpenRouter returned None content "
                    f"(finish_reason={resp.choices[0].finish_reason})."
                )
            raw = content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            if parsed.get("is_contradiction"):
                n_confirmed += 1
                cand["llm_confirmed"] = True
                cand["contradiction_type"] = parsed.get("contradiction_type", "unknown")
        except Exception as e:
            cand["llm_error"] = str(e)

    m_contr = n_confirmed / len(candidates) if candidates else 0.0
    return {
        "m_contr":      round(m_contr, 4),
        "n_candidates": len(candidates),
        "n_confirmed":  n_confirmed,
    }


# ── A.2 — Terminological inconsistency ───────────────────────────────────────

_TERM_PROMPT = """\
You are evaluating terminological consistency in a scientific survey.

The following surface forms were clustered together as potentially referring to the same concept:
{list_of_forms}

Context snippets where each form appears:
{context_snippets}

Determine whether these forms represent:
1. Legitimate stylistic variation — all forms refer to exactly the same concept, and using different forms is normal scientific writing practice
2. Genuine terminological inconsistency — the forms refer to subtly different concepts, or their interchangeable use creates ambiguity

Important:
- Abbreviations and their full forms are NOT inconsistency
- Different levels of specificity may or may not be inconsistent depending on context
- Only flag as inconsistency when the variation actually causes confusion

Respond with a JSON object:
{{"is_inconsistent": true | false, "reasoning": "brief explanation", "severity": "none" | "minor" | "major"}}"""


def compute_m_term(
    sections: list[dict],
    ner_pipe,
    specter_model,
    client: OpenAI,
    cfg: dict,
) -> dict:
    """Compute terminological inconsistency rate (A.2, exploratory).

    NER → SPECTER embedding → HDBSCAN clustering → LLM judge for inconsistency.

    Args:
        sections: List of sections with sentences.
        ner_pipe: NER pipeline for entity extraction.
        specter_model: SPECTER sentence encoder for embeddings.
        client: OpenAI client for LLM judge.
        cfg: Config with similarity threshold and cluster settings.

    Returns:
        Dict with m_term (rate), n_clusters, n_inconsistent, exploratory flag.
    """
    sim_threshold   = cfg.get("term_similarity_threshold", 0.85)
    min_cluster     = cfg.get("term_min_cluster_size", 2)
    model           = cfg["judge_model"]

    from sklearn.metrics.pairwise import cosine_similarity
    try:
        import hdbscan
    except ImportError:
        return {"m_term": None, "error": "hdbscan not installed", "exploratory": True}

    # Collect all (entity_text, sentence, section) triples
    records: list[dict] = []
    for sec in sections:
        for sent in sec["sentences"]:
            try:
                ents = ner_pipe(sent[:512])
                for e in ents:
                    word = e.get("word", "").strip("##").strip()
                    if len(word) > 2:
                        records.append({
                            "term": word,
                            "sentence": sent,
                            "section": sec["title"],
                        })
            except Exception:
                pass

    if len(records) < min_cluster * 2:
        return {"m_term": 0.0, "n_clusters": 0, "n_inconsistent": 0, "exploratory": True}

    terms = [r["term"] for r in records]
    embeddings = specter_model.encode(terms, batch_size=64, show_progress_bar=False)

    # Agglomerative clustering by cosine similarity
    dist_matrix = 1.0 - cosine_similarity(embeddings)
    dist_matrix = np.clip(dist_matrix, 0.0, 2.0).astype(np.float64)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster,
        metric="precomputed",
    )
    labels = clusterer.fit_predict(dist_matrix)

    # Group records by cluster
    clusters: dict[int, list[dict]] = {}
    for rec, label in zip(records, labels):
        if label == -1:
            continue  # noise
        clusters.setdefault(label, []).append(rec)

    # Only clusters with multiple distinct surface forms
    multi_form_clusters = {
        k: v for k, v in clusters.items()
        if len({r["term"].lower() for r in v}) > 1
    }

    if not multi_form_clusters:
        return {"m_term": 0.0, "n_clusters": 0, "n_inconsistent": 0, "exploratory": True}

    n_inconsistent = 0
    for cluster_recs in multi_form_clusters.values():
        forms = list({r["term"] for r in cluster_recs})[:8]
        snippets = "\n".join(
            f'- "{r["term"]}" in [{r["section"]}]: "{r["sentence"][:200]}"'
            for r in cluster_recs[:6]
        )
        prompt = _TERM_PROMPT.format(
            list_of_forms=", ".join(f'"{f}"' for f in forms),
            context_snippets=snippets,
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = resp.choices[0].message.content
            if content is None:
                print(f"[ERROR] OpenRouter None content (finish_reason={resp.choices[0].finish_reason}).\n"
                      f"Full response: {resp}", file=sys.stderr)
                raise RuntimeError(
                    f"OpenRouter returned None content "
                    f"(finish_reason={resp.choices[0].finish_reason})."
                )
            raw = content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            if parsed.get("is_inconsistent"):
                n_inconsistent += 1
        except Exception:
            pass

    n_clusters = len(multi_form_clusters)
    m_term = n_inconsistent / n_clusters if n_clusters else 0.0
    return {
        "m_term":        round(m_term, 4),
        "n_clusters":    n_clusters,
        "n_inconsistent": n_inconsistent,
        "exploratory":   True,
    }


# ── A.3 — Repetition detection ────────────────────────────────────────────────

def compute_m_rep(
    sections: list[dict],
    specter_model,
    nli_pipe,
    cfg: dict,
) -> dict:
    """Compute cross-section repetition rate (A.3).

    SPECTER embedding pre-filter → bi-directional NLI entailment check.

    Args:
        sections: List of sections with sentences.
        specter_model: SPECTER sentence encoder for pre-filtering.
        nli_pipe: NLI pipeline for entailment verification.
        cfg: Config with similarity and entailment thresholds.

    Returns:
        Dict with m_rep (rate), n_total_sentences, n_candidates, n_duplicates.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    emb_threshold = cfg.get("rep_embedding_prefilter", 0.7)
    nli_threshold = cfg.get("rep_nli_threshold", 0.5)

    # Flatten all sentences with section labels
    all_sents: list[tuple[str, str]] = []
    for sec in sections:
        for sent in sec["sentences"]:
            all_sents.append((sent, sec["title"]))

    n_total = len(all_sents)
    if n_total < 2:
        return {"m_rep": 0.0, "n_total_sentences": n_total, "n_duplicates": 0}

    texts = [s for s, _ in all_sents]
    titles = [t for _, t in all_sents]

    # Encode all sentences
    embs = specter_model.encode(texts, batch_size=64, show_progress_bar=False)
    sims = cos_sim(embs, embs)

    # Find pre-filter candidates (different sections, high similarity)
    candidates = []
    for i in range(n_total):
        for j in range(i + 1, n_total):
            if titles[i] == titles[j]:
                continue  # same section
            if sims[i, j] >= emb_threshold:
                candidates.append((i, j))

    # Bi-directional NLI check
    duplicate_indices: set[int] = set()
    for i, j in candidates:
        try:
            fwd = nli_scores(nli_pipe, texts[i], texts[j])
            bwd = nli_scores(nli_pipe, texts[j], texts[i])
            if (fwd.get("ENTAILMENT", 0) >= nli_threshold and
                    bwd.get("ENTAILMENT", 0) >= nli_threshold):
                duplicate_indices.add(i)
                duplicate_indices.add(j)
        except Exception:
            pass

    m_rep = len(duplicate_indices) / n_total if n_total else 0.0
    return {
        "m_rep":              round(m_rep, 4),
        "n_total_sentences":  n_total,
        "n_candidates":       len(candidates),
        "n_duplicates":       len(duplicate_indices),
    }


# ── Per-survey processing ─────────────────────────────────────────────────────

def process_survey(
    gen: dict,
    out_path: Path,
    cfg: dict,
    ner_pipe,
    nli_pipe,
    specter_model,
    client: OpenAI,
) -> dict | None:
    survey_id = gen["id"]
    out_file  = out_path / f"{survey_id}.json"

    if cfg.get("resume") and out_file.exists():
        try:
            with open(out_file) as f:
                existing = json.load(f)
            print(f"  [SKIP] {survey_id} — already scored")
            return existing
        except Exception:
            print(f"  [WARN] {survey_id} — corrupt cache, re-processing")

    if not gen.get("success", False):
        print(f"  [SKIP] {survey_id} — generation not successful")
        return None

    text = gen.get("text", "").strip()
    if not text:
        print(f"  [SKIP] {survey_id} — empty text")
        return None

    print(f"  [PROC] {survey_id} | {gen.get('query', '')[:60]}")
    t0 = time.time()

    sections = split_sections(text)
    n_sents  = sum(len(s["sentences"]) for s in sections)
    print(f"         {len(sections)} sections, {n_sents} sentences")

    try:
        contr = compute_m_contr(sections, ner_pipe, nli_pipe, client, cfg)
    except Exception as e:
        contr = {"m_contr": None, "error": str(e)}
        logger.exception(f"Error computing m_contr for {survey_id}")

    try:
        term = compute_m_term(sections, ner_pipe, specter_model, client, cfg)
    except Exception as e:
        term = {"m_term": None, "error": str(e), "exploratory": True}
        logger.exception(f"Error computing m_term for {survey_id}")

    try:
        rep = compute_m_rep(sections, specter_model, nli_pipe, cfg)
    except Exception as e:
        rep = {"m_rep": None, "error": str(e)}
        logger.exception(f"Error computing m_rep for {survey_id}")

    result = {
        "survey_id":  survey_id,
        "dataset_id": gen["dataset_id"],
        "model_id":   gen["model_id"],
        "query":      gen.get("query", ""),
        "n_sections": len(sections),
        "n_sentences": n_sents,
        **{f"contr_{k}": v for k, v in contr.items()},
        **{f"term_{k}":  v for k, v in term.items()},
        **{f"rep_{k}":   v for k, v in rep.items()},
        "latency_sec": round(time.time() - t0, 1),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

    with open(out_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"         m_contr={contr.get('m_contr')}  "
        f"m_term={term.get('m_term')}  "
        f"m_rep={rep.get('m_rep')}  "
        f"({result['latency_sec']}s)"
    )
    return result


# ── Summary CSV ───────────────────────────────────────────────────────────────

def write_summary(results: list[dict], out_path: Path) -> None:
    csv_path = out_path / "summary.csv"
    fields = [
        "survey_id", "query",
        "contr_m_contr", "contr_n_candidates", "contr_n_confirmed",
        "term_m_term",   "term_n_clusters",    "term_n_inconsistent",
        "rep_m_rep",     "rep_n_total_sentences", "rep_n_duplicates",
        "latency_sec",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\n[structural] summary → {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Structural quality metrics (A.1, A.2, A.3)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model",   required=True)
    args = parser.parse_args()

    cfg    = load_config(CONFIG)
    client = make_client(cfg)

    print("\n[structural] Loading models...")
    ner_pipe     = load_ner_model(cfg["ner_model_path"])
    nli_pipe     = load_nli_model(cfg["nli_model_path"])
    specter_model = load_specter(cfg["specter_model_path"])

    gen_dir = ROOT / "results" / "generations" / f"{args.dataset}_{args.model}"
    if not gen_dir.exists():
        print(f"[ERROR] Generation dir not found: {gen_dir}", file=sys.stderr)
        sys.exit(1)

    run_id  = f"{cfg['judge_id']}_{cfg['judge_comment']}"
    out_dir = ROOT / "results" / "scores" / f"{args.dataset}_{args.model}_structural_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_files = sorted(gen_dir.glob("*.json"))
    gen_files = [f for f in gen_files if not re.search(r"_(raw|old)\.json$", f.name)]

    print(f"\n[structural] {args.dataset} / {args.model}")
    print(f"             {len(gen_files)} surveys → {out_dir}\n")

    all_results = []
    for gf in gen_files:
        with open(gf) as f:
            gen = json.load(f)

        res = process_survey(gen, out_dir, cfg, ner_pipe, nli_pipe, specter_model, client)
        if res:
            all_results.append(res)

    if all_results:
        write_summary(all_results, out_dir)

    ok = len(all_results)
    n_err = len(gen_files) - ok
    print(f"\n[structural] done — ok={ok}  err={n_err}")


if __name__ == "__main__":
    main()
