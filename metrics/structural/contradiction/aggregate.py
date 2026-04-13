# metrics/structural/contradiction/aggregate.py
# Orchestrates all 3 M_contr stages for a single survey.
# Intermediate results are persisted to stage_dir for resumability.

import json
from pathlib import Path

from .candidates import generate_candidates
from .check import run_contradiction_check
from .topic_filter import run_topic_filter


def compute_m_contr(
    survey_id: str,
    sections: list[dict],
    embedder,
    client,
    cfg: dict,
    cache,
    stage_dir: Path,
) -> dict:
    """Three-stage contradiction pipeline for one survey.

    Stage 1: SPECTER similarity filter → candidate pairs        → candidates.json
    Stage 2: LLM topic filter → same-subject pairs              → topic_filtered.json
    Stage 3: LLM contradiction check → confirmed contradictions → contradictions.json

    Each stage loads from disk if its output file already exists (resume support).
    Failed pairs are excluded from both numerator and denominator of M_contr.

    Args:
        survey_id:  Survey identifier (used for logging only).
        sections:   List of {"title": str, "sentences": list[str]}.
        embedder:   SentenceTransformer model (reused from M_rep).
        client:     OpenAI client.
        cfg:        Config dict with thresholds and judge settings.
        cache:      diskcache.Cache instance for LLM call caching.
        stage_dir:  Directory for intermediate JSON files.

    Returns:
        Dict with m_contr, counts, and contradiction records.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)

    threshold  = cfg.get("similarity_threshold", 0.6)
    batch_size = cfg.get("embedding_batch_size", 32)

    # ── Stage 1 — SPECTER similarity candidates ───────────────────────────────
    cands_file = stage_dir / "candidates.json"
    if cands_file.exists():
        candidates = json.loads(cands_file.read_text())
    else:
        candidates = generate_candidates(sections, embedder, threshold, batch_size)
        cands_file.write_text(json.dumps(candidates, ensure_ascii=False))

    if not candidates:
        return {
            "m_contr":              None,
            "n_candidates_stage1":  0,
            "n_after_topic_filter": 0,
            "n_contradictions":     0,
            "n_failed":             0,
            "status":               "no_candidates",
            "contradictions":       [],
        }

    # ── Stage 2 — LLM topic filter ────────────────────────────────────────────
    topic_file = stage_dir / "topic_filtered.json"
    if topic_file.exists():
        after_topic = json.loads(topic_file.read_text())
    else:
        after_topic = run_topic_filter(candidates, client, cfg, cache)
        topic_file.write_text(json.dumps(after_topic, ensure_ascii=False))

    same_subject   = [p for p in after_topic if p.get("same_subject") and p.get("status") != "failed"]
    n_failed_topic = sum(1 for p in after_topic if p.get("status") == "failed")

    # ── Stage 3 — LLM contradiction check ────────────────────────────────────
    contr_file = stage_dir / "contradictions.json"
    if contr_file.exists():
        checked = json.loads(contr_file.read_text())
    else:
        checked = run_contradiction_check(same_subject, client, cfg, cache) if same_subject else []
        contr_file.write_text(json.dumps(checked, ensure_ascii=False))

    contradictions = [p for p in checked if p.get("is_contradiction") and p.get("status") != "failed"]
    n_failed_check = sum(1 for p in checked if p.get("status") == "failed")

    n_candidates     = len(candidates)
    n_after_topic    = len(same_subject)
    n_contradictions = len(contradictions)
    n_failed         = n_failed_topic + n_failed_check
    m_contr = round(n_contradictions / n_after_topic, 4) if n_after_topic > 0 else None

    contradiction_records = [
        {
            "sentence_i":         {"section": p["t1"], "text": p["s1"]},
            "sentence_j":         {"section": p["t2"], "text": p["s2"]},
            "similarity":         p["similarity"],
            "contradiction_type": p.get("contradiction_type", "none"),
            "reasoning":          p.get("reasoning", ""),
        }
        for p in contradictions
    ]

    return {
        "m_contr":              m_contr,
        "n_candidates_stage1":  n_candidates,
        "n_after_topic_filter": n_after_topic,
        "n_contradictions":     n_contradictions,
        "n_failed":             n_failed,
        "status":               "ok",
        "contradictions":       contradiction_records,
    }
