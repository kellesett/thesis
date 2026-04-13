# metrics/structural/contradiction/aggregate.py
# Orchestrates all 3 M_contr stages for a single survey.
# Intermediate results are persisted to stage_dir for resumability.

import json
from pathlib import Path

from .candidates import generate_candidates
from .check import run_contradiction_check
from .llm_utils import TokenCounter
from .topic_filter import run_topic_filter


def compute_m_contr(
    survey_id: str,
    sections: list[dict],
    embedder,
    client,
    cfg: dict,
    cache,
    stage_dir: Path,
    specter_bar=None,
    topic_bar=None,
    contr_bar=None,
) -> dict:
    """Three-stage contradiction pipeline for one survey.

    Stage 1: SPECTER similarity filter → candidate pairs        → candidates.json
    Stage 2: LLM topic filter → same-subject pairs              → topic_filtered.json
    Stage 3: LLM contradiction check → confirmed contradictions → contradictions.json

    Each stage loads from disk if its output file already exists (resume support).
    Failed pairs are excluded from both numerator and denominator of M_contr.

    Args:
        survey_id:   Survey identifier (used for logging only).
        sections:    List of {"title": str, "sentences": list[str]}.
        embedder:    SentenceTransformer model (reused from M_rep).
        client:      OpenAI client.
        cfg:         Config dict with thresholds and judge settings.
        cache:       diskcache.Cache instance for LLM call caching.
        stage_dir:   Directory for intermediate JSON files.
        specter_bar: Optional tqdm bar for stage 1 (SPECTER pair iteration).
        topic_bar:   Optional tqdm bar for stage 2 (LLM topic filter).
        contr_bar:   Optional tqdm bar for stage 3 (LLM contradiction check).

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
        if specter_bar is not None:
            # Loaded from cache — fast-forward bar and show cached count
            n = sum(len(s["sentences"]) for s in sections)
            n_pairs = n * (n - 1) // 2
            specter_bar.reset(total=n_pairs)
            specter_bar.update(n_pairs)
            specter_bar.set_postfix_str(f"→ {len(candidates)} sel (cached)")
    else:
        n = sum(len(s["sentences"]) for s in sections)
        n_pairs = n * (n - 1) // 2
        if specter_bar is not None:
            specter_bar.reset(total=n_pairs)
        candidates = generate_candidates(sections, embedder, threshold, batch_size, pbar=specter_bar)
        cands_file.write_text(json.dumps(candidates, ensure_ascii=False))

    if not candidates:
        # Zero out remaining bars if no candidates
        for bar in [topic_bar, contr_bar]:
            if bar is not None:
                bar.reset(total=1)
                bar.update(1)
                bar.set_postfix_str("0 sel (no candidates)")
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
    topic_counter = TokenCounter()

    if topic_file.exists():
        after_topic = json.loads(topic_file.read_text())
        n_same = sum(1 for p in after_topic if p.get("same_subject") and p.get("status") != "failed")
        if topic_bar is not None:
            topic_bar.reset(total=len(candidates))
            topic_bar.update(len(candidates))
            topic_bar.set_postfix_str(f"{n_same}/{len(candidates)} sel (cached)")
    else:
        if topic_bar is not None:
            topic_bar.reset(total=len(candidates))
        log_reasoning = cfg.get("judge_log_reasoning", True)
        after_topic = run_topic_filter(
            candidates, client, cfg, cache,
            pbar=topic_bar, token_counter=topic_counter,
            log_reasoning=log_reasoning,
        )
        topic_file.write_text(json.dumps(after_topic, ensure_ascii=False))

    same_subject   = [p for p in after_topic if p.get("same_subject") and p.get("status") != "failed"]
    n_failed_topic = sum(1 for p in after_topic if p.get("status") == "failed")

    # ── Stage 3 — LLM contradiction check ────────────────────────────────────
    contr_file = stage_dir / "contradictions.json"
    contr_counter = TokenCounter()

    if contr_file.exists():
        checked = json.loads(contr_file.read_text())
        n_confirmed = sum(1 for p in checked if p.get("is_contradiction") and p.get("status") != "failed")
        if contr_bar is not None:
            n_total = len(same_subject) or 1
            contr_bar.reset(total=n_total)
            contr_bar.update(n_total)
            contr_bar.set_postfix_str(f"{n_confirmed}/{n_total} confirmed (cached)")
    else:
        if contr_bar is not None:
            contr_bar.reset(total=max(len(same_subject), 1))
        checked = run_contradiction_check(
            same_subject, client, cfg, cache,
            pbar=contr_bar, token_counter=contr_counter,
            log_reasoning=log_reasoning,
        ) if same_subject else []
        if not same_subject and contr_bar is not None:
            contr_bar.update(contr_bar.total)
            contr_bar.set_postfix_str("0 confirmed (no same-subject pairs)")
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
