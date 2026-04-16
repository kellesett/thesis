#!/usr/bin/env python3
"""
scripts/build_surge_reference.py

Final assembly step for the ``SurGE_reference`` pseudo-generator: combines the
LaTeX body and the SS-enriched reference mapping into the unified generation
format consumed by all metrics.

Inputs (per survey):
    1. ``datasets/surge/latex_src/<arxiv_id>/merged.tex``      (from merge_latex.sh)
    2. ``datasets/surge/latex_src/<arxiv_id>/ss_matches_<mode>.json``
                                                              (from match_ss_to_bibitems.py)

Outputs (per survey):
    1. ``datasets/surge/latex_src/<arxiv_id>/merged.md``
        — clean GFM markdown with inline ``[N]`` citations; intermediate
        artifact that can be inspected with any markdown viewer.
    2. ``results/generations/SurGE_reference/<sid>.json``
        — final generation record matching the perplexity_dr schema. Body is
        the same ``merged.md`` content; ``meta.references[]`` is enriched
        from the SS mapping.

Reference schema (every reference has ALL keys below; missing values are null):

    idx                  int    — 1-indexed, matches ``[N]`` in the body
    title                str    — prefer SS's clean title; fallback latex-extracted
    canonical_title      str    — same as title when SS matched; normalized latex otherwise
    url                  str?   — prefer SS url; fallback latex-extracted
    arxiv_id             str?   — REQUIRED KEY. SS's ArXiv externalId, else latex-parsed, else null
    semantic_scholar_id  str?   — REQUIRED KEY. SS's paperId, else null
    doi                  str?   — from SS if available
    year                 int?   — from SS if available
    doc_id               int?   — from SurGE title_index (corpus.json lookup)

Both ``arxiv_id`` and ``semantic_scholar_id`` are present as keys on every
entry; their value is ``null`` when SS couldn't resolve that reference and
no arxiv id could be extracted from LaTeX.

Usage::

    # all surveys with both merged.tex and ss_matches_hybrid.json on disk:
    python3 scripts/build_surge_reference.py --mode hybrid

    # first 40 surveys, overwrite existing outputs:
    python3 scripts/build_surge_reference.py --mode hybrid --limit 40 --force

    # single survey, debug:
    python3 scripts/build_surge_reference.py --mode hybrid --id 0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scripts.parse_reference_md import parse as parse_merged_tex  # noqa: E402
from src.log_setup import setup_logging  # noqa: E402


# ── Paths ─────────────────────────────────────────────────────────────────────

SURVEYS_JSON     = ROOT / "datasets" / "SurGE" / "surveys.json"
TITLE_INDEX_JSON = ROOT / "datasets" / "SurGE" / "title_index.json"
LATEX_SRC_DIR    = ROOT / "datasets" / "surge" / "latex_src"
ARXIV_CACHE      = LATEX_SRC_DIR / "arxiv_cache.json"
OUT_DIR          = ROOT / "results" / "generations" / "SurGE_reference"


# ── title_index helper ───────────────────────────────────────────────────────

_TITLE_KEY_RE = re.compile(r"[^a-z0-9]+")


def _title_key(title: str | None) -> str | None:
    """Normalize to the same alnum-lower key scheme used by title_index.json."""
    if not title:
        return None
    return _TITLE_KEY_RE.sub("", title.lower()) or None


def _lookup_doc_id(title_index: dict[str, int], candidates: list[str | None]) -> int | None:
    """Return first title_index hit among the candidate titles, else None.

    Candidates are tried in order — callers should pass the cleanest title
    first (e.g. SS's canonical_title before the latex-extracted variant) so
    ties go to the higher-quality source.
    """
    for c in candidates:
        k = _title_key(c)
        if k and k in title_index:
            return int(title_index[k])
    return None


# ── Reference enrichment ──────────────────────────────────────────────────────


def _build_enriched_references(
    parse_refs: list[dict],
    ss_mapping: list[dict],
    title_index: dict[str, int],
) -> tuple[list[dict], dict[str, int]]:
    """Merge parse_reference_md refs with SS mapping into the output schema.

    Args:
        parse_refs:  ``references`` field from ``parse_merged_tex`` result.
        ss_mapping:  ``mapping`` field from ``ss_matches_<mode>.json``.
        title_index: loaded SurGE ``title_index.json``.

    Returns:
        ``(refs, counters)`` where ``counters`` tracks how many refs got each
        kind of enrichment — used for the per-survey log line.
    """
    # ss_mapping entries carry ``latex_idx`` matching our ``ref.idx``. Build a
    # fast lookup once per survey.
    ss_by_idx: dict[int, dict] = {m["latex_idx"]: m for m in ss_mapping}

    counters = {
        "ss_matched":          0,
        "with_arxiv_id":       0,
        "with_ss_paper_id":    0,
        "with_doc_id":         0,
        "fetchable":           0,  # arxiv_id OR ss_paper_id
    }

    out: list[dict] = []
    for ref in parse_refs:
        idx             = ref["idx"]
        latex_title     = ref.get("title")
        latex_canon     = ref.get("canonical_title")
        latex_url       = ref.get("url")
        latex_arxiv_id  = ref.get("arxiv_id")

        ss = ss_by_idx.get(idx)
        if ss is not None:
            counters["ss_matched"] += 1
            title           = ss.get("ss_title") or latex_title
            # SS title is already presentation-quality, so use it as the
            # canonical_title too; fall back to latex's normalized variant.
            canonical_title = ss.get("ss_title") or latex_canon
            # arxiv_id priority: SS (authoritative) → latex-parsed → null.
            arxiv_id        = ss.get("ss_arxiv_id") or latex_arxiv_id
            ss_paper_id     = ss.get("ss_paper_id")
            doi             = ss.get("ss_doi")
            year            = ss.get("ss_year")
            url             = ss.get("ss_url") or latex_url
        else:
            title           = latex_title
            canonical_title = latex_canon
            arxiv_id        = latex_arxiv_id
            ss_paper_id     = None
            doi             = None
            year            = None
            url             = latex_url

        # doc_id: try cleanest title first. canonical_title is either SS's
        # clean string or the normalized latex form; latex title is the raw
        # bibitem-extracted string (dirty but sometimes the only hit).
        doc_id = _lookup_doc_id(title_index, [canonical_title, title, latex_title])

        entry = {
            "idx":                 idx,
            "title":               title,
            "canonical_title":     canonical_title,
            "url":                 url,
            "arxiv_id":            arxiv_id,          # mandatory key
            "semantic_scholar_id": ss_paper_id,       # mandatory key
            "doi":                 doi,
            "year":                year,
            "doc_id":              doc_id,
        }
        out.append(entry)

        if arxiv_id:    counters["with_arxiv_id"]    += 1
        if ss_paper_id: counters["with_ss_paper_id"] += 1
        if doc_id:      counters["with_doc_id"]      += 1
        if arxiv_id or ss_paper_id:
            counters["fetchable"] += 1

    return out, counters


# ── Per-survey orchestration ──────────────────────────────────────────────────


def _process_one(
    *,
    survey:        dict,
    arxiv_id:      str,
    mode:          str,
    title_index:   dict[str, int],
    generated_at:  str,
    force:         bool,
) -> tuple[str, dict]:
    """Build merged.md + generation JSON for one survey.

    Returns:
        ``(status, info)`` where status is one of
        ``"written"`` / ``"skipped_cache"`` / ``"skipped_no_tex"``
        / ``"skipped_no_match"``, and info carries per-survey counters for
        the run summary.
    """
    sid       = str(survey["survey_id"])
    paper_dir = LATEX_SRC_DIR / arxiv_id
    merged_tex = paper_dir / "merged.tex"
    merged_md  = paper_dir / "merged.md"
    ss_matches = paper_dir / f"ss_matches_{mode}.json"
    out_file   = OUT_DIR / f"{sid}.json"

    if out_file.exists() and not force:
        return "skipped_cache", {}
    if not merged_tex.is_file():
        logger.warning("[sid=%s arxiv=%s] no merged.tex — run scripts/merge_latex.sh first", sid, arxiv_id)
        return "skipped_no_tex", {}
    if not ss_matches.is_file():
        logger.warning(
            "[sid=%s arxiv=%s] no %s — run match_ss_to_bibitems.py --mode %s first",
            sid, arxiv_id, ss_matches.name, mode,
        )
        return "skipped_no_match", {}

    # 1. parse merged.tex → (body_md, references)
    parse_result = parse_merged_tex(merged_tex)
    if not parse_result.body_md.strip():
        raise RuntimeError(f"parse_merged_tex produced empty body for {merged_tex}")

    # 2. write merged.md (intermediate artifact, canonical GFM output)
    merged_md.write_text(parse_result.body_md, encoding="utf-8")

    # 3. load the SS mapping and build enriched references
    ss_data    = json.loads(ss_matches.read_text(encoding="utf-8"))
    ss_mapping = ss_data.get("mapping") or []
    refs, counters = _build_enriched_references(
        parse_result.references, ss_mapping, title_index,
    )

    # 4. assemble the generation record
    record = {
        "id":         sid,
        "dataset_id": "SurGE",
        "model_id":   "SurGE_reference",
        "query":      survey.get("survey_title") or "",
        "text":       parse_result.body_md,
        "success":    True,
        "meta": {
            "model":        "human",
            "generated_at": generated_at,
            "latency_sec":  None,
            "cost_usd":     None,
            "error":        None,
            "source":       "merged_tex",
            "match_mode":   mode,
            # Keep a snapshot of the match-stage quality for post-hoc analysis
            # without forcing consumers to open ss_matches_*.json.
            "match_stats":  ss_data.get("stats"),
            "references":   refs,
            "authors":      survey.get("authors"),
            "year":         survey.get("year"),
            "arxiv_id":     arxiv_id,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    info = {
        "n_refs":          len(refs),
        "parse_stats":     parse_result.stats,
        **counters,
    }
    return "written", info


# ── Entry point ───────────────────────────────────────────────────────────────


def _load_surveys_sorted() -> list[dict]:
    data = json.loads(SURVEYS_JSON.read_text(encoding="utf-8"))
    data.sort(key=lambda s: s["survey_id"])
    return data


def _load_arxiv_cache() -> dict:
    if not ARXIV_CACHE.exists():
        logger.error("arxiv_cache.json missing — run scripts/fetch_reference_latex.py first")
        return {}
    return json.loads(ARXIV_CACHE.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=("string", "llm", "hybrid"), default="hybrid",
                        help="Which ss_matches_<mode>.json to consume.")
    parser.add_argument("--id", type=int, default=None,
                        help="Process only this survey_id.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only surveys with survey_id <= LIMIT (inclusive, id-based).")
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if the output generation JSON already exists.")
    args = parser.parse_args()

    setup_logging("build_surge_reference")

    # Reserve stderr for tqdm — logs go to the file only; tail -f to follow.
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

    # 1. Gather and filter surveys
    surveys = _load_surveys_sorted()
    if args.id is not None:
        surveys = [s for s in surveys if s["survey_id"] == args.id]
    elif args.limit is not None:
        # id-based inclusive filter: survey_id <= LIMIT. See veriscore's note.
        surveys = [s for s in surveys if s["survey_id"] <= args.limit]
    if not surveys:
        logger.warning("no surveys to process (filters produced empty set)")
        return 0

    arxiv_cache = _load_arxiv_cache()
    logger.info("Loading title_index.json ...")
    title_index = json.loads(TITLE_INDEX_JSON.read_text(encoding="utf-8"))

    generated_at = datetime.now(timezone.utc).isoformat()

    summary = {
        "total":             0,
        "written":           0,
        "skipped_cache":     0,
        "skipped_no_tex":    0,
        "skipped_no_match":  0,
        "skipped_no_id":     0,
        "failed":            0,
        "n_refs_total":      0,
        "ss_matched_total":  0,
        "with_arxiv_total":  0,
        "with_ss_id_total":  0,
        "with_doc_id_total": 0,
        "fetchable_total":   0,
    }

    outer = tqdm(surveys, desc="surveys", unit="surv", dynamic_ncols=True)
    try:
        for survey in outer:
            summary["total"] += 1
            sid = str(survey["survey_id"])
            entry = arxiv_cache.get(sid)
            if not entry or not entry.get("arxiv_id"):
                summary["skipped_no_id"] += 1
                continue
            arxiv_id = entry["arxiv_id"]
            outer.set_description(f"surveys [{sid} → {arxiv_id}]")

            try:
                status, info = _process_one(
                    survey=survey, arxiv_id=arxiv_id, mode=args.mode,
                    title_index=title_index, generated_at=generated_at,
                    force=args.force,
                )
            except Exception as e:
                logger.exception("[sid=%s arxiv=%s] failed: %s", sid, arxiv_id, e)
                summary["failed"] += 1
                continue

            summary[status] = summary.get(status, 0) + 1
            if status != "written":
                continue

            summary["n_refs_total"]      += info["n_refs"]
            summary["ss_matched_total"]  += info["ss_matched"]
            summary["with_arxiv_total"]  += info["with_arxiv_id"]
            summary["with_ss_id_total"]  += info["with_ss_paper_id"]
            summary["with_doc_id_total"] += info["with_doc_id"]
            summary["fetchable_total"]   += info["fetchable"]

            logger.info(
                "[sid=%s arxiv=%s] refs=%d ss_matched=%d "
                "arxiv_id=%d ss_paper_id=%d doc_id=%d fetchable=%d",
                sid, arxiv_id,
                info["n_refs"], info["ss_matched"],
                info["with_arxiv_id"], info["with_ss_paper_id"],
                info["with_doc_id"], info["fetchable"],
            )

            # Outer postfix: running counts across processed surveys.
            outer.set_postfix_str(
                f"refs={summary['n_refs_total']} "
                f"arxiv={summary['with_arxiv_total']} "
                f"ss={summary['with_ss_id_total']} "
                f"doc={summary['with_doc_id_total']} "
                f"fetch={summary['fetchable_total']}"
            )
    finally:
        outer.close()

    logger.info("=" * 60)
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    logger.info("Output : %s", OUT_DIR)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
