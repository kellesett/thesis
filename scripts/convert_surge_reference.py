#!/usr/bin/env python3
"""
scripts/convert_surge_reference.py

Converts SurGE human-written reference surveys into the unified generation
format at ``results/generations/SurGE_reference/<survey_id>.json``, matching
the schema of ``results/generations/SurGE_perplexity_dr/<id>.json``.

Primary path (preferred, high-fidelity):
    fetch → merge → parse_reference_md.parse(merged.tex)
    ↳ inline numeric [N] citations + a canonical references[] list

Fallback path (when no LaTeX source is available yet):
    build a plain markdown body from surveys.json ``structure[]`` with
    ``references=[]`` — downstream metrics that require actual references
    will skip the survey.

Run::

    python3 scripts/convert_surge_reference.py                 # all surveys
    python3 scripts/convert_surge_reference.py --id 0          # single
    python3 scripts/convert_surge_reference.py --force         # ignore cache
    python3 scripts/convert_surge_reference.py --limit 10      # first 10 by idx
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

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scripts.parse_reference_md import parse as parse_merged_tex  # noqa: E402
from src.log_setup import setup_logging  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────

SURVEYS_JSON     = ROOT / "datasets" / "SurGE" / "surveys.json"
CORPUS_JSON      = ROOT / "datasets" / "SurGE" / "corpus.json"
TITLE_INDEX_JSON = ROOT / "datasets" / "SurGE" / "title_index.json"
LATEX_SRC_DIR    = ROOT / "datasets" / "surge" / "latex_src"
ARXIV_CACHE      = LATEX_SRC_DIR / "arxiv_cache.json"
OUT_DIR          = ROOT / "results" / "generations" / "SurGE_reference"


# ── Title index matching ──────────────────────────────────────────────────────


_TITLE_KEY_RE = re.compile(r"[^a-z0-9]+")


def _title_key(title: str | None) -> str | None:
    """Normalize a title into the same key scheme as ``title_index.json``.

    title_index uses ``lowercase(title) with every non-alphanumeric char
    stripped``, so ``"A Survey on Edge Computing"`` → ``"asurveyonedgecomputing"``.
    """
    if not title:
        return None
    return _TITLE_KEY_RE.sub("", title.lower()) or None


def _enrich_references(
    refs: list[dict],
    title_index: dict[str, int],
) -> tuple[list[dict], int]:
    """Try to match each reference's title against SurGE's title_index.

    For matched references we add a ``doc_id`` field (int) — downstream
    metrics (factuality, diversity) use ``doc_id`` to fetch the abstract from
    ``corpus.json``.

    Returns ``(refs, n_matched)`` where ``refs`` is the same list with
    possibly-added ``doc_id`` keys and ``n_matched`` is the count of matched
    entries.
    """
    matched = 0
    for ref in refs:
        key = _title_key(ref.get("title"))
        if not key:
            continue
        doc_id = title_index.get(key)
        if doc_id is None:
            # title_index uses the raw lowercased-alnum form; try the cleaner
            # canonical_title as a fallback.
            key_alt = _title_key(ref.get("canonical_title"))
            if key_alt and key_alt != key:
                doc_id = title_index.get(key_alt)
        if doc_id is not None:
            ref["doc_id"] = int(doc_id)
            matched += 1
    return refs, matched


# ── Fallback builder (surveys.json → markdown) ────────────────────────────────


def _fallback_body(survey: dict) -> str:
    """Build a naive markdown body from surveys.json ``structure[]``.

    Used when no merged.tex is available (arxiv search failed or tarball
    download errored). The content field of SurGE sections is raw LaTeX that
    was stripped of its \\cite{...} markers during dataset construction, so the
    fallback body has inherently poorer citation coverage — hence we emit
    ``references=[]`` to signal to downstream tools that this survey cannot
    drive citation-dependent metrics.
    """
    parts: list[str] = []
    title = survey.get("survey_title") or ""
    if title:
        parts.append(f"# {title}")
    for section in survey.get("structure") or []:
        name = (section.get("title") or "").strip()
        content = (section.get("content") or "").strip()
        if name and name.lower() != "root":
            parts.append(f"## {name}")
        if content:
            parts.append(content)
    return "\n\n".join(parts).strip() + "\n"


# ── Main per-survey conversion ────────────────────────────────────────────────


def _load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_merged_tex(sid: str, cache: dict) -> Path | None:
    """Return the path to ``merged.tex`` for survey ``sid``, if it exists."""
    entry = cache.get(sid)
    if not entry:
        return None
    arxiv_id = entry.get("arxiv_id")
    if not arxiv_id:
        return None
    candidate = LATEX_SRC_DIR / arxiv_id / "merged.tex"
    return candidate if candidate.is_file() else None


def _build_record(
    survey: dict,
    title_index: dict[str, int],
    cache: dict,
    generated_at: str,
) -> tuple[dict, dict]:
    """Build the generation-format record for a single survey.

    Returns ``(record, info)`` where ``info`` is a small dict of diagnostic
    counters used by the caller to build the run summary.
    """
    sid = str(survey["survey_id"])
    query = survey.get("survey_title") or ""

    info = {
        "source": "fallback",
        "references": 0,
        "matched": 0,
        "cites_total": 0,
        "cites_resolved": 0,
    }

    merged_tex = _resolve_merged_tex(sid, cache)
    if merged_tex is not None:
        try:
            result = parse_merged_tex(merged_tex)
        except Exception as e:
            logger.warning("[%s] parse_merged_tex failed (%s) → falling back", sid, e)
            merged_tex = None
        else:
            refs, matched = _enrich_references(result.references, title_index)
            info.update(
                source         = "merged_tex",
                references     = len(refs),
                matched        = matched,
                cites_total    = result.stats.get("cites_total", 0),
                cites_resolved = result.stats.get("cites_resolved", 0),
            )
            body = result.body_md
            references = refs

    if merged_tex is None:
        body = _fallback_body(survey)
        references = []

    record = {
        "id":         sid,
        "dataset_id": "SurGE",
        "model_id":   "SurGE_reference",
        "query":      query,
        "text":       body,
        "success":    bool(body and body.strip()),
        "meta": {
            "model":        "human",
            "generated_at": generated_at,
            "latency_sec":  None,
            "cost_usd":     None,
            "error":        None if (body and body.strip()) else "Empty body",
            "source":       info["source"],
            "references":   references,
            "authors":      survey.get("authors"),
            "year":         survey.get("year"),
        },
    }
    return record, info


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> int:
    """Convert SurGE reference surveys to unified generation format."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--id", type=int, default=None,
                        help="Process a single survey_id.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N surveys by ascending idx.")
    parser.add_argument("--force", action="store_true",
                        help="Re-convert surveys even if an output file exists.")
    args = parser.parse_args()

    setup_logging("convert_surge_reference")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading surveys.json ...")
    surveys = _load_json(SURVEYS_JSON)
    # Canonical ordering by survey_id — matches perplexity_dr output layout
    # and ensures --limit N means "first N by idx".
    surveys.sort(key=lambda s: s["survey_id"])

    logger.info("Loading title_index.json ...")
    title_index = _load_json(TITLE_INDEX_JSON)

    cache: dict
    if ARXIV_CACHE.exists():
        cache = _load_json(ARXIV_CACHE)
    else:
        logger.warning("No arxiv_cache.json — every survey will use fallback path.")
        cache = {}

    if args.id is not None:
        surveys = [s for s in surveys if s["survey_id"] == args.id]
    elif args.limit is not None:
        surveys = surveys[: args.limit]

    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "total":        0,
        "written":      0,
        "skipped":      0,
        "failed":       0,
        "merged_tex":   0,
        "fallback":     0,
        "refs_total":   0,
        "refs_matched": 0,
    }

    for survey in surveys:
        sid = str(survey["survey_id"])
        summary["total"] += 1
        out_file = OUT_DIR / f"{sid}.json"

        if out_file.exists() and not args.force:
            summary["skipped"] += 1
            continue

        try:
            record, info = _build_record(survey, title_index, cache, generated_at)
        except Exception as e:
            summary["failed"] += 1
            logger.exception("[%s] failed: %s", sid, e)
            continue

        out_file.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["written"] += 1
        summary[info["source"]] += 1
        summary["refs_total"]   += info["references"]
        summary["refs_matched"] += info["matched"]

        logger.info(
            "[%s] %-11s refs=%-3d matched=%-3d cites=%d/%d",
            sid,
            info["source"],
            info["references"],
            info["matched"],
            info["cites_resolved"],
            info["cites_total"],
        )

    logger.info("=" * 60)
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    logger.info("Output : %s", OUT_DIR)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
