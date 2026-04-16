#!/usr/bin/env python3
"""
scripts/enrich_surge_reference.py

Post-process ``results/generations/SurGE_reference/<id>.json`` files by
searching arxiv for each reference's title and, on a strong match, filling in
``arxiv_id`` / ``url`` / ``canonical_title`` in place.

Rationale
---------
``parse_reference_md.py`` extracts references directly from ``\\bibitem`` blocks
in ``merged.tex``. Most bibitems carry a title but no machine-readable arxiv
pointer, so the initial conversion emits ``arxiv_id=None`` for almost every
reference. This script closes that gap by calling the arxiv Atom API keyed on
the reference title — the same direction ``perplexity_dr`` goes (``url ->
arxiv_id -> canonical_title``) but starting one step earlier because we only
have a title.

Behaviour
---------
* Idempotent: already-enriched references (``arxiv_id`` set) are skipped.
* Globally cached: ``scripts.arxiv_title_search`` persists hits *and* misses
  to ``datasets/surge/latex_src/ref_arxiv_cache.json``. Re-running costs zero
  network I/O for previously-seen titles.
* After a successful enrichment, re-runs ``title_index.json`` matching so that
  the freshly-obtained ``canonical_title`` can also populate ``doc_id`` for
  references that weren't matched during initial conversion.

Runtime note
------------
Every new (cache-miss) reference costs ~3.2 s of rate-limited wait. A survey
with ~100 references therefore takes ~5 minutes on a cold cache. Plan
accordingly — ``--limit`` and ``--id`` exist for incremental runs.

Usage::

    python3 scripts/enrich_surge_reference.py                   # all files
    python3 scripts/enrich_surge_reference.py --limit 40        # first 40 by idx
    python3 scripts/enrich_surge_reference.py --id 0            # single survey
    python3 scripts/enrich_surge_reference.py --dry-run         # no write
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tqdm import tqdm  # noqa: E402

from scripts.arxiv_title_search import search_by_title  # noqa: E402
from src.log_setup import setup_logging  # noqa: E402

GEN_DIR          = ROOT / "results" / "generations" / "SurGE_reference"
TITLE_INDEX_JSON = ROOT / "datasets" / "SurGE" / "title_index.json"

_TITLE_KEY_RE = re.compile(r"[^a-z0-9]+")


# ── title_index.json helper (mirrors convert_surge_reference) ────────────────


def _title_key(title: str | None) -> str | None:
    """Normalize a title into the same key scheme as ``title_index.json``."""
    if not title:
        return None
    return _TITLE_KEY_RE.sub("", title.lower()) or None


def _rematch_doc_id(ref: dict, title_index: dict) -> bool:
    """Try to fill ``doc_id`` using the (possibly refreshed) canonical_title.

    Returns ``True`` if a new ``doc_id`` was assigned.
    """
    if "doc_id" in ref:
        return False
    # Prefer the canonical_title we just got from arxiv — it's whitespace-
    # normalized and drops LaTeX quirks, so it's the cleanest candidate.
    for candidate in (ref.get("canonical_title"), ref.get("title")):
        key = _title_key(candidate)
        if not key:
            continue
        doc_id = title_index.get(key)
        if doc_id is not None:
            ref["doc_id"] = int(doc_id)
            return True
    return False


# ── Per-file enrichment ───────────────────────────────────────────────────────


def enrich_file(path: Path, title_index: dict, *, dry_run: bool = False) -> dict:
    """Enrich references in a single generation JSON.

    Returns a dict of diagnostic counters used for the run summary.

    Prints a per-reference tqdm bar on stderr so a multi-minute run is visible
    in real time — arxiv rate-limiting (3.2s/request) means that without a
    progress indicator the script appears frozen for ~5 min per survey.
    """
    stats = {
        "total_refs":  0,
        "pre_arxivid": 0,  # references already carrying arxiv_id (skipped)
        "hit":         0,  # new arxiv_id populated this run
        "miss":        0,
        "doc_id_new":  0,
    }
    gen = json.loads(path.read_text(encoding="utf-8"))
    refs = gen.get("meta", {}).get("references", [])
    changed = False

    bar = tqdm(
        refs,
        desc=f"[{path.stem}]",
        unit="ref",
        leave=False,
        dynamic_ncols=True,
    )
    for ref in bar:
        stats["total_refs"] += 1
        if ref.get("arxiv_id"):
            stats["pre_arxivid"] += 1
            # Still try title_index match in case an earlier run missed it.
            if _rematch_doc_id(ref, title_index):
                stats["doc_id_new"] += 1
                changed = True
            bar.set_postfix(hit=stats["hit"], miss=stats["miss"], pre=stats["pre_arxivid"])
            continue

        title = ref.get("title") or ""
        hit = search_by_title(title)
        if hit is None:
            stats["miss"] += 1
            bar.set_postfix(hit=stats["hit"], miss=stats["miss"], pre=stats["pre_arxivid"])
            continue

        # Populate fields — never overwrite a pre-existing non-None value.
        ref["arxiv_id"] = hit["arxiv_id"]
        if not ref.get("url"):
            ref["url"] = hit["url"]
        # canonical_title is the authoritative one from arxiv, so we *do*
        # overwrite the heuristic one we derived from \bibitem content.
        ref["canonical_title"] = hit["canonical_title"]
        stats["hit"] += 1
        changed = True

        if _rematch_doc_id(ref, title_index):
            stats["doc_id_new"] += 1

        bar.set_postfix(hit=stats["hit"], miss=stats["miss"], pre=stats["pre_arxivid"])

    if changed and not dry_run:
        path.write_text(
            json.dumps(gen, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return stats


# ── Entry point ───────────────────────────────────────────────────────────────


def _numeric_key(p: Path) -> int:
    """Sort ``<id>.json`` filenames by numeric id, pushing non-numeric to the end."""
    return int(p.stem) if p.stem.isdigit() else 10**9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N files by ascending id.")
    parser.add_argument("--id", type=int, default=None,
                        help="Process a single survey id.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Search without writing changes back to disk.")
    args = parser.parse_args()

    setup_logging("enrich_surge_reference")
    title_index = json.loads(TITLE_INDEX_JSON.read_text(encoding="utf-8"))

    files = sorted(GEN_DIR.glob("*.json"), key=_numeric_key)
    if args.id is not None:
        files = [f for f in files if f.stem == str(args.id)]
    elif args.limit is not None:
        files = files[: args.limit]

    if not files:
        logger.warning("No generation files to process.")
        return 0

    totals = {
        "files":       0,
        "total_refs":  0,
        "pre_arxivid": 0,
        "hit":         0,
        "miss":        0,
        "doc_id_new":  0,
    }

    for f in files:
        try:
            stats = enrich_file(f, title_index, dry_run=args.dry_run)
        except Exception as e:
            logger.exception("%s: %s", f.name, e)
            continue

        totals["files"] += 1
        for k in ("total_refs", "pre_arxivid", "hit", "miss", "doc_id_new"):
            totals[k] += stats[k]

        logger.info(
            "[%s] refs=%-3d hit=%-3d miss=%-3d pre=%-3d doc_id+=%d",
            f.stem,
            stats["total_refs"],
            stats["hit"],
            stats["miss"],
            stats["pre_arxivid"],
            stats["doc_id_new"],
        )

    logger.info("=" * 60)
    logger.info("Summary: %s", json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
