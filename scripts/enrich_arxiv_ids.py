#!/usr/bin/env python3
"""
scripts/enrich_arxiv_ids.py

Fill missing ``arxiv_id`` in ``results/generations/SurGE_reference/<sid>.json``.

Rationale
---------
Sanity check (doc_id vs all_cites) confirms that every reference we matched to
SurGE's ``corpus.json`` is on arxiv — the corpus itself is an arxiv snapshot.
But a reference can have ``doc_id != null`` while still missing ``arxiv_id``,
because Semantic Scholar doesn't always populate ``externalIds.ArXiv`` even
when the paper is available on arxiv (common for papers whose primary venue
was IEEE/ACM with an arxiv mirror).

This script closes that gap:
    for each ref with doc_id BUT no arxiv_id:
        title = corpus[doc_id].Title           # curated, clean
        hit   = arxiv_title_search(title)      # STRONG_MATCH_SCORE=0.98, cached
        if hit:
            ref.arxiv_id = hit.arxiv_id
            ref.url      = hit.url  (if not already set)

Properties
----------
* Exact-after-normalize threshold (0.98) on a clean curated title → near-zero
  false-positive rate.
* Idempotent: references that already carry ``arxiv_id`` are skipped; re-runs
  cost zero network I/O thanks to the global title-search cache at
  ``datasets/surge/latex_src/ref_arxiv_cache.json``.
* In-place update of the generation JSON. ``meta.match_stats`` is a snapshot
  of the match stage and is intentionally left unchanged — downstream stat
  aggregation should recount ``arxiv_id`` from ``meta.references[]`` if
  current counts are needed.

Usage::

    python3 scripts/enrich_arxiv_ids.py               # all SurGE_reference JSONs
    python3 scripts/enrich_arxiv_ids.py --limit 40    # first 40 by survey_id
    python3 scripts/enrich_arxiv_ids.py --id 12       # single survey
    python3 scripts/enrich_arxiv_ids.py --dry-run     # no writes
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.arxiv_title_search import search_by_title  # noqa: E402
from src.log_setup import setup_logging                  # noqa: E402

GEN_DIR     = ROOT / "results" / "generations" / "SurGE_reference"
CORPUS_JSON = ROOT / "datasets" / "SurGE" / "corpus.json"


# ── Corpus title index ────────────────────────────────────────────────────────


def _load_corpus_titles() -> dict[int, str]:
    """Return ``{doc_id: Title}`` from corpus.json.

    corpus.json is ~1 M entries and close to 500 MB in memory if we keep every
    field. We keep only ``doc_id → Title`` because that is all we need for
    downstream arxiv search.
    """
    logger.info("Loading corpus.json ...")
    corpus = json.loads(CORPUS_JSON.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for e in corpus:
        doc_id = e.get("doc_id")
        title  = e.get("Title")
        if doc_id is not None and title:
            out[int(doc_id)] = title
    logger.info("Indexed %d corpus titles", len(out))
    return out


# ── Per-file enrichment ───────────────────────────────────────────────────────


def _numeric_key(p: Path) -> int:
    return int(p.stem) if p.stem.isdigit() else 10**9


def enrich_file(
    path: Path,
    corpus_titles: dict[int, str],
    *,
    dry_run: bool = False,
    inner_bar: tqdm | None = None,
) -> dict:
    """Enrich a single generation JSON in place. Returns per-file counters.

    When ``inner_bar`` is provided, it is ``reset()`` to the number of
    candidate refs for this file and ``update(1)`` is called as each arxiv
    lookup completes. The postfix shows running hits/misses so a long cold
    run stays visibly alive.
    """
    stats = {
        "refs_total":      0,
        "candidates":      0,  # doc_id set AND arxiv_id null
        "no_corpus_title": 0,  # shouldn't happen, but guard
        "hits":            0,
        "misses":          0,
    }
    d = json.loads(path.read_text(encoding="utf-8"))
    refs = d.get("meta", {}).get("references", [])
    stats["refs_total"] = len(refs)

    # Identify refs needing an arxiv_id lookup up front — avoids iterating
    # everyone through search_by_title and makes the progress counter honest.
    candidates = [r for r in refs if r.get("doc_id") and not r.get("arxiv_id")]
    stats["candidates"] = len(candidates)

    if inner_bar is not None:
        # total=max(N,1) so the bar still renders for surveys with 0 candidates
        # (they finish instantly with a single update below).
        inner_bar.reset(total=max(len(candidates), 1))
        inner_bar.set_postfix_str("hits=0 misses=0")
        if not candidates:
            inner_bar.update(1)

    changed = False
    for ref in candidates:
        title = corpus_titles.get(int(ref["doc_id"]))
        if not title:
            stats["no_corpus_title"] += 1
        else:
            hit = search_by_title(title)
            if hit is None:
                stats["misses"] += 1
            else:
                ref["arxiv_id"] = hit["arxiv_id"]
                if not ref.get("url"):
                    ref["url"] = hit["url"]
                stats["hits"] += 1
                changed = True
        if inner_bar is not None:
            inner_bar.update(1)
            inner_bar.set_postfix_str(
                f"hits={stats['hits']} misses={stats['misses']}"
            )

    if changed and not dry_run:
        path.write_text(
            json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    return stats


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--id",    type=int, default=None,
                        help="Process only this survey_id.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only surveys with survey_id <= LIMIT (inclusive, id-based).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write changes back to the generation JSONs.")
    args = parser.parse_args()

    setup_logging("enrich_arxiv_ids")

    # Reserve stderr for tqdm — logs go to the file only.
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

    files = sorted(GEN_DIR.glob("*.json"), key=_numeric_key)
    if args.id is not None:
        files = [f for f in files if f.stem == str(args.id)]
    elif args.limit is not None:
        # id-based inclusive filter: survey_id <= LIMIT. See veriscore's note.
        files = [
            f for f in files
            if f.stem.isdigit() and int(f.stem) <= args.limit
        ]
    if not files:
        logger.warning("no generation JSONs to process (filters produced empty set)")
        return 0

    corpus_titles = _load_corpus_titles()

    totals = {
        "files":             0,
        "refs_total":        0,
        "candidates_total":  0,
        "hits":              0,
        "misses":            0,
        "no_corpus_title":   0,
    }

    outer = tqdm(
        files, desc="surveys", unit="surv",
        position=0, dynamic_ncols=True,
    )
    # Reusable inner bar for per-arxiv-lookup progress. total=1 is a harmless
    # placeholder — enrich_file calls reset() immediately with the real count.
    inner = tqdm(
        total=1, desc="arxiv-lookup", unit="ref",
        position=1, leave=False, dynamic_ncols=True,
    )
    try:
        for f in outer:
            outer.set_description(f"surveys [{f.stem}]")
            try:
                stats = enrich_file(
                    f, corpus_titles,
                    dry_run=args.dry_run, inner_bar=inner,
                )
            except Exception as e:
                logger.exception("%s failed: %s", f.name, e)
                continue
            totals["files"]            += 1
            totals["refs_total"]       += stats["refs_total"]
            totals["candidates_total"] += stats["candidates"]
            totals["hits"]             += stats["hits"]
            totals["misses"]           += stats["misses"]
            totals["no_corpus_title"]  += stats["no_corpus_title"]
            if stats["candidates"]:
                logger.info(
                    "[sid=%s] refs=%-3d candidates=%-2d hits=%-2d misses=%-2d",
                    f.stem, stats["refs_total"], stats["candidates"],
                    stats["hits"], stats["misses"],
                )
            outer.set_postfix_str(
                f"cand={totals['candidates_total']} "
                f"hits={totals['hits']} "
                f"misses={totals['misses']}"
            )
    finally:
        outer.close()
        inner.close()

    logger.info("=" * 60)
    logger.info("Summary: %s", json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
