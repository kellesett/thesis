#!/usr/bin/env python3
"""
scripts/arxiv_title_search.py

Title-first arxiv lookup for bibliography entries.

Used by ``scripts/enrich_surge_reference.py`` to fill ``arxiv_id`` / ``url`` /
``canonical_title`` in SurGE_reference generations, where reference entries
are extracted from ``\\bibitem`` blocks and rarely carry an explicit arxiv URL.

Design
------
* Reuses title cleaning + scoring helpers from ``fetch_reference_latex`` —
  same normalization is used for both survey-level and bibitem-level lookups,
  so thresholds are comparable.
* Adds an extra **query-side** cleaning pass: bibitem titles extracted by
  ``parse_reference_md.py`` often carry a venue tail (e.g. ``"… In Proceedings
  of the Fifth Conference on Machine Translation. 940-946"``). Feeding that
  verbatim to ``ti:"…"`` fails because arxiv does a literal substring match.
  :func:`clean_title_for_query` truncates at common venue markers and drops
  trailing page ranges, keeping only the title proper.
* Global disk cache keyed on the normalized cleaned title. Both hits and
  misses are cached — re-running an enrichment pass is free of network I/O
  for entries we have already seen.
* Respects arxiv's rate-limit guideline (≥3 s between queries) via a process-
  global timestamp. Retries with exponential backoff inherited from
  ``fetch_with_retry``.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the same network/parsing helpers the survey-level fetcher uses so
# scoring semantics stay identical across the two code paths.
from scripts.fetch_reference_latex import (  # noqa: E402
    SEARCH_DELAY,
    STRONG_MATCH_SCORE,
    _query_arxiv,
    normalize,
    title_score,
)

DEFAULT_MIN_SCORE = STRONG_MATCH_SCORE  # 0.98 — effectively "exact after normalize"
CACHE_PATH = ROOT / "datasets" / "surge" / "latex_src" / "ref_arxiv_cache.json"


# ── Query-side title cleaning ─────────────────────────────────────────────────

_YEAR_PREFIX_RE = re.compile(r"^\s*\(?\d{4}\)?\.?\s*")
_PAGE_RANGE_RE  = re.compile(r"\s+\d+\s*[\-–—]+\s*\d+\s*\.?\s*$")

# Tokens that almost always introduce the venue part of a bibitem and should
# truncate the title. Matched case-insensitively, only when they appear at
# least 10 chars into the string (to avoid truncating titles that legitimately
# begin with "Proceedings" etc.).
_VENUE_MARKERS = re.compile(
    r"""(?ix)\s+(?:
        In\s+(?:Proceedings|Advances|Proc\.?|Workshop|IEEE|ACM|20\d{2}|19\d{2})
        | Proceedings\s+of
        | Proc\.\s+of
        | Conference\s+on
        | \[Online\]
        | Available:
        | pp\.\s
        | vol\.\s
        | doi:
        | arXiv:
        | URL\s
    )"""
)


def clean_title_for_query(title: str) -> str:
    """Aggressively clean a bibitem title for an arxiv ``ti:"…"`` query.

    Heuristics, applied in order:
        1. Drop a leading ``(YEAR)`` or ``YEAR.`` prefix (common in ACM styles).
        2. Truncate at the first venue marker (``"In Proceedings …"``,
           ``"vol. …"``, ``"[Online] …"`` etc.) when it sits past char 10.
        3. Drop trailing page ranges like ``680-686``.
        4. Collapse whitespace and strip edge punctuation.
        5. Hard-cap at 20 words — very long strings almost always mean the
           extractor captured a whole sentence, and arxiv's ``ti:`` search
           becomes brittle on them.

    Returns an empty string for empty / noise-only input.
    """
    if not title:
        return ""
    t = title.strip()
    t = _YEAR_PREFIX_RE.sub("", t)
    m = _VENUE_MARKERS.search(t)
    if m and m.start() > 10:
        t = t[: m.start()]
    t = _PAGE_RANGE_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip(" .,;:-\"'")
    words = t.split()
    if len(words) > 20:
        t = " ".join(words[:20])
    return t


# ── Cache ─────────────────────────────────────────────────────────────────────


def _cache_key(cleaned_title: str) -> str:
    """Canonical key for the cache: alnum-only lowercase of a cleaned title."""
    return normalize(cleaned_title)


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Corrupt %s — starting fresh", CACHE_PATH)
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Search ────────────────────────────────────────────────────────────────────

_cache: dict | None = None
_last_request_at: float = 0.0


def _rate_limit() -> None:
    """Block until the arxiv rate-limit window has elapsed."""
    global _last_request_at
    now = time.monotonic()
    wait = SEARCH_DELAY - (now - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def search_by_title(
    title: str,
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    use_cache: bool = True,
) -> dict | None:
    """Search arxiv by a bibitem title, return a high-confidence match or None.

    Args:
        title:     Raw title string (from a ``\\bibitem`` extraction).
                   Cleaned internally via :func:`clean_title_for_query`.
        min_score: Minimum :func:`title_score` required. Defaults to the
                   same STRONG_MATCH_SCORE (0.98) used by the survey-level
                   fetcher.
        use_cache: If True, hit the disk cache first and update it on new
                   lookups. Disable only for debugging.

    Returns:
        ``{arxiv_id, canonical_title, url, score, strategy}`` on hit, or
        ``None`` on miss / below-threshold / empty query.
    """
    global _cache

    query_title = clean_title_for_query(title)
    if len(query_title) < 8:  # too short → noise
        return None

    key = _cache_key(query_title)
    if not key:
        return None

    if use_cache:
        if _cache is None:
            _cache = _load_cache()
        cached = _cache.get(key)
        if cached == "MISS":
            return None
        if isinstance(cached, dict):
            return cached

    _rate_limit()
    candidates = _query_arxiv(f'ti:"{query_title}"', max_results=5)

    result: dict | None = None
    if candidates:
        best, best_score = None, 0.0
        for c in candidates:
            score = title_score(query_title, c["title"])
            if score > best_score:
                best_score, best = score, c
        if best and best_score >= min_score:
            result = {
                "arxiv_id":        best["arxiv_id"],
                "canonical_title": best["title"],
                "url":             f"https://arxiv.org/abs/{best['arxiv_id']}",
                "score":           round(best_score, 3),
                "strategy":        "title_exact_quoted",
            }

    if use_cache:
        _cache[key] = result if result else "MISS"
        _save_cache(_cache)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Standalone arxiv search by bibitem title (debug tool)."
    )
    ap.add_argument("title", help="Raw bibitem title string.")
    ap.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print(f"cleaned query: {clean_title_for_query(args.title)!r}")
    result = search_by_title(
        args.title,
        min_score=args.min_score,
        use_cache=not args.no_cache,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(_cli())
