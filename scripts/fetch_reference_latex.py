#!/usr/bin/env python3
"""
scripts/fetch_reference_latex.py

For each of the 205 SurGE reference surveys:
  1. Clean the survey title from LaTeX artifacts (\\\\, \\cmd{...}, ~, etc.)
  2. Search arxiv API by cleaned title, verify match by first author
  3. Download source tarball from arxiv
  4. Extract to datasets/surge/latex_src/<arxiv_id>/

Results:
  datasets/surge/latex_src/
    arxiv_cache.json          — {survey_id: {arxiv_id, found_title, score, strategy} | null}
    fetch_failures.log        — structured log of failures (one JSON per line)
    <arxiv_id>/               — extracted tarball contents
      *.tex, *.bib, figures/, ...

Usage:
    python scripts/fetch_reference_latex.py                        # all 205 surveys
    python scripts/fetch_reference_latex.py --limit 5              # first N surveys only
    python scripts/fetch_reference_latex.py --id 0                 # single survey_id
    python scripts/fetch_reference_latex.py --retry-not-found      # retry cached nulls
    python scripts/fetch_reference_latex.py --retry-not-found --id 4

Design notes:
  - Title cleaning now strips `\\\\` (LaTeX linebreak), `~`, LaTeX commands, braces.
    This was the main cause of not-found cases in the old version (e.g. survey 4
    "Deep Learning for Image Super-resolution:\\\\A Survey").
  - Multi-strategy search: exact quoted title → unquoted title words → title+first-author.
    First match with score >= ACCEPT_SCORE and valid author-check wins.
  - Retry with exponential backoff on network errors.
  - Cache stores the strategy that produced the match (for auditability).
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import shutil
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

import requests
from tqdm import tqdm

ROOT     = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.log_setup import setup_logging  # noqa: E402

SURVEYS  = ROOT / "datasets" / "SurGE" / "surveys.json"
OUT_DIR  = ROOT / "datasets" / "surge" / "latex_src"
CACHE    = OUT_DIR / "arxiv_cache.json"
FAILLOG  = OUT_DIR / "fetch_failures.log"

ARXIV_SEARCH = "https://export.arxiv.org/api/query"
ARXIV_SRC    = "https://arxiv.org/src/{arxiv_id}"
ATOM_NS      = "http://www.w3.org/2005/Atom"

# Pacing (be polite to arxiv — their guidelines say 3s between requests)
SEARCH_DELAY   = 3.2
DOWNLOAD_DELAY = 3.2
MAX_RETRIES    = 3
BACKOFF_BASE   = 2.0

# Matching thresholds
ACCEPT_SCORE        = 0.85   # below this — not accepted even with author match
STRONG_MATCH_SCORE  = 0.98   # exact or near-exact title match, trust fully
AUTHOR_REQUIRED_BELOW = STRONG_MATCH_SCORE  # require author-check when score < 0.98

logger = logging.getLogger(__name__)


# ── Title cleaning ─────────────────────────────────────────────────────────────

def clean_title(raw: str) -> str:
    """Strip LaTeX artifacts from a survey title for arxiv search.

    Handles:
      - `\\\\` (LaTeX line break)          → space
      - `\\cmd{...}` (LaTeX commands)      → contents preserved, command stripped
      - `{...}` (grouping braces)          → contents preserved
      - `~` (non-breaking space)           → space
      - `$...$` (inline math)              → contents preserved
      - multiple whitespace                → collapsed
      - trailing periods / colons          → stripped
    """
    t = raw

    # LaTeX linebreaks → space (this was the main bug — unhandled)
    t = re.sub(r"\\\\(?:\[[^\]]*\])?", " ", t)

    # LaTeX commands with arguments: \cmd{text} → text
    t = re.sub(r"\\[a-zA-Z]+\*?\s*\{([^{}]*)\}", r"\1", t)
    # LaTeX commands without arguments: \alpha, \ldots → (removed)
    t = re.sub(r"\\[a-zA-Z]+\*?", "", t)

    # Grouping braces
    t = t.replace("{", "").replace("}", "")

    # Non-breaking spaces
    t = t.replace("~", " ")

    # Inline math $...$
    t = re.sub(r"\$([^$]*)\$", r"\1", t)

    # Collapse whitespace, strip punctuation edges
    t = re.sub(r"\s+", " ", t).strip()
    t = t.strip(" .:;,")

    return t


def normalize(text: str) -> str:
    """Alphanumeric-only lowercase for fuzzy comparison."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def title_score(query: str, candidate: str) -> float:
    """Similarity between two titles in [0.0, 1.0]."""
    q_norm = normalize(query)
    c_norm = normalize(candidate)
    if not q_norm or not c_norm:
        return 0.0
    if q_norm == c_norm:
        return 1.0
    if q_norm in c_norm or c_norm in q_norm:
        return 0.9
    # Word Jaccard
    q_words = set(normalize(query).split()) if " " in query else {q_norm}
    c_words = set(normalize(candidate).split()) if " " in candidate else {c_norm}
    if q_words and c_words:
        return len(q_words & c_words) / max(len(q_words), len(c_words))
    return 0.0


def author_match(survey_authors: list[str], entry_authors: list[str]) -> bool:
    """Check if at least one survey author appears in the arxiv entry's author list.

    Uses last-name matching to tolerate formatting differences:
      - "John Doe"    → "doe"  (last token)
      - "J. Doe"      → "doe"  (last alphabetic token, skipping initials)
      - "Doe, John"   → "doe"  (token before comma)
    """
    if not survey_authors or not entry_authors:
        return False

    def last_name(name: str) -> str | None:
        # "Doe, John" → take part before comma
        if "," in name:
            head = name.split(",", 1)[0]
            tokens = re.findall(r"[A-Za-zА-Яа-я]{2,}", head)
            if tokens:
                return tokens[-1].lower()
        # "Jane Doe" / "J. Doe" → last token of length ≥ 2 (skips initials like "J")
        tokens = re.findall(r"[A-Za-zА-Яа-я]{2,}", name)
        if tokens:
            return tokens[-1].lower()
        return None

    sv = {ln for n in survey_authors if (ln := last_name(n))}
    ev = {ln for n in entry_authors  if (ln := last_name(n))}
    return bool(sv & ev)


# ── Network helpers ────────────────────────────────────────────────────────────

def fetch_with_retry(url: str, *, params: dict | None = None, stream: bool = False,
                     timeout: int = 30) -> requests.Response | None:
    """GET with exponential backoff. Returns None if all retries fail."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=timeout, stream=stream)
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            if attempt == MAX_RETRIES - 1:
                logger.warning("network failure for %s (%s): %s", url, params, e)
                return None
            backoff = BACKOFF_BASE ** attempt
            logger.info("retry %d/%d after %.1fs (err=%s)", attempt + 1, MAX_RETRIES, backoff, e)
            time.sleep(backoff)
    return None


def _parse_arxiv_response(xml_text: str) -> list[dict]:
    """Parse arxiv Atom feed → list of {arxiv_id, title, authors, year}."""
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning("arxiv XML parse error: %s", e)
        return out

    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        entry_id = (entry.findtext(f"{{{ATOM_NS}}}id", "") or "").strip()
        m = re.search(r"arxiv\.org/abs/([^v\s]+)", entry_id)
        if not m:
            continue

        title = re.sub(r"\s+", " ", (entry.findtext(f"{{{ATOM_NS}}}title", "") or "").strip())
        published = entry.findtext(f"{{{ATOM_NS}}}published", "") or ""
        authors = [
            (a.findtext(f"{{{ATOM_NS}}}name", "") or "").strip()
            for a in entry.findall(f"{{{ATOM_NS}}}author")
        ]
        out.append({
            "arxiv_id": m.group(1),
            "title":    title,
            "authors":  authors,
            "year":     published[:4],
        })
    return out


# ── Search strategies ──────────────────────────────────────────────────────────

def _query_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Low-level arxiv query. Returns parsed candidates."""
    params = {
        "search_query": query,
        "max_results":  max_results,
        "sortBy":       "relevance",
    }
    r = fetch_with_retry(ARXIV_SEARCH, params=params, timeout=20)
    if r is None:
        return []
    return _parse_arxiv_response(r.text)


def _pick_best(candidates: list[dict], clean: str, year: str,
               survey_authors: list[str]) -> tuple[dict, float] | None:
    """Score candidates, apply author-check for ambiguous ones, return (best, score)."""
    best, best_score = None, 0.0
    for c in candidates:
        score = title_score(clean, c["title"])

        # Year bonus (small — just breaks ties)
        if year and c["year"] == year:
            score = min(1.0, score + 0.02)

        # For weak-to-medium matches, require author overlap
        if score < AUTHOR_REQUIRED_BELOW and not author_match(survey_authors, c["authors"]):
            continue

        if score > best_score:
            best_score = score
            best = c

    if best and best_score >= ACCEPT_SCORE:
        return best, best_score
    return None


# English stopwords to drop in loose fallback searches (keeps arxiv query focused
# on content words, which tolerates typos and punctuation differences like
# "Web Pages" vs "Webpages").
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with",
    "from", "at", "by", "as", "is", "are", "via", "using", "towards",
    "survey", "review", "overview", "brief", "comprehensive", "systematic",
}


def _keywords(clean: str) -> list[str]:
    """Extract content keywords from a cleaned title (drop stopwords, punctuation)."""
    # Strip ": A Survey"-style suffix first (common pattern)
    head = re.split(r":", clean, maxsplit=1)[0]
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]*", head)
    kept = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) >= 2]
    return kept if kept else tokens  # don't strip everything


def search_arxiv(survey: dict) -> dict | None:
    """Multi-strategy arxiv search for a single survey.

    Tries strategies in order, returns the first acceptable match.
    Returned dict: {arxiv_id, found_title, score, strategy}.
    """
    raw_title      = survey["survey_title"]
    clean          = clean_title(raw_title)
    year           = survey.get("year", "") or ""
    survey_authors = survey.get("authors", []) or []

    if not clean:
        logger.warning("empty title after cleaning: %r", raw_title)
        return None

    keywords = _keywords(clean)

    strategies = [
        # 1. Exact quoted full title — the most precise, works for ~95% of cases.
        ("exact_quoted", f'ti:"{clean}"'),
    ]
    # 2. Loose keyword title search — tolerates punctuation / stopword diffs
    #    (e.g. "Web Pages" vs "Webpages"). Only if we have ≥2 keywords.
    if len(keywords) >= 2:
        kw_query = " ".join(keywords)
        strategies.append(("keywords_title", f"ti:({kw_query})"))

    # 3. Title + first-author surname (only if survey has authors)
    if survey_authors:
        first_last = re.findall(r"[A-Za-z]{2,}", survey_authors[0])
        if first_last:
            surname = first_last[-1]  # last alphabetic token = last name
            strategies.append(("title_author", f'ti:"{clean}" AND au:{surname}'))

    for strategy, query in strategies:
        candidates = _query_arxiv(query)
        time.sleep(SEARCH_DELAY)
        if not candidates:
            continue

        picked = _pick_best(candidates, clean, year, survey_authors)
        if picked:
            best, score = picked
            return {
                "arxiv_id":    best["arxiv_id"],
                "found_title": best["title"],
                "score":       round(score, 3),
                "strategy":    strategy,
            }

    return None


# ── Source download ────────────────────────────────────────────────────────────

def download_source(arxiv_id: str, out_dir: Path) -> tuple[bool, str]:
    """Download and extract arxiv source tarball. Returns (ok, reason)."""
    dest = out_dir / arxiv_id
    if dest.exists() and any(dest.iterdir()):
        return True, "already_extracted"

    url = ARXIV_SRC.format(arxiv_id=arxiv_id)
    r = fetch_with_retry(url, timeout=60)
    if r is None:
        return False, "network_error"

    content_type = r.headers.get("Content-Type", "")
    raw = r.content
    dest.mkdir(parents=True, exist_ok=True)

    # Try tar.gz / gzipped formats
    if "gzip" in content_type or "tar" in content_type or raw[:2] == b"\x1f\x8b":
        try:
            with gzip.open(BytesIO(raw)) as gz:
                inner = gz.read()
            try:
                with tarfile.open(fileobj=BytesIO(inner)) as tar:
                    tar.extractall(dest, filter="data")
                return True, "tar_gz"
            except tarfile.TarError:
                # Single .tex file
                (dest / "main.tex").write_bytes(inner)
                return True, "single_tex_gz"
        except Exception:
            pass

        # Try direct tarfile on raw (un-gzipped .tar)
        try:
            with tarfile.open(fileobj=BytesIO(raw)) as tar:
                tar.extractall(dest, filter="data")
            return True, "tar"
        except tarfile.TarError:
            pass

    if "pdf" in content_type:
        shutil.rmtree(dest)
        return False, "pdf_only_no_latex"

    # Unknown — save raw for later inspection
    (dest / "source.bin").write_bytes(raw)
    return False, f"unknown_content_type:{content_type}"


# ── Cache I/O ──────────────────────────────────────────────────────────────────

def load_cache() -> dict:
    if CACHE.exists():
        with open(CACHE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def log_failure(survey_id: str, reason: str, **extra) -> None:
    """Append a structured failure record to fetch_failures.log."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"survey_id": survey_id, "reason": reason, **extra}
    with open(FAILLOG, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only surveys with survey_id <= LIMIT (inclusive, id-based; default: all 205).")
    parser.add_argument("--id", type=int, default=None,
                        help="Process a single survey_id")
    parser.add_argument("--retry-not-found", action="store_true",
                        help="Re-query arxiv for surveys cached as null (not-found)")
    args = parser.parse_args()

    setup_logging("fetch_reference_latex")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SURVEYS) as f:
        surveys = json.load(f)

    # surveys.json is NOT sorted by survey_id in-file. We iterate strictly by
    # ascending survey_id so that:
    #   (a) the same id → same output path `results/generations/SurGE_*/<id>.json`
    #       convention used everywhere else in the codebase;
    #   (b) `--limit N` means "first N surveys by id", not "first N in file order".
    surveys.sort(key=lambda s: s["survey_id"])

    if args.id is not None:
        surveys = [s for s in surveys if s["survey_id"] == args.id]
    elif args.limit is not None:
        # id-based inclusive filter: survey_id <= LIMIT. Natural on sparse id
        # sets where "first N by position" would drop a different subset.
        surveys = [s for s in surveys if s["survey_id"] <= args.limit]

    cache = load_cache()

    stats = {"found_new": 0, "found_cached": 0, "not_found": 0,
             "downloaded": 0, "already_there": 0, "download_failed": 0}

    bar = tqdm(surveys, unit="survey")
    for s in bar:
        sid   = str(s["survey_id"])
        title = s["survey_title"]

        bar.set_description(f"[{sid}] {title[:40]}")

        # Step 1: find arxiv_id ───────────────────────────────────────────────
        cached = cache.get(sid)

        if cached is None and sid in cache and not args.retry_not_found:
            # Previously cached as not-found; skip unless retry requested
            stats["not_found"] += 1
            continue

        if cached is not None:
            arxiv_id = cached["arxiv_id"]
            stats["found_cached"] += 1
        else:
            result = search_arxiv(s)
            if result is None:
                tqdm.write(f"  [NOT FOUND] sid={sid}: {clean_title(title)[:60]}")
                cache[sid] = None
                save_cache(cache)
                log_failure(sid, "arxiv_search_not_found",
                            cleaned_title=clean_title(title), raw_title=title)
                stats["not_found"] += 1
                continue

            arxiv_id = result["arxiv_id"]
            cache[sid] = result
            save_cache(cache)
            stats["found_new"] += 1
            tqdm.write(
                f"  [FOUND] sid={sid} → {arxiv_id} "
                f"(score={result['score']:.2f}, {result['strategy']}): "
                f"{result['found_title'][:60]}"
            )

        # Step 2: download & extract ──────────────────────────────────────────
        dest = OUT_DIR / arxiv_id
        if dest.exists() and any(dest.iterdir()):
            stats["already_there"] += 1
            continue

        ok, reason = download_source(arxiv_id, OUT_DIR)
        time.sleep(DOWNLOAD_DELAY)

        if ok:
            n_files = sum(1 for _ in dest.rglob("*") if _.is_file())
            tqdm.write(f"  [OK] {arxiv_id} → {n_files} files ({reason})")
            stats["downloaded"] += 1
        else:
            tqdm.write(f"  [DL FAIL] sid={sid} {arxiv_id}: {reason}")
            log_failure(sid, "download_failed", arxiv_id=arxiv_id, detail=reason)
            stats["download_failed"] += 1

    print()
    print("─" * 50)
    print(f"  Found (new)       : {stats['found_new']}")
    print(f"  Found (cached)    : {stats['found_cached']}")
    print(f"  Not found         : {stats['not_found']}")
    print(f"  Downloaded (new)  : {stats['downloaded']}")
    print(f"  Already extracted : {stats['already_there']}")
    print(f"  Download failed   : {stats['download_failed']}")
    print()
    print(f"  Cache   : {CACHE}")
    print(f"  Failures: {FAILLOG}")


if __name__ == "__main__":
    main()
