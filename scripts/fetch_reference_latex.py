#!/usr/bin/env python3
"""
scripts/fetch_reference_latex.py

For each of the 205 SurGE reference surveys:
  1. Find arxiv_id via arxiv search API (by title)
  2. Download source tarball from arxiv
  3. Extract to datasets/surge/latex_src/<arxiv_id>/

Results:
  datasets/surge/latex_src/
    arxiv_cache.json          — {survey_id: {arxiv_id, title, score}} cache
    <arxiv_id>/               — extracted tarball contents
      *.tex, *.bib, figures/, ...

Usage:
    python scripts/fetch_reference_latex.py
    python scripts/fetch_reference_latex.py --limit 5      # first N surveys only
    python scripts/fetch_reference_latex.py --id 0         # single survey_id
"""
import argparse
import gzip
import json
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
SURVEYS  = ROOT / "datasets" / "surge" / "surveys.json"
OUT_DIR  = ROOT / "datasets" / "surge" / "latex_src"
CACHE    = OUT_DIR / "arxiv_cache.json"

ARXIV_SEARCH  = "https://export.arxiv.org/api/query"
ARXIV_SRC     = "https://arxiv.org/src/{arxiv_id}"

# Be polite to arxiv
SEARCH_DELAY  = 1.5   # seconds between search API calls
DOWNLOAD_DELAY = 2.0  # seconds between source downloads


# ── Helpers ────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip non-alphanum for fuzzy title matching."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def title_score(query: str, candidate: str) -> float:
    """Simple overlap score between normalized titles."""
    q_words = set(normalize(query).split()) if " " in query else {normalize(query)}
    c_words = set(normalize(candidate).split()) if " " in candidate else {normalize(candidate)}
    q_norm = normalize(query)
    c_norm = normalize(candidate)
    # Exact match
    if q_norm == c_norm:
        return 1.0
    # Prefix/substring match
    if q_norm in c_norm or c_norm in q_norm:
        return 0.9
    # Word overlap
    if q_words and c_words:
        return len(q_words & c_words) / max(len(q_words), len(c_words))
    return 0.0


def search_arxiv(title: str, year: str) -> dict | None:
    """Search arxiv API by title, return best match or None."""
    # Strip LaTeX from title
    clean_title = re.sub(r"\\[a-zA-Z]+\{?", "", title).replace("}", "").strip()
    clean_title = re.sub(r"\s+", " ", clean_title)

    params = {
        "search_query": f'ti:"{clean_title}"',
        "max_results":  5,
        "sortBy":       "relevance",
    }
    try:
        resp = requests.get(ARXIV_SEARCH, params=params, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"    [WARN] arxiv search failed: {e}", flush=True)
        return None

    ns = "http://www.w3.org/2005/Atom"
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        print(f"    [WARN] XML parse error: {e}", flush=True)
        return None

    best = None
    best_score = 0.0

    for entry in root.findall(f"{{{ns}}}entry"):
        entry_id  = entry.findtext(f"{{{ns}}}id", "").strip()
        entry_title = entry.findtext(f"{{{ns}}}title", "").strip()
        entry_title = re.sub(r"\s+", " ", entry_title)

        # arxiv_id: extract from URL like https://arxiv.org/abs/1911.02794v2
        m = re.search(r"arxiv\.org/abs/([^v\s]+)", entry_id)
        if not m:
            continue
        arxiv_id = m.group(1)

        score = title_score(clean_title, entry_title)

        # Year bonus: entry published year should match
        published = entry.findtext(f"{{{ns}}}published", "")
        entry_year = published[:4] if published else ""
        if entry_year == year:
            score = min(1.0, score + 0.05)

        if score > best_score:
            best_score = score
            best = {"arxiv_id": arxiv_id, "found_title": entry_title, "score": score}

    if best and best_score >= 0.7:
        return best
    return None


def download_source(arxiv_id: str, out_dir: Path) -> bool:
    """Download and extract arxiv source tarball to out_dir/<arxiv_id>/."""
    dest = out_dir / arxiv_id
    if dest.exists() and any(dest.iterdir()):
        return True  # already downloaded

    url = ARXIV_SRC.format(arxiv_id=arxiv_id)
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"    [WARN] download failed for {arxiv_id}: {e}", flush=True)
        return False

    content_type = resp.headers.get("Content-Type", "")
    raw = resp.content

    dest.mkdir(parents=True, exist_ok=True)

    # Try tar.gz
    if "gzip" in content_type or "tar" in content_type or raw[:2] == b"\x1f\x8b":
        try:
            # Could be plain .gz (single file) or .tar.gz
            with gzip.open(BytesIO(raw)) as gz:
                inner = gz.read()
            # Check if the inner is a tar
            try:
                with tarfile.open(fileobj=BytesIO(inner)) as tar:
                    tar.extractall(dest, filter="data")
                return True
            except tarfile.TarError:
                # Single .tex file
                (dest / "main.tex").write_bytes(inner)
                return True
        except Exception:
            pass

        # Try direct tarfile on raw
        try:
            with tarfile.open(fileobj=BytesIO(raw)) as tar:
                tar.extractall(dest, filter="data")
            return True
        except tarfile.TarError:
            pass

    # PDF fallback (no LaTeX source available)
    if "pdf" in content_type:
        print(f"    [INFO] {arxiv_id} — only PDF source available (no LaTeX)", flush=True)
        shutil.rmtree(dest)
        return False

    # Unknown — save raw and hope for the best
    (dest / "source.bin").write_bytes(raw)
    print(f"    [WARN] {arxiv_id} — unknown content type: {content_type}", flush=True)
    return False


def load_cache() -> dict:
    if CACHE.exists():
        with open(CACHE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N surveys")
    parser.add_argument("--id",    type=int, default=None, help="Process single survey_id")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SURVEYS) as f:
        surveys = json.load(f)

    if args.id is not None:
        surveys = [s for s in surveys if s["survey_id"] == args.id]
    elif args.limit:
        surveys = surveys[:args.limit]

    cache = load_cache()

    stats = {"found": 0, "not_found": 0, "downloaded": 0, "failed": 0, "skipped": 0}

    bar = tqdm(surveys, unit="survey")
    for s in bar:
        sid   = str(s["survey_id"])
        title = s["survey_title"]
        year  = s.get("year", "")

        bar.set_description(f"[{sid}] {title[:40]}")

        # ── Step 1: find arxiv_id ──────────────────────────────────────────────
        if sid in cache:
            entry = cache[sid]
            if entry is None:
                stats["not_found"] += 1
                continue
            arxiv_id = entry["arxiv_id"]
            stats["skipped"] += 1
        else:
            result = search_arxiv(title, year)
            time.sleep(SEARCH_DELAY)

            if result is None:
                tqdm.write(f"  [NOT FOUND] survey_id={sid}: {title[:60]}")
                cache[sid] = None
                save_cache(cache)
                stats["not_found"] += 1
                continue

            arxiv_id = result["arxiv_id"]
            cache[sid] = {"arxiv_id": arxiv_id, "found_title": result["found_title"], "score": result["score"]}
            save_cache(cache)
            stats["found"] += 1
            tqdm.write(f"  [FOUND] survey_id={sid} → {arxiv_id} (score={result['score']:.2f}): {result['found_title'][:60]}")

        # ── Step 2: download & extract ─────────────────────────────────────────
        dest = OUT_DIR / arxiv_id
        if dest.exists() and any(dest.iterdir()):
            continue  # already extracted

        ok = download_source(arxiv_id, OUT_DIR)
        time.sleep(DOWNLOAD_DELAY)

        if ok:
            stats["downloaded"] += 1
            n_files = sum(1 for _ in dest.rglob("*") if _.is_file())
            tqdm.write(f"  [OK] {arxiv_id} → {n_files} files extracted")
        else:
            stats["failed"] += 1

    print(f"\n{'─' * 50}")
    print(f"  Found in arxiv  : {stats['found'] + stats['skipped']}")
    print(f"  Not found       : {stats['not_found']}")
    print(f"  Downloaded      : {stats['downloaded']}")
    print(f"  Failed download : {stats['failed']}")
    print(f"  Cache hits      : {stats['skipped']}")
    print(f"\n  Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
