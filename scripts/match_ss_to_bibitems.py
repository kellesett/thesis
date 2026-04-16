#!/usr/bin/env python3
"""
scripts/match_ss_to_bibitems.py

Batch-match bibliography entries between two sources across SurGE surveys:

* LaTeX ``\\bibitem`` blocks parsed from each survey's ``merged.tex``
  (produced by ``scripts/merge_latex.sh``).
* Semantic Scholar ``/paper/ARXIV:<id>/references`` response, keyed on the
  survey's arxiv_id (resolved via ``datasets/surge/latex_src/arxiv_cache.json``).

Output: one standalone mapping JSON per survey, at
``datasets/surge/latex_src/<arxiv_id>/ss_matches_<mode>.json``. A later
integration step reads it to patch
``results/generations/SurGE_reference/<sid>.json`` (renumbering citation
markers and swapping in clean SS metadata).

Two matching strategies, selected by ``--mode``:

* ``string`` (default) — normalized longest-common-substring. Both the
  full bibitem raw text and each SS title are lowercased and reduced to
  ``[a-z0-9]+`` tokens joined by single spaces; ``score`` is the LCS length
  divided by the normalized SS-title length (so bibitem clutter doesn't
  deflate the score). Bipartite assignment is greedy by descending score with
  a configurable ``--threshold`` (default 0.85). Deterministic, no network.

* ``llm`` — one OpenRouter chat call per SS title with the whole bibitem list
  numbered in the prompt. The model returns either a bibitem number or null.
  Calls are parallelised via a thread pool (``--parallel``, default 50).
  Multiple SS titles may land on the same bibitem; conflicts are resolved by
  keeping the SS index that finished first (stable sort by SS-position) and
  logged in the output. Cost + token usage is reported per-survey in the
  output JSON and aggregated across the whole run.

Progress display:

* ``string`` mode — one tqdm bar iterating surveys.
* ``llm`` mode — hierarchical: outer bar iterating surveys (postfix shows
  cumulative matched count and global tokens/cost), inner bar iterating LLM
  calls for the current survey (postfix shows per-survey tokens/cost). The
  inner bar is **reused** across surveys via ``tqdm.reset(total=...)`` so the
  number of visible bars stays at 2 regardless of how many surveys are
  processed.

Usage::

    # all surveys, string mode:
    python3 scripts/match_ss_to_bibitems.py --mode string

    # survey_id 0 only, LLM mode:
    OPENROUTER_API_KEY=... python3 scripts/match_ss_to_bibitems.py \\
        --mode llm --id 0 --parallel 50

    # first 40 surveys by ascending survey_id, LLM with a custom model:
    python3 scripts/match_ss_to_bibitems.py \\
        --mode llm --limit 40 --model google/gemma-3-27b-it --parallel 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics.utils import TokenCounter  # noqa: E402
from scripts.parse_reference_md import BibItem, parse_bibliography  # noqa: E402
from src.log_setup import setup_logging  # noqa: E402


# ── Constants ────────────────────────────────────────────────────────────────

SS_BASE   = "https://api.semanticscholar.org/graph/v1"
SS_FIELDS = "title,externalIds,authors,year"  # authors is a nested array; no sub-field syntax accepted
SS_LIMIT  = 100        # max page size allowed by SS graph API
SS_DELAY  = 3.2        # seconds between paginated SS requests (no-auth guideline)

OPENROUTER_URL = "https://openrouter.ai/api/v1"

DEFAULT_THRESHOLD = 0.85
DEFAULT_MODEL     = "google/gemma-3-27b-it"   # closest real OpenRouter Gemma; override via --model
DEFAULT_PARALLEL  = 50
DEFAULT_CACHE_DIR = ROOT / "datasets" / "surge" / "latex_src"

# Max chars of each bibitem we show to the LLM. 500 easily fits
# authors+title+venue while keeping the prompt under context limits at 50
# concurrent requests. Surveys with ~200 bibitems × 500 chars ≈ 100k chars =
# ~25k tokens per call, comfortably inside Gemma-3-27b's 128k window.
LLM_RAW_TRUNC = 500


# ── IO helper (kept local; avoids importing a private name) ───────────────────


def _read_tex(path: Path) -> str:
    """Decode a .tex file as UTF-8 with latin-1 fallback (pre-2020 arxiv quirk)."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        logger.debug("latin-1 fallback on %s", path)
        return data.decode("latin-1", errors="replace")


# ── Semantic Scholar fetching ─────────────────────────────────────────────────


def fetch_ss_references(
    arxiv_id: str,
    cache_dir: Path,
    ss_api_key: str | None = None,
) -> list[dict]:
    """Fetch (or load cached) Semantic Scholar references for an arxiv paper.

    Cache lives at ``<cache_dir>/<arxiv_id>/ss_references.json`` and stores the
    flat list of ``citedPaper`` dicts — one per successfully-resolved
    reference. Empty entries (SS couldn't resolve the reference) are dropped
    here so callers don't need to re-check.

    Args:
        arxiv_id:   Bare arxiv id (``YYMM.NNNNN``).
        cache_dir:  Parent directory for the ``<arxiv_id>/ss_references.json``
                    cache file.
        ss_api_key: Optional Semantic Scholar API key. Without a key the
                    unauth limit is strict (~100 req / 5 min / IP) and you
                    will hit back-pressure almost immediately. Apply at
                    https://www.semanticscholar.org/product/api#api-key-form.

    Notes:
        SS occasionally returns the rate-limit message in the **body** with
        HTTP 200 (``{"code": "429", "message": ...}``). We detect and honor
        that exactly like a real 429.

    Returns:
        list of SS ``citedPaper`` dicts with at least ``title`` populated.
        Empty list if SS knows nothing about this arxiv_id.
    """
    paper_dir  = cache_dir / arxiv_id
    cache_file = paper_dir / "ss_references.json"
    if cache_file.exists():
        logger.debug("loading cached SS refs from %s", cache_file)
        return json.loads(cache_file.read_text(encoding="utf-8"))

    # ARXIV: is the current canonical prefix in the Graph API; arXiv: / arxiv:
    # are rejected with a 400 on some shards as of 2024+.
    url     = f"{SS_BASE}/paper/ARXIV:{arxiv_id}/references"
    headers = {"x-api-key": ss_api_key} if ss_api_key else {}

    out: list[dict] = []
    offset = 0
    first  = True
    while True:
        params = {"fields": SS_FIELDS, "limit": SS_LIMIT, "offset": offset}
        if not first:
            time.sleep(SS_DELAY)
        first = False

        r = requests.get(url, params=params, headers=headers, timeout=30)
        rate_limited = r.status_code == 429
        if not rate_limited and r.status_code == 200:
            # Check for the 200-with-429-in-body pattern (no JSON parse if not json).
            try:
                body = r.json()
            except ValueError:
                body = None
            if isinstance(body, dict) and str(body.get("code", "")) == "429":
                rate_limited = True
        if rate_limited:
            wait = 60 if ss_api_key else 300
            logger.info(
                "SS rate-limited (offset=%d, key=%s) — waiting %ds",
                offset, "yes" if ss_api_key else "no", wait,
            )
            time.sleep(wait)
            continue
        if r.status_code == 404:
            logger.warning("SS has no record for arxiv_id=%s", arxiv_id)
            break
        r.raise_for_status()
        data = r.json()
        page = [(e.get("citedPaper") or {}) for e in data.get("data", [])]
        out.extend([p for p in page if p.get("title")])
        next_off = data.get("next")
        if next_off is None:
            break
        offset = int(next_off)

    paper_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


# ── Normalization ─────────────────────────────────────────────────────────────


def _norm(s: str) -> str:
    """Lowercase + keep only ``[a-z0-9]`` tokens, joined by single spaces.

    Same normalization is applied to bibitem raw text and SS titles before
    any string comparison — keeps alphabet and digits, collapses everything
    else (LaTeX commands, punctuation, whitespace) to a single space.
    """
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# ── Variant 1: longest-common-substring ratio ─────────────────────────────────


def _lcs_ratio(ss_norm: str, bib_norm: str) -> float:
    """score = len(LCS(ss, bib)) / len(ss).  In [0, 1]; 1.0 == perfect containment.

    Dividing by ``len(ss)`` (not ``len(bib)``) is deliberate: bibitems contain
    authors, venue, years, etc. around the title, which we treat as permitted
    "noise" rather than evidence against a match.
    """
    if not ss_norm or not bib_norm:
        return 0.0
    m = SequenceMatcher(None, ss_norm, bib_norm).find_longest_match(
        0, len(ss_norm), 0, len(bib_norm),
    )
    return m.size / len(ss_norm)


def match_by_string(
    bibitems: list[BibItem],
    ss_refs: list[dict],
    threshold: float,
) -> dict[int, tuple[int, float]]:
    """Greedy bipartite matching by LCS ratio.

    Returns a mapping ``{bibitem_index: (ss_index, score)}``. Every bibitem
    appears at most once; every ss_ref appears at most once.
    """
    bib_norms = [_norm(b.raw) for b in bibitems]
    ss_norms  = [_norm(s.get("title") or "") for s in ss_refs]

    # Full N×M score matrix, filtered by threshold up front so the sort below
    # processes only viable candidates. N=80, M=80 → ~6.4k pairs — fine.
    pairs: list[tuple[float, int, int]] = []
    for j, sn in enumerate(ss_norms):
        if not sn:
            continue
        for k, bn in enumerate(bib_norms):
            sc = _lcs_ratio(sn, bn)
            if sc >= threshold:
                pairs.append((sc, k, j))
    # Sort by score desc, then by (k, j) for deterministic tie-breaking so
    # that re-running the same inputs yields exactly the same mapping.
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))

    taken_k: set[int] = set()
    taken_j: set[int] = set()
    mapping: dict[int, tuple[int, float]] = {}
    for score, k, j in pairs:
        if k in taken_k or j in taken_j:
            continue
        mapping[k] = (j, score)
        taken_k.add(k)
        taken_j.add(j)
    return mapping


# ── Variant 2: LLM-as-dispatcher over OpenRouter ──────────────────────────────


_LLM_SYSTEM = (
    "You match a single paper title to one entry from a numbered list of "
    "bibliography entries. Pick the entry number that refers to the SAME "
    "paper as the given title (venue/year/formatting differences are fine). "
    "If no entry clearly refers to the same paper, respond with null. "
    'Respond with exactly one JSON object and nothing else: '
    '{"match": <integer between 1 and N, or null>}.'
)

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _make_session(parallel: int) -> requests.Session:
    """HTTP session with a connection pool sized for the chosen concurrency.

    Default ``requests`` pool is 10 — at ``--parallel 50`` we'd otherwise see
    ``Connection pool is full, discarding connection`` warnings and suffer
    TCP re-handshakes. Sizing the pool to ``parallel`` avoids that.
    """
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=parallel, pool_maxsize=parallel)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _strip_json_fence(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _llm_pick_one(
    session: requests.Session,
    api_key: str,
    model: str,
    ss_title: str,
    bib_lines: list[str],
    token_counter: TokenCounter | None = None,
    max_retries: int = 3,
) -> int | None:
    """One LLM call; returns 0-indexed bibitem index or None.

    When ``token_counter`` is supplied, input/output tokens and the
    OpenRouter-reported cost are accumulated atomically. ``"usage":
    {"include": true}`` must be in the request body for the ``cost`` field to
    appear (OpenRouter omits it by default to save response bytes).
    """
    n = len(bib_lines)
    user = (
        f"Paper title:\n{ss_title}\n\n"
        f"Bibliography entries:\n" + "\n".join(bib_lines) +
        f'\n\nRespond with: {{"match": <integer 1..{n} or null>}}'
    )
    body = {
        "model":       model,
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user",   "content": user},
        ],
        "max_tokens":  30,
        "temperature": 0.0,
        # Ask OpenRouter to echo the dollar cost alongside the token counts.
        # Without this, usage.cost is absent from the response.
        "usage": {"include": True},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = session.post(
                f"{OPENROUTER_URL}/chat/completions",
                json=body, headers=headers, timeout=60,
            )
            r.raise_for_status()
            payload = r.json()
            content = payload["choices"][0]["message"]["content"] or ""
            content = _strip_json_fence(content)
            # Record usage on successful completion only (retries don't double-count
            # because we only enter this block after a clean 200 response).
            if token_counter is not None:
                usage = payload.get("usage") or {}
                token_counter.add(
                    int(usage.get("prompt_tokens")     or 0),
                    int(usage.get("completion_tokens") or 0),
                    cost=float(usage.get("cost")       or 0.0),
                )
            # Gemma sometimes trails a comment or leading whitespace — grab the
            # first {...} blob rather than json.loads the whole content.
            blob = _JSON_OBJECT_RE.search(content)
            if blob is None:
                raise ValueError(f"no JSON object in content: {content[:200]!r}")
            data = json.loads(blob.group(0))
            match = data.get("match")
            if match is None:
                return None
            if isinstance(match, int) and 1 <= match <= n:
                return match - 1
            # LLM returned a bogus number — treat as no match rather than retrying
            # (it's not going to self-correct on the same prompt).
            logger.debug("LLM returned out-of-range match=%r for title=%r",
                         match, ss_title[:60])
            return None
        except Exception as e:
            last_err = e
            backoff = 2.0 ** attempt
            logger.debug("LLM retry %d/%d (err=%s), backoff=%.1fs",
                         attempt + 1, max_retries, e, backoff)
            time.sleep(backoff)
    logger.warning("LLM permanently failed for ss_title=%r: %s",
                   ss_title[:60], last_err)
    return None


def match_by_llm(
    bibitems: list[BibItem],
    ss_refs: list[dict],
    *,
    model: str,
    api_key: str,
    session: requests.Session,
    executor: ThreadPoolExecutor,
    inner_bar: tqdm,
    global_tokens: TokenCounter,
) -> tuple[dict[int, tuple[int, float]], list[dict], TokenCounter]:
    """Parallel LLM matching over OpenRouter for a single survey.

    This function does NOT own the HTTP session, thread pool, or inner progress
    bar — those are created once in :func:`main` and reused across surveys so
    we keep a stable, two-line tqdm display regardless of survey count.

    Args:
        bibitems:       Parsed LaTeX bibitems.
        ss_refs:        Flat list of SS ``citedPaper`` dicts.
        model:          OpenRouter model slug.
        api_key:        OpenRouter API key.
        session:        Requests session with a connection pool sized to
                        ``executor``'s ``max_workers``.
        executor:       Long-lived ``ThreadPoolExecutor`` shared across surveys.
        inner_bar:      Long-lived tqdm bar. ``reset(total=...)`` is called at
                        the start of this survey; ``update(1)`` is called as
                        each LLM call completes.
        global_tokens:  Run-wide token/cost accumulator. The per-survey totals
                        are added to it once all calls for this survey have
                        completed.

    Returns:
        ``(mapping, conflicts, per_survey_tokens)`` — same shapes as before,
        plus a fresh :class:`TokenCounter` scoped to this survey only (also
        embedded into the output JSON under ``usage``).
    """
    bib_lines = [
        f"[{i + 1}] " + re.sub(r"\s+", " ", b.raw[:LLM_RAW_TRUNC]).strip()
        for i, b in enumerate(bibitems)
    ]
    jobs = [
        (j, (s.get("title") or "").strip())
        for j, s in enumerate(ss_refs)
        if (s.get("title") or "").strip()
    ]

    per_tokens = TokenCounter()
    # Reset the bar for this survey. total=0 shows 0/0 (empty survey guard).
    inner_bar.reset(total=max(len(jobs), 1))
    inner_bar.set_postfix_str(f"surv: {per_tokens.fmt()}  total: {global_tokens.fmt()}")

    if not jobs:
        inner_bar.update(1)  # close out the dummy total so the bar reads "1/1"
        return {}, [], per_tokens

    # j → k (0-indexed); None-results simply skipped.
    results: dict[int, int] = {}
    futs = {
        executor.submit(
            _llm_pick_one, session, api_key, model, title, bib_lines,
            token_counter=per_tokens,
        ): j
        for j, title in jobs
    }
    for fut in as_completed(futs):
        j = futs[fut]
        try:
            k = fut.result()
        except Exception as e:
            logger.warning("LLM job crashed for ss[%d]: %s", j, e)
            k = None
        if k is not None:
            results[j] = k
        inner_bar.update(1)
        # Recompute the postfix after every completion — cheap, and shows
        # the running per-survey spend next to the global cumulative.
        inner_bar.set_postfix_str(
            f"surv: {per_tokens.fmt()}  total: {global_tokens.fmt()}"
        )

    # Roll this survey's totals into the run-wide counter.
    global_tokens.add(
        per_tokens.in_tokens,
        per_tokens.out_tokens,
        cost=per_tokens.cost_usd,
    )

    # Detect conflicts: multiple SS titles → same bibitem.
    by_k: dict[int, list[int]] = {}
    for j, k in results.items():
        by_k.setdefault(k, []).append(j)

    mapping: dict[int, tuple[int, float]] = {}
    conflicts: list[dict] = []
    for k, js in by_k.items():
        js_sorted = sorted(js)
        winner = js_sorted[0]
        # LLM gives no confidence score; 1.0 is a placeholder so downstream
        # consumers can sort / threshold uniformly with the string-mode output.
        mapping[k] = (winner, 1.0)
        if len(js_sorted) > 1:
            conflicts.append({
                "bibitem_idx":        bibitems[k].idx,
                "winner_ss_index":    winner,
                "loser_ss_indices":   js_sorted[1:],
                "winner_title":       ss_refs[winner].get("title"),
                "loser_titles":       [ss_refs[x].get("title") for x in js_sorted[1:]],
            })
    return mapping, conflicts, per_tokens


# ── Output builder ────────────────────────────────────────────────────────────


def _mk_mapping_entry(bib: BibItem, ss: dict, score: float) -> dict:
    ext   = ss.get("externalIds") or {}
    arxiv = ext.get("ArXiv")
    doi   = ext.get("DOI")
    url   = (
        f"https://arxiv.org/abs/{arxiv}" if arxiv
        else f"https://doi.org/{doi}"    if doi
        else None
    )
    return {
        "latex_idx":   bib.idx,
        "latex_key":   bib.key,
        "latex_title": bib.title,
        "ss_title":    ss.get("title"),
        "ss_arxiv_id": arxiv,
        "ss_doi":      doi,
        "ss_url":      url,
        "ss_year":     ss.get("year"),
        "score":       round(score, 4),
    }


def build_output(
    *,
    arxiv_id:  str,
    mode:      str,
    model:     str | None,
    threshold: float | None,
    bibitems:  list[BibItem],
    ss_refs:   list[dict],
    mapping:   dict[int, tuple[int, float]],
    conflicts: list[dict],
    tokens:    TokenCounter | None = None,
) -> dict:
    """Assemble the final mapping JSON (see module docstring for schema).

    When ``tokens`` is provided (LLM mode) the counts and cost are embedded
    under ``usage`` so a later aggregation step can total cost across all
    surveys without re-running anything.
    """
    matched_k = set(mapping.keys())
    matched_j = {j for j, _ in mapping.values()}

    mapping_out = [
        _mk_mapping_entry(bibitems[k], ss_refs[j], score)
        for k, (j, score) in sorted(mapping.items())
    ]
    unmatched_latex = [
        {"idx": b.idx, "key": b.key, "title": b.title}
        for k, b in enumerate(bibitems) if k not in matched_k
    ]
    unmatched_ss = [
        {
            "ss_index": j,
            "title":    s.get("title"),
            "arxiv_id": (s.get("externalIds") or {}).get("ArXiv"),
            "doi":      (s.get("externalIds") or {}).get("DOI"),
            "year":     s.get("year"),
        }
        for j, s in enumerate(ss_refs) if j not in matched_j
    ]

    out: dict = {
        "arxiv_id":  arxiv_id,
        "mode":      mode,
        "model":     model,
        "threshold": threshold,
        "stats": {
            "n_latex":         len(bibitems),
            "n_ss":            len(ss_refs),
            "matched":         len(mapping),
            "unmatched_latex": len(unmatched_latex),
            "unmatched_ss":    len(unmatched_ss),
            "conflicts":       len(conflicts),
        },
        "mapping":         mapping_out,
        "unmatched_latex": unmatched_latex,
        "unmatched_ss":    unmatched_ss,
        "conflicts":       conflicts,
    }
    if tokens is not None:
        out["usage"] = {
            "in_tokens":  tokens.in_tokens,
            "out_tokens": tokens.out_tokens,
            "cost_usd":   round(tokens.cost_usd, 6),
        }
    return out


# ── Per-survey orchestration ──────────────────────────────────────────────────


def _resolve_merged_tex(arxiv_id: str, cache_dir: Path) -> Path:
    """Path to ``merged.tex`` for the given arxiv paper (may or may not exist)."""
    return cache_dir / arxiv_id / "merged.tex"


def _process_one(
    *,
    arxiv_id:       str,
    merged_tex:     Path,
    mode:           str,
    ss_api_key:     str | None,
    cache_dir:      Path,
    # string-mode knob
    threshold:      float,
    # llm-mode plumbing (all None in string mode)
    model:          str | None,
    api_key:        str | None,
    session:        requests.Session | None,
    executor:       ThreadPoolExecutor | None,
    inner_bar:      tqdm | None,
    global_tokens:  TokenCounter | None,
) -> dict:
    """Process one survey end-to-end. Returns the mapping JSON dict.

    Called once per survey by :func:`main` inside the outer tqdm loop.
    Exceptions propagate so the caller can mark the survey as failed and
    continue with the next one.
    """
    tex = _read_tex(merged_tex)
    bibitems, _span = parse_bibliography(tex)
    if not bibitems:
        raise RuntimeError(f"no \\bibitem entries in {merged_tex}")

    ss_refs = fetch_ss_references(arxiv_id, cache_dir, ss_api_key=ss_api_key)

    conflicts: list[dict] = []
    per_tokens: TokenCounter | None = None
    if mode == "string":
        mapping = match_by_string(bibitems, ss_refs, threshold=threshold)
    else:
        assert session is not None and executor is not None \
            and inner_bar is not None and global_tokens is not None \
            and api_key is not None and model is not None, \
            "llm mode requires session/executor/inner_bar/global_tokens/api_key/model"
        mapping, conflicts, per_tokens = match_by_llm(
            bibitems, ss_refs,
            model=model, api_key=api_key,
            session=session, executor=executor,
            inner_bar=inner_bar, global_tokens=global_tokens,
        )

    return build_output(
        arxiv_id=arxiv_id,
        mode=mode,
        model=(model if mode == "llm" else None),
        threshold=(threshold if mode == "string" else None),
        bibitems=bibitems,
        ss_refs=ss_refs,
        mapping=mapping,
        conflicts=conflicts,
        tokens=per_tokens,
    )


# ── Entry point ───────────────────────────────────────────────────────────────


SURVEYS_JSON = ROOT / "datasets" / "SurGE" / "surveys.json"
ARXIV_CACHE  = DEFAULT_CACHE_DIR / "arxiv_cache.json"


def _load_surveys_sorted() -> list[dict]:
    """surveys.json is NOT sorted by survey_id in-file; force ascending order."""
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

    # Batch selection
    parser.add_argument("--id", type=int, default=None,
                        help="Process only this survey_id.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N surveys by ascending survey_id.")
    parser.add_argument("--force", action="store_true",
                        help="Re-match even if an output file already exists.")

    # Strategy + knobs
    parser.add_argument("--mode", choices=("string", "llm"), default="string",
                        help="Matching strategy.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"LCS/len(ss) threshold for --mode string (default {DEFAULT_THRESHOLD}).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"OpenRouter model slug (default {DEFAULT_MODEL}).")
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL,
                        help=f"Concurrent LLM calls per survey (default {DEFAULT_PARALLEL}).")
    parser.add_argument("--api-key-env", type=str, default="OPENROUTER_API_KEY",
                        help="Env var holding the OpenRouter key (--mode llm only).")

    # Paths
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help=f"Root of per-paper dirs (default {DEFAULT_CACHE_DIR}).")
    parser.add_argument("--ss-api-key-env", type=str, default="SEMANTIC_SCHOLAR_API_KEY",
                        help="Env var holding a Semantic Scholar API key (recommended).")
    args = parser.parse_args()

    setup_logging("match_ss_to_bibitems")

    # Reserve stderr for tqdm: drop the console StreamHandler that setup_logging
    # attached so nothing stomps the live bars. All log records keep flowing to
    # results/logs/match_ss_to_bibitems.log — tail -f to follow in real time.
    # FileHandler subclasses StreamHandler, so guard with a type check.
    log_file: Path | None = None
    for h in list(logging.root.handlers):
        if isinstance(h, logging.FileHandler):
            log_file = Path(getattr(h, "baseFilename", "")) or log_file
        elif isinstance(h, logging.StreamHandler):
            logging.root.removeHandler(h)
    if log_file is not None:
        print(
            f"Logs → {log_file}  (tail -f to follow; stderr reserved for tqdm)",
            file=sys.stderr,
            flush=True,
        )

    # 1. Gather and filter surveys ─────────────────────────────────────────────
    surveys = _load_surveys_sorted()
    if args.id is not None:
        surveys = [s for s in surveys if s["survey_id"] == args.id]
    elif args.limit is not None:
        surveys = surveys[: args.limit]
    if not surveys:
        logger.warning("no surveys to process (filters produced empty set)")
        return 0

    arxiv_cache = _load_arxiv_cache()
    ss_api_key  = os.environ.get(args.ss_api_key_env) or None

    # 2. LLM prerequisites — created once, reused across surveys ──────────────
    openrouter_key: str | None     = None
    session:  requests.Session     | None = None
    executor: ThreadPoolExecutor   | None = None
    inner_bar: tqdm                | None = None
    global_tokens: TokenCounter    | None = None
    if args.mode == "llm":
        openrouter_key = os.environ.get(args.api_key_env)
        if not openrouter_key:
            logger.error("env %s is empty — required for --mode llm", args.api_key_env)
            return 2
        session       = _make_session(args.parallel)
        executor      = ThreadPoolExecutor(max_workers=args.parallel)
        global_tokens = TokenCounter()
        # position=1 keeps the inner bar under the outer; reset() is called
        # per-survey inside match_by_llm so only two lines ever render.
        inner_bar     = tqdm(
            total=1, desc="llm-match", unit="call",
            position=1, leave=False, dynamic_ncols=True,
        )

    # 3. Iterate ──────────────────────────────────────────────────────────────
    summary = {
        "total":         0,
        "processed":     0,
        "skipped_cache": 0,   # output file already exists
        "skipped_no_tex": 0,  # merged.tex missing
        "skipped_no_id": 0,   # no arxiv_id in cache
        "failed":        0,
        "n_latex_total": 0,
        "n_ss_total":    0,
        "matched_total": 0,
        "conflicts_total": 0,
    }

    outer_bar = tqdm(
        surveys, desc="surveys", unit="surv",
        position=0, dynamic_ncols=True,
    )

    try:
        for survey in outer_bar:
            summary["total"] += 1
            sid = str(survey["survey_id"])

            entry = arxiv_cache.get(sid)
            if not entry or not entry.get("arxiv_id"):
                summary["skipped_no_id"] += 1
                continue
            arxiv_id   = entry["arxiv_id"]
            merged_tex = _resolve_merged_tex(arxiv_id, args.cache_dir)
            if not merged_tex.is_file():
                summary["skipped_no_tex"] += 1
                continue

            out_file = args.cache_dir / arxiv_id / f"ss_matches_{args.mode}.json"
            if out_file.exists() and not args.force:
                summary["skipped_cache"] += 1
                continue

            outer_bar.set_description(f"surveys [{sid} → {arxiv_id}]")

            try:
                result = _process_one(
                    arxiv_id=arxiv_id,
                    merged_tex=merged_tex,
                    mode=args.mode,
                    ss_api_key=ss_api_key,
                    cache_dir=args.cache_dir,
                    threshold=args.threshold,
                    model=(args.model if args.mode == "llm" else None),
                    api_key=openrouter_key,
                    session=session,
                    executor=executor,
                    inner_bar=inner_bar,
                    global_tokens=global_tokens,
                )
            except Exception as e:
                logger.exception("[sid=%s arxiv=%s] failed: %s", sid, arxiv_id, e)
                summary["failed"] += 1
                continue

            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            s = result["stats"]
            summary["processed"]       += 1
            summary["n_latex_total"]   += s["n_latex"]
            summary["n_ss_total"]      += s["n_ss"]
            summary["matched_total"]   += s["matched"]
            summary["conflicts_total"] += s["conflicts"]

            # Per-survey log line (keeps a searchable record independent of tqdm)
            logger.info(
                "[sid=%s arxiv=%s] latex=%d ss=%d matched=%d "
                "(%.0f%% of latex, %.0f%% of ss) conflicts=%d",
                sid, arxiv_id,
                s["n_latex"], s["n_ss"], s["matched"],
                100 * s["matched"] / max(s["n_latex"], 1),
                100 * s["matched"] / max(s["n_ss"], 1),
                s["conflicts"],
            )

            # Outer postfix: cumulative matched + global usage (llm only).
            postfix = f"matched={summary['matched_total']}"
            if global_tokens is not None:
                postfix += f" {global_tokens.fmt()}"
            outer_bar.set_postfix_str(postfix)
    finally:
        outer_bar.close()
        if inner_bar is not None:
            inner_bar.close()
        if executor is not None:
            executor.shutdown(wait=True)
        if session is not None:
            session.close()

    # 4. Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    if global_tokens is not None:
        logger.info(
            "Total LLM usage: in=%d out=%d cost=$%.6f  (%s)",
            global_tokens.in_tokens, global_tokens.out_tokens,
            global_tokens.cost_usd, global_tokens.fmt(),
        )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
