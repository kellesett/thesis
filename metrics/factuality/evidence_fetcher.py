"""metrics/factuality/evidence_fetcher.py

Resolve a reference to an evidence-text suitable for AlignScore premise.

Two sources (modes):
    * ``abstract``  — paper abstract. Fast (free when doc_id hits
                     ``corpus.json``; SS API otherwise).
    * ``full_text`` — full paper body parsed from the arxiv LaTeX source.
                     Expensive — tarball download + latexpand + pandoc.
                     Without an ``arxiv_id`` we currently skip (GROBID /
                     openAccessPdf PDF-parsing is future work).

Output shape is uniform across modes::

    [
      {"ref":    {"idx": 5, "arxiv_id": ..., "semantic_scholar_id": ..., "doc_id": ...},
       "text":   "Clean evidence text"  # or None on miss
       "source": "corpus" | "ss_paperid" | "ss_arxiv" | "arxiv_latex"
                 | "miss" | "cached:<prev_source>" | "no_arxiv_id" | "<error>"}
      ...
    ]

Downstream (factuality/main.py) decides whether to feed these to AlignScore
separately (per-ref, enables attribution) or concatenated (faster, one
score per claim). This module is agnostic to that choice — it just fills
the ``text`` field, caches aggressively, and never blocks the caller on an
LLM.

Cache layout (persistent, shared across runs)::

    <cache_dir>/
        abstracts.json          # {key: {text, source}}
        full_texts/<key>.txt    # one plain-text file per ref

Cache key precedence: ``arxiv_id`` (stripped of version suffix) →
``semantic_scholar_id`` → ``doc_id``. This keeps refs unique across
channels — the same paper found via two different ids collapses to one
cache entry as soon as the higher-priority id arrives.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent

# Reused pieces from the reference-pipeline scripts:
#   * fetch_with_retry  — requests.GET + exponential backoff
#   * _clean_latex_fragment — LaTeX-artifact stripper for arxiv summaries
# Both already live in scripts/, adding them here via sys.path injection
# keeps dependency surface identical to existing pipeline code.
sys.path.insert(0, str(ROOT))
from scripts.fetch_reference_latex import fetch_with_retry  # noqa: E402
from scripts.parse_reference_md import _clean_latex_fragment  # noqa: E402


# ── Constants ────────────────────────────────────────────────────────────────

SS_BASE              = "https://api.semanticscholar.org/graph/v1"
SS_DELAY             = 1.1   # minimum sec between SS requests (polite)
SS_THROTTLE_UNAUTH   = 3     # secs to wait on 429 w/o api-key (dropped
                             # from 300 — without a key every request will
                             # 429 anyway, so a long sleep just delays the
                             # inevitable transition to miss)
SS_THROTTLE_AUTH     = 60    # secs to wait on 429 w/ api-key

# Arxiv Atom API. Per https://info.arxiv.org/help/api/user-manual.html
# they ask for a 3-second gap between requests.
ARXIV_QUERY_URL      = "https://export.arxiv.org/api/query"
ARXIV_ATOM_NS        = "http://www.w3.org/2005/Atom"
ARXIV_DELAY          = 3.2   # min sec between arxiv requests

# Default paths — relative to repo root. Caller may override.
DEFAULT_CACHE_DIR    = ROOT / "datasets" / "factuality_cache"
DEFAULT_LATEX_SRC    = ROOT / "datasets" / "surge" / "latex_src"
DEFAULT_MERGE_SH     = ROOT / "scripts" / "merge_latex.sh"


# ── Cache-key / filename helpers ─────────────────────────────────────────────


# Pre-compiled to keep it out of hot-path f-strings (PEP 701 only landed in
# 3.12 — under 3.11 a raw string with backslashes inside an f-string
# expression is a SyntaxError, and docker runs 3.11).
_ARXIV_VERSION_SUFFIX_RE = re.compile(r"v\d+$")


def _bare_arxiv(arxiv_id: str) -> str:
    """Strip the trailing ``v1`` / ``v2`` / ... version from an arxiv id.

    The same paper should cache once across submitted versions; the bare
    form (``"2010.12345"`` instead of ``"2010.12345v3"``) is canonical.
    """
    return _ARXIV_VERSION_SUFFIX_RE.sub("", str(arxiv_id))


def _ref_key(ref: dict) -> str:
    """Canonical cache key for a reference triple.

    Precedence: arxiv_id > semantic_scholar_id > doc_id. Same paper
    reached through different channels collapses to the highest-priority
    key available. ``arxiv_id`` version suffix (``v1``, ``v2`` …) is
    stripped — the same paper shouldn't cache twice over versions.
    """
    if ref.get("arxiv_id"):
        return f"arxiv:{_bare_arxiv(ref['arxiv_id'])}"
    if ref.get("semantic_scholar_id"):
        return f"ss:{ref['semantic_scholar_id']}"
    if ref.get("doc_id") is not None:
        return f"doc:{ref['doc_id']}"
    # Last-resort key so mapping is never empty; shouldn't normally happen.
    return f"unknown:idx{ref.get('idx', '?')}"


def _safe_filename(key: str) -> str:
    """Map cache key → filesystem-safe basename. Colons become underscores."""
    return re.sub(r"[^\w\-.]", "_", key)


# ── Abstract cache (single JSON) ─────────────────────────────────────────────


def load_abstract_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> dict:
    """Load the persistent abstract cache or return an empty dict.

    Schema: ``{key: {"text": str | None, "source": str}}``. ``text = None``
    means a recorded miss (SS/corpus had no abstract); re-querying such
    keys is wasted network I/O, so we keep them explicitly.
    """
    path = cache_dir / "abstracts.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("corrupt abstract cache at %s — starting fresh", path)
        return {}


def save_abstract_cache(cache: dict, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Persist the abstract cache to disk atomically.

    Atomic-via-rename keeps the cache file readable even if the process
    crashes mid-write (callers that save every-N refs benefit from this).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "abstracts.json"
    tmp  = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(path)


# ── SS API client ────────────────────────────────────────────────────────────


_last_ss_request_at: float = 0.0


def _ss_throttle() -> None:
    """Block until the minimum inter-request interval has elapsed."""
    global _last_ss_request_at
    wait = SS_DELAY - (time.monotonic() - _last_ss_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_ss_request_at = time.monotonic()


def _ss_get_abstract(
    paper_id: str,
    session: requests.Session,
    api_key: str | None,
) -> str | None:
    """GET /paper/<paper_id>?fields=abstract, returning the abstract string
    or None on miss/error.

    ``paper_id`` is the full SS external-id form like ``ARXIV:1512.03385``,
    ``DOI:10.1109/...``, or a bare SS paperId. Handles two rate-limit
    patterns: HTTP 429 and the 200-with-"code":"429" body form.
    """
    headers = {"x-api-key": api_key} if api_key else {}
    url     = f"{SS_BASE}/paper/{paper_id}"
    for _ in range(3):   # at most two retries after a rate-limit wait
        _ss_throttle()
        try:
            r = session.get(
                url, params={"fields": "abstract"},
                headers=headers, timeout=30,
            )
        except requests.RequestException as e:
            logger.debug("SS request failed for %s: %s", paper_id, e)
            return None

        rate_limited = (r.status_code == 429)
        if not rate_limited and r.status_code == 200:
            # 200 with body code "429" — occurs on unauth traffic spikes.
            try:
                body = r.json()
            except ValueError:
                body = None
            if isinstance(body, dict) and str(body.get("code", "")) == "429":
                rate_limited = True
                r = None  # force body re-read next iteration
        if rate_limited:
            wait = SS_THROTTLE_AUTH if api_key else SS_THROTTLE_UNAUTH
            logger.info("SS rate-limited for %s, sleeping %ds", paper_id, wait)
            time.sleep(wait)
            continue
        if r.status_code != 200:
            logger.debug("SS %s → HTTP %s", paper_id, r.status_code)
            return None
        data = r.json()
        if isinstance(data, dict):
            abstract = data.get("abstract")
            # SS returns None for papers where abstract text isn't in their
            # graph (common for older non-arxiv sources). Signal clearly:
            # ``None`` here → caller records a miss.
            return abstract if abstract else None
        return None
    return None


# ── arxiv API client ─────────────────────────────────────────────────────────


_last_arxiv_request_at: float = 0.0


def _arxiv_throttle() -> None:
    """Block until the minimum inter-request interval has elapsed.

    arxiv's user manual (user-manual.html §"API policies") asks for a
    3-second gap between requests. We honour it with a per-process counter —
    concurrent fetchers from the same process all funnel through one queue,
    identical to the SS pattern above.
    """
    global _last_arxiv_request_at
    wait = ARXIV_DELAY - (time.monotonic() - _last_arxiv_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_arxiv_request_at = time.monotonic()


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_abstract(raw: str) -> str:
    """Strip LaTeX artifacts and collapse whitespace in an arxiv abstract.

    arxiv abstracts are prose with occasional inline LaTeX (``\\emph{}``,
    ``$x^2$``, ``\\cite{...}``, ``~`` non-breaking space, etc.). Running
    them through :func:`scripts.parse_reference_md._clean_latex_fragment`
    — the same cleaner our LaTeX→markdown pipeline uses — keeps the text
    consistent with other cached abstracts (corpus.json, SS) which are
    stored as plain prose.
    """
    cleaned = _clean_latex_fragment(raw)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _arxiv_get_abstract(arxiv_id: str) -> str | None:
    """GET arxiv Atom API by id, return the summary text or None on miss.

    Args:
        arxiv_id: Bare arxiv id (``"1512.03385"``); version suffix is
                  stripped by the caller (:func:`_bare_arxiv`).

    Returns:
        Normalized abstract text, or ``None`` when the paper id is
        unknown / the ``<summary>`` element is empty / network failed.

    Implementation:
        * Uses :func:`scripts.fetch_reference_latex.fetch_with_retry` for
          transient-error handling (3 attempts, exponential backoff).
        * Parses the Atom response with stdlib ``xml.etree.ElementTree``.
        * Normalizes the text through :func:`_normalize_abstract`.

    Failure modes (all return ``None``, caller falls through to next tier):
        * Network: timeout / HTTP error after all retries.
        * Withdrawn / non-existent paper: no ``<entry>`` in feed.
        * Empty ``<summary>``: stub or metadata-only record.
        * XML parse error (unlikely, but we log and move on).
    """
    _arxiv_throttle()
    r = fetch_with_retry(
        ARXIV_QUERY_URL,
        params={"id_list": arxiv_id},
        timeout=20,
    )
    if r is None:
        return None

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        logger.warning("arxiv XML parse error for %s: %s", arxiv_id, e)
        return None

    entry = root.find(f"{{{ARXIV_ATOM_NS}}}entry")
    if entry is None:
        return None

    summary_el = entry.find(f"{{{ARXIV_ATOM_NS}}}summary")
    if summary_el is None or not (summary_el.text or "").strip():
        return None

    return _normalize_abstract(summary_el.text) or None


# ── Abstract resolution (multi-tier) ─────────────────────────────────────────


def fetch_abstract(
    ref: dict,
    cache: dict,
    *,
    corpus_index: dict[str, str] | None = None,
    session: requests.Session | None = None,
    ss_api_key: str | None = None,
    ss_enabled: bool = False,
) -> tuple[str | None, str]:
    """Resolve one ref → (abstract text, source).

    Priority chain (stops at first non-``None`` hit):
        1. ``cache[key]``           — any prior outcome (hit or recorded miss)
        2. ``corpus_index[doc_id]`` — local arxiv-corpus lookup (free)
        3. arxiv Atom API by ``arxiv_id``
        4. SS API by ``semantic_scholar_id`` or ``ARXIV:<id>``
           — only if ``ss_enabled=True``. Without an API key SS is
           effectively unusable (shared anon quota at ~100 req/5min/IP),
           so this tier defaults to off. Set in config when/if a key is
           available.

    Cache policy:
        Every terminal outcome — a hit OR an exhaustion-of-channels miss —
        is persisted. Re-runs therefore do no redundant network work.
        The user is responsible for deleting ``abstracts.json`` (or
        specific entries) to force a re-fetch after policy changes.

    Args:
        ref:          Reference dict with idx / arxiv_id / ss_id / doc_id.
        cache:        In-memory abstract cache (:func:`load_abstract_cache`
                      result). Mutated in place; caller should persist via
                      :func:`save_abstract_cache`.
        corpus_index: Optional ``{doc_id_str: "title\\n\\nabstract"}``
                      map from :func:`factuality.main.build_corpus_index`.
        session:      Shared ``requests.Session`` for SS calls.
        ss_api_key:   Semantic Scholar API key; may be ``None`` even with
                      ``ss_enabled=True`` — SS then runs in anon mode.
        ss_enabled:   If ``False``, tier 4 is skipped entirely (no network
                      calls, no throttle). Default ``False``.

    Returns:
        ``(text, source)``. ``text`` is ``None`` only when every tier miss'd.
        ``source`` identifies the tier that produced the answer:
        ``"corpus"``, ``"arxiv_api"``, ``"ss_paperid"``, ``"ss_arxiv"``,
        ``"cached:<prev_source>"``, ``"miss"``.
    """
    key = _ref_key(ref)

    # --- Tier 1: disk cache (any entry, including a recorded miss) --------
    # A cached miss is an intentional "we tried, nothing was available"
    # record — honouring it avoids hammering remote APIs on every re-run.
    # To force re-fetch, delete cache entries on disk.
    cached = cache.get(key)
    if cached is not None:
        return (cached.get("text"), f"cached:{cached.get('source', 'unknown')}")

    # --- Tier 2: local corpus.json lookup ---------------------------------
    if corpus_index is not None and ref.get("doc_id") is not None:
        text = corpus_index.get(str(ref["doc_id"]))
        if text:
            cache[key] = {"text": text, "source": "corpus"}
            return (text, "corpus")

    # --- Tier 3: arxiv Atom API by arxiv_id -------------------------------
    if ref.get("arxiv_id"):
        text = _arxiv_get_abstract(_bare_arxiv(ref["arxiv_id"]))
        if text:
            cache[key] = {"text": text, "source": "arxiv_api"}
            return (text, "arxiv_api")

    # --- Tier 4: SS API (optional) ----------------------------------------
    if ss_enabled:
        if session is None:
            session = requests.Session()

        # Try paperId first (most specific). Fall back to ARXIV:{id}.
        # NB: SS paperId is the bare 40-char hash — no "PAPERID:" prefix.
        # Valid prefixes are ARXIV / DOI / CorpusId / MAG / ACL / PMID /
        # PMCID / URL (per SS API docs). Unknown prefixes triggered fake
        # 429 responses that the detector below then slept on; goal here
        # is to never re-introduce that bug.
        for source_label, param in (
            ("ss_paperid", str(ref["semantic_scholar_id"])
                            if ref.get("semantic_scholar_id") else None),
            ("ss_arxiv",   f"ARXIV:{_bare_arxiv(ref['arxiv_id'])}"
                            if ref.get("arxiv_id") else None),
        ):
            if param is None:
                continue
            text = _ss_get_abstract(param, session, ss_api_key)
            if text is not None:
                cache[key] = {"text": text, "source": source_label}
                return (text, source_label)

    # All enabled channels exhausted — persist the miss.
    cache[key] = {"text": None, "source": "miss"}
    return (None, "miss")


# ── Full-text resolution (arxiv-sources only, for now) ───────────────────────


def fetch_full_text(
    ref: dict,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    latex_src_dir: Path = DEFAULT_LATEX_SRC,
    merge_sh: Path = DEFAULT_MERGE_SH,
) -> tuple[str | None, str]:
    """Resolve one ref → (full-text markdown, source).

    Pipeline (arxiv-only):
        1. If ``<cache_dir>/full_texts/<key>.txt`` exists → cache hit.
        2. If ``<latex_src_dir>/<arxiv_id>/`` lacks the tarball → call
           :func:`scripts.fetch_reference_latex.download_source`.
        3. If ``merged.tex`` missing → invoke ``merge_latex.sh <arxiv_id>``.
        4. Parse with :func:`scripts.parse_reference_md.parse` → body_md.
        5. Cache the result to ``<cache_dir>/full_texts/<key>.txt``.

    Refs without ``arxiv_id`` return ``(None, "no_arxiv_id")`` — GROBID or
    SS-openAccessPdf fallback is not implemented yet. This ceiling matters:
    on the ``SurGE_reference`` data we have ~50% arxiv-fetchable refs; the
    rest are effectively evidence-less in full-text mode today.

    Errors are surfaced as non-empty source labels (``download_failed:...``,
    ``merge_failed``, ``parse_failed:...``) so the caller can bucket them.
    """
    arxiv_id = ref.get("arxiv_id")
    if not arxiv_id:
        return (None, "no_arxiv_id")

    arxiv_id = _bare_arxiv(arxiv_id)
    key = f"arxiv:{arxiv_id}"
    cache_file = cache_dir / "full_texts" / f"{_safe_filename(key)}.txt"

    # Disk cache hit
    if cache_file.exists():
        try:
            return (cache_file.read_text(encoding="utf-8"), "cached:arxiv_latex")
        except OSError as e:
            logger.warning("cache read failed for %s: %s", cache_file, e)

    # Ensure tarball is on disk
    paper_dir = latex_src_dir / arxiv_id
    if not paper_dir.exists() or not any(paper_dir.iterdir()):
        from scripts.fetch_reference_latex import download_source  # noqa: E402
        latex_src_dir.mkdir(parents=True, exist_ok=True)
        ok, reason = download_source(arxiv_id, latex_src_dir)
        if not ok:
            return (None, f"download_failed:{reason}")

    # Ensure merged.tex exists
    merged_tex = paper_dir / "merged.tex"
    if not merged_tex.is_file():
        try:
            proc = subprocess.run(
                ["bash", str(merge_sh), arxiv_id],
                capture_output=True, text=True, timeout=120,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return (None, f"merge_failed:{type(e).__name__}")
        if proc.returncode != 0 or not merged_tex.is_file():
            return (None, "merge_failed")

    # Parse LaTeX → markdown body
    from scripts.parse_reference_md import parse as parse_merged_tex  # noqa: E402
    try:
        result = parse_merged_tex(merged_tex)
    except Exception as e:
        return (None, f"parse_failed:{type(e).__name__}:{str(e)[:80]}")

    text = (result.body_md or "").strip()
    if not text:
        return (None, "parse_empty")

    # Persist to cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_file.write_text(text, encoding="utf-8")
    except OSError as e:
        logger.warning("failed to cache full text at %s: %s", cache_file, e)
    return (text, "arxiv_latex")


# ── Batch dispatcher ─────────────────────────────────────────────────────────


def fetch_evidence(
    refs: list[dict],
    mode: str,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    abstract_cache: dict | None = None,
    corpus_index: dict[str, str] | None = None,
    ss_api_key: str | None = None,
    ss_enabled: bool = False,
    session: requests.Session | None = None,
    save_every: int = 20,
    progress_bar = None,
) -> list[dict]:
    """Fetch evidence for a batch of refs. Returns the uniform entry list.

    Args:
        refs:           Reference dicts to resolve.
        mode:           ``"abstract"`` or ``"full_text"``.
        cache_dir:      Root of the persistent cache tree.
        abstract_cache: Pre-loaded cache (caller may share across batches);
                        if ``None``, this function loads+saves on its own.
        corpus_index:   Optional ``{doc_id_str: abstract}`` — speeds up
                        abstract mode enormously when many refs are
                        corpus-resolvable (SurGE_reference: ~90% coverage).
        ss_api_key:     Semantic Scholar API key (set via env elsewhere,
                        passed in here for explicitness).
        session:        Shared ``requests.Session``; created on demand.
        save_every:     Persist the abstract cache every N fresh fetches —
                        crash-safe without a save-per-ref overhead.
        progress_bar:   Optional tqdm (or any obj with ``.update(n)``) —
                        advanced by 1 per ref. If the caller wants pretty
                        postfix strings it should manage them itself; we
                        only drive the counter.

    Returns:
        List of entries ``[{"ref": ref, "text": str|None, "source": str}, ...]``
        parallel to ``refs``.
    """
    if mode not in ("abstract", "full_text"):
        raise ValueError(f"unknown mode: {mode!r}")

    # Abstract mode needs the shared JSON cache; full_text is per-file.
    owns_cache = False
    if mode == "abstract" and abstract_cache is None:
        abstract_cache = load_abstract_cache(cache_dir)
        owns_cache = True

    if session is None and mode == "abstract":
        session = requests.Session()

    out: list[dict] = []
    n_fresh = 0  # fresh fetches since last save — drives periodic persistence
    for ref in refs:
        if mode == "abstract":
            text, source = fetch_abstract(
                ref, abstract_cache,   # type: ignore[arg-type]
                corpus_index=corpus_index,
                session=session,
                ss_api_key=ss_api_key,
                ss_enabled=ss_enabled,
            )
        else:  # full_text
            text, source = fetch_full_text(ref, cache_dir=cache_dir)

        out.append({"ref": ref, "text": text, "source": source})
        if progress_bar is not None:
            progress_bar.update(1)

        # Periodic save — only matters for abstract mode (full_text is
        # already file-per-ref atomic).
        if mode == "abstract" and not source.startswith("cached:") and source != "corpus":
            n_fresh += 1
            if owns_cache and n_fresh % save_every == 0:
                save_abstract_cache(abstract_cache, cache_dir)   # type: ignore[arg-type]

    if mode == "abstract" and owns_cache and n_fresh > 0:
        save_abstract_cache(abstract_cache, cache_dir)   # type: ignore[arg-type]

    return out


# ── High-level: unified per-survey sources pipeline ──────────────────────────
#
# Both internal and external factuality modes go through a single per-survey
# JSON artefact (see metrics/factuality/sources_io.py for schema). The
# function below is the only entry point callers should use — it hides the
# mode-dispatch, file-io, and downstream conversion behind one signature.


from datetime import datetime, timezone   # noqa: E402  (late import deliberate)
from metrics.factuality.sources_io import (                  # noqa: E402
    SCHEMA_VERSION, build_empty_entry, canonical_sources_dir,
    load_gen_sources, save_gen_sources,
)


def _sources_to_key_evidence(
    sources: dict, evidence_source: str,
) -> dict[str, tuple[str | None, str]]:
    """Convert a loaded sources dict into ``{ref_key: (text, source)}``.

    ``ref_key`` mirrors :func:`_ref_key`'s priority rule so the same keys
    work across in-memory `refs` dicts and the persisted sources file.
    ``evidence_source`` selects which field (``abstract`` or ``text``) to
    surface as the downstream AlignScore premise.
    """
    if evidence_source == "abstract":
        text_field, source_field = "abstract", "abs_source"
    elif evidence_source == "full_text":
        text_field, source_field = "text", "text_source"
    else:
        raise ValueError(f"unknown evidence_source: {evidence_source!r}")

    out: dict[str, tuple[str | None, str]] = {}
    for entry in sources.get("refs", {}).values():
        key = _ref_key(entry)
        text = entry.get(text_field)
        src  = entry.get(source_field) or "miss"
        out[key] = (text, src)
    return out


def _build_sources_internal(
    gen: dict,
    *,
    evidence_source: str,
    cache_dir: Path,
    abstract_cache: dict,
    corpus_index: dict[str, str] | None,
    ss_api_key: str | None,
    ss_enabled: bool,
    progress_bar = None,
) -> dict:
    """Run the internal waterfall for every ref in ``gen.meta.references``,
    assemble a unified-schema dict, and return it (without saving).

    Unlike the older :func:`fetch_evidence` this function:
      * fetches **all** refs in the generation, not scope-derived uniques;
      * produces the unified per-survey schema (see ``sources_io``);
      * leaves per-tier error trails (``abs_errors`` / ``text_errors``) as
        ``None`` — internal fetchers don't track reasons at the moment.
    """
    refs = gen.get("meta", {}).get("references", [])
    per_ref: dict[str, dict] = {}

    # Shared session for SS calls within this survey.
    session = requests.Session()

    for ref in refs:
        entry = build_empty_entry(ref)
        if evidence_source == "abstract":
            text, source = fetch_abstract(
                ref, abstract_cache,
                corpus_index=corpus_index,
                session=session,
                ss_api_key=ss_api_key,
                ss_enabled=ss_enabled,
            )
            if text:
                entry["abstract"]   = text
                entry["abs_source"] = source
        elif evidence_source == "full_text":
            text, source = fetch_full_text(ref, cache_dir=cache_dir)
            if text:
                entry["text"]         = text
                entry["text_source"]  = source
        else:
            raise ValueError(f"unknown evidence_source: {evidence_source!r}")

        per_ref[str(ref["idx"])] = entry
        if progress_bar is not None:
            progress_bar.update(1)

    # Persist the abstract cache after this survey's fetches — crash-safety
    # beats the old end-of-run-only flush when processing hundreds of surveys.
    save_abstract_cache(abstract_cache, cache_dir)

    return {
        "survey_id":            gen["id"],
        "dataset_id":           gen["dataset_id"],
        "model_id":             gen["model_id"],
        "fetched_at":           datetime.now(timezone.utc).isoformat(),
        "fetch_provider":       f"internal_{evidence_source}",
        "n_refs_total":         len(refs),
        "n_refs_with_abstract": sum(1 for e in per_ref.values() if e["abstract"]),
        "n_refs_with_text":     sum(1 for e in per_ref.values() if e["text"]),
        "n_refs_no_identifier": sum(
            1 for e in per_ref.values()
            if not (e.get("arxiv_id") or e.get("semantic_scholar_id"))
        ),
        # Internal fetchers don't currently propagate per-tier reasons;
        # colab-produced files fill these in when present.
        "abs_error_summary":    None,
        "text_error_summary":   None,
        "refs":                 per_ref,
    }


def prepare_key_evidence(
    gen: dict,
    *,
    evidence_mode: str,
    evidence_source: str,
    sources_dir: Path,
    # internal-mode params (ignored in external mode):
    cache_dir: Path = DEFAULT_CACHE_DIR,
    abstract_cache: dict | None = None,
    corpus_index: dict[str, str] | None = None,
    ss_api_key: str | None = None,
    ss_enabled: bool = False,
    progress_bar = None,
) -> dict[str, tuple[str | None, str]]:
    """Single entry point from factuality's process_survey into the evidence
    layer. Ensures the per-survey sources file exists, loads it, and returns
    the ``key_to_evidence`` dict ready for the AlignScore stage.

    **Both modes use the same canonical path** —
    ``results/generations/<dataset_id>_<model_id>/sources/<sid>_sources.json``.
    The difference is only what happens when the file is absent:

    * ``"internal"`` — run the waterfall for every ref in the generation,
      build a fresh file at the canonical path, return it. An existing
      file there is used as-is (checkpoint/resume semantics).
    * ``"external"`` — require the file to be present. Missing file
      raises :class:`FileNotFoundError` with an explicit message; the
      run for this survey fails. Missing *refs* inside the file are
      downgraded to evidence misses downstream (caller may WARN).

    Put Colab-produced files straight under
    ``results/generations/<ds>_<mdl>/sources/`` and switch to
    ``evidence_mode: external`` to consume them.

    ``progress_bar`` — optional tqdm-like object. In internal mode the bar
    advances once per ref. In external mode (no network) it sweeps to
    completion after the load.

    Returns ``{ref_key: (text, source)}`` — same shape the old
    :func:`fetch_evidence` produced, so the align stage is unchanged.
    """
    refs      = gen.get("meta", {}).get("references", [])
    survey_id = str(gen["id"])

    # Try to load existing sources file (both modes share this path).
    sources = None
    try:
        sources = load_gen_sources(sources_dir, survey_id)
    except ValueError as e:
        # Schema mismatch — warn, let mode-specific logic decide next step.
        logger.warning("%s", e)
        sources = None

    if sources is not None:
        logger.info(
            "[RESUME] sid=%s — sources file present at %s, skipping fetch",
            survey_id, sources_dir / f"{survey_id}_sources.json",
        )
        if progress_bar is not None:
            progress_bar.update(max(len(refs), 1))
        return _sources_to_key_evidence(sources, evidence_source)

    # File missing — mode decides.
    if evidence_mode == "external":
        raise FileNotFoundError(
            f"evidence_mode=external but sources file is missing for "
            f"survey {survey_id}: expected "
            f"{sources_dir / f'{survey_id}_sources.json'}. "
            f"Put the colab_bulk_fetch output there or switch to "
            f"evidence_mode=internal to build it."
        )
    if evidence_mode != "internal":
        raise ValueError(f"unknown evidence_mode: {evidence_mode!r}")

    # Internal mode + no existing file → build fresh.
    if abstract_cache is None:
        abstract_cache = load_abstract_cache(cache_dir)
    sources = _build_sources_internal(
        gen,
        evidence_source=evidence_source,
        cache_dir=cache_dir,
        abstract_cache=abstract_cache,
        corpus_index=corpus_index,
        ss_api_key=ss_api_key,
        ss_enabled=ss_enabled,
        progress_bar=progress_bar,
    )
    save_gen_sources(sources_dir, survey_id, sources)
    logger.info(
        "[PROC] sid=%s — sources written to %s "
        "(abstracts=%d/%d, texts=%d/%d, no_id=%d)",
        survey_id, out_dir / f"{survey_id}_sources.json",
        sources["n_refs_with_abstract"], sources["n_refs_total"],
        sources["n_refs_with_text"],     sources["n_refs_total"],
        sources["n_refs_no_identifier"],
    )
    return _sources_to_key_evidence(sources, evidence_source)
