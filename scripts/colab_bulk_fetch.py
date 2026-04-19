"""scripts/colab_bulk_fetch.py

Standalone Colab-runnable script: given an archive (or directory) of
generation JSON files for one (dataset, model), fetch abstracts + full
texts for every ``[N]`` citation and emit per-survey sources files in
the unified factuality schema.

Designed for copy-paste into a single Colab cell. No CLI. Constants at
the top. Output files are compatible with ``metrics/factuality`` in
``evidence_mode: external``.

Pipeline:
  1. Load generations (extract archive if needed).
  2. For every ref in every generation, pick an identifier (ss_id
     priority > ARXIV:arxiv_id > skip). Refs without either are still
     emitted in the output with all evidence fields ``null``.
  3. Dedupe identifiers and POST them all to SS /paper/batch.
  4. For each unique identifier, run the abstract + full-text waterfall:

     Abstract:  SS response → arxiv Atom API → Crossref → OpenAlex
     Full text: arxiv PDF   → SS openAccessPdf → Unpaywall → (pymupdf)

  5. Group results by survey_id and write
     ``<OUT_DIR>/<survey_id>_sources.json`` (atomic, schema_version=1).

Colab preamble (run once before this cell):
    !pip install -q requests tqdm pymupdf

Input formats supported:
  * ``.tar.gz`` / ``.tgz``
  * ``.tar``
  * ``.zip``
  * plain directory with ``<survey_id>.json`` files (no archive)

Resume:
  On startup the script scans ``OUT_DIR`` and skips surveys that already
  have a complete ``<sid>_sources.json`` (schema_version matches). To
  force a re-fetch, delete the target file.

NOTE: schema stays in sync with ``metrics/factuality/sources_io.py``
manually. Bump :data:`SCHEMA_VERSION` here and in ``sources_io.py``
together.
"""

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — edit before running
# ════════════════════════════════════════════════════════════════════════════

# Path to an archive or directory of generations. Directory must contain
# ``<survey_id>.json`` files at top level.
GENERATIONS_INPUT = "generations.tar.gz"

# Directory for per-survey output files. Created if missing.
OUT_DIR = "sources_out"

# Dataset/model tags saved inside each sources file. Optional but recommended;
# when None they're pulled from the first generation's ``dataset_id`` /
# ``model_id`` fields.
DATASET_ID_OVERRIDE = None
MODEL_ID_OVERRIDE   = None

# Full-text fetching toggle (PDFs + pymupdf parse). False = ~20x faster,
# abstracts only.
FETCH_FULL_TEXT = True

# Contact email for polite pool on Crossref / OpenAlex / Unpaywall — they
# don't validate it, just use it as a good-citizen marker.
USER_EMAIL = "you@example.com"

# Per-API rate limits (seconds between consecutive calls, separate buckets).
RATE_LIMIT_SS        = 1.1
RATE_LIMIT_ARXIV     = 3.2   # arxiv's own recommendation
RATE_LIMIT_CROSSREF  = 0.1
RATE_LIMIT_OPENALEX  = 0.1
RATE_LIMIT_UNPAYWALL = 0.1

# SS /paper/batch accepts up to 500 IDs per POST.
SS_BATCH_SIZE = 500

# PDF-parse caps — guard against 500-page theses bloating output.
PDF_MAX_PAGES = 80
PDF_MAX_CHARS = 200_000

TIMEOUT_SEC  = 60
MAX_RETRIES  = 3
BACKOFF_BASE = 2.0


# ════════════════════════════════════════════════════════════════════════════
# Code — no config below this line.
# ════════════════════════════════════════════════════════════════════════════

import json
import re
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm.auto import tqdm

# pymupdf is only needed when FETCH_FULL_TEXT is True; load lazily at
# module-import time and let main() decide whether the absence is fatal
# (it raises if FETCH_FULL_TEXT ends up True but fitz is None).
try:
    import fitz  # pymupdf
except ImportError:
    fitz = None  # type: ignore


# Schema version — MUST match metrics/factuality/sources_io.py::SCHEMA_VERSION.
# Bump here and there together on any schema change.
SCHEMA_VERSION = 1

SS_BATCH_URL  = "https://api.semanticscholar.org/graph/v1/paper/batch"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
CROSSREF_URL  = "https://api.crossref.org/works/{doi}"
OPENALEX_URL  = "https://api.openalex.org/works/doi:{doi}"
UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"

ATOM_NS = "http://www.w3.org/2005/Atom"
UA      = f"thesis-bulk-fetch/1.0 (mailto:{USER_EMAIL})"


# ── Throttling ──────────────────────────────────────────────────────────────

_last_call_at: dict[str, float] = {}


def throttle(bucket: str, min_interval: float) -> None:
    """Block until `min_interval` seconds have passed since the last call
    tagged with `bucket`. Separate buckets throttle independently."""
    last = _last_call_at.get(bucket, 0.0)
    wait = min_interval - (time.monotonic() - last)
    if wait > 0:
        time.sleep(wait)
    _last_call_at[bucket] = time.monotonic()


def http_get(url: str, *, bucket: str, delay: float, **kwargs) -> requests.Response | None:
    """GET with throttle + bounded retry/backoff. Returns None on final fail."""
    kwargs.setdefault("timeout", TIMEOUT_SEC)
    kwargs.setdefault("headers", {}).setdefault("User-Agent", UA)
    for attempt in range(MAX_RETRIES):
        throttle(bucket, delay)
        try:
            r = requests.get(url, **kwargs)
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(BACKOFF_BASE ** attempt)
    return None


def http_post(url: str, *, bucket: str, delay: float, **kwargs) -> requests.Response | None:
    kwargs.setdefault("timeout", TIMEOUT_SEC)
    kwargs.setdefault("headers", {}).setdefault("User-Agent", UA)
    for attempt in range(MAX_RETRIES):
        throttle(bucket, delay)
        try:
            r = requests.post(url, **kwargs)
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(BACKOFF_BASE ** attempt)
    return None


# ── Text helpers ────────────────────────────────────────────────────────────

_WS_RE   = re.compile(r"\s+")
# Control chars (0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F) — keep \t=0x09, \n=0x0A,
# \r=0x0D. PDFs sometimes embed these (form feeds, start-of-header), and
# they make downstream JSON fragile on some readers.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_JATS_RE = re.compile(r"<[^>]+>")


def normalize_ws(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def strip_control_chars(s: str | None) -> str | None:
    if not s:
        return s
    return _CTRL_RE.sub(" ", s)


def strip_jats(s: str) -> str:
    return normalize_ws(_JATS_RE.sub("", s))


def reconstruct_inverted_index(inv: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as ``{word: [positions]}``. Reverse it."""
    if not inv:
        return None
    pairs: list[tuple[int, str]] = []
    for word, positions in inv.items():
        for p in positions:
            pairs.append((p, word))
    pairs.sort(key=lambda t: t[0])
    return normalize_ws(" ".join(w for _, w in pairs)) or None


# ── Arxiv-id normalization ──────────────────────────────────────────────────

_ARXIV_VER_RE = re.compile(r"v\d+$")

def bare_arxiv(arxiv_id: str) -> str:
    return _ARXIV_VER_RE.sub("", str(arxiv_id))


# ── SS batch ────────────────────────────────────────────────────────────────

def ss_batch_fetch(ids: list[str]) -> tuple[list[dict | None] | None, str | None]:
    """POST /paper/batch for up to SS_BATCH_SIZE ids.

    Returns ``(results, batch_error)``:
      * ``results`` — list parallel to input, each entry a dict or None,
        or **whole list is None** when the HTTP call itself failed.
      * ``batch_error`` — short tag when ``results is None``, else None.
    """
    fields = "title,abstract,externalIds,openAccessPdf"
    r = http_post(
        SS_BATCH_URL,
        bucket="ss", delay=RATE_LIMIT_SS,
        params={"fields": fields},
        json={"ids": ids},
    )
    if r is None:
        return None, "http_error"
    try:
        data = r.json()
    except ValueError:
        return None, "parse_error"
    if not isinstance(data, list):
        return None, "unexpected_response"
    padded = list(data) + [None] * max(0, len(ids) - len(data))
    return padded, None


# ── Abstract sources (return (text|None, error|None)) ───────────────────────

def arxiv_abstract(arxiv_id: str) -> tuple[str | None, str | None]:
    r = http_get(
        ARXIV_API_URL, bucket="arxiv", delay=RATE_LIMIT_ARXIV,
        params={"id_list": arxiv_id},
    )
    if r is None:
        return None, "http_error"
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return None, "xml_parse_error"
    entry = root.find(f"{{{ATOM_NS}}}entry")
    if entry is None:
        return None, "no_entry"
    summary = entry.find(f"{{{ATOM_NS}}}summary")
    if summary is None or not (summary.text or "").strip():
        return None, "empty_summary"
    return normalize_ws(summary.text), None


def crossref_abstract(doi: str) -> tuple[str | None, str | None]:
    r = http_get(
        CROSSREF_URL.format(doi=quote(doi, safe="")),
        bucket="crossref", delay=RATE_LIMIT_CROSSREF,
        params={"mailto": USER_EMAIL},
    )
    if r is None:
        return None, "http_error"
    try:
        msg = r.json().get("message", {})
    except ValueError:
        return None, "parse_error"
    raw = msg.get("abstract")
    if not raw:
        return None, "no_abstract_field"
    cleaned = strip_jats(raw)
    if not cleaned:
        return None, "empty_after_strip"
    return cleaned, None


def openalex_abstract(doi: str) -> tuple[str | None, str | None]:
    r = http_get(
        OPENALEX_URL.format(doi=quote(doi, safe="")),
        bucket="openalex", delay=RATE_LIMIT_OPENALEX,
        params={"mailto": USER_EMAIL},
    )
    if r is None:
        return None, "http_error"
    try:
        data = r.json()
    except ValueError:
        return None, "parse_error"
    inv = data.get("abstract_inverted_index")
    if not inv:
        return None, "no_inverted_index"
    reconstructed = reconstruct_inverted_index(inv)
    if not reconstructed:
        return None, "empty_after_reconstruct"
    return reconstructed, None


# ── Full-text sources ───────────────────────────────────────────────────────

def extract_pdf_text(pdf_bytes: bytes) -> str | None:
    """Parse PDF bytes into plain text via pymupdf, capped to PDF_MAX_PAGES /
    PDF_MAX_CHARS. Strips control chars (survives JSON roundtrip cleanly)."""
    if fitz is None:
        return None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return None
    try:
        parts: list[str] = []
        total = 0
        for i, page in enumerate(doc):
            if i >= PDF_MAX_PAGES:
                break
            try:
                t = page.get_text() or ""
            except Exception:
                continue
            if not t:
                continue
            parts.append(t)
            total += len(t)
            if total >= PDF_MAX_CHARS:
                break
    finally:
        doc.close()
    if not parts:
        return None
    out = normalize_ws("\n".join(parts))
    out = strip_control_chars(out) or ""
    return out[:PDF_MAX_CHARS] if len(out) > PDF_MAX_CHARS else out or None


def fetch_pdf_bytes(url: str, bucket: str, delay: float) -> tuple[bytes | None, str | None]:
    r = http_get(url, bucket=bucket, delay=delay, stream=False)
    if r is None:
        return None, "http_error"
    ctype = r.headers.get("Content-Type", "").lower()
    if "pdf" not in ctype and "octet-stream" not in ctype:
        if not r.content.startswith(b"%PDF"):
            return None, "not_pdf"
    return r.content, None


def arxiv_pdf_text(arxiv_id: str) -> tuple[str | None, str | None]:
    pdf, err = fetch_pdf_bytes(
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        bucket="arxiv", delay=RATE_LIMIT_ARXIV,
    )
    if pdf is None:
        return None, err
    text = extract_pdf_text(pdf)
    return (text, None) if text else (None, "pdf_parse_failed")


def oa_pdf_text(url: str) -> tuple[str | None, str | None]:
    pdf, err = fetch_pdf_bytes(url, bucket="oa_pdf", delay=0.1)
    if pdf is None:
        return None, err
    text = extract_pdf_text(pdf)
    return (text, None) if text else (None, "pdf_parse_failed")


def unpaywall_pdf_text(doi: str) -> tuple[str | None, str | None]:
    r = http_get(
        UNPAYWALL_URL.format(doi=quote(doi, safe="")),
        bucket="unpaywall", delay=RATE_LIMIT_UNPAYWALL,
        params={"email": USER_EMAIL},
    )
    if r is None:
        return None, "http_error"
    try:
        data = r.json()
    except ValueError:
        return None, "parse_error"
    best = data.get("best_oa_location") or {}
    url = best.get("url_for_pdf") or best.get("url")
    if not url:
        return None, "no_oa_url"
    return oa_pdf_text(url)


# ── Identifier selection ────────────────────────────────────────────────────

def ss_identifier_for(ref: dict) -> str | None:
    """Pick the identifier to send to SS /paper/batch for one ref.

    Priority: ss_id > ARXIV:arxiv_id > None (skip).
    """
    ss_id = ref.get("semantic_scholar_id")
    if ss_id:
        return str(ss_id)
    arxiv = ref.get("arxiv_id")
    if arxiv:
        return f"ARXIV:{bare_arxiv(arxiv)}"
    return None


# ── Per-ref waterfall ───────────────────────────────────────────────────────

def resolve_abstract_for_ref(
    ref: dict, ss_info: dict | None, ss_error: str | None,
) -> tuple[str | None, str | None, list[dict]]:
    """Abstract waterfall for one ref. Returns (text, source, errors)."""
    errors: list[dict] = []

    # Tier 1: SS
    if ss_info is None:
        errors.append({"source": "ss", "reason": ss_error or "unknown"})
    else:
        abs_ss = ss_info.get("abstract")
        if abs_ss:
            return normalize_ws(abs_ss), "ss", errors
        errors.append({"source": "ss", "reason": "no_abstract"})

    # Identifiers — use ones from ref first, fall back to SS's externalIds.
    ext       = (ss_info or {}).get("externalIds") or {}
    arxiv_id  = ref.get("arxiv_id") or ext.get("ArXiv")
    doi       = ref.get("doi") or ext.get("DOI")

    # Tier 2: arxiv
    if not arxiv_id:
        errors.append({"source": "arxiv", "reason": "no_arxiv_id"})
    else:
        text, err = arxiv_abstract(bare_arxiv(arxiv_id))
        if text:
            return text, "arxiv", errors
        errors.append({"source": "arxiv", "reason": err or "unknown"})

    # Tier 3: Crossref
    if not doi:
        errors.append({"source": "crossref", "reason": "no_doi"})
    else:
        text, err = crossref_abstract(doi)
        if text:
            return text, "crossref", errors
        errors.append({"source": "crossref", "reason": err or "unknown"})

    # Tier 4: OpenAlex
    if not doi:
        errors.append({"source": "openalex", "reason": "no_doi"})
    else:
        text, err = openalex_abstract(doi)
        if text:
            return text, "openalex", errors
        errors.append({"source": "openalex", "reason": err or "unknown"})

    return None, None, errors


def resolve_text_for_ref(
    ref: dict, ss_info: dict | None,
) -> tuple[str | None, str | None, list[dict]]:
    """Full-text waterfall for one ref. Returns (text, source, errors)."""
    errors: list[dict] = []

    ext       = (ss_info or {}).get("externalIds") or {}
    arxiv_id  = ref.get("arxiv_id") or ext.get("ArXiv")
    doi       = ref.get("doi") or ext.get("DOI")
    oa        = (ss_info or {}).get("openAccessPdf") or {}
    oa_url    = oa.get("url")

    # Tier 1: arxiv PDF
    if not arxiv_id:
        errors.append({"source": "arxiv_pdf", "reason": "no_arxiv_id"})
    else:
        t, err = arxiv_pdf_text(bare_arxiv(arxiv_id))
        if t:
            return t, "arxiv_pdf", errors
        errors.append({"source": "arxiv_pdf", "reason": err or "unknown"})

    # Tier 2: SS openAccessPdf
    if not oa_url:
        errors.append({"source": "ss_oa_pdf", "reason": "no_oa_url"})
    else:
        t, err = oa_pdf_text(oa_url)
        if t:
            return t, "ss_oa_pdf", errors
        errors.append({"source": "ss_oa_pdf", "reason": err or "unknown"})

    # Tier 3: Unpaywall
    if not doi:
        errors.append({"source": "unpaywall_pdf", "reason": "no_doi"})
    else:
        t, err = unpaywall_pdf_text(doi)
        if t:
            return t, "unpaywall_pdf", errors
        errors.append({"source": "unpaywall_pdf", "reason": err or "unknown"})

    return None, None, errors


# ── Archive / generation loading ────────────────────────────────────────────

def prepare_generations_dir(input_path: str) -> Path:
    """Return a Path to a directory containing generation ``<id>.json`` files.

    Accepts a plain directory or an archive (.tar.gz, .tgz, .tar, .zip).
    Archives are extracted to a temp dir. If the archive has a single
    top-level directory, we drill into it so the caller gets the
    JSONs directly.
    """
    p = Path(input_path)
    if p.is_dir():
        return p
    if not p.is_file():
        raise FileNotFoundError(f"GENERATIONS_INPUT not found: {input_path}")

    tmp = Path(tempfile.mkdtemp(prefix="colab_fetch_gens_"))
    name = p.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if name.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(p, mode) as tar:
            tar.extractall(tmp)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(p) as zf:
            zf.extractall(tmp)
    else:
        raise ValueError(
            f"Unsupported archive type: {name}. Use .tar.gz / .tar / .zip, "
            f"or pass a plain directory."
        )

    # If the archive has a single top-level dir, drill in one level.
    entries = [e for e in tmp.iterdir() if e.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return tmp


def load_generations(gens_dir: Path) -> list[dict]:
    """Load every <id>.json file from `gens_dir`. Skips files that don't
    look like a generation (missing ``id`` or ``meta.references``)."""
    out: list[dict] = []
    for path in sorted(gens_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] skipping unparsable {path.name}: {e}")
            continue
        if "id" not in data or not isinstance(data.get("meta", {}).get("references"), list):
            print(f"[warn] skipping {path.name}: not a generation (missing id/meta.references)")
            continue
        out.append(data)
    return out


# ── Output writing ──────────────────────────────────────────────────────────

def save_sources_file(out_dir: Path, survey_id: str, data: dict) -> Path:
    """Atomic write — schema_version injected here so callers can't forget."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{survey_id}_sources.json"
    payload = dict(data)
    payload["schema_version"] = SCHEMA_VERSION
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    tmp.replace(path)
    return path


def already_done(out_dir: Path, survey_id: str) -> bool:
    """True iff `<survey_id>_sources.json` exists AND schema_version matches."""
    p = out_dir / f"{survey_id}_sources.json"
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("schema_version") == SCHEMA_VERSION


def build_empty_entry(ref: dict) -> dict:
    """Skeleton ref entry with identifiers pre-filled, evidence fields None."""
    return {
        "idx":                 ref.get("idx"),
        "arxiv_id":            ref.get("arxiv_id"),
        "semantic_scholar_id": ref.get("semantic_scholar_id"),
        "doc_id":              ref.get("doc_id"),
        "doi":                 ref.get("doi"),
        "abstract":            None,
        "text":                None,
        "abs_source":          None,
        "text_source":         None,
        "abs_errors":          None,
        "text_errors":         None,
    }


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
#
# Every config constant above is also a kwarg on `main()`. That lets you
# call the script with overrides directly from a notebook:
#
#     main(generations_input="my_gens.tgz",
#          fetch_full_text=False,
#          rate_limit_arxiv=5.0)
#
# The module-level constants stay as sensible defaults. Rate limits,
# timeouts, retries, PDF caps and user email are used deep inside helper
# functions that read them as module globals — `main()` overwrites those
# globals when you pass a different value, and restores them on exit.


def main(
    *,
    generations_input:     str  = GENERATIONS_INPUT,
    out_dir:               str  = OUT_DIR,
    dataset_id_override:   str | None = DATASET_ID_OVERRIDE,
    model_id_override:     str | None = MODEL_ID_OVERRIDE,
    fetch_full_text:       bool = FETCH_FULL_TEXT,
    user_email:            str  = USER_EMAIL,
    rate_limit_ss:         float = RATE_LIMIT_SS,
    rate_limit_arxiv:      float = RATE_LIMIT_ARXIV,
    rate_limit_crossref:   float = RATE_LIMIT_CROSSREF,
    rate_limit_openalex:   float = RATE_LIMIT_OPENALEX,
    rate_limit_unpaywall:  float = RATE_LIMIT_UNPAYWALL,
    ss_batch_size:         int   = SS_BATCH_SIZE,
    pdf_max_pages:         int   = PDF_MAX_PAGES,
    pdf_max_chars:         int   = PDF_MAX_CHARS,
    timeout_sec:           int   = TIMEOUT_SEC,
    max_retries:           int   = MAX_RETRIES,
    backoff_base:          float = BACKOFF_BASE,
) -> None:
    """Run the full bulk-fetch pipeline. All behaviour knobs are kwargs
    with module-level defaults.

    Args:
        generations_input: Archive (.tar.gz / .tar / .zip) or directory
            of ``<survey_id>.json`` generations.
        out_dir: Where to write ``<survey_id>_sources.json`` files.
        dataset_id_override / model_id_override: Tags written inside each
            sources file. When None, pulled from the first generation.
        fetch_full_text: If False, skip PDF fetching entirely
            (~20× faster). ``text`` fields remain None; ``text_errors``
            remains None.
        user_email: Polite-pool contact address for Crossref / OpenAlex /
            Unpaywall. Not validated by them; any address you control.
        rate_limit_*: Minimum seconds between consecutive calls per API.
        ss_batch_size: IDs per /paper/batch POST (max 500).
        pdf_max_pages / pdf_max_chars: Cut-offs for PDF-to-text parsing.
        timeout_sec / max_retries / backoff_base: HTTP retry policy.
    """
    # ── Override module globals that helpers consume ─────────────────────
    # Helpers (http_get, throttle, resolve_*, extract_pdf_text) read these
    # from the module namespace; mutating them here keeps the helper
    # signatures small while still letting the caller tune everything.
    global USER_EMAIL, UA
    global RATE_LIMIT_SS, RATE_LIMIT_ARXIV, RATE_LIMIT_CROSSREF
    global RATE_LIMIT_OPENALEX, RATE_LIMIT_UNPAYWALL
    global SS_BATCH_SIZE, PDF_MAX_PAGES, PDF_MAX_CHARS
    global TIMEOUT_SEC, MAX_RETRIES, BACKOFF_BASE, FETCH_FULL_TEXT

    # pymupdf presence check — only matters when we'll actually fetch PDFs.
    if fetch_full_text and fitz is None:
        raise RuntimeError(
            "fetch_full_text=True but pymupdf is not installed. "
            "Run `!pip install pymupdf` in a Colab cell above, or pass "
            "fetch_full_text=False to skip full-text fetching."
        )

    USER_EMAIL           = user_email
    UA                   = f"thesis-bulk-fetch/1.0 (mailto:{user_email})"
    RATE_LIMIT_SS        = rate_limit_ss
    RATE_LIMIT_ARXIV     = rate_limit_arxiv
    RATE_LIMIT_CROSSREF  = rate_limit_crossref
    RATE_LIMIT_OPENALEX  = rate_limit_openalex
    RATE_LIMIT_UNPAYWALL = rate_limit_unpaywall
    SS_BATCH_SIZE        = ss_batch_size
    PDF_MAX_PAGES        = pdf_max_pages
    PDF_MAX_CHARS        = pdf_max_chars
    TIMEOUT_SEC          = timeout_sec
    MAX_RETRIES          = max_retries
    BACKOFF_BASE         = backoff_base
    FETCH_FULL_TEXT      = fetch_full_text

    # ── Load generations ─────────────────────────────────────────────────
    gens_dir = prepare_generations_dir(generations_input)
    generations = load_generations(gens_dir)
    if not generations:
        raise SystemExit(f"No valid generation JSONs found under {gens_dir}")
    print(f"Loaded {len(generations)} generations from {gens_dir}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Resume — skip surveys that already have a correctly-versioned output.
    skipped, todo = [], []
    for g in generations:
        (skipped if already_done(out_path, str(g["id"])) else todo).append(g)
    if skipped:
        print(f"Resuming: {len(skipped)} sources files already present, "
              f"{len(todo)} to produce.")
    if not todo:
        print("Nothing to do — all outputs present.")
        return

    dataset_id = dataset_id_override or todo[0].get("dataset_id", "unknown")
    model_id   = model_id_override   or todo[0].get("model_id",   "unknown")

    # ── Phase 1: Collect unique SS-batch identifiers ─────────────────────
    # An identifier can be the same across multiple refs in multiple surveys
    # (same paper cited by many). We dedup at batch level — one SS call per
    # identifier, then cross-walk back per (survey_id, ref_idx) pair.
    ident_to_ref_example: dict[str, dict] = {}
    refs_by_ident: dict[str, list[tuple[str, int]]] = defaultdict(list)
    ident_for: dict[tuple[str, int], str | None] = {}
    for gen in todo:
        sid = str(gen["id"])
        for ref in gen.get("meta", {}).get("references", []):
            ident = ss_identifier_for(ref)
            ident_for[(sid, ref["idx"])] = ident
            if ident is None:
                continue
            refs_by_ident[ident].append((sid, ref["idx"]))
            if ident not in ident_to_ref_example:
                ident_to_ref_example[ident] = ref

    unique_idents = list(ident_to_ref_example)
    n_no_ident = sum(1 for v in ident_for.values() if v is None)
    print(f"Phase 1: {len(unique_idents)} unique SS identifiers "
          f"(across {sum(len(v) for v in refs_by_ident.values())} ref-occurrences); "
          f"{n_no_ident} refs have no ss_id or arxiv_id (skipped).")

    # ── Phase 2: SS /paper/batch ────────────────────────────────────────
    ss_meta:   dict[str, dict] = {}
    ss_errors: dict[str, str]  = {}
    print(f"Phase 2: SS batch ({len(unique_idents)} ids, "
          f"chunks of {SS_BATCH_SIZE}) ...")
    for i in tqdm(range(0, len(unique_idents), SS_BATCH_SIZE), desc="ss_batch"):
        chunk = unique_idents[i:i + SS_BATCH_SIZE]
        results, batch_err = ss_batch_fetch(chunk)
        if results is None:
            for ident in chunk:
                ss_errors[ident] = batch_err or "unknown"
            continue
        for ident, res in zip(chunk, results):
            if isinstance(res, dict):
                ss_meta[ident] = res
            else:
                ss_errors[ident] = "not_found"

    # ── Phase 3: Per-identifier waterfall (abstract + text) ─────────────
    print(f"Phase 3: per-identifier waterfall "
          f"(abstract{' + full_text' if FETCH_FULL_TEXT else ''}) ...")
    ident_resolved: dict[str, dict] = {}
    for ident in tqdm(unique_idents, desc="papers"):
        ref_example = ident_to_ref_example[ident]
        ss_info     = ss_meta.get(ident)
        ss_err      = ss_errors.get(ident)

        # Merge identifiers we learned from SS back onto the example ref
        # so the waterfall can also use SS-provided arxiv_id/DOI for refs
        # that started with only an ss_id.
        ext = (ss_info or {}).get("externalIds") or {}
        enriched = dict(ref_example)
        enriched.setdefault("arxiv_id", ext.get("ArXiv"))
        enriched.setdefault("doi",      ext.get("DOI"))

        abstract, abs_src, abs_errs = resolve_abstract_for_ref(enriched, ss_info, ss_err)
        if FETCH_FULL_TEXT:
            text, text_src, text_errs = resolve_text_for_ref(enriched, ss_info)
        else:
            text, text_src, text_errs = None, None, None

        abstract = strip_control_chars(abstract)
        text     = strip_control_chars(text)

        ident_resolved[ident] = {
            "abstract":    abstract,
            "text":        text,
            "abs_source":  abs_src,
            "text_source": text_src,
            "abs_errors":  abs_errs if abstract is None else None,
            "text_errors": text_errs if text is None else None,
            # Echo identifiers we ended up with — SS may have enriched
            # arxiv_id/DOI beyond what the ref carried originally.
            "_enriched_arxiv_id": enriched.get("arxiv_id"),
            "_enriched_doi":      enriched.get("doi"),
        }

    # ── Phase 4: Group per survey + write files ──────────────────────────
    print(f"Phase 4: assembling per-survey sources files → {out_path} ...")
    for gen in tqdm(todo, desc="surveys"):
        sid  = str(gen["id"])
        refs = gen.get("meta", {}).get("references", [])

        per_ref:          dict[str, dict] = {}
        abs_err_summary:  dict[str, Counter] = defaultdict(Counter)
        text_err_summary: dict[str, Counter] = defaultdict(Counter)
        n_with_abstract = n_with_text = n_no_id = 0

        for ref in refs:
            entry = build_empty_entry(ref)
            ident = ident_for[(sid, ref["idx"])]
            if ident is None:
                # Refs with no usable identifier — emit explicit error
                # trail so downstream knows why nothing was tried.
                entry["abs_errors"] = [
                    {"source": "ss",       "reason": "no_identifier"},
                    {"source": "arxiv",    "reason": "no_arxiv_id"},
                    {"source": "crossref", "reason": "no_doi"},
                    {"source": "openalex", "reason": "no_doi"},
                ]
                entry["text_errors"] = [
                    {"source": "arxiv_pdf",     "reason": "no_arxiv_id"},
                    {"source": "ss_oa_pdf",     "reason": "no_oa_url"},
                    {"source": "unpaywall_pdf", "reason": "no_doi"},
                ]
                n_no_id += 1
            else:
                r = ident_resolved[ident]
                if entry["arxiv_id"] is None and r.get("_enriched_arxiv_id"):
                    entry["arxiv_id"] = r["_enriched_arxiv_id"]
                if entry["doi"] is None and r.get("_enriched_doi"):
                    entry["doi"] = r["_enriched_doi"]
                entry["abstract"]    = r["abstract"]
                entry["text"]        = r["text"]
                entry["abs_source"]  = r["abs_source"]
                entry["text_source"] = r["text_source"]
                entry["abs_errors"]  = r["abs_errors"]
                entry["text_errors"] = r["text_errors"]

            if entry["abstract"]: n_with_abstract += 1
            if entry["text"]:     n_with_text     += 1
            for err in entry.get("abs_errors") or []:
                abs_err_summary[err["source"]][err["reason"]] += 1
            for err in entry.get("text_errors") or []:
                text_err_summary[err["source"]][err["reason"]] += 1

            per_ref[str(ref["idx"])] = entry

        data = {
            "survey_id":             sid,
            "dataset_id":            dataset_id,
            "model_id":              model_id,
            "fetched_at":            datetime.now(timezone.utc).isoformat(),
            "fetch_provider":        "colab",
            "n_refs_total":          len(refs),
            "n_refs_with_abstract":  n_with_abstract,
            "n_refs_with_text":      n_with_text,
            "n_refs_no_identifier":  n_no_id,
            "abs_error_summary":     {k: dict(v) for k, v in abs_err_summary.items()} or None,
            "text_error_summary":    {k: dict(v) for k, v in text_err_summary.items()} or None,
            "refs":                  per_ref,
        }
        save_sources_file(out_path, sid, data)

    print(f"\nDone. Wrote {len(todo)} sources files to {out_path}/")


# ════════════════════════════════════════════════════════════════════════════
# Invocation — edit kwargs here to override defaults without touching the
# config block at the top.
# ════════════════════════════════════════════════════════════════════════════

main()
