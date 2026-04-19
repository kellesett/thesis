"""scripts/colab_fetch_stats.py

Summary statistics over a directory of per-survey sources files
(``<OUT_DIR>/<survey_id>_sources.json``) produced by
``colab_bulk_fetch.py``.

Drops straight into a Colab cell — constants at the top, no CLI,
pure stdlib. Reports:

  * number of surveys scanned + total refs across them
  * abstract coverage (found / None) + per-source breakdown
  * full-text coverage (found / None) + per-source breakdown
  * joint distribution (both / abstract-only / text-only / neither)
  * text-length summary (min / median / mean / max chars)
  * failure-reason breakdown per tier
  * survey_id + ref_idx pairs where nothing was resolved (for manual follow-up)
"""

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

IN_DIR = "sources_out"

# How many "nothing found" (survey_id, ref_idx) pairs to print at the end.
# Set to 0 to suppress, or large to see the full list.
SHOW_MISSING_IDS = 30


# ════════════════════════════════════════════════════════════════════════════
# Code
# ════════════════════════════════════════════════════════════════════════════

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def load_sources_dir(in_dir: str) -> list[dict]:
    """Load every <sid>_sources.json from `in_dir`. Skips corrupt files."""
    p = Path(in_dir)
    if not p.is_dir():
        raise SystemExit(f"Input directory not found: {in_dir}")
    out: list[dict] = []
    for path in sorted(p.glob("*_sources.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] skipping unparsable {path.name}: {e}")
            continue
        out.append(data)
    return out


def iter_ref_entries(surveys: list[dict]):
    """Yield (survey_id, ref_idx, entry) for every ref in every survey."""
    for s in surveys:
        sid = s.get("survey_id", "?")
        for idx_str, entry in (s.get("refs") or {}).items():
            yield sid, idx_str, entry


def fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return "   —  "
    return f"{100 * n / total:5.1f}%"


def fmt_count_row(label: str, n: int, total: int) -> str:
    return f"  {label:<28}{n:>6}   {fmt_pct(n, total)}"


def print_source_breakdown(title: str, sources: Counter, total: int) -> None:
    print(title)
    if not sources:
        print("  (nothing found)")
        return
    # Sort by count desc, with None last.
    ordered = sorted(sources.items(), key=lambda kv: (kv[0] is None, -kv[1]))
    for src, cnt in ordered:
        label = src if src is not None else "(None — not found)"
        print(fmt_count_row(label, cnt, total))


def text_length_stats(entries: list[dict], field: str) -> None:
    lens = [len(e[field]) for e in entries if e.get(field)]
    if not lens:
        print(f"  {field}: no non-empty entries")
        return
    print(f"  {field}: n={len(lens)}")
    print(f"    min    = {min(lens):>8,} chars")
    print(f"    median = {int(median(lens)):>8,} chars")
    print(f"    mean   = {int(mean(lens)):>8,} chars")
    print(f"    max    = {max(lens):>8,} chars")


def count_errors(entries: list[dict], field: str) -> dict[str, Counter]:
    """Aggregate error trails into ``{source: {reason: count}}``.

    Only entries where the target field came back None carry an ``*_errors``
    trail — this is a view over failures only.
    """
    acc: dict[str, Counter] = defaultdict(Counter)
    for e in entries:
        for err in e.get(field) or []:
            src    = err.get("source", "?")
            reason = err.get("reason", "?")
            acc[src][reason] += 1
    return acc


_ABS_SOURCES  = ("ss",        "arxiv",     "crossref",      "openalex")
_TEXT_SOURCES = ("arxiv_pdf", "ss_oa_pdf", "unpaywall_pdf")


def print_error_breakdown(
    title: str, errs: dict[str, Counter], canonical: tuple[str, ...],
) -> None:
    print(title)
    if not errs:
        print("  (no failed entries)")
        return
    ordered = [s for s in canonical if s in errs] + \
              [s for s in errs if s not in canonical]
    for src in ordered:
        reasons = errs[src]
        total = sum(reasons.values())
        print(f"  {src:<18} ({total} failures)")
        for reason, cnt in reasons.most_common():
            print(f"    {reason:<28}{cnt:>5}")


# ── Main ────────────────────────────────────────────────────────────────────

surveys = load_sources_dir(IN_DIR)
n_surveys = len(surveys)
print(f"Surveys in {IN_DIR}: {n_surveys}\n")
if n_surveys == 0:
    raise SystemExit("Empty input — nothing to report.")

# Collapse to flat list of ref entries across all surveys for global stats.
all_entries: list[dict] = []
all_triples: list[tuple[str, str, dict]] = []
for sid, idx, entry in iter_ref_entries(surveys):
    all_entries.append(entry)
    all_triples.append((sid, idx, entry))

total = len(all_entries)
print(f"Total refs across surveys: {total}\n")
if total == 0:
    raise SystemExit("No ref entries found.")

# ── Coverage ────────────────────────────────────────────────────────────────
n_abs  = sum(1 for e in all_entries if e.get("abstract"))
n_text = sum(1 for e in all_entries if e.get("text"))

print("Coverage:")
print(fmt_count_row("abstracts found",    n_abs,          total))
print(fmt_count_row("abstracts missing",  total - n_abs,  total))
print(fmt_count_row("full texts found",   n_text,         total))
print(fmt_count_row("full texts missing", total - n_text, total))
print()

# ── Joint ───────────────────────────────────────────────────────────────────
both      = sum(1 for e in all_entries if e.get("abstract") and e.get("text"))
abs_only  = sum(1 for e in all_entries if e.get("abstract") and not e.get("text"))
text_only = sum(1 for e in all_entries if not e.get("abstract") and e.get("text"))
nothing   = sum(1 for e in all_entries if not e.get("abstract") and not e.get("text"))

print("Joint (abstract × text):")
print(fmt_count_row("both present",      both,      total))
print(fmt_count_row("abstract only",     abs_only,  total))
print(fmt_count_row("text only",         text_only, total))
print(fmt_count_row("neither",           nothing,   total))
print()

# ── Source breakdown ────────────────────────────────────────────────────────
abs_sources  = Counter(e.get("abs_source")  for e in all_entries)
text_sources = Counter(e.get("text_source") for e in all_entries)

print_source_breakdown("Abstract source breakdown:",  abs_sources,  total)
print()
print_source_breakdown("Full-text source breakdown:", text_sources, total)
print()

# ── Text length ─────────────────────────────────────────────────────────────
print("Text length (non-empty entries):")
text_length_stats(all_entries, "abstract")
text_length_stats(all_entries, "text")
print()

# ── Failure reason breakdown ────────────────────────────────────────────────
abs_errs  = count_errors(all_entries, "abs_errors")
text_errs = count_errors(all_entries, "text_errors")
print_error_breakdown(
    "Abstract failure reasons (entries where abstract is None):",
    abs_errs, _ABS_SOURCES,
)
print()
print_error_breakdown(
    "Full-text failure reasons (entries where text is None):",
    text_errs, _TEXT_SOURCES,
)
print()

# ── Per-survey aggregate counters (stored inside each file) ─────────────────
# Aggregate the top-level counters that colab_bulk_fetch writes for faster
# cross-survey inspection.
n_no_id_total = sum(s.get("n_refs_no_identifier", 0) for s in surveys)
print("Per-survey top-level (from file headers):")
print(f"  sum n_refs_no_identifier: {n_no_id_total}")
# Histogram of (n_refs_with_abstract / n_refs_total) per survey
print(f"  per-survey abstract coverage (non-zero denom):")
rates = [
    s["n_refs_with_abstract"] / s["n_refs_total"]
    for s in surveys if s.get("n_refs_total")
]
if rates:
    print(f"    min={min(rates):.2%}  median={median(rates):.2%}  "
          f"mean={mean(rates):.2%}  max={max(rates):.2%}")
print()

# ── Missing refs (survey_id, ref_idx) pairs ─────────────────────────────────
missing = [
    (sid, idx) for sid, idx, e in all_triples
    if not e.get("abstract") and not e.get("text")
]
if missing and SHOW_MISSING_IDS > 0:
    print(f"Refs with nothing resolved ({len(missing)} total, "
          f"showing up to {SHOW_MISSING_IDS}):")
    for sid, idx in missing[:SHOW_MISSING_IDS]:
        print(f"  survey={sid}  ref_idx={idx}")
    if len(missing) > SHOW_MISSING_IDS:
        print(f"  ... and {len(missing) - SHOW_MISSING_IDS} more")
