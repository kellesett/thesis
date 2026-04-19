#!/usr/bin/env python3
"""scripts/plot_section_lengths.py

EDA tool — histogram of **section-scope lengths (chars)** across all claims
in a claims directory, matched against their source generations.

Why: before running AlignScore-backed factcheck we want to know how big the
evidence-pool-context windows are going to be. Section scope (claim's own
heading-section + every ancestor's own heading-section, siblings excluded —
see :mod:`metrics.factuality.claim_scope`) sets how much text we'll feed
into AlignScore-per-candidate-ref lookups; long scopes = more refs = more
inference cost; tiny scopes = thin evidence pool.

Inputs:
    --claims-dir      — ``results/scores/<ds>_<mdl>_claims/``
    --generations-dir — ``results/generations/<ds>_<mdl>/``

Matches claims/<sid>.json to generations/<sid>.json by filename stem, pulls
``claim.sources[] = [{sentence, sentence_idx}, ...]`` (new veriscore schema,
post-source-linkage patch) and unions section-scope spans per claim. Old
claims files (``source_sentence: ""``) are transparently handled — they'd
be skipped-as-unresolvable, which is itself a useful diagnostic.

Output:
    * Stdout summary: total / skipped / failed / resolved counts + median/mean.
    * Matplotlib histogram via ``plt.show()``.

Example::

    .venv/bin/python3 scripts/plot_section_lengths.py \\
        --claims-dir      results/scores/SurGE_reference_claims \\
        --generations-dir results/generations/SurGE_reference
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics.factuality.claim_scope import (  # noqa: E402
    _find_source_offset, _section_scope_spans,
)


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent (start, end) spans so char-length sums
    don't double-count regions hit by multiple sources.

    For a claim mentioned in 2 paragraphs, the ancestor prefaces often
    overlap (same top-level heading). Without merging we'd count that
    shared prose twice. Standard interval-merge.
    """
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def section_length_for_claim(
    text: str,
    sources: list[dict],
    max_ancestor_depth: int | None = None,
) -> int | None:
    """Total char-length of the section scope for one claim.

    Sources are the claim's ``sources[]`` entries; each resolves to a set of
    section-scope spans. Spans are unioned (across sources) before summing
    lengths — avoids over-counting when the same ancestor section covers
    multiple source occurrences.

    ``max_ancestor_depth`` bounds how many heading-tree levels above the
    claim's own section we include: ``None`` = unlimited, ``0`` = own-only,
    etc. See :func:`metrics.factuality.claim_scope._section_scope_spans`.

    Returns ``None`` if none of the sources could be located in ``text``
    (the caller can count these as "failed resolution").
    """
    all_spans: list[tuple[int, int]] = []
    any_resolved = False
    for src in sources:
        off, _ = _find_source_offset(text, src.get("sentence") or "")
        if off is None:
            continue
        any_resolved = True
        all_spans.extend(_section_scope_spans(
            text, off, max_ancestor_depth=max_ancestor_depth,
        ))
    if not any_resolved:
        return None
    return sum(e - s for s, e in _merge_spans(all_spans))


def _normalized_sources(claim: dict) -> list[dict]:
    """Return claim['sources'] for the new schema; synthesize a single-source
    list from the legacy ``source_sentence[_idx]`` fields if present; return
    an empty list if neither is available (claim from pre-patch veriscore with
    empty source_sentence — unresolvable by this tool)."""
    sources = claim.get("sources") or []
    if sources:
        return sources
    legacy = claim.get("source_sentence")
    if legacy:
        return [{
            "sentence":     legacy,
            "sentence_idx": claim.get("source_sentence_idx", 0),
        }]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--claims-dir",      type=Path, required=True,
                        help="results/scores/<ds>_<mdl>_claims/")
    parser.add_argument("--generations-dir", type=Path, required=True,
                        help="results/generations/<ds>_<mdl>/")
    parser.add_argument("--bins",            type=int, default=40,
                        help="Histogram bin count (default 40).")
    parser.add_argument("--unit", choices=("chars", "kchars"), default="chars",
                        help="Display x-axis in raw chars or thousands (default chars).")
    parser.add_argument("--log-y", action="store_true",
                        help="Log scale on y-axis (counts can be heavy-tailed).")
    parser.add_argument("--max-ancestor-depth", type=int, default=None,
                        help="Cap ancestor inclusion depth in section scope. "
                             "0 = own section only; 1 = immediate parent; "
                             "default = None (unlimited, all ancestors).")
    args = parser.parse_args()

    if not args.claims_dir.is_dir():
        print(f"[ERROR] claims-dir not found: {args.claims_dir}", file=sys.stderr)
        return 2
    if not args.generations_dir.is_dir():
        print(f"[ERROR] generations-dir not found: {args.generations_dir}", file=sys.stderr)
        return 2

    # Walk claims files, numeric-by-sid so summary counts are deterministic.
    claim_files = sorted(
        (f for f in args.claims_dir.glob("*.json") if f.stem != "summary"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9,
    )
    if not claim_files:
        print(f"[ERROR] no *.json claim files in {args.claims_dir}", file=sys.stderr)
        return 2

    lengths: list[int] = []
    # Counters track every claim we touched, regardless of outcome — so the
    # final ratio "resolved / total" is a diagnostic of how well the
    # claim-scope locator is doing on this data.
    n_total           = 0
    n_no_gen          = 0
    n_no_sources      = 0   # claim has neither sources[] nor source_sentence
    n_failed_resolve  = 0   # sources exist but none match in gen text

    for cf in claim_files:
        gen_file = args.generations_dir / cf.name
        if not gen_file.exists():
            n_no_gen += 1
            continue

        claims_data = json.loads(cf.read_text(encoding="utf-8"))
        gen         = json.loads(gen_file.read_text(encoding="utf-8"))
        text        = gen.get("text", "")

        for claim in claims_data.get("claims", []):
            n_total += 1
            sources = _normalized_sources(claim)
            if not sources:
                n_no_sources += 1
                continue
            L = section_length_for_claim(
                text, sources, max_ancestor_depth=args.max_ancestor_depth,
            )
            if L is None:
                n_failed_resolve += 1
                continue
            lengths.append(L)

    # ── Stdout summary ────────────────────────────────────────────────────────
    print(f"\n[section-lengths]  {args.claims_dir.name}")
    print(f"  claim files scanned: {len(claim_files)}")
    print(f"  total claims:        {n_total}")
    print(f"  no matching gen:     {n_no_gen}  (files)")
    print(f"  no sources field:    {n_no_sources}  (claims — old schema or empty)")
    print(f"  failed to resolve:   {n_failed_resolve}  (claims — sentence not found in text)")
    print(f"  resolved & plotted:  {len(lengths)}")
    if lengths:
        print(f"\n  median: {median(lengths):>7.0f} chars")
        print(f"  mean:   {mean(lengths):>7.0f} chars")
        print(f"  min:    {min(lengths):>7d} chars")
        print(f"  max:    {max(lengths):>7d} chars")
    print()

    if not lengths:
        print("Nothing to plot.", file=sys.stderr)
        return 1

    # ── Histogram ─────────────────────────────────────────────────────────────
    if args.unit == "kchars":
        data = [L / 1000 for L in lengths]
        xlabel = "Section scope length (thousands of chars)"
    else:
        data = lengths
        xlabel = "Section scope length (chars)"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=args.bins, edgecolor="black", alpha=0.75, color="#3b78c2")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of claims")
    depth_tag = (
        "all ancestors" if args.max_ancestor_depth is None
        else f"depth={args.max_ancestor_depth}"
    )
    ax.set_title(
        f"Section-scope length — {args.claims_dir.name}  [{depth_tag}]\n"
        f"{len(lengths)} claims resolved (of {n_total})  ·  "
        f"median {median(lengths):.0f} chars, mean {mean(lengths):.0f} chars"
    )
    ax.grid(axis="y", alpha=0.3)
    if args.log_y:
        ax.set_yscale("log")
    # Median + mean markers — visual anchor for the distribution
    ax.axvline(
        median(data), color="darkorange", linestyle="--", linewidth=1.2,
        label=f"median ({median(data):.0f})",
    )
    ax.axvline(
        mean(data), color="firebrick", linestyle=":", linewidth=1.2,
        label=f"mean ({mean(data):.0f})",
    )
    ax.legend()
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
