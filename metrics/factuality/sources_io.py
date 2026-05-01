"""metrics/factuality/sources_io.py

Single source of truth for the per-survey sources file schema.

For every generation (one paper) we persist a single JSON describing which
``[N]`` citations in that paper's text map to which evidence (abstract +
full text), who fetched them, and which waterfall tiers failed.

Path (canonical, for both producers/consumers)::

    results/generations/<dataset_id>_<model_id>/sources/<survey_id>_sources.json

Two producers write these files:

* ``scripts/colab_bulk_fetch.py``    — runs in Colab (SS reachable there).
* ``metrics/factuality/evidence_fetcher.py`` (internal mode) — runs at
  factuality launch, per-survey.

Two consumers read them:

* ``metrics/factuality/evidence_fetcher.py`` (both modes — after load,
  builds the ``key_to_evidence`` dict that the align stage consumes).
* ``scripts/colab_fetch_stats.py`` — dashboards coverage + failure stats
  across all surveys.

Schema is versioned via :data:`SCHEMA_VERSION`. Bump it on any schema
change; loaders refuse mismatches explicitly instead of letting downstream
code silently degrade.

Schema (top-level)::

    {
      "schema_version":        1,
      "survey_id":             "0",
      "dataset_id":            "SurGE",
      "model_id":              "reference",
      "fetched_at":            "2026-04-19T14:22:03Z",
      "fetch_provider":        "colab" | "internal_abstract" | "internal_full_text"
                               | "internal_full_text_or_abstract",

      # Aggregate counters (summaries of `refs` below — redundant but
      # cheap to re-use at dashboard time).
      "n_refs_total":          int,
      "n_refs_with_abstract":  int,
      "n_refs_with_text":      int,
      "n_refs_no_identifier":  int,     # neither ss_id nor arxiv_id

      # Error aggregation by tier (None for internal-mode files — internal
      # tiers don't currently track reasons; populated by colab).
      "abs_error_summary":     {"<tier>": {"<reason>": count, ...}, ...} | null,
      "text_error_summary":    {"<tier>": {"<reason>": count, ...}, ...} | null,

      # Per-ref dict, keyed by ref_idx as a string (JSON keys are strings).
      "refs": {
        "1": {
          "idx":                 1,
          "arxiv_id":            str | None,
          "semantic_scholar_id": str | None,
          "doc_id":              int | None,
          "doi":                 str | None,
          "abstract":            str | None,
          "text":                str | None,
          "abs_source":          "ss" | "arxiv" | "crossref" | "openalex"
                                 | "corpus" | "arxiv_api" | "ss_paperid"
                                 | "ss_arxiv" | None,
          "text_source":         "arxiv_pdf" | "ss_oa_pdf" | "unpaywall_pdf"
                                 | "arxiv_latex" | None,
          "abs_errors":          [{"source": ..., "reason": ...}, ...] | null,
          "text_errors":         [{"source": ..., "reason": ...}, ...] | null
        },
        "2": {...},
        ...
      }
    }

Notes:
    * ``<survey_id>_sources.json`` contains **every** ref in the generation's
      ``meta.references``, not only the ones cited by claims. This keeps the
      file decoupled from claim-extraction choices (claimify / veriscore).
    * Empty evidence fields (``abstract`` / ``text``) are stored as ``None``;
      downstream treats them as "no evidence for this ref".
    * For colab-produced files the matching ``*_errors`` field lists every
      tier that was attempted and failed (in priority order). For internal-
      mode files these are usually ``null`` — internal fetchers don't
      propagate per-tier reasons today.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 1


# ── Helpers ──────────────────────────────────────────────────────────────────


def canonical_sources_dir(root: Path, dataset_id: str, model_id: str) -> Path:
    """Return the canonical sources directory for a (dataset, model) pair.

    Centralised so both the producer (internal evidence_fetcher) and any
    consumer computes the same path. ``root`` is the repo root.
    """
    return root / "results" / "generations" / f"{dataset_id}_{model_id}" / "sources"


# ── Save / load ──────────────────────────────────────────────────────────────


def save_gen_sources(out_dir: Path, survey_id: str, data: dict) -> Path:
    """Atomic write of ``<survey_id>_sources.json``.

    ``data`` should already follow the schema documented at the top of this
    module; ``schema_version`` is injected here automatically so callers
    cannot forget it. Written via ``.tmp → rename`` to survive
    mid-write crashes.

    Returns the absolute path that was written.
    """
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


def load_gen_sources(in_dir: Path, survey_id: str) -> dict | None:
    """Load ``<survey_id>_sources.json`` from ``in_dir``.

    Returns:
        The parsed dict on success, or ``None`` when the file is absent.

    Raises:
        ValueError: when the file exists but ``schema_version`` doesn't
        match :data:`SCHEMA_VERSION`. The caller should either delete the
        file to force re-generation, or run a migration script.
    """
    path = in_dir / f"{survey_id}_sources.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    got = data.get("schema_version")
    if got != SCHEMA_VERSION:
        raise ValueError(
            f"sources schema mismatch at {path}: "
            f"file is v{got!r}, expected v{SCHEMA_VERSION}. "
            f"Delete the file and re-run the producer, or run a migration."
        )
    return data


def build_empty_entry(ref: dict) -> dict:
    """Skeleton ref entry — all fields present, values pulled from `ref`
    when available, evidence fields null.

    Use this when populating a sources file; callers fill in ``abstract``,
    ``text``, sources, errors as they resolve.
    """
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
