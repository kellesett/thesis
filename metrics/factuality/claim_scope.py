"""metrics/factuality/claim_scope.py

Given a claim's source sentence(s) and the generation's markdown body,
resolve the set of inline ``[N]`` citations that co-occur with the claim in
its local scope — either the **paragraph** or the whole **section** it sits
in — and map each ``[N]`` back to a reference record (arxiv_id / SS id /
doc_id).

Downstream the factcheck pipeline feeds those refs into evidence fetching
(abstract via SS, full-text via arxiv) and AlignScore-based support
verification. This module is the glue between veriscore's claim → source
linkage and the evidence-pool construction.

Design notes
------------
* No external deps beyond the stdlib ``re`` — the module is intentionally
  tiny and deterministic. spaCy is not re-tokenized here; the caller already
  supplies ``source_sentence`` as the exact string extracted by veriscore.
* Citation regex excludes markdown-link forms like ``[1](#anchor)`` so
  numeric bracket markers in figure/table refs don't contaminate the pool.
* Resolution failure (sentence not found in text) is reported back via
  counters, not an exception — a claim with 3 sources, 2 of which resolve
  cleanly, should still contribute its 2 scope-windows to the evidence pool.
"""
from __future__ import annotations

import re


# ── Regexes ──────────────────────────────────────────────────────────────────

# Markdown heading: leading ``#``'s determine level. ``^`` anchors to line
# start (multiline mode); we also require at least one whitespace after.
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+\S.*$", re.MULTILINE)

# Inline citation: "[N]" or "[N1, N2, ...]". Negative lookahead ``(?!\()``
# rejects markdown-link forms like "[1](#fig:systemCategory)" where the
# bracket is really the display text of a link, not a citation. We still
# accept "[1][2]" (two adjacent citations) because the lookahead sees the
# second ``[`` after the first ``]``.
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\](?!\()")


# ── Source-sentence locator ──────────────────────────────────────────────────


def _find_source_offset(text: str, source_sentence: str) -> tuple[int, str] | tuple[None, None]:
    """Locate ``source_sentence`` in ``text`` and report how we found it.

    Returns ``(offset, resolution)`` where ``resolution`` is ``"exact"`` or
    ``"fallback"``, or ``(None, None)`` if neither strategy finds the sentence.

    Strategy:
        1. Exact substring match. veriscore stores ``source_sentence`` as the
           stripped ``.text`` of a spaCy ``Sent``, so when the same markdown
           body is re-read it's almost always a direct substring.
        2. Fallback: first ~10 content words. Handles rare whitespace / edge
           punctuation drift between extraction and retrieval.
    """
    if not source_sentence:
        return (None, None)

    off = text.find(source_sentence)
    if off != -1:
        return (off, "exact")

    # Fallback: chain of decreasing-length word-prefix anchors (10 → 5 → 3).
    # Handles common drifts:
    #   * trailing punctuation / whitespace — caught at k=10
    #   * ending paraphrase — caught at k=5 if the middle is preserved
    #   * heavy tail rewrite — k=3 is a last-resort prefix
    # The anchor must be at least 10 chars to avoid matching random
    # 3-word snippets that happen to occur elsewhere in the text.
    words = source_sentence.split()
    for k in (10, 5, 3):
        if len(words) < k and k > 3:
            continue  # try a shorter anchor — avoid "10 words" === full sentence
        anchor = " ".join(words[: min(k, len(words))])
        if len(anchor) < 10:
            continue
        off = text.find(anchor)
        if off != -1:
            return (off, "fallback")

    return (None, None)


# ── Scope spans ──────────────────────────────────────────────────────────────


def _paragraph_span(text: str, offset: int) -> tuple[int, int]:
    """Bounds of the markdown paragraph containing ``offset``.

    Paragraph separator is ``\\n\\n`` (two consecutive newlines) — the
    canonical markdown-blank-line form. Span is inclusive on start, exclusive
    on end (standard Python slice semantics).
    """
    prev_break = text.rfind("\n\n", 0, offset)
    start = prev_break + 2 if prev_break != -1 else 0
    next_break = text.find("\n\n", offset)
    end = next_break if next_break != -1 else len(text)
    return (start, end)


def _section_scope_spans(
    text: str,
    offset: int,
    max_ancestor_depth: int | None = None,
) -> list[tuple[int, int]]:
    """All text spans that form the "section scope" of a claim at ``offset``.

    A **section** is defined strictly as the text between two adjacent
    headings of any level — i.e. the heading's own local content, containing
    no nested headings. Under that rule the scope of a claim is the union
    of:

      1. the claim's own section (its immediate heading → next heading), and
      2. every ancestor's own section (each ancestor heading → its next
         heading — which is necessarily the first deeper child of that
         ancestor, so we capture the parent's "preface" prose that set up
         the subtree).

    Siblings of any ancestor are explicitly excluded: they branch off a
    different sub-topic and their citations are not evidence for this claim.

    Args:
        text:   The markdown body.
        offset: Character position of the claim in ``text``.
        max_ancestor_depth:
            How many ancestor levels to include in the scope. ``None`` = no
            limit (all ancestors up to the document root, default behaviour).
            ``0`` = no ancestors, own section only. ``1`` = immediate parent
            only. ``2`` = parent + grandparent. Etc.

            Rationale: on survey-length markdown the root ``# Introduction``
            can carry 5-10 kB of shared prose with dozens of citations,
            which inflates deeply-nested claims' scopes into the 20k-char
            range. Capping the depth trims that long tail when the upper-
            level context is too broad for a specific claim.

    Example — claim sitting in Header3's prose (level 3)::

        ## Header1             ← ancestor (L2)
        text1                  ← included (Header1's own span)
        ### Header2            ← sibling of Header3 (same L3 level)
        text2                  ← EXCLUDED
        ### Header3            ← claim's heading
        text3                  ← included (Header3's own span)
        #### Header4           ← child of Header3
        text4                  ← EXCLUDED (would be Header4's own span)

    Result with ``max_ancestor_depth=None``: scope = text1 + text3.
    Result with ``max_ancestor_depth=0``: scope = text3 only (no ancestors).

    Edge cases:
      * Claim in text before any heading → return only the preamble span
        ``(0, first_heading_off)``.
      * No headings at all → the whole document is one span.
      * Claim under a heading that has no descendants and no next heading
        (e.g. a lone ``# Title`` followed by prose to end of doc) → span
        runs to end of document. There's no "next heading" to bound it.
    """
    headings = [(m.start(), len(m.group(1))) for m in _HEADING_RE.finditer(text)]
    if not headings:
        return [(0, len(text))]

    # Index of the claim's immediate heading (nearest at or before offset).
    current_idx: int | None = None
    for i, (h_off, _) in enumerate(headings):
        if h_off > offset:
            break
        current_idx = i

    if current_idx is None:
        # Claim is in the preamble (before any heading). The preamble itself
        # is the entire scope — it doesn't belong to any heading, and under
        # the strict "between two headings" rule we can't pull in anything
        # from below.
        return [(0, headings[0][0])]

    # Stack-based ancestor detection: for each heading up to and including
    # the current one, pop siblings and same-or-deeper headings before push.
    # The stack invariant is strictly decreasing levels top-to-bottom.
    stack: list[int] = []  # indices into ``headings``
    for i, (_, h_level) in enumerate(headings):
        if i > current_idx:
            break
        while stack and headings[stack[-1]][1] >= h_level:
            stack.pop()
        stack.append(i)

    # stack[-1] is current, stack[:-1] are strict ancestors of the claim
    # (ordered root-first, since they were pushed in document order and
    # popping never removes genuine ancestors — only siblings at or below).
    ancestor_idxs = stack[:-1]
    if max_ancestor_depth is not None:
        # Keep only the N closest ancestors (rightmost — closest to current
        # in the heading hierarchy). max_ancestor_depth=0 drops all.
        ancestor_idxs = (
            ancestor_idxs[-max_ancestor_depth:] if max_ancestor_depth > 0 else []
        )
    scope_idxs = ancestor_idxs + [current_idx]

    spans: list[tuple[int, int]] = []
    for idx in scope_idxs:
        h_off = headings[idx][0]
        h_end = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
        spans.append((h_off, h_end))
    return spans


# ── Citation extraction ──────────────────────────────────────────────────────


def _citations_in_span(text: str, start: int, end: int) -> list[int]:
    """All ``[N]`` / ``[N1, N2]`` citation indices inside ``text[start:end]``.

    Order of appearance is preserved (caller will typically dedup into a
    set); markdown-link brackets ``[N](...)`` are skipped via the negative
    lookahead baked into ``_CITATION_RE``.
    """
    out: list[int] = []
    for m in _CITATION_RE.finditer(text, start, end):
        for num in m.group(1).split(","):
            out.append(int(num.strip()))
    return out


# ── Ref triple lookup ────────────────────────────────────────────────────────


def _ref_triple(idx: int, references: list[dict]) -> dict:
    """Collect the three canonical ids for citation ``[idx]``.

    Output always has the three keys — ``arxiv_id``, ``semantic_scholar_id``,
    ``doc_id`` — with ``None`` where a particular source doesn't have that
    identifier, or when the citation index doesn't resolve to any reference
    at all (orphan ``[N]``). Consumers should treat ``None``s as "not
    fetchable via this channel" and fall back to another channel.
    """
    ref = next((r for r in references if r.get("idx") == idx), None)
    return {
        "idx":                 idx,
        "arxiv_id":            (ref.get("arxiv_id")            if ref else None),
        "semantic_scholar_id": (ref.get("semantic_scholar_id") if ref else None),
        "doc_id":              (ref.get("doc_id")              if ref else None),
    }


# ── Public entrypoint ────────────────────────────────────────────────────────


def resolve_claim_scope(
    text: str,
    sources: list[dict],
    references: list[dict],
    *,
    max_ancestor_depth: int | None = None,
) -> dict:
    """Union-of-scopes citation resolution for one claim.

    For each source sentence of the claim we compute two evidence scopes:

      * **paragraph** — the single markdown paragraph (between ``\\n\\n``
        boundaries) that contains the sentence.
      * **section** — the claim's own heading-section (text between its
        immediate heading and the next heading of any level) plus up to
        ``max_ancestor_depth`` closest ancestors' own heading-sections.
        Siblings at all levels are excluded. ``None`` = all ancestors (the
        heading-tree root is the furthest one). See
        :func:`_section_scope_spans` for the detailed rule.

    When the claim occurs in multiple sentences (see veriscore's
    ``sources[]`` output for duplicate-claim handling), we take the union of
    citations across all occurrence sites for both scopes — a claim backed
    by citations at any of its occurrence locations is validly
    evidence-covered.

    Args:
        text:       The generation's markdown body (``gen["text"]``).
        sources:    ``[{"sentence": str, "sentence_idx": int}, ...]`` from
                    ``claims[i]["sources"]``.
        references: ``gen["meta"]["references"]`` — canonical reference
                    list with ``idx`` / ``arxiv_id`` / ``semantic_scholar_id``
                    / ``doc_id`` fields.

    Returns:
        A dict::

            {
              "paragraph_refs": [
                {"idx": 5, "arxiv_id": ..., "semantic_scholar_id": ..., "doc_id": ...},
                ...
              ],                      # unioned across sources, sorted by idx
              "section_refs":   [...],
              "paragraph_spans": [(start, end), ...],   # one per resolved
                                                        # source, in source
                                                        # order; used by the
                                                        # caller to slice
                                                        # ``text`` for the
                                                        # classify-context
                                                        # prompt (surrounding
                                                        # survey prose, not
                                                        # the cited paper's
                                                        # abstract).
              "n_sources_resolved": int,     # how many sources were located
              "n_sources_total":    int,     # == len(sources)
              "resolution":         "exact" | "fallback" | "mixed" | "failed",
            }

        ``resolution`` aggregates per-source status:
          * ``"exact"``   — every source found by direct substring
          * ``"fallback"``— all resolved, at least one via the 10-word anchor
          * ``"mixed"``   — some resolved, some not (partial coverage)
          * ``"failed"``  — no source resolved; lists are empty
    """
    paragraph_idxs:  set[int]             = set()
    section_idxs:    set[int]             = set()
    paragraph_spans: list[tuple[int, int]] = []
    resolutions:     list[str]             = []

    for src in sources:
        sentence = src.get("sentence") or ""
        offset, res = _find_source_offset(text, sentence)
        if offset is None:
            resolutions.append("failed")
            continue
        resolutions.append(res)

        p_start, p_end = _paragraph_span(text, offset)
        paragraph_spans.append((p_start, p_end))
        paragraph_idxs.update(_citations_in_span(text, p_start, p_end))

        # Section scope is the union of the claim's own heading-section and
        # each included ancestor's own heading-section. Depth capped by
        # ``max_ancestor_depth`` (None = all ancestors).
        for s_start, s_end in _section_scope_spans(
            text, offset, max_ancestor_depth=max_ancestor_depth,
        ):
            section_idxs.update(_citations_in_span(text, s_start, s_end))

    n_resolved = sum(1 for r in resolutions if r != "failed")
    if n_resolved == 0:
        aggregate = "failed"
    elif "failed" in resolutions:
        aggregate = "mixed"
    elif any(r == "fallback" for r in resolutions):
        aggregate = "fallback"
    else:
        aggregate = "exact"

    return {
        "paragraph_refs":     [_ref_triple(i, references) for i in sorted(paragraph_idxs)],
        "section_refs":       [_ref_triple(i, references) for i in sorted(section_idxs)],
        "paragraph_spans":    paragraph_spans,
        "n_sources_resolved": n_resolved,
        "n_sources_total":    len(sources),
        "resolution":         aggregate,
    }
