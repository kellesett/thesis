#!/usr/bin/env python3
"""
scripts/parse_reference_md.py

Parses a merged LaTeX source (produced by ``scripts/merge_latex.sh``) into the
unified generation payload used by ``results/generations/<dataset>_<model>/``:

    (body_md, references) = parse(merged_tex_path)

where:

* ``body_md``         is clean GFM markdown with inline citations rewritten as
                      ``[N]`` / ``[1, 2]`` numeric references matching
                      ``references[k].idx``;
* ``references``      is ``list[dict]`` following the schema declared in
                      ``CLAUDE.md`` (``idx``, ``title``, ``url``, ``arxiv_id``,
                      ``canonical_title``). Extra keys (``raw``) may be added
                      for debugging.

Why a single Python script instead of pandoc --citeproc:
    * Many SurGE papers ship broken or non-UTF-8 .bib files → citeproc fails.
    * latexpand --expand-bbl already inlines the full thebibliography block, so
      we can extract references directly from merged.tex and do NOT need a
      separate .bib file.
    * Replacing \\cite{key} with a unique placeholder *before* calling pandoc
      lets us keep pandoc fully offline (``pandoc -f latex -t gfm``) and then
      map placeholders to numeric [N] in the markdown output, matching the
      perplexity_dr format exactly.

The script can also be used as a CLI for a quick eyeball check::

    python3 scripts/parse_reference_md.py datasets/surge/latex_src/1911.02794/merged.tex
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Bibitem parsing ───────────────────────────────────────────────────────────


@dataclass
class BibItem:
    """Parsed representation of one \\bibitem.

    Attributes:
        idx:              1-indexed position in the bibliography (matches [N]).
        key:              LaTeX citation key (argument of \\bibitem{...}).
        raw:              Full raw LaTeX content of the bibitem (for debugging
                          and as a fallback title when heuristics fail).
        title:            Best-effort plain-text title extracted from ``raw``.
        canonical_title:  Normalized title used for corpus matching (lowercased,
                          punctuation-stripped, whitespace-collapsed).
        url:              First URL found in ``raw`` (http/https), or None.
        arxiv_id:         Bare arxiv id (``YYMM.NNNNN`` or ``arch-ive/YYMMNNN``)
                          extracted from ``raw``, or None.
    """

    idx: int
    key: str
    raw: str
    title: str
    canonical_title: str | None
    url: str | None
    arxiv_id: str | None

    def to_reference(self) -> dict:
        """Return the reference entry in the canonical meta.references schema."""
        return {
            "idx":             self.idx,
            "title":           self.title,
            "url":             self.url,
            "arxiv_id":        self.arxiv_id,
            "canonical_title": self.canonical_title,
        }


_ARXIV_RE = re.compile(
    r"""(?ix)
    (?:arxiv[:\s]*|abs/|arxiv\.org/(?:abs|pdf)/)
    (\d{4}\.\d{4,5}) (?:v\d+)?
    """,
)
# Old-style arxiv identifiers like ``cs.CV/0703011`` are intentionally ignored
# here; SurGE's corpus is dominated by the post-2007 ``YYMM.NNNNN`` scheme.

_URL_RE = re.compile(r"https?://[^\s}\\]+")


def _clean_latex_fragment(s: str) -> str:
    """Turn a raw LaTeX fragment into a plain-text line.

    This is deliberately conservative: we strip the commands we actually see in
    SurGE bibitems and leave the rest alone rather than trying to be a full
    LaTeX interpreter.
    """
    if not s:
        return ""
    t = s
    # ACM bib helpers → keep inner content.
    t = re.sub(r"\\bibinfo\s*\{[^{}]*\}\s*\{", "", t)
    t = re.sub(r"\\bibfield\s*\{[^{}]*\}\s*\{", "", t)
    # Common single-arg wrappers — keep content.
    t = re.sub(
        r"\\(?:emph|textit|textbf|textsc|mbox|text|href|url|texttt)\s*\{([^{}]*)\}",
        r"\1",
        t,
    )
    # Line break / spacing macros.
    t = re.sub(r"\\\\(?:\[[^\]]*\])?", " ", t)
    t = re.sub(r"\\newblock\b", " ", t)
    t = re.sub(r"\\natexlab\s*\{[^{}]*\}", "", t)
    t = re.sub(r"\\BIBentry[A-Za-z]+", "", t)
    t = re.sub(r"\\shownote\s*\{[^{}]*\}", "", t)
    # Remaining backslash commands w/ optional brace arg: drop the command name.
    t = re.sub(r"\\[A-Za-z@]+\*?\s*\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\[A-Za-z@]+\*?", "", t)
    # Unbalanced / leftover braces and ties.
    t = t.replace("~", " ").replace("{", "").replace("}", "")
    # LaTeX quotes → ascii.
    t = t.replace("``", '"').replace("''", '"').replace("`", "'")
    t = re.sub(r"\s+", " ", t).strip(" .,;:")
    return t


def _normalize_title(title: str) -> str | None:
    """Produce a canonical_title suitable for matching against title_index."""
    if not title:
        return None
    t = title.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


def _extract_title(raw: str) -> str:
    """Best-effort extraction of a title string from a raw bibitem body."""
    # 1. ACM SIGCONF: \bibinfo{title}{...}
    m = re.search(r"\\bibinfo\s*\{\s*title\s*\}\s*\{([^{}]+)\}", raw, re.DOTALL)
    if m:
        t = _clean_latex_fragment(m.group(1))
        if t:
            return t
    # 2. IEEE tran: ``Title,''
    m = re.search(r"``([^`']{8,}?)''", raw, re.DOTALL)
    if m:
        t = _clean_latex_fragment(m.group(1))
        if t:
            return t
    # 3. \newblock Title\newblock
    parts = [p.strip() for p in re.split(r"\\newblock", raw) if p.strip()]
    if len(parts) >= 2:
        t = _clean_latex_fragment(parts[1])
        if t:
            return t.rstrip(".")
    # 4. Fallback: first ~160 chars of cleaned raw content.
    cleaned = _clean_latex_fragment(raw)
    return cleaned[:160] if cleaned else raw[:160]


def _extract_url_and_arxiv(raw: str) -> tuple[str | None, str | None]:
    """Extract the first URL and (if any) arxiv id from a raw bibitem body."""
    url: str | None = None
    m = re.search(r"\\url\s*\{([^{}]+)\}", raw)
    if m:
        url = m.group(1).strip()
    else:
        m = re.search(r"\\href\s*\{([^{}]+)\}\s*\{", raw)
        if m:
            url = m.group(1).strip()
        else:
            m = _URL_RE.search(raw)
            if m:
                url = m.group(0).rstrip(".,;:")

    arxiv_id: str | None = None
    m = _ARXIV_RE.search(raw)
    if m:
        arxiv_id = m.group(1)
    elif url:
        m = _ARXIV_RE.search(url)
        if m:
            arxiv_id = m.group(1)

    return url, arxiv_id


def _split_bibitem_head(after_bibitem: str) -> tuple[str | None, str] | None:
    """Parse the head of a substring that starts just after ``\\bibitem``.

    Returns ``(key, rest)`` where ``rest`` is the content of the bibitem
    (excluding the optional ``[label]`` and the mandatory ``{key}``).
    Returns ``None`` if no ``{key}`` could be located.

    ``[label]`` can contain arbitrary LaTeX including nested brackets, so we
    walk the string with a depth counter rather than using a regex.
    """
    s = after_bibitem.lstrip()
    i = 0
    # Skip optional [label] with bracket depth tracking.
    if i < len(s) and s[i] == "[":
        depth = 0
        while i < len(s):
            ch = s[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        else:
            return None  # unterminated [label]
    # Mandatory {key} — first balanced pair starting here.
    while i < len(s) and s[i] != "{":
        if not s[i].isspace():
            return None
        i += 1
    if i >= len(s) or s[i] != "{":
        return None
    depth = 0
    key_start = i + 1
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                key_end = i
                return s[key_start:key_end], s[i + 1:]
        i += 1
    return None


def parse_bibliography(tex: str) -> tuple[list[BibItem], tuple[int, int] | None]:
    """Extract bibitems from ``tex``.

    Returns ``(items, span)`` where ``span`` is the ``(start, end)`` character
    offsets of the ``\\begin{thebibliography}...\\end{thebibliography}`` block
    in ``tex`` (or ``None`` if no such block exists).

    ``items`` are numbered starting from 1 in the order they appear in the
    bibliography — this is the same ordering perplexity_dr uses for its ``[N]``
    citations.
    """
    m = re.search(
        r"\\begin\s*\{thebibliography\}.*?\\end\s*\{thebibliography\}",
        tex,
        re.DOTALL,
    )
    if not m:
        return [], None
    body = m.group(0)
    span = (m.start(), m.end())

    items: list[BibItem] = []
    chunks = body.split(r"\bibitem")[1:]
    idx = 0
    for chunk in chunks:
        parsed = _split_bibitem_head(chunk)
        if parsed is None:
            logger.debug("bibitem without parsable {key}: %r", chunk[:80])
            continue
        key, rest = parsed
        # Trim the trailing \end{thebibliography} from the last chunk.
        rest = re.sub(r"\\end\s*\{thebibliography\}.*$", "", rest, flags=re.DOTALL)
        raw = rest.strip()
        idx += 1
        title = _extract_title(raw)
        url, arxiv_id = _extract_url_and_arxiv(raw)
        items.append(
            BibItem(
                idx=idx,
                key=key.strip(),
                raw=raw,
                title=title,
                canonical_title=_normalize_title(title),
                url=url,
                arxiv_id=arxiv_id,
            )
        )
    return items, span


# ── \cite replacement ─────────────────────────────────────────────────────────


_CITE_CMD = re.compile(
    r"""\\(?:cite|citep|citet|citeauthor|citeyear|citealp|autocite|textcite|parencite|nocite)
        (?:\*)?                         # starred variants
        (?:\[[^\]]*\])?                 # optional prenote
        (?:\[[^\]]*\])?                 # optional postnote
        \s*\{([^{}]*)\}                 # required key(s)
    """,
    re.VERBOSE,
)

# Unique marker that survives pandoc verbatim — pandoc won't touch text that
# doesn't look like a command, so any ascii string is fine, but we pick a
# distinctive one to make post-processing unambiguous.
_CITE_PLACEHOLDER_OPEN = "\u2983CITE:"
_CITE_PLACEHOLDER_CLOSE = "\u2984"
_CITE_PLACEHOLDER_RE = re.compile(
    re.escape(_CITE_PLACEHOLDER_OPEN) + r"([0-9,\- ]+)" + re.escape(_CITE_PLACEHOLDER_CLOSE)
)


def _rewrite_cites(tex: str, key_to_idx: dict[str, int]) -> tuple[str, int, int]:
    """Replace every ``\\cite{...}`` with a numeric placeholder.

    The placeholder contains a comma-separated list of ``idx`` values in the
    same order as the original keys. Unknown keys are silently dropped; if all
    keys in a single \\cite{...} are unknown the whole command is removed.

    Returns ``(new_tex, total_cites, resolved_cites)`` where ``total_cites`` is
    the number of \\cite macros encountered and ``resolved_cites`` is how many
    resolved to at least one known key.
    """
    total = 0
    resolved = 0

    def repl(m: re.Match) -> str:
        nonlocal total, resolved
        total += 1
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        idxs = [str(key_to_idx[k]) for k in keys if k in key_to_idx]
        if not idxs:
            return ""
        resolved += 1
        return f"{_CITE_PLACEHOLDER_OPEN}{','.join(idxs)}{_CITE_PLACEHOLDER_CLOSE}"

    return _CITE_CMD.sub(repl, tex), total, resolved


# ── pandoc invocation ─────────────────────────────────────────────────────────


def _read_tex(path: Path) -> str:
    """Read a .tex file tolerating UTF-8 and latin-1 encodings.

    Many pre-2020 arXiv sources are Latin-1 (or effectively undistinguishable
    from it), which makes ``open(..., encoding='utf-8')`` blow up. We prefer
    UTF-8 but fall back to latin-1 on failure — losing no characters, since
    latin-1 is a total byte→codepoint mapping.
    """
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        logger.debug("latin-1 fallback on %s", path)
        return data.decode("latin-1", errors="replace")


def _run_pandoc(tex: str) -> str:
    """Run ``pandoc -f latex -t gfm --wrap=none`` on the supplied LaTeX string.

    Uses a temp file because some pandoc versions treat stdin input as a single
    line and mishandle ``\\input{}``-style includes (already resolved here, but
    still: file input is the common path).

    Raises:
        RuntimeError: If pandoc cannot be found or exits non-zero.
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".tex", encoding="utf-8", delete=False
    ) as fh:
        fh.write(tex)
        tmp_path = fh.name
    try:
        proc = subprocess.run(
            [
                "pandoc",
                "-f", "latex",
                "-t", "gfm-raw_html",
                "--wrap=none",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"pandoc failed ({proc.returncode}): {proc.stderr.strip()[:500]}"
            )
        return proc.stdout
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Markdown post-processing ──────────────────────────────────────────────────


_HEADER_JUNK_RE = re.compile(r"^\s*\{#[^}]*\}\s*$", re.MULTILINE)
_ATTR_SUFFIX_RE = re.compile(r"\s*\{[^{}\n]*\}\s*$", re.MULTILINE)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _render_citations(md: str) -> str:
    """Replace ``⟦CITE:1,2,3⟧`` placeholders with ``[1][2][3]`` in markdown."""

    def repl(m: re.Match) -> str:
        nums = [n.strip() for n in m.group(1).split(",") if n.strip()]
        return "".join(f"[{n}]" for n in nums)

    return _CITE_PLACEHOLDER_RE.sub(repl, md)


def _cleanup_markdown(md: str) -> str:
    """Strip pandoc attribute suffixes and collapse excessive blank lines.

    Pandoc's gfm writer annotates headers with ``{#sec:foo}`` attributes and
    images with ``{width=…}`` suffixes; neither carries useful info for metric
    evaluation. We drop them.
    """
    md = _render_citations(md)
    md = _HEADER_JUNK_RE.sub("", md)
    md = _ATTR_SUFFIX_RE.sub("", md)
    md = _MULTI_BLANK_RE.sub("\n\n", md)
    return md.strip() + "\n"


# ── Public API ────────────────────────────────────────────────────────────────


@dataclass
class ParseResult:
    """Return value of :func:`parse`.

    Attributes:
        body_md:          Cleaned markdown body (no references section).
        references:       List of reference dicts (canonical schema).
        stats:            Diagnostic counters — useful for logging/summaries.
    """

    body_md: str
    references: list[dict]
    stats: dict = field(default_factory=dict)


def parse(merged_tex: Path | str) -> ParseResult:
    """Parse a ``merged.tex`` file into ``(body_md, references, stats)``.

    Args:
        merged_tex: Path to a merged LaTeX file produced by ``merge_latex.sh``.

    Returns:
        A :class:`ParseResult`. ``body_md`` is safe to use directly as the
        ``text`` field of a generation JSON.
    """
    merged_tex = Path(merged_tex)
    tex = _read_tex(merged_tex)

    bibitems, span = parse_bibliography(tex)
    if span is not None:
        tex_body = tex[: span[0]] + tex[span[1]:]
    else:
        tex_body = tex

    key_to_idx = {b.key: b.idx for b in bibitems}
    tex_body, total_cites, resolved_cites = _rewrite_cites(tex_body, key_to_idx)

    md = _run_pandoc(tex_body)
    md = _cleanup_markdown(md)

    refs = [b.to_reference() for b in bibitems]
    stats = {
        "bibitems":       len(bibitems),
        "cites_total":    total_cites,
        "cites_resolved": resolved_cites,
        "tex_chars":      len(tex),
        "md_chars":       len(md),
        "had_bib_block":  span is not None,
    }
    return ParseResult(body_md=md, references=refs, stats=stats)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Parse a merged.tex into markdown + canonical references.",
    )
    ap.add_argument("merged_tex", type=Path)
    ap.add_argument("--out-md",   type=Path, default=None,
                    help="Write the body markdown to this path.")
    ap.add_argument("--out-refs", type=Path, default=None,
                    help="Write references as JSON to this path.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    result = parse(args.merged_tex)
    print(json.dumps(result.stats, indent=2))
    if result.references:
        preview = result.references[:3]
        print("references preview:")
        print(json.dumps(preview, indent=2, ensure_ascii=False))
    if args.out_md:
        args.out_md.write_text(result.body_md, encoding="utf-8")
    if args.out_refs:
        args.out_refs.write_text(
            json.dumps(result.references, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
