"""
src/evaluators/citation.py
Corpus-based citation metrics for survey generation evaluation.

Metrics:
  citation_count       — raw number of references in the generation
  corpus_match_rate    — fraction of references whose titles were found in the
                         SurGE corpus (proxy for academic source quality)
  coverage             — SurGE Coverage: fraction of all_cites (target survey's
                         reference list) that appear in the generation
  reference_self_cited — 1 if the generation cites the target survey itself, else 0

Usage:
    evaluator = CitationEvaluator(corpus_index_path)
    result = evaluator.evaluate(generation, target_survey_meta)
    # result = {"citation_count": 50, "corpus_match_rate": 0.20, "coverage": 0.10}

Notes:
  - The title index must be pre-built once with build_title_index().
  - Perplexity reference titles are cleaned before matching (strip arXiv IDs,
    "[PDF]" markers, site suffixes, and truncation "...").
  - normalize_string() mirrors SurGE's own normalization (letters only, lowercase).
  - Coverage can be 0 when the model cites recent papers not present in the
    target survey's era — this is an expected and scientifically meaningful result.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path


# ── SurGE normalization (matches evaluator.py in SurGE repo) ─────────────────

def normalize_string(s: str) -> str:
    """Keep only ASCII letters, lowercase — mirrors SurGE's normalize_string."""
    letters = re.findall(r'[a-zA-Z]', s)
    return ''.join(letters).lower()


# ── Title cleaning for Perplexity-returned references ────────────────────────

def clean_perplexity_title(title: str | None) -> str:
    """
    Strip noise from titles returned by Perplexity via OpenRouter annotations.

    Perplexity titles often contain:
      - "[1911.02794] Title - arXiv"
      - "[PDF] Title"
      - "Title ... - arXiv.org"   (truncated)
      - "Title - ar5iv"

    After cleaning, the title should match the corpus Title field.
    """
    if not title:
        return ""
    t = title.strip()
    # Remove [PDF] prefix
    t = re.sub(r'^\[PDF\]\s*', '', t, flags=re.IGNORECASE)
    # Remove [arXiv_id] prefix like [1911.02794]
    t = re.sub(r'^\[\d{4}\.\d{4,6}\]\s*', '', t)
    # Remove trailing " - SiteName" suffixes
    t = re.sub(
        r'\s*[-–]\s*'
        r'(arXiv(?:\.org)?|ar5iv|Semantic\s*Scholar|ResearchGate'
        r'|ACM|IEEE|Springer|CVPR|NeurIPS|ICLR|ICML|AAAI|EMNLP|ACL|NAACL)\b.*$',
        '', t, flags=re.IGNORECASE,
    )
    # Remove trailing " - domain.tld" pattern
    t = re.sub(r'\s*[-–]\s*\S+\.(com|org|edu|io|net)\S*\s*$', '', t, flags=re.IGNORECASE)
    # Remove trailing "..." (truncated titles — cannot do exact match)
    t = re.sub(r'\s*\.\.\.\s*$', '', t)
    return t.strip()


# ── Self-citation detection ───────────────────────────────────────────────────

def detect_self_citation(references: list[dict], survey_title: str) -> bool:
    """
    Return True if any reference matches the target survey's own title.

    This detects the case where a deep research agent (e.g. Perplexity) finds
    and cites the very survey it is being evaluated against — a form of
    reference leakage that can inflate text-similarity metrics.

    Matching uses normalize_string (letters only, lowercase) for robustness
    against minor title variations. Both canonical_title (from arxiv API) and
    the cleaned Perplexity title are checked.
    """
    target_key = normalize_string(survey_title)
    for ref in references:
        # Check canonical title first (most reliable)
        canonical = ref.get("canonical_title")
        if canonical and normalize_string(canonical) == target_key:
            return True
        # Fallback: cleaned Perplexity title
        clean = clean_perplexity_title(ref.get("title"))
        if clean and normalize_string(clean) == target_key:
            return True
    return False


# ── Index builder (run once) ──────────────────────────────────────────────────

def build_title_index(corpus_path: Path, index_path: Path) -> None:
    """
    Stream through corpus.json and build a {normalized_title: doc_id} index.
    Saves the result to index_path as a JSON file (~80 MB).
    Avoids loading the full 1.6 GB corpus into memory.
    """
    title_index: dict[str, int] = {}
    title_re = re.compile(r'"Title"\s*:\s*"(.*?)"(?:,\s*)?$')
    docid_re  = re.compile(r'"doc_id"\s*:\s*(\d+)')

    title_val = None
    docid_val = None
    count = 0
    start = time.time()

    print(f"Building title index from {corpus_path} ...")
    with open(corpus_path, encoding='utf-8') as f:
        for line in f:
            t = title_re.search(line)
            if t:
                title_val = t.group(1)
            d = docid_re.search(line)
            if d:
                docid_val = int(d.group(1))
            if title_val is not None and docid_val is not None:
                title_index[normalize_string(title_val)] = docid_val
                title_val = None
                docid_val = None
                count += 1

    elapsed = time.time() - start
    print(f"  Indexed {count:,} entries in {elapsed:.1f}s")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(title_index, f, ensure_ascii=False)
    size_mb = index_path.stat().st_size / 1024 / 1024
    print(f"  Saved index → {index_path} ({size_mb:.1f} MB)")


# ── Main evaluator ────────────────────────────────────────────────────────────

class CitationEvaluator:
    """
    Computes citation metrics for a generated survey using the SurGE corpus index.

    Args:
        index_path: path to pre-built title_index.json
    """

    def __init__(self, index_path: Path) -> None:
        if not index_path.exists():
            raise FileNotFoundError(
                f"Title index not found at {index_path}.\n"
                f"Build it first with: build_title_index(corpus_path, index_path)"
            )
        print(f"Loading title index from {index_path} ...")
        with open(index_path, encoding='utf-8') as f:
            self._index: dict[str, int] = json.load(f)
        print(f"  {len(self._index):,} titles loaded.")

    def _match_references(self, references: list[dict]) -> list[int | None]:
        """
        Map each reference dict to a corpus doc_id or None.

        Lookup priority:
          1. canonical_title — fetched from arxiv API at generation time (clean)
          2. Cleaned Perplexity title — fallback when canonical not available
        """
        results: list[int | None] = []
        for ref in references:
            # Prefer canonical title (arxiv API) over noisy Perplexity title
            canonical = ref.get("canonical_title")
            if canonical:
                key = normalize_string(canonical)
                doc_id = self._index.get(key)
                if doc_id is not None:
                    results.append(doc_id)
                    continue
            # Fallback: clean Perplexity title
            clean = clean_perplexity_title(ref.get('title'))
            key   = normalize_string(clean)
            results.append(self._index.get(key))
        return results

    def match_references(self, references: list[dict]) -> list[int | None]:
        """Public wrapper for callers that need the same title→doc_id mapping."""
        return self._match_references(references)

    def evaluate(
        self,
        generation: dict,
        target_survey_meta: dict,
    ) -> dict[str, float | int | None]:
        """
        Compute citation metrics for one generation.

        Args:
            generation:          unified generation dict (must have meta.references)
            target_survey_meta:  dataset instance .meta dict (must have all_cites)

        Returns dict with keys:
            citation_count        (int)
            corpus_match_rate     (float 0–1)
            coverage              (float 0–1)
            reference_self_cited  (int: 1 or 0)
        """
        references: list[dict] = (generation.get('meta') or {}).get('references', [])
        all_cites: list[int]   = target_survey_meta.get('all_cites', [])
        survey_title: str      = target_survey_meta.get('survey_title', '')

        citation_count = len(references)
        self_cited = detect_self_citation(references, survey_title)

        if citation_count == 0:
            return {
                'citation_count':       0,
                'corpus_match_rate':    None,
                'coverage':             0.0 if all_cites else None,
                'reference_self_cited': int(self_cited),
            }

        doc_ids = self._match_references(references)
        matched_doc_ids = [d for d in doc_ids if d is not None]

        corpus_match_rate = len(matched_doc_ids) / citation_count

        if not all_cites:
            coverage = None
        else:
            all_cites_set = set(all_cites)
            found_in_target = sum(1 for d in matched_doc_ids if d in all_cites_set)
            coverage = found_in_target / len(all_cites_set)

        return {
            'citation_count':       citation_count,
            'corpus_match_rate':    round(corpus_match_rate, 4),
            'coverage':             round(coverage, 4) if coverage is not None else None,
            'reference_self_cited': int(self_cited),
        }
