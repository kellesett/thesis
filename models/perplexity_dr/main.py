#!/usr/bin/env python
"""
models/perplexity_dr/main.py
Generation runner for Perplexity Sonar Deep Research.

Usage (inside Docker):
    python models/perplexity_dr/main.py --dataset SurGE
"""
import os
import sys
import json
import argparse
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.base import BaseModel


# ── Module-level helper (used by scripts/enrich_references.py) ────────────────

def replace_or_append_references(text: str, references: list[dict]) -> str:
    """Replace existing References section or append one at the end.

    If a References heading is found, replaces only that section —
    preserving any subsequent sections at the same or higher heading level.
    """
    if not references:
        return text
    import re
    ref_section = _build_references_section(references)
    match = re.search(r"^(#{1,3})\s*references\s*$", text, re.IGNORECASE | re.MULTILINE)
    if match:
        heading_level = len(match.group(1))
        before = text[:match.start()].rstrip()
        after_refs = text[match.end():]
        next_heading = re.search(
            rf"^#{{1,{heading_level}}}\s+\S",
            after_refs,
            re.MULTILINE,
        )
        after = after_refs[next_heading.start():] if next_heading else ""
        return before + "\n\n" + ref_section + ("\n\n" + after.strip() if after else "")
    return text.rstrip() + "\n\n" + ref_section


def _build_references_section(references: list[dict]) -> str:
    """Build a ## References section in SurGE-compatible format: [N] Title."""
    if not references:
        return ""
    lines = ["## References", ""]
    for ref in references:
        title = ref.get("canonical_title") or ref["title"]
        lines.append(f"[{ref['idx']}] {title}")
    return "\n".join(lines)


# ── Model class ───────────────────────────────────────────────────────────────

class PerplexityDR(BaseModel):
    """Perplexity Sonar Deep Research via OpenAI-compatible API."""

    def __init__(self) -> None:
        super().__init__(Path(__file__).parent)

        api_key_env = self.cfg["api_key_env"]
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise SystemExit(f"Env var '{api_key_env}' is not set")

        self.client = OpenAI(base_url=self.cfg["base_url"], api_key=api_key)
        self.model: str = self.cfg["model"]
        self.search_domain_filter: list[str] = self.cfg.get("search_domain_filter") or []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def extract_annotations(self, response) -> list[dict]:
        """Extract url_citation annotations from response as [{idx, title, url}].

        idx is the 1-based position in the FULL annotations array (including
        non-url_citation entries).  Perplexity embeds citation markers like [1],
        [2], … in the text by positional index, so this must reflect the full
        array position — not just the count of url_citations seen so far.
        """
        annotations = getattr(response.choices[0].message, "annotations", None) or []
        return [
            {
                "idx":   i + 1,   # positional index in full annotations list
                "title": a.url_citation.title,
                "url":   a.url_citation.url,
            }
            for i, a in enumerate(annotations)
            if getattr(a, "type", None) == "url_citation"
        ]

    def enrich_with_arxiv_titles(self, references: list[dict]) -> list[dict]:
        """
        For references with arxiv URLs, fetch canonical titles via arxiv API.

        Adds two fields to each ref:
          arxiv_id        — e.g. "1911.02794" (or None)
          canonical_title — official arxiv title (or None if not found / not arxiv)

        Makes a single batch request to minimize API calls.
        """
        import re
        import urllib.request
        import xml.etree.ElementTree as ET

        arxiv_re = re.compile(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,6})')

        idx_to_arxiv: dict[int, str] = {}
        for ref in references:
            m = arxiv_re.search(ref.get("url", ""))
            if m:
                idx_to_arxiv[ref["idx"]] = m.group(1)

        arxiv_id_to_title: dict[str, str] = {}
        if idx_to_arxiv:
            ids = list(set(idx_to_arxiv.values()))
            url = f"https://export.arxiv.org/api/query?id_list={','.join(ids)}&max_results={len(ids)}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "thesis-research/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    xml_data = resp.read()
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                root = ET.fromstring(xml_data)
                for entry in root.findall("atom:entry", ns):
                    id_el    = entry.find("atom:id", ns)
                    title_el = entry.find("atom:title", ns)
                    if id_el is not None and title_el is not None:
                        raw_id  = id_el.text.strip().split("/")[-1]
                        bare_id = re.sub(r'v\d+$', '', raw_id)
                        title   = " ".join(title_el.text.split())
                        arxiv_id_to_title[bare_id] = title
            except Exception as e:
                print(f"  [WARN] arxiv API error: {e}")

        enriched = []
        for ref in references:
            arxiv_id = idx_to_arxiv.get(ref["idx"])
            canonical = arxiv_id_to_title.get(arxiv_id) if arxiv_id else None
            enriched.append({
                **ref,
                "arxiv_id":        arxiv_id,
                "canonical_title": canonical,
            })
        return enriched

    # ── BaseModel interface ───────────────────────────────────────────────────

    def generate(self, instance) -> dict:
        """Call Perplexity Sonar Deep Research and return unified generation dict."""
        import time

        query = instance.query
        t0 = time.time()
        try:
            extra: dict = {}
            if self.search_domain_filter:
                extra["search_domain_filter"] = self.search_domain_filter

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": query}],
                stream=False,
                extra_body=extra or None,
            )
            text  = response.choices[0].message.content or ""
            usage = response.usage
            cost  = 0.0
            if usage:
                # Approximate cost: $1/M input + $5/M output tokens
                cost = (usage.prompt_tokens * 1 + usage.completion_tokens * 5) / 1_000_000

            references = self.enrich_with_arxiv_titles(self.extract_annotations(response))
            text_with_refs = replace_or_append_references(text, references)

            # Save raw response for debugging
            if self.out_dir is not None:
                raw_file = self.out_dir / f"{instance.id}_raw.json"
                raw_file.write_text(
                    json.dumps(response.model_dump(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            return {
                "text":    text_with_refs,
                "success": bool(text.strip()),
                "meta": {
                    "model":                self.model,
                    "latency_sec":          round(time.time() - t0, 2),
                    "cost_usd":             round(cost, 5),
                    "error":                None,
                    "search_domain_filter": self.search_domain_filter or None,
                    "references":           references,
                },
            }

        except Exception as e:
            import traceback
            return {
                "text":    "",
                "success": False,
                "meta": {
                    "model":                self.model,
                    "latency_sec":          round(time.time() - t0, 2),
                    "cost_usd":             0.0,
                    "error":                f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                    "search_domain_filter": self.search_domain_filter or None,
                    "references":           [],
                },
            }


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate surveys with Perplexity Deep Research")
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. SurGE)")
    args = parser.parse_args()

    PerplexityDR().run(args.dataset)
