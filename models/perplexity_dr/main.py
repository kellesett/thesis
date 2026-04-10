#!/usr/bin/env python
"""
models/perplexity_dr/main.py
Generation runner for Perplexity Sonar Deep Research.

Reads config.yaml, loads the requested dataset, generates surveys via
OpenRouter, and saves unified generation JSONs to:
  results/generations/<dataset_id>_<model_id>/

Usage (inside Docker):
    python models/perplexity_dr/main.py --dataset SurGE
"""
import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT       = Path(__file__).parent.parent.parent
CONFIG     = Path(__file__).parent / "config.yaml"
sys.path.insert(0, str(ROOT))

from src.datasets import load_dataset as load_dataset_cls
from src.datasets.base import DatasetInstance


def load_registry(registry_path: Path) -> dict[str, str]:
    """Return {dataset_id: path} from datasets/registry.yaml."""
    import yaml
    with open(registry_path, encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    return {entry["id"]: entry["path"] for entry in reg["datasets"]}


def extract_annotations(response) -> list[dict]:
    """Extract url_citation annotations from response as [{idx, title, url}]."""
    annotations = getattr(response.choices[0].message, "annotations", None) or []
    return [
        {
            "idx":   i + 1,
            "title": a.url_citation.title,
            "url":   a.url_citation.url,
        }
        for i, a in enumerate(annotations)
        if getattr(a, "type", None) == "url_citation"
    ]


def enrich_with_arxiv_titles(references: list[dict]) -> list[dict]:
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

    # Collect arxiv IDs per reference index
    idx_to_arxiv: dict[int, str] = {}
    for ref in references:
        m = arxiv_re.search(ref.get("url", ""))
        if m:
            idx_to_arxiv[ref["idx"]] = m.group(1)

    # Batch query arxiv API
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
                    # Extract bare ID from URL like http://arxiv.org/abs/1911.02794v2
                    raw_id = id_el.text.strip().split("/")[-1]
                    bare_id = re.sub(r'v\d+$', '', raw_id)  # strip version suffix
                    title   = " ".join(title_el.text.split())  # normalize whitespace
                    arxiv_id_to_title[bare_id] = title
        except Exception as e:
            print(f"  [WARN] arxiv API error: {e}")

    # Annotate references
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


def build_references_section(references: list[dict]) -> str:
    """Build a ## References section in SurGE-compatible format: [N] Title.

    Uses canonical_title (from arxiv API) when available, falling back to the
    original Perplexity-returned title.
    """
    if not references:
        return ""
    lines = ["## References", ""]
    for ref in references:
        title = ref.get("canonical_title") or ref["title"]
        lines.append(f"[{ref['idx']}] {title}")
    return "\n".join(lines)


def replace_or_append_references(text: str, references: list[dict]) -> str:
    """Replace existing References section or append one at the end.

    If a References heading is found, replaces only that section —
    preserving any subsequent sections at the same or higher heading level.
    """
    if not references:
        return text
    import re
    ref_section = build_references_section(references)
    match = re.search(r"^(#{1,3})\s*references\s*$", text, re.IGNORECASE | re.MULTILINE)
    if match:
        heading_level = len(match.group(1))  # number of # in found heading
        before = text[:match.start()].rstrip()
        after_refs = text[match.end():]
        # Find next heading at same or higher level (fewer or equal #)
        next_heading = re.search(
            rf"^#{{1,{heading_level}}}\s+\S",
            after_refs,
            re.MULTILINE,
        )
        after = after_refs[next_heading.start():] if next_heading else ""
        return before + "\n\n" + ref_section + ("\n\n" + after.strip() if after else "")
    return text.rstrip() + "\n\n" + ref_section


def generate_survey(client: OpenAI, model: str, query: str,
                    search_domain_filter: list[str] | None = None) -> dict:
    """
    Call the model and return a dict with text, references, raw_response, cost, latency, and success flag.
    Perplexity Deep Research returns the full survey as a single message.
    """
    import time
    t0 = time.time()
    try:
        extra: dict = {}
        if search_domain_filter:
            extra["search_domain_filter"] = search_domain_filter

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            stream=False,
            extra_body=extra or None,
        )
        text    = response.choices[0].message.content or ""
        usage   = response.usage
        cost    = 0.0
        if usage:
            # Approximate cost: $1/M input + $5/M output tokens (sonar-deep-research pricing)
            cost = (usage.prompt_tokens * 1 + usage.completion_tokens * 5) / 1_000_000

        references = enrich_with_arxiv_titles(extract_annotations(response))

        return {
            "text":         replace_or_append_references(text, references),
            "success":      bool(text.strip()),
            "latency_sec":  round(time.time() - t0, 2),
            "cost_usd":     round(cost, 5),
            "error":        None,
            "references":   references,
            "raw_response": response.model_dump(),
        }
    except Exception as e:
        return {
            "text":         "",
            "success":      False,
            "latency_sec":  round(time.time() - t0, 2),
            "cost_usd":     0.0,
            "error":        str(e),
            "references":   [],
            "raw_response": None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate surveys with Perplexity Deep Research")
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. SurGE)")
    args = parser.parse_args()

    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_id             = cfg["model_id"]
    model                = cfg["model"]
    n_surveys            = cfg.get("n_surveys", 1)
    resume               = cfg.get("resume", True)
    search_domain_filter = cfg.get("search_domain_filter") or []

    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        raise SystemExit(f"Env var '{cfg['api_key_env']}' is not set")
    client = OpenAI(base_url=cfg["base_url"], api_key=api_key)

    # ── Load dataset ──────────────────────────────────────────────────────────
    registry = load_registry(ROOT / "datasets" / "registry.yaml")
    if args.dataset not in registry:
        raise SystemExit(f"Dataset '{args.dataset}' not in registry. Available: {list(registry)}")

    dataset = load_dataset_cls(args.dataset, registry[args.dataset])
    print(f"Dataset  : {args.dataset} ({len(dataset)} surveys, using first {n_surveys})")

    # ── Prepare output dir ────────────────────────────────────────────────────
    out_dir = ROOT / "results" / "generations" / f"{args.dataset}_{model_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model    : {model}")
    print(f"Output   : {out_dir}\n")

    # ── Generate ──────────────────────────────────────────────────────────────
    instances = list(dataset)[:n_surveys]
    for instance in instances:
        out_file = out_dir / f"{instance.id}.json"

        if resume and out_file.exists():
            try:
                existing = json.loads(out_file.read_text(encoding="utf-8"))
                if existing.get("success") and existing.get("text"):
                    print(f"  [SKIP] {instance.id}")
                    continue
            except Exception:
                pass

        inp = dataset.input_prepare(instance)
        print(f"  [GEN]  {instance.id} | {instance.query[:60]}")

        result = generate_survey(client, model, inp["query"],
                                 search_domain_filter=search_domain_filter)

        # ── Save raw response for debugging ──────────────────────────────────
        if result["raw_response"] is not None:
            raw_file = out_dir / f"{instance.id}_raw.json"
            raw_file.write_text(
                json.dumps(result["raw_response"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        generation = {
            "id":         instance.id,
            "dataset_id": args.dataset,
            "model_id":   model_id,
            "query":      inp["query"],
            "text":       result["text"],
            "success":    result["success"],
            "meta": {
                "model":                model,
                "generated_at":         datetime.now(timezone.utc).isoformat(),
                "latency_sec":          result["latency_sec"],
                "cost_usd":             result["cost_usd"],
                "error":                result["error"],
                "search_domain_filter": search_domain_filter or None,
                "references":           result["references"],
            },
        }

        out_file.write_text(
            json.dumps(generation, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        status = "OK" if result["success"] else "FAIL"
        print(
            f"    [{status}] latency={result['latency_sec']}s"
            + (f"  error={result['error']}" if result["error"] else "")
        )

    print(f"\nGenerations saved to: {out_dir}")


if __name__ == "__main__":
    main()
