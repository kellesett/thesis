#!/usr/bin/env python3
"""
scripts/enrich_references.py
Retroactively enrich already-generated files with canonical arxiv titles
and self-citation flags.

For each generation JSON in results/generations/<dir>/,
  1. Finds references with arxiv URLs and fetches canonical titles via arxiv API.
  2. Detects whether the generation cites the target survey itself
     (reference_self_cited flag in meta).

Requires the SurGE dataset to be present for self-citation detection.

Usage:
    python scripts/enrich_references.py --dir SurGE_perplexity_dr
    python scripts/enrich_references.py  # all generation directories
"""
import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from models.perplexity_dr.main import replace_or_append_references

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ARXIV_RE = re.compile(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,6})')


def fetch_arxiv_titles(arxiv_ids: list[str], max_retries: int = 2) -> dict[str, str]:
    """Batch-fetch canonical titles from arxiv API. Returns {arxiv_id: title}.

    Retries on 429 (rate limit): waits 5s then 10s before giving up.

    Args:
        arxiv_ids: List of arxiv IDs (e.g., ["2301.00001", "2301.00002"]).
        max_retries: Maximum number of retry attempts on rate limit.

    Returns:
        Dictionary mapping arxiv ID to canonical title, or empty dict on failure.
    """
    if not arxiv_ids:
        return {}
    url = (
        f"https://export.arxiv.org/api/query"
        f"?id_list={','.join(arxiv_ids)}&max_results={len(arxiv_ids)}"
    )
    logger.info(f"\n    [arxiv] GET {url[:80]}...")
    delay = 5
    for attempt in range(max_retries):
        logger.info(f"    [arxiv] attempt {attempt + 1}/{max_retries} ...", end=" ")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "thesis-research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                status = resp.status
                xml_data = resp.read()
            logger.info(f"HTTP {status}, {len(xml_data)} bytes received")

            ns   = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(xml_data)
            entries = root.findall("atom:entry", ns)
            logger.info(f"    [arxiv] parsed {len(entries)} entries")

            result = {}
            for entry in entries:
                id_el    = entry.find("atom:id", ns)
                title_el = entry.find("atom:title", ns)
                if id_el is not None and title_el is not None:
                    raw_id  = id_el.text.strip().split("/")[-1]
                    bare_id = re.sub(r'v\d+$', '', raw_id)
                    title   = " ".join(title_el.text.split())
                    result[bare_id] = title
            logger.info(f"    [arxiv] matched {len(result)}/{len(arxiv_ids)} IDs to titles")
            return result

        except urllib.error.HTTPError as e:
            logger.error(f"HTTP {e.code} {e.reason}")
            if e.code == 429 and attempt < max_retries - 1:
                next_delay = delay * 2
                logger.info(f"    [429] rate limited, waiting {delay}s before retry...")
                time.sleep(delay)
                delay = next_delay
            else:
                logger.warning(f"    [WARN] giving up after HTTP {e.code}")
                return {}
        except urllib.error.URLError as e:
            logger.error(f"URLError: {e.reason}")
            logger.warning(f"    [WARN] network error, giving up")
            return {}
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            logger.warning(f"    [WARN] bad response from arxiv, giving up")
            return {}
        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}")
            logger.warning(f"    [WARN] unexpected error, giving up")
            return {}
    return {}


def enrich_file(path: Path, survey_title: str | None = None) -> bool:
    """
    Enrich references in one generation JSON with canonical titles and self-citation flag.
    Returns True if the file was modified.

    Args:
        path:         path to generation JSON
        survey_title: title of the target survey (for self-citation detection).
                      If None, self-citation detection is skipped.
    """
    from src.evaluators.citation import detect_self_citation

    with open(path, encoding="utf-8") as f:
        gen = json.load(f)

    meta = gen.setdefault("meta", {})
    refs = meta.get("references", [])

    changed = False

    # ── Step 1: Enrich with canonical arxiv titles ────────────────────────────
    needs_enrichment = refs and not all("canonical_title" in r for r in refs)
    if needs_enrichment:
        idx_to_arxiv: dict[int, str] = {}
        for ref in refs:
            m = ARXIV_RE.search(ref.get("url", ""))
            if m:
                idx_to_arxiv[ref["idx"]] = m.group(1)

        unique_ids = list(set(idx_to_arxiv.values()))
        titles = fetch_arxiv_titles(unique_ids)

        if titles:
            for ref in refs:
                if "canonical_title" in ref:
                    continue
                arxiv_id = idx_to_arxiv.get(ref["idx"])
                ref["arxiv_id"]        = arxiv_id
                ref["canonical_title"] = titles.get(arxiv_id) if arxiv_id else None
            changed = True
        else:
            print("    [WARN] no titles fetched from arxiv API", end=" ")

    # ── Step 2: Self-citation detection ──────────────────────────────────────
    if survey_title and "reference_self_cited" not in meta:
        self_cited = detect_self_citation(refs, survey_title)
        meta["reference_self_cited"] = self_cited
        changed = True

    # ── Step 3: Rebuild ## References section in text with canonical titles ──
    if refs and all("canonical_title" in r for r in refs):
        new_text = replace_or_append_references(gen.get("text", ""), refs)
        if new_text != gen.get("text", ""):
            gen["text"] = new_text
            changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(gen, f, ensure_ascii=False, indent=2)

    return changed


def load_survey_titles(dataset_id: str) -> dict[str, str]:
    """Return {survey_id: survey_title} for a given dataset, or {} if not available.

    Args:
        dataset_id: Dataset identifier (e.g., 'SurGE').

    Returns:
        Dictionary mapping survey ID to survey title, or empty dict on error.
    """
    try:
        import yaml
        from src.datasets import load_dataset as load_ds
        with open(ROOT / "datasets" / "registry.yaml") as f:
            reg = yaml.safe_load(f)
        registry = {e["id"]: e["path"] for e in reg["datasets"]}
        if dataset_id not in registry:
            return {}
        ds = load_ds(dataset_id, registry[dataset_id])
        return {str(inst.id): inst.meta.get("survey_title", "") for inst in ds}
    except Exception as e:
        logger.warning(f"  [WARN] Could not load dataset '{dataset_id}': {e}")
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich generation references with arxiv canonical titles and self-citation flags"
    )
    parser.add_argument("--dir", help="Generation directory name (e.g. SurGE_perplexity_dr). "
                                      "If omitted, processes all directories.")
    args = parser.parse_args()

    gen_root = ROOT / "results" / "generations"
    if args.dir:
        dirs = [gen_root / args.dir]
    else:
        dirs = sorted(d for d in gen_root.iterdir() if d.is_dir())

    for gen_dir in dirs:
        if not gen_dir.exists():
            logger.error(f"Directory not found: {gen_dir}")
            continue

        gen_files = sorted(f for f in gen_dir.glob("*.json") if re.fullmatch(r'\d+\.json', f.name))
        if not gen_files:
            continue

        # Infer dataset_id from directory name (e.g. "SurGE_perplexity_dr" → "SurGE")
        dataset_id = gen_dir.name.split("_")[0]
        survey_titles = load_survey_titles(dataset_id)

        logger.info(f"\n{gen_dir.name} ({len(gen_files)} files, dataset={dataset_id})")
        for gen_file in gen_files:
            try:
                with open(gen_file) as f:
                    gid = json.load(f).get("id")
                survey_title = survey_titles.get(str(gid))

                logger.info(f"  {gen_file.name} ... ", end="")
                modified = enrich_file(gen_file, survey_title=survey_title)
                if modified:
                    self_cited = json.loads(gen_file.read_text())["meta"].get("reference_self_cited")
                    flag = " ⚠ self-cited!" if self_cited else ""
                    logger.info(f"enriched{flag}")
                    time.sleep(1)  # be polite to arxiv API
                else:
                    # Even if titles were already fetched, re-check self-citation flag
                    with open(gen_file) as f:
                        gen = json.load(f)
                    if survey_title and "reference_self_cited" not in gen.get("meta", {}):
                        from src.evaluators.citation import detect_self_citation
                        refs = gen["meta"].get("references", [])
                        self_cited = detect_self_citation(refs, survey_title)
                        gen["meta"]["reference_self_cited"] = self_cited
                        gen_file.write_text(json.dumps(gen, ensure_ascii=False, indent=2))
                        flag = " ⚠ self-cited!" if self_cited else ""
                        logger.info(f"flag added{flag}")
                    else:
                        self_cited = gen.get("meta", {}).get("reference_self_cited")
                        flag = " ⚠ self-cited!" if self_cited else ""
                        logger.info(f"skipped{flag}")
            except Exception as e:
                logger.error(f"  Error processing {gen_file.name} (ID {gid}): {e}", exc_info=True)


if __name__ == "__main__":
    main()
