#!/usr/bin/env python3
"""
metrics/claimify/main.py
Atomic claim decomposition for generated surveys.

Splits each survey into sections, then uses an LLM to extract atomic claims
per section. Results are saved to:

  results/scores/<dataset_id>_<model_id>_claims/<survey_id>.json

Format:
  {
    "survey_id": 0,
    "dataset_id": "SurGE",
    "model_id":   "perplexity_dr",
    "query":      "...",
    "n_sections": 8,
    "n_claims":   312,
    "claims": [
      {"claim_id": 0, "claim": "...", "section": "Introduction", "source_sentence": "..."},
      ...
    ],
    "judge_model": "openai/gpt-4o-mini",
    "timestamp":   "2026-04-11T..."
  }

factuality/ and expert/ metrics read this cache and fail if it's absent.

Usage (inside Docker):
    python metrics/claimify/main.py --dataset SurGE --model perplexity_dr
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT   = Path(__file__).parent.parent.parent
CONFIG = Path(__file__).parent / "config.yaml"
sys.path.insert(0, str(ROOT))


# ── Config & client ───────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def make_client(cfg: dict) -> OpenAI:
    api_key = os.environ.get(cfg["judge_api_key_env"], "")
    if not api_key:
        raise RuntimeError(
            f"API key not set: env var '{cfg['judge_api_key_env']}'"
        )
    return OpenAI(api_key=api_key, base_url=cfg["judge_base_url"])


# ── Markdown section splitter ─────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def split_sections(text: str) -> list[dict]:
    """
    Split Markdown survey text into sections.
    Returns list of {"title": str, "text": str}.
    Sections with fewer than 100 chars of content are skipped.
    """
    headings = list(_HEADING_RE.finditer(text))
    sections = []

    for i, m in enumerate(headings):
        title  = m.group(2).strip()
        start  = m.end()
        end    = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body   = text[start:end].strip()

        # Skip short / empty bodies (e.g. sub-headings with no content)
        if len(body) < 100:
            continue

        sections.append({"title": title, "text": body})

    # If no headings found, treat whole text as one section
    if not sections:
        sections = [{"title": "Full text", "text": text.strip()}]

    return sections


def chunk_section(section: dict, max_chars: int) -> list[dict]:
    """
    Split a long section into chunks of ≤ max_chars for safe LLM processing.
    Splits on paragraph boundaries where possible.
    """
    text = section["text"]
    if len(text) <= max_chars:
        return [section]

    chunks = []
    paragraphs = re.split(r"\n{2,}", text)
    current, buf = "", []
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            buf.append(para)
            current += para + "\n\n"
        else:
            if buf:
                chunks.append({"title": section["title"], "text": current.strip()})
            buf   = [para]
            current = para + "\n\n"
    if buf:
        chunks.append({"title": section["title"], "text": current.strip()})

    return chunks


# ── LLM decomposition ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert at decomposing scientific survey text into atomic claims. "
    "An atomic claim is a single, self-contained factual assertion that cannot be "
    "further broken down without losing meaning. Each claim must be a complete sentence."
)

_USER_TEMPLATE = """\
Extract all atomic claims from the following section of a scientific survey.

Section title: "{section_title}"

Text:
\"\"\"{text}\"\"\"

Rules:
- Each claim must be a single, self-contained sentence expressing one fact or assertion
- Strip inline citation markers like [1], [Author et al., 2023], etc. — do NOT include them
- Do NOT paraphrase beyond what the text says; stay faithful to the original meaning
- Headings, figure captions, and reference lists are NOT claims — skip them
- Abbreviations should be kept as-is (e.g. "BERT" not "Bidirectional Encoder Representations from Transformers")

Return a JSON array (and nothing else):
[
  {{"claim": "...", "source_sentence": "..."}},
  ...
]
where "source_sentence" is the verbatim sentence (or short passage) the claim was drawn from."""


def decompose_section(
    section: dict,
    client: OpenAI,
    model: str,
    max_retries: int,
) -> list[dict]:
    """Call LLM to extract atomic claims from one section chunk."""
    prompt = _USER_TEMPLATE.format(
        section_title=section["title"],
        text=section["text"][:4000],  # hard cap for safety
    )

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()

            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            claims = json.loads(raw)
            if not isinstance(claims, list):
                raise ValueError("Expected JSON array")
            return claims

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print(f"    [WARN] parse failed after {max_retries} attempts: {e}")
                return []
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print(f"    [WARN] LLM call failed: {e}")
                return []

    return []


# ── Per-survey processing ─────────────────────────────────────────────────────

def process_survey(
    gen: dict,
    out_path: Path,
    cfg: dict,
    client: OpenAI,
) -> dict | None:
    """
    Decompose one survey into atomic claims and save to out_path.
    Returns summary dict or None on skip/error.
    """
    survey_id = gen["id"]
    out_file  = out_path / f"{survey_id}.json"

    if cfg.get("resume") and out_file.exists():
        try:
            with open(out_file) as f:
                existing = json.load(f)
            print(f"  [SKIP] {survey_id} — {existing['n_claims']} claims already saved")
            return existing
        except Exception:
            print(f"  [WARN] {survey_id} — corrupt cache, re-processing")

    if not gen.get("success", False):
        print(f"  [SKIP] {survey_id} — generation not successful")
        return None

    text = gen.get("text", "").strip()
    if not text:
        print(f"  [SKIP] {survey_id} — empty text")
        return None

    print(f"  [PROC] {survey_id} | {gen.get('query', '')[:60]}")

    sections = split_sections(text)
    max_chars = cfg.get("max_section_chars", 4000)
    max_retries = cfg.get("max_retries", 3)

    all_claims: list[dict] = []
    claim_id = 0

    for section in sections:
        chunks = chunk_section(section, max_chars)
        for chunk in chunks:
            raw_claims = decompose_section(
                chunk, client, cfg["judge_model"], max_retries
            )
            for c in raw_claims:
                if isinstance(c, dict) and c.get("claim"):
                    all_claims.append({
                        "claim_id":       claim_id,
                        "claim":          c["claim"].strip(),
                        "section":        section["title"],
                        "source_sentence": c.get("source_sentence", "").strip(),
                    })
                    claim_id += 1

    result = {
        "survey_id":   survey_id,
        "dataset_id":  gen["dataset_id"],
        "model_id":    gen["model_id"],
        "query":       gen.get("query", ""),
        "n_sections":  len(sections),
        "n_claims":    len(all_claims),
        "claims":      all_claims,
        "judge_model": cfg["judge_model"],
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

    with open(out_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"         → {len(all_claims)} claims from {len(sections)} sections")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Atomic claim decomposition")
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. SurGE)")
    parser.add_argument("--model",   required=True, help="Model id (e.g. perplexity_dr)")
    args = parser.parse_args()

    cfg    = load_config()
    client = make_client(cfg)

    gen_dir = ROOT / "results" / "generations" / f"{args.dataset}_{args.model}"
    if not gen_dir.exists():
        print(f"[ERROR] Generation dir not found: {gen_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = ROOT / "results" / "scores" / f"{args.dataset}_{args.model}_claims"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_files = sorted(gen_dir.glob("*.json"))
    # Exclude _raw.json and _old.json side-files
    gen_files = [f for f in gen_files if not re.search(r"_(raw|old)\.json$", f.name)]

    print(f"\n[claimify] {args.dataset} / {args.model}")
    print(f"           {len(gen_files)} surveys → {out_dir}\n")

    n_ok, n_skip, n_err = 0, 0, 0
    total_claims = 0

    for gf in gen_files:
        try:
            with open(gf) as f:
                gen = json.load(f)
        except Exception as e:
            print(f"  [ERROR] reading {gf.name}: {e}")
            n_err += 1
            continue

        try:
            result = process_survey(gen, out_dir, cfg, client)
            if result is None:
                n_skip += 1
            else:
                n_ok += 1
                total_claims += result["n_claims"]
        except Exception as e:
            print(f"  [ERROR] {gf.stem}: {e}")
            traceback.print_exc()
            n_err += 1

    print(f"\n[claimify] done — ok={n_ok} skip={n_skip} err={n_err}")
    print(f"           total claims: {total_claims}")
    if n_ok > 0:
        print(f"           avg per survey: {total_claims // n_ok}")


if __name__ == "__main__":
    main()
