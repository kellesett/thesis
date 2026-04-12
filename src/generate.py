"""
Test survey generation systems via OpenRouter and/or local server.

Installation:
    pip install openai datasets tqdm python-dotenv

.env:
    OPENROUTER_API_KEY=sk-or-...
    LOCAL_API_BASE=http://localhost:8000/v1   # URL of local server (DeepSeek etc.)
    LOCAL_API_KEY=anything                   # key (usually any string)
    LOCAL_MODEL=deepseek-chat                # model name on local server

Usage:
    python base.py                                   # all systems via OpenRouter
    python base.py --dry_run                         # 1 topic, all systems
    python base.py --systems openai                  # OpenAI DR only
    python base.py --systems autosurvey --local      # academic pipeline on local DeepSeek
    python base.py --topics 3 --local                # first 3 topics, academic locally

Backend logic:
    DR-systems (openai, perplexity) → always OpenRouter
    Academic-systems → OpenRouter by default, or local server with --local
"""

import os
import json
import time
import argparse
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Model configuration ──────────────────────────────────────────────────────

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# backend: "openrouter"  — always via OpenRouter
#          "academic"    — OpenRouter by default; switches to local server with --local
SYSTEMS = {
    # ── Deep Research systems (always OpenRouter) ──────────────────────────────
    "openai_o4_mini_dr": {
        "model": "openai/o4-mini-deep-research",
        "category": "deep_research",
        "backend": "openrouter",
        "price_in": 2.0,
        "price_out": 8.0,
        "note": "OpenAI Deep Research (budget-friendly)",
    },
    "perplexity_dr": {
        "model": "perplexity/sonar-deep-research",
        "category": "deep_research",
        "backend": "openrouter",
        "price_in": 2.0,
        "price_out": 8.0,
        "note": "Perplexity Sonar Deep Research",
    },
    # ── Academic pipeline (OpenRouter or local with --local) ──────────────────
    "autosurvey_gpt4o": {
        "model": "openai/gpt-4o",
        "category": "academic",
        "backend": "academic",
        "price_in": 2.50,
        "price_out": 10.0,
        "note": "AutoSurvey pipeline (GPT-4o backend)",
    },
    "autosurvey_gemini": {
        "model": "google/gemini-2.5-pro",
        "category": "academic",
        "backend": "academic",
        "price_in": 1.25,
        "price_out": 10.0,
        "note": "AutoSurvey pipeline (Gemini 2.5 Pro backend)",
    },
    "autosurvey_local": {
        "model": None,           # taken from LOCAL_MODEL in .env
        "category": "academic",
        "backend": "academic",
        "price_in": 0.0,
        "price_out": 0.0,
        "note": "AutoSurvey pipeline (local server)",
    },
}


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class SurveyResult:
    system_id: str
    model: str
    category: str
    topic_id: str
    topic: str
    generated_text: str
    references: list[str] = field(default_factory=list)
    arxiv_ids: dict = field(default_factory=dict)  # {arxiv_id: {"arxivId": ...}} для SurveyForge
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_sec: float = 0.0
    error: Optional[str] = None

    @property
    def success(self):
        return self.error is None and len(self.generated_text) > 200

    @property
    def word_count(self):
        return len(self.generated_text.split())

    @property
    def reference_count(self):
        return len(self.references)

    def to_dict(self):
        return asdict(self)


# ─── Промпты ──────────────────────────────────────────────────────────────────

DEEP_RESEARCH_PROMPT = """Write a comprehensive scientific survey on the following topic:

Topic: {topic}

Requirements:
1. Cover major contributions, milestones, and key papers in this area
2. Describe the main methods, models, and approaches
3. Discuss important datasets and benchmarks
4. Identify open challenges and future directions
5. Cite relevant papers with proper references in [Author et al., Year] format

Structure the survey with clear sections (Introduction, Background, Methods, \
Applications, Challenges, Conclusion). Include a References section at the end \
with all cited works."""

ACADEMIC_OUTLINE_PROMPT = """Create a detailed outline for a comprehensive scientific survey on: {topic}

Generate 6-8 main sections with 2-4 subsections each.
Format as a numbered list. Be specific about what each section should cover."""

ACADEMIC_SECTION_PROMPT = """You are writing a section of a scientific survey on: {topic}

Full survey outline:
{outline}

Write the section: {section}

Requirements:
- 400-700 words
- Cite relevant papers as [Author et al., Year]
- Be technically accurate and specific
- Connect to other sections where appropriate"""


# ─── Клиент OpenRouter ────────────────────────────────────────────────────────

def make_client(api_key: str, base_url: str = OPENROUTER_BASE) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def chat(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int = 8000,
    temperature: float = 0.3,
) -> tuple[str, int, int]:
    """Call the model. Return (text, input_tokens, output_tokens)."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_headers={
            "HTTP-Referer": "https://github.com/survey-benchmark",
            "X-Title": "Survey Generation Benchmark",
        },
    )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    return text, usage.prompt_tokens, usage.completion_tokens


# ─── Generators ───────────────────────────────────────────────────────────────────

def generate_deep_research(
    client: OpenAI, model: str, topic: str
) -> tuple[str, int, int]:
    """Single request — full survey. For DR models."""
    prompt = DEEP_RESEARCH_PROMPT.format(topic=topic)
    return chat(client, model, [{"role": "user", "content": prompt}], max_tokens=12000)


def generate_academic_pipeline(
    client: OpenAI, model: str, topic: str
) -> tuple[str, int, int]:
    """
    Simplified AutoSurvey pipeline:
    1. Generate outline
    2. Generate 4 key sections
    3. Assemble
    """
    total_in, total_out = 0, 0

    # Step 1: outline
    outline, inp, out = chat(
        client, model,
        [{"role": "user", "content": ACADEMIC_OUTLINE_PROMPT.format(topic=topic)}],
        max_tokens=1500,
    )
    total_in += inp
    total_out += out

    # Step 2: parse sections (take first 4 main ones)
    sections = _parse_sections(outline)[:4]
    if not sections:
        sections = ["Introduction and Background", "Methods and Approaches",
                    "Applications and Results", "Challenges and Future Directions"]

    # Step 3: generate sections
    section_texts = {}
    for sec in sections:
        text, inp, out = chat(
            client, model,
            [{"role": "user", "content": ACADEMIC_SECTION_PROMPT.format(
                topic=topic, outline=outline, section=sec
            )}],
            max_tokens=1200,
        )
        section_texts[sec] = text
        total_in += inp
        total_out += out
        time.sleep(1)  # rate limiting

    # Assembly
    full_text = f"# A Survey on {topic}\n\n"
    full_text += f"## Outline\n\n{outline}\n\n---\n\n"
    for sec, text in section_texts.items():
        full_text += f"## {sec}\n\n{text}\n\n"
    full_text += "## References\n\n[See inline citations above]"

    return full_text, total_in, total_out


def _parse_sections(outline: str) -> list[str]:
    sections = []
    for line in outline.split("\n"):
        line = line.strip()
        m = re.match(r"^(?:\d+[\.\)]\s*|#{1,3}\s*)(.+)", line)
        if m:
            title = m.group(1).strip().rstrip(":")
            if len(title) > 5 and not title[0].isspace():
                sections.append(title)
    return sections


def _extract_refs(text: str) -> list[str]:
    """Extract text references like [Author et al., 2023]."""
    inline = re.findall(r"\[([A-Z][^,\[\]]{2,40},\s*\d{4}[^\]]*)\]", text)
    ref_section = []
    in_refs = False
    for line in text.split("\n"):
        s = line.strip()
        if re.search(r"^#*\s*(references|bibliography)", s, re.I):
            in_refs = True
            continue
        if in_refs and s and not s.startswith("#"):
            ref_section.append(s)
    all_refs = list(dict.fromkeys(inline + ref_section))
    return all_refs[:150]


def _extract_arxiv_ids(text: str) -> dict:
    """
    Extract arxiv ID from survey text.
    Matches patterns: arxiv:YYMM.NNNNN, arxiv.org/abs/..., arXiv:...
    Returns dict {arxiv_id: {"arxivId": arxiv_id}} — SurveyForge ref.json format.
    """
    patterns = [
        r"arxiv[:\s/]+(\d{4}\.\d{4,5}(?:v\d+)?)",  # arxiv:2301.12345 or arxiv/2301.12345
        r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",  # arxiv.org/abs/2301.12345
        r"\b(\d{4}\.\d{4,5})\b",  # bare ID like 2301.12345 (less reliable)
    ]
    found = {}
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            raw_id = m.group(1)
            # Remove version (v1, v2...)
            clean_id = re.sub(r"v\d+$", "", raw_id)
            # Basic validation: YYMM.NNNNN — year 00-99, month 01-12
            parts = clean_id.split(".")
            if len(parts) == 2 and len(parts[0]) == 4 and len(parts[1]) in (4, 5):
                year = int(parts[0][:2])
                month = int(parts[0][2:4])
                # arxiv started in April 1991 (9704), validate year range
                if (7 <= year <= 30) and (1 <= month <= 12):
                    found[clean_id] = {"arxivId": clean_id}
    return found


# ─── Main runner ──────────────────────────────────────────────────────────────

def run_system(
    client: OpenAI,
    system_id: str,
    config: dict,
    topic_id: str,
    topic: str,
    local_client: Optional[OpenAI] = None,
) -> SurveyResult:
    model = config["model"]
    category = config["category"]
    backend = config.get("backend", "openrouter")
    start = time.time()

    # Client selection: DR-systems always OpenRouter; academic — local if provided
    active_client = client
    if backend == "academic" and local_client is not None:
        active_client = local_client
        model = config.get("local_model") or model  # local_model can override

    try:
        if category == "deep_research":
            text, inp, out = generate_deep_research(active_client, model, topic)
        else:
            text, inp, out = generate_academic_pipeline(active_client, model, topic)

        latency = time.time() - start
        cost = (inp * config["price_in"] + out * config["price_out"]) / 1_000_000

        arxiv_ids = _extract_arxiv_ids(text)
        return SurveyResult(
            system_id=system_id,
            model=model,
            category=category,
            topic_id=topic_id,
            topic=topic,
            generated_text=text,
            references=_extract_refs(text),
            arxiv_ids=arxiv_ids,
            input_tokens=inp,
            output_tokens=out,
            cost_usd=cost,
            latency_sec=latency,
        )

    except Exception as e:
        logger.exception(f"Generation failed for {system_id}/{topic_id}: {e}")
        return SurveyResult(
            system_id=system_id,
            model=model,
            category=category,
            topic_id=topic_id,
            topic=topic,
            generated_text="",
            latency_sec=time.time() - start,
            error=str(e),
        )


# ─── Load topics ──────────────────────────────────────────────────────────────

# Mapping topic → human review title (from test.py SurveyForge)
TOPIC_TO_HUMAN_TITLE = {
    "3D Gaussian Splatting": "A Survey on 3D Gaussian Splatting",
    "3D Object Detection in Autonomous Driving": "3D Object Detection for Autonomous Driving: A Comprehensive Survey",
    "Evaluation of Large Language Models": "A Survey on Evaluation of Large Language Models",
    "LLM-based Multi-Agent": "A survey on large language model based autonomous agents",
    "Generative Diffusion Models": "A survey on generative diffusion models",
    "Graph Neural Networks": "Graph neural networks: Taxonomy, advances, and trends",
    "Hallucination in Large Language Models": "Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models",
    "Multimodal Large Language Models": "A Survey on Multimodal Large Language Models",
    "Retrieval-Augmented Generation for Large Language Models": "Retrieval-augmented generation for large language models: A survey",
    "Vision Transformers": "A survey of visual transformers",
}

ROOT_DIR        = Path(__file__).parent.parent
SURVEYBENCH_DIR = ROOT_DIR / "repos" / "SurveyForge" / "SurveyBench"


def load_topics(n: int = 10) -> list[dict]:
    """
    Load topics from SurveyForge/SurveyBench/topics.txt.
    topic_id = topic slug (used in result filenames).
    """
    topics_file = SURVEYBENCH_DIR / "topics.txt"
    if not topics_file.exists():
        raise FileNotFoundError(
            f"Not found: {topics_file}\n"
            "Ensure SurveyForge is cloned to repos/SurveyForge/"
        )
    topics = []
    with open(topics_file) as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            # slug for filename: remove special chars
            slug = re.sub(r"[^\w]", "_", name).strip("_")
            topics.append({
                "topic_id": slug,
                "topic": name,
                "human_title": TOPIC_TO_HUMAN_TITLE.get(name, name),
            })
    topics = topics[:n]
    logger.info(f"Loaded {len(topics)} topics from {topics_file}")
    return topics


# ─── CLI and entry point ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Survey generation benchmark via OpenRouter / local server")
    parser.add_argument("--systems", default="all",
                        help="Systems comma-separated or 'all': openai,perplexity,autosurvey_gpt4o,autosurvey_gemini,autosurvey_local")
    parser.add_argument("--topics", type=int, default=5,
                        help="Number of topics from SurveyBench (default: 5)")
    parser.add_argument("--dry_run", action="store_true",
                        help="One topic, all selected systems")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-generated results")
    parser.add_argument("--out", default=str(ROOT_DIR / "results"),
                        help="Directory for results")
    parser.add_argument("--local", action="store_true",
                        help="Run academic systems on local server (LOCAL_API_BASE from .env)")
    args = parser.parse_args()

    # ── OpenRouter client (always needed for DR systems) ────────────────────────
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in .env")
    client = make_client(api_key)

    # ── Local client (optional) ────────────────────────────────────────────────
    local_client = None
    local_model = os.getenv("LOCAL_MODEL", "deepseek-chat")
    if args.local:
        local_base = os.getenv("LOCAL_API_BASE")
        local_key = os.getenv("LOCAL_API_KEY", "local")
        if not local_base:
            raise SystemExit("Set LOCAL_API_BASE in .env (e.g., http://localhost:8000/v1)")
        local_client = make_client(local_key, local_base)
        print(f"Local server: {local_base}, model: {local_model}")
        # Set local_model for academic systems with model=None
        for cfg in SYSTEMS.values():
            if cfg.get("backend") == "academic" and cfg["model"] is None:
                cfg["model"] = local_model

    # Select systems
    if args.systems == "all":
        selected = SYSTEMS
    else:
        keys = [s.strip() for s in args.systems.split(",")]
        # Support short aliases
        alias = {
            "openai": "openai_o4_mini_dr",
            "perplexity": "perplexity_dr",
            "autosurvey": "autosurvey_gpt4o",
            "gemini": "autosurvey_gemini",
        }
        selected = {alias.get(k, k): SYSTEMS[alias.get(k, k)]
                    for k in keys if alias.get(k, k) in SYSTEMS}

    if not selected:
        raise SystemExit(f"Unknown systems: {args.systems}. Available: {list(SYSTEMS)}")

    # Load topics
    n_topics = 1 if args.dry_run else args.topics
    topics = load_topics(n_topics)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cost forecast
    print(f"\nSystems: {list(selected.keys())}")
    print(f"Topics: {len(topics)}")
    print(f"Requests: {len(selected) * len(topics)}")
    print(f"Results directory: {out_dir}/\n")

    # Main loop
    total_cost = 0.0
    all_results = []

    for topic_data in tqdm(topics, desc="Topics"):
        topic_id = topic_data["topic_id"]
        topic = topic_data["topic"]

        for sys_id, config in selected.items():
            out_file = out_dir / f"{sys_id}__{topic_id}.json"

            if args.resume and out_file.exists():
                try:
                    with open(out_file) as f:
                        data = json.load(f)
                    if data.get("generated_text"):
                        tqdm.write(f"  [SKIP] {sys_id} / {topic[:40]}")
                        continue
                except Exception:
                    pass

            backend_label = "local" if (args.local and config.get("backend") == "academic") else "openrouter"
            tqdm.write(f"  [{config['category']}|{backend_label}] {sys_id} / {topic[:50]}")
            result = run_system(client, sys_id, config, topic_id, topic, local_client=local_client)

            # Save results
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

            all_results.append(result)
            total_cost += result.cost_usd

            status = "OK" if result.success else "FAIL"
            tqdm.write(
                f"    [{status}] {result.word_count} words | "
                f"{result.reference_count} refs | "
                f"${result.cost_usd:.3f} | {result.latency_sec:.1f}s"
            )
            if result.error:
                tqdm.write(f"    Error: {result.error}")

            # Pause between requests
            time.sleep(3)

    # Summary table
    print(f"\n{'='*65}")
    print(f"{'System':<30} {'Topics':>4} {'Avg Words':>8} {'Avg Refs':>10} {'Cost':>10}")
    print(f"{'-'*65}")

    from collections import defaultdict
    by_system = defaultdict(list)
    for r in all_results:
        if r.success:
            by_system[r.system_id].append(r)

    for sys_id, results in by_system.items():
        avg_words = int(sum(r.word_count for r in results) / len(results))
        avg_refs = int(sum(r.reference_count for r in results) / len(results))
        total = sum(r.cost_usd for r in results)
        print(f"{sys_id:<30} {len(results):>4} {avg_words:>8} {avg_refs:>10} ${total:>9.3f}")

    print(f"{'='*65}")
    print(f"Total spent: ${total_cost:.3f}")
    print(f"Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()