#!/usr/bin/env python
"""
models/surveyforge/main.py
Generation runner for SurveyForge (ACL 2025).

Reads config.yaml, loads the requested dataset, generates surveys via the
SurveyForge pipeline (FAISS RAG + outline heuristics + LLM), and saves
unified generation JSONs to:
  results/generations/<dataset_id>_<model_id>/

Input mapping:
  instance.query  →  SurveyForge topic string (= survey title from dataset)

References format (unified, same as perplexity_dr):
  [{"idx": N, "title": "...", "url": "https://arxiv.org/abs/...", "arxiv_id": "..."}]

Usage (inside Docker):
    python models/surveyforge/main.py --dataset SurGE
"""
import os
import sys
import json
import re
import tempfile
import argparse
import types
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT   = Path(__file__).parent.parent.parent
CONFIG = Path(__file__).parent / "config.yaml"

# sys.path order matters: both /app/src/ and SF_CODE/src/ are namespace packages,
# and `utils` exists in both. We need SF_CODE first so SurveyForge's internal
# `from src.utils import tokenCounter` resolves to its own utils.py — not ours.
#
# Strategy: import our src.datasets FIRST (caches it in sys.modules), then put
# SF_CODE at the front. Python won't re-resolve already-cached modules.
SF_CODE = ROOT / "repos" / "SurveyForge" / "code"
sys.path.insert(0, str(ROOT))
from src.datasets import load_dataset as load_dataset_cls  # cached before SF_CODE takes over

sys.path.insert(0, str(SF_CODE))  # SF_CODE is now first → SurveyForge finds its own src/utils


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_registry(registry_path: Path) -> dict[str, str]:
    """Return {dataset_id: path} from datasets/registry.yaml."""
    with open(registry_path, encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    return {entry["id"]: entry["path"] for entry in reg["datasets"]}


def resolve_backend(cfg: dict) -> tuple[str, str]:
    """Return (api_url, api_key) for the configured backend."""
    backend = cfg.get("backend", "local")
    if backend == "local":
        base    = os.getenv("LOCAL_API_BASE", "http://localhost:8000/v1").rstrip("/")
        key     = os.getenv("LOCAL_API_KEY", "none")
        api_url = f"{base}/chat/completions"
    else:
        key     = os.getenv("OPENROUTER_API_KEY", "")
        api_url = "https://openrouter.ai/api/v1/chat/completions"
    return api_url, key


def make_sf_args(saving_path: str, cfg: dict) -> types.SimpleNamespace:
    """Build the argparse-style namespace SurveyForge internals expect."""
    return types.SimpleNamespace(
        saving_path=saving_path,
        debug=False,
        gpu="0",
        model=cfg["model"],
        ckpt="",
        section_num=cfg.get("section_num", 6),
        subsection_len=cfg.get("subsection_len", 500),
        outline_reference_num=cfg.get("outline_reference_num", 1500),
        rag_num=cfg.get("rag_num", 100),
        rag_max_out=cfg.get("rag_max_out", 60),
        db_path=cfg["db_path"],
        survey_outline_path=cfg.get("survey_outline_path", cfg["db_path"]),
        embedding_model=cfg["embedding_model"],
    )


def normalize_references(sf_references: dict) -> list[dict]:
    """
    Convert SurveyForge reference dict {int → arxiv_id} to unified list.

    SurveyForge output:  {1: "2301.12345", 2: "2204.56789", ...}
    Unified format:      [{"idx": 1, "arxiv_id": "...", "url": "...", "title": None}, ...]

    Title is not available at this stage (SurveyForge stores only the arxiv ID
    in the references dict; the title appears in the ## References section of
    the text).  We leave it None so downstream tools can enrich if needed.
    """
    refs = []
    for idx, arxiv_id in sorted(sf_references.items(), key=lambda x: x[0]):
        refs.append({
            "idx":             int(idx),
            "title":           None,        # embedded in survey text
            "url":             f"https://arxiv.org/abs/{arxiv_id}",
            "arxiv_id":        str(arxiv_id),
            "canonical_title": None,        # can be enriched later
        })
    return refs


# ── SurveyForge DB loader (called once) ──────────────────────────────────────

def load_surveyforge_db(cfg: dict) -> dict:
    """Load FAISS indices and paper/survey databases. Takes ~1–2 min."""
    from src.database import database, database_survey
    from src.rag import GeneralRAG_langchain

    db_path = cfg["db_path"]
    emb     = cfg["embedding_model"]

    print("  [DB] Loading paper database …")
    db_paper  = database(db_path=db_path, embedding_model=emb)
    print("  [DB] Loading survey database …")
    db_survey = database_survey(db_path=db_path, embedding_model=emb)

    abs_index   = f"{db_path}/faiss_paper_title_abs_embeddings_FROM_2012_0101_TO_240926.bin"
    title_index = f"{db_path}/faiss_paper_title_embeddings_FROM_2012_0101_TO_240926.bin"
    doc_db      = f"{db_path}/arxiv_paper_db_with_cc.json"
    id2idx      = f"{db_path}/arxivid_to_index_abs.json"

    # Shared args stub for GeneralRAG_langchain
    dummy_args = types.SimpleNamespace(
        embedding_model=emb,
        db_path=db_path,
        debug=False,
        saving_path="/tmp",
        gpu="0",
    )

    print("  [DB] Building abstract RAG index …")
    rag_abs = GeneralRAG_langchain(
        args=dummy_args,
        retriever_type="vectorstore",
        index_db_path=abs_index,
        doc_db_path=doc_db,
        arxivid_to_index_path=id2idx,
        embedding_model=emb,
    )
    print("  [DB] Building title RAG index …")
    rag_title = GeneralRAG_langchain(
        args=dummy_args,
        retriever_type="vectorstore",
        index_db_path=title_index,
        doc_db_path=doc_db,
        arxivid_to_index_path=id2idx,
        embedding_model=emb,
    )

    return {
        "paper":            db_paper,
        "survey":           db_survey,
        "rag_outline":      rag_abs,
        "rag_suboutline":   rag_abs,
        "rag_subsection":   rag_abs,
        "rag_title4citation": rag_title,
    }


# ── Core generation ───────────────────────────────────────────────────────────

def generate_survey(
    topic: str,
    db: dict,
    cfg: dict,
    api_url: str,
    api_key: str,
    tmp_dir: str,
) -> dict:
    """
    Run the full SurveyForge pipeline for one topic.

    Returns unified dict:
      text, references, success, latency_sec, error
    """
    import time
    from src.agents.outline_writer import outlineWriter
    from src.agents.writer import subsectionWriter

    model      = cfg["model"]
    refinement = cfg.get("refinement", False)
    sf_args    = make_sf_args(tmp_dir, cfg)

    t0 = time.time()
    try:
        # ── Stage 1: Outline ────────────────────────────────────────────────
        print(f"    [SF] Stage 1 — outline …")
        ol_writer = outlineWriter(
            args=sf_args, model=model, ckpt="",
            api_key=api_key, api_url=api_url, database=db,
        )
        outline_with_desc = ol_writer.draft_outline(
            topic,
            cfg.get("outline_reference_num", 1500),
            30000,
            cfg.get("section_num", 6),
        )

        # duplicate_first_last_sections (same as SurveyForge main.py)
        outline_with_desc = _duplicate_first_last_sections(outline_with_desc)

        # ── Stage 2: Content (SANA) ─────────────────────────────────────────
        print(f"    [SF] Stage 2 — content …")
        sub_writer = subsectionWriter(
            args=sf_args, model=model, ckpt="",
            api_key=api_key, api_url=api_url, database=db,
        )

        if refinement:
            _, text_with_refs, _, _, refined_text, refined_refs = sub_writer.write(
                topic, outline_with_desc,
                subsection_len=cfg.get("subsection_len", 500),
                rag_num=cfg.get("rag_num", 100),
                rag_max_out=cfg.get("rag_max_out", 60),
                refining=True,
            )
            final_text = refined_text
            sf_refs    = refined_refs
        else:
            _, final_text, sf_refs = sub_writer.write(
                topic, outline_with_desc,
                subsection_len=cfg.get("subsection_len", 500),
                rag_num=cfg.get("rag_num", 100),
                rag_max_out=cfg.get("rag_max_out", 60),
                refining=False,
            )

        references = normalize_references(sf_refs if isinstance(sf_refs, dict) else {})

        return {
            "text":        final_text.replace("---", "").strip(),
            "references":  references,
            "success":     bool(final_text.strip()),
            "latency_sec": round(time.time() - t0, 2),
            "error":       None,
        }

    except Exception as e:
        import traceback
        return {
            "text":        "",
            "references":  [],
            "success":     False,
            "latency_sec": round(time.time() - t0, 2),
            "error":       f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


def _duplicate_first_last_sections(markdown_content: str) -> str:
    """Mirror of SurveyForge main.py helper — adds subsection titles to first/last section."""
    pattern  = r'(## \d+\.?\s*.*?(?=\n##|\Z))'
    sections = re.findall(pattern, markdown_content, re.DOTALL)
    if len(sections) < 2:
        return markdown_content

    def _augment(section: str) -> str:
        num   = re.search(r'## (\d+)', section).group(1)
        title = section.split('\n')[0].strip()
        body  = '\n'.join(section.split('\n')[1:]).strip()
        return f"{title}\n{body}\n\n### {num}.1 {title.split(maxsplit=2)[-1]}\nDescription: {body}\n\n"

    markdown_content = markdown_content.replace(sections[0],  _augment(sections[0]))
    markdown_content = markdown_content.replace(sections[-1], _augment(sections[-1]))
    return markdown_content


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate surveys with SurveyForge")
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. SurGE)")
    args = parser.parse_args()

    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_id   = cfg["model_id"]
    n_surveys  = cfg.get("n_surveys", 5)
    resume     = cfg.get("resume", True)

    api_url, api_key = resolve_backend(cfg)

    # ── Load dataset ──────────────────────────────────────────────────────────
    registry = load_registry(ROOT / "datasets" / "registry.yaml")
    if args.dataset not in registry:
        raise SystemExit(f"Dataset '{args.dataset}' not in registry. Available: {list(registry)}")

    dataset = load_dataset_cls(args.dataset, registry[args.dataset])
    print(f"Dataset  : {args.dataset} ({len(dataset)} surveys, using first {n_surveys})")

    # ── Prepare output dir ────────────────────────────────────────────────────
    out_dir = ROOT / "results" / "generations" / f"{args.dataset}_{model_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model    : {cfg['model']}")
    print(f"Backend  : {cfg.get('backend', 'local')}")
    print(f"Output   : {out_dir}\n")

    # ── Validate DB path ──────────────────────────────────────────────────────
    db_path = Path(cfg["db_path"])
    if not db_path.exists():
        raise SystemExit(
            f"SurveyForge DB not found: {db_path}\n"
            f"Run: make sfdb"
        )

    # ── Load DB once (expensive) ──────────────────────────────────────────────
    print("Loading SurveyForge database …")
    db = load_surveyforge_db(cfg)
    print("Database loaded.\n")

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

        # instance.query = survey title (used as SurveyForge topic)
        topic = instance.query
        print(f"  [GEN]  {instance.id} | {topic[:70]}")

        with tempfile.TemporaryDirectory() as tmp:
            result = generate_survey(topic, db, cfg, api_url, api_key, tmp)

        generation = {
            "id":         instance.id,
            "dataset_id": args.dataset,
            "model_id":   model_id,
            "query":      topic,
            "text":       result["text"],
            "success":    result["success"],
            "meta": {
                "model":        cfg["model"],
                "backend":      cfg.get("backend", "local"),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "latency_sec":  result["latency_sec"],
                "cost_usd":     None,       # local inference — no cost
                "error":        result["error"],
                "references":   result["references"],
                # SurveyForge-specific params for reproducibility
                "sf_params": {
                    "section_num":            cfg.get("section_num", 6),
                    "subsection_len":         cfg.get("subsection_len", 500),
                    "outline_reference_num":  cfg.get("outline_reference_num", 1500),
                    "rag_num":                cfg.get("rag_num", 100),
                    "rag_max_out":            cfg.get("rag_max_out", 60),
                    "refinement":             cfg.get("refinement", False),
                },
            },
        }

        out_file.write_text(
            json.dumps(generation, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        status = "OK" if result["success"] else "FAIL"
        print(
            f"    [{status}] latency={result['latency_sec']}s"
            + (f"  refs={len(result['references'])}" if result["success"] else "")
            + (f"  error={result['error'][:80]}" if result["error"] else "")
        )

    print(f"\nGenerations saved to: {out_dir}")


if __name__ == "__main__":
    main()
