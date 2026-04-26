#!/usr/bin/env python
"""
models/surveyforge/main.py
Generation runner for SurveyForge (ACL 2025).

Usage (inside Docker):
    python models/surveyforge/main.py --dataset SurGE
"""
import os
import sys
import re
import tempfile
import argparse
import types
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT   = Path(__file__).parent.parent.parent

# sys.path order matters: both /app/src/ and SF_CODE/src/ are namespace packages,
# and `utils` exists in both. We need SF_CODE first so SurveyForge's internal
# `from src.utils import tokenCounter` resolves to its own utils.py — not ours.
#
# Strategy: import our src.datasets FIRST (caches it in sys.modules), then put
# SF_CODE at the front. Python won't re-resolve already-cached modules.
SF_CODE = ROOT / "repos" / "SurveyForge" / "code"
sys.path.insert(0, str(ROOT))
from src.models.base import BaseModel          # cached before SF_CODE takes over
from src.datasets import load_dataset as _     # noqa: F401 — force cache of src.datasets

sys.path.insert(0, str(SF_CODE))  # SF_CODE is now first → SurveyForge finds its own src/utils
# ⚠️  Do NOT reorder the imports above — the caching trick breaks if SF_CODE is inserted before
#     src.datasets is imported. Linters / auto-sort tools must not touch this block.


# ── Helper ────────────────────────────────────────────────────────────────────

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


# ── Model class ───────────────────────────────────────────────────────────────

class SurveyForge(BaseModel):
    """SurveyForge pipeline: FAISS RAG + outline heuristics + LLM."""

    def __init__(self) -> None:
        super().__init__(Path(__file__).parent)

        self.api_url, self.api_key = self._resolve_backend()

        # Validate DB path before loading (fail fast)
        db_path = Path(self.cfg["db_path"])
        if not db_path.exists():
            raise SystemExit(
                f"SurveyForge DB not found: {db_path}\n"
                f"Run: make sfdb"
            )

        # Load FAISS indices once — expensive (~1-2 min, ~10 GB RAM)
        print("Loading SurveyForge database …")
        self.db = self._load_db()
        print("Database loaded.\n")

    # ── Setup helpers ─────────────────────────────────────────────────────────

    def _resolve_backend(self) -> tuple[str, str]:
        """Return (api_url, api_key) for the configured backend."""
        backend = self.cfg.get("backend", "local")
        if backend == "local":
            base    = os.getenv("LOCAL_API_BASE", "http://localhost:8000/v1").rstrip("/")
            key     = os.getenv("LOCAL_API_KEY", "none")
            api_url = f"{base}/chat/completions"
        else:
            key     = os.getenv("OPENROUTER_API_KEY", "")
            api_url = "https://openrouter.ai/api/v1/chat/completions"
        return api_url, key

    def _make_sf_args(self, saving_path: str) -> types.SimpleNamespace:
        """Build the argparse-style namespace SurveyForge internals expect."""
        cfg = self.cfg
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

    def _load_db(self) -> dict:
        """Load FAISS indices and paper/survey databases."""
        from src.database import database, database_survey
        from src.rag import GeneralRAG_langchain

        cfg   = self.cfg
        emb   = cfg["embedding_model"]
        db_p  = cfg["db_path"]

        print("  [DB] Loading paper database …")
        db_paper  = database(db_path=db_p, embedding_model=emb)
        print("  [DB] Loading survey database …")
        db_survey = database_survey(db_path=db_p, embedding_model=emb)

        abs_index   = f"{db_p}/faiss_paper_title_abs_embeddings_FROM_2012_0101_TO_240926.bin"
        title_index = f"{db_p}/faiss_paper_title_embeddings_FROM_2012_0101_TO_240926.bin"
        doc_db      = f"{db_p}/arxiv_paper_db_with_cc.json"
        id2idx      = f"{db_p}/arxivid_to_index_abs.json"

        dummy_args = types.SimpleNamespace(
            embedding_model=emb, db_path=db_p, debug=False,
            saving_path="/tmp", gpu="0",
        )

        print("  [DB] Building abstract RAG index …")
        rag_abs = GeneralRAG_langchain(
            args=dummy_args, retriever_type="vectorstore",
            index_db_path=abs_index, doc_db_path=doc_db,
            arxivid_to_index_path=id2idx, embedding_model=emb,
        )
        print("  [DB] Building title RAG index …")
        rag_title = GeneralRAG_langchain(
            args=dummy_args, retriever_type="vectorstore",
            index_db_path=title_index, doc_db_path=doc_db,
            arxivid_to_index_path=id2idx, embedding_model=emb,
        )

        return {
            "paper":              db_paper,
            "survey":             db_survey,
            "rag_outline":        rag_abs,
            "rag_suboutline":     rag_abs,
            "rag_subsection":     rag_abs,
            "rag_title4citation": rag_title,
        }

    # ── BaseModel interface ───────────────────────────────────────────────────

    def generate(self, instance) -> dict:
        """Run the full SurveyForge pipeline for one instance."""
        import time
        from src.agents.outline_writer import outlineWriter
        from src.agents.writer import subsectionWriter

        cfg        = self.cfg
        refinement = cfg.get("refinement", False)
        topic      = instance.query
        t0         = time.time()

        try:
            with tempfile.TemporaryDirectory() as tmp:
                sf_args = self._make_sf_args(tmp)

                # ── Stage 1: Outline ──────────────────────────────────────────
                print(f"    [SF] Stage 1 — outline …")
                ol_writer = outlineWriter(
                    args=sf_args, model=cfg["model"], ckpt="",
                    api_key=self.api_key, api_url=self.api_url, database=self.db,
                )
                outline_with_desc = ol_writer.draft_outline(
                    topic,
                    cfg.get("outline_reference_num", 1500),
                    30000,
                    cfg.get("section_num", 6),
                )
                outline_with_desc = _duplicate_first_last_sections(outline_with_desc)

                # ── Stage 2: Content ──────────────────────────────────────────
                print(f"    [SF] Stage 2 — content …")
                sub_writer = subsectionWriter(
                    args=sf_args, model=cfg["model"], ckpt="",
                    api_key=self.api_key, api_url=self.api_url, database=self.db,
                )

                if refinement:
                    _, text_with_refs, _, _, final_text, sf_refs = sub_writer.write(
                        topic, outline_with_desc,
                        subsection_len=cfg.get("subsection_len", 500),
                        rag_num=cfg.get("rag_num", 100),
                        rag_max_out=cfg.get("rag_max_out", 60),
                        refining=True,
                    )
                else:
                    _, final_text, sf_refs = sub_writer.write(
                        topic, outline_with_desc,
                        subsection_len=cfg.get("subsection_len", 500),
                        rag_num=cfg.get("rag_num", 100),
                        rag_max_out=cfg.get("rag_max_out", 60),
                        refining=False,
                    )

            references = self.normalize_arxiv_references(
                sf_refs if isinstance(sf_refs, dict) else {}
            )

            return {
                "text":    final_text.replace("---", "").strip(),
                "success": bool(final_text.strip()),
                "meta": {
                    "model":       cfg["model"],
                    "backend":     cfg.get("backend", "local"),
                    "latency_sec": round(time.time() - t0, 2),
                    "cost_usd":    None,
                    "error":       None,
                    "references":  references,
                    "sf_params": {
                        "section_num":           cfg.get("section_num", 6),
                        "subsection_len":        cfg.get("subsection_len", 500),
                        "outline_reference_num": cfg.get("outline_reference_num", 1500),
                        "rag_num":               cfg.get("rag_num", 100),
                        "rag_max_out":           cfg.get("rag_max_out", 60),
                        "refinement":            refinement,
                    },
                },
            }

        except Exception as e:
            import traceback
            return {
                "text":    "",
                "success": False,
                "meta": {
                    "model":       cfg["model"],
                    "backend":     cfg.get("backend", "local"),
                    "latency_sec": round(time.time() - t0, 2),
                    "cost_usd":    None,
                    "error":       f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                    "references":  [],
                    "sf_params":   {},
                },
            }


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate surveys with SurveyForge")
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. SurGE)")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Generate only surveys with survey_id <= LIMIT "
                             "(inclusive, id-based — not positional).")
    args = parser.parse_args()

    SurveyForge().run(args.dataset, limit=args.limit)
