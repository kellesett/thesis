#!/usr/bin/env python
"""
models/surveygen_i/main.py
Generation runner for SurveyGen-I (IJCNLP-AACL 2025).

Pipeline: multi-stage Literature Retrieval (Semantic Scholar) →
          PlanEvo (dynamic outline) → CaM-Writing (citation-aware subsections).

Output: unified generation JSON, survey text from final_survey_refined.md,
        references parsed from references_master.bib.

Usage (inside Docker):
    python models/surveygen_i/main.py --dataset SurGE
"""
import os
import re
import sys
import asyncio
import argparse
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT    = Path(__file__).parent.parent.parent
SGI_DIR = ROOT / "repos" / "SurveyGen-I"  # repo root (prompts, acm/ template)

sys.path.insert(0, str(ROOT))
from src.models.base import BaseModel


# ── OpenRouter compatibility patch ────────────────────────────────────────────

class _ChatCompletionWrapper:
    """
    Adapts a chat.completions response to look like an OpenAI Responses API
    response so that call_llm() in llm_tools.py works transparently.
    """
    def __init__(self, response):
        self.output_text = response.choices[0].message.content or ""
        usage = response.usage
        self.usage = type("Usage", (), {
            "input_tokens":  usage.prompt_tokens     if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        })()


def _patch_llm_service_for_openrouter(base_url: str) -> None:
    """
    Monkey-patch LlmServiceAsync to use chat.completions instead of
    responses.create so that OpenRouter-compatible endpoints work.
    Must be called BEFORE the graph is built.
    """
    from openai import AsyncOpenAI
    import surveygen_i.tools.llm_service as _mod

    class PatchedLlmServiceAsync(_mod.LlmServiceAsync):
        def __init__(self, api_key: str, model: str = "gpt-4o-mini-2024-07-18"):
            self.api_key = api_key
            self.model   = model
            self.client  = AsyncOpenAI(api_key=api_key, base_url=base_url)

        async def ask(self, prompt: str, mode: str = "default") -> _ChatCompletionWrapper:
            system = (
                "You are a thoughtful, expert scientist, and you are knowledgeable "
                "about carefully crafting a search phrase to find useful papers in a "
                "search engine."
            ) if mode == "keywords" else "You are an academic assistant."

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
            )
            return _ChatCompletionWrapper(response)

    _mod.LlmServiceAsync = PatchedLlmServiceAsync


# ── Reference parsing from BibTeX ─────────────────────────────────────────────

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,6})")


def _parse_bib_references(bib_path: Path) -> list[dict]:
    """
    Parse references_master.bib and build unified reference list.

    Looks for arxiv IDs in:
      - eprint field  (when archiveprefix = arXiv)
      - url field     (https://arxiv.org/abs/XXXX.XXXXX)

    Returns list of unified refs:
      [{"idx": N, "title": str|None, "url": str, "arxiv_id": str|None, "canonical_title": None}]
    """
    if not bib_path.exists():
        return []

    try:
        import bibtexparser
        with open(bib_path, encoding="utf-8", errors="replace") as f:
            bib_db = bibtexparser.load(f)
    except Exception as e:
        print(f"  [WARN] Could not parse BibTeX: {e}")
        return []

    refs = []
    for i, entry in enumerate(bib_db.entries, 1):
        title   = entry.get("title",   "").strip("{}")
        url     = entry.get("url",     "")
        eprint  = entry.get("eprint",  "")
        arch    = entry.get("archiveprefix", "").lower()

        arxiv_id = None

        # arxiv eprint field
        if arch == "arxiv" and eprint:
            m = _ARXIV_ID_RE.search(eprint)
            if m:
                arxiv_id = m.group(1)

        # arxiv URL fallback
        if arxiv_id is None and "arxiv.org" in url:
            m = _ARXIV_ID_RE.search(url)
            if m:
                arxiv_id = m.group(1)

        refs.append({
            "idx":             i,
            "title":           title or None,
            "url":             url or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""),
            "arxiv_id":        arxiv_id,
            "canonical_title": None,
        })

    return refs


# ── Model class ───────────────────────────────────────────────────────────────

class SurveyGenI(BaseModel):
    """SurveyGen-I: memory-guided, citation-traced survey generation pipeline."""

    def __init__(self) -> None:
        super().__init__(Path(__file__).parent)

        if not SGI_DIR.exists():
            raise SystemExit(
                f"SurveyGen-I repo not found: {SGI_DIR}\n"
                f"Run: git clone https://github.com/SurveyGens/SurveyGen-I repos/SurveyGen-I"
            )

        api_key = os.getenv(self.cfg["api_key_env"])
        if not api_key:
            raise SystemExit(f"Env var '{self.cfg['api_key_env']}' is not set")
        self.api_key = api_key

        # Patch LlmServiceAsync if OpenRouter base_url is configured
        base_url = self.cfg.get("base_url")
        if base_url:
            # Must happen before importing the graph
            sys.path.insert(0, str(SGI_DIR))
            _patch_llm_service_for_openrouter(base_url)
            print(f"  [SGI] OpenRouter patch active: {base_url}")
        else:
            sys.path.insert(0, str(SGI_DIR))

    # ── Core generation ───────────────────────────────────────────────────────

    def generate(self, instance) -> dict:
        """Run the SurveyGen-I pipeline for one instance."""
        import time
        t0 = time.time()

        topic = instance.query

        # Dedicated temp output dir per survey
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "sgi_output"
            result = asyncio.run(self._run_pipeline(topic, str(output_dir)))

        latency = round(time.time() - t0, 2)
        result["meta"]["latency_sec"] = latency
        return result

    async def _run_pipeline(self, topic: str, output_dir: str) -> dict:
        """Async wrapper around the SurveyGen-I LangGraph pipeline."""
        import uuid
        from sentence_transformers import CrossEncoder
        from langchain_huggingface import HuggingFaceEmbeddings
        from surveygen_i.graph.pipeline_graph import build_graph
        from surveygen_i.graph.state import PaperPipelineState
        from surveygen_i.tools.llm_service import LlmServiceAsync

        cfg = self.cfg

        # Prepare output dirs — must run from SGI_DIR (relative paths for acm/ template)
        old_cwd = os.getcwd()
        os.chdir(SGI_DIR)
        try:
            from run_pipeline import prepare_output_dirs
            prepare_output_dirs(output_dir)
        finally:
            os.chdir(old_cwd)

        # Services
        llm = LlmServiceAsync(api_key=self.api_key, model=cfg["model"])

        emb_cfg = {"model_name": "BAAI/bge-small-en-v1.5", "device": "cpu"}
        embedding = HuggingFaceEmbeddings(
            model_name=emb_cfg["model_name"],
            model_kwargs={"device": emb_cfg["device"]},
            encode_kwargs={"normalize_embeddings": True},
        )
        reranker = CrossEncoder("BAAI/bge-reranker-base")

        state: PaperPipelineState = {
            "research_topic":          topic,
            "explanation":             "",
            "openai_api_key":          self.api_key,
            "start_year":              cfg.get("start_year", 2018),
            "end_year":                cfg.get("end_year", 2025),
            "paper_limit_per_keyword": cfg.get("paper_limit_per_keyword", 30),
            "expand_top_n_papers":     cfg.get("expand_top_n_papers", 10),
            "text_column":             "abstract",
            "id_column":               "paperId",
            "output_base_dir":         output_dir,
            "citation_trace":          cfg.get("citation_trace", True),
            "enable_outline_update":   cfg.get("enable_outline_update", True),
            "arxiv_only":              cfg.get("arxiv_only", False),
        }

        graph = build_graph(llm, embedding, reranker)
        thread_config = {
            "recursion_limit": 100000,
            "configurable": {"thread_id": str(uuid.uuid4())},
        }

        # Run pipeline — must be in SGI_DIR for relative prompt paths
        os.chdir(SGI_DIR)
        try:
            async for _ in graph.astream(state, config=thread_config):
                pass
        finally:
            os.chdir(old_cwd)

        return self._collect_output(Path(output_dir))

    def _collect_output(self, output_dir: Path) -> dict:
        """Read generated markdown and BibTeX from pipeline output dir."""
        acm_dir = output_dir / "acm"

        # Prefer refined over original
        text = ""
        for name in ("final_survey_refined.md", "final_survey_original.md"):
            md_path = acm_dir / name
            if md_path.exists():
                text = md_path.read_text(encoding="utf-8").strip()
                if text:
                    break

        references = _parse_bib_references(acm_dir / "references_master.bib")

        return {
            "text":    text,
            "success": bool(text),
            "meta": {
                "model":       self.cfg["model"],
                "latency_sec": None,   # filled in generate()
                "cost_usd":    None,   # tracked internally by llm_tools.py
                "error":       None if text else "No output markdown found",
                "references":  references,
                "citation_trace":        self.cfg.get("citation_trace", True),
                "enable_outline_update": self.cfg.get("enable_outline_update", True),
            },
        }


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate surveys with SurveyGen-I")
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. SurGE)")
    args = parser.parse_args()

    SurveyGenI().run(args.dataset)
