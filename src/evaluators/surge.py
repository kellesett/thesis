"""
src/evaluators/surge.py
SurGE evaluator: SH-Recall, Structure_Quality, Logic.

Wraps the core evaluation logic (monkey-patching, LLM judge, metrics)
previously in src/surge_evaluate.py into a BaseEvaluator subclass.

The evaluator:
  1. Looks up the reference survey via self.dataset.get_by_id(generation["id"])
  2. Writes generation["text"] to a temp .md file (SurGE's parser expects a path)
  3. Runs the requested metrics
  4. Returns {"scores": {...}, "judge_log": [...]}
"""
from __future__ import annotations

import os
import re
import sys
import time
import tempfile
import logging
import threading
from pathlib import Path
from typing import Optional

from openai import OpenAI

from src.datasets.base import BaseDataset
from .base import BaseEvaluator
from .citation import CitationEvaluator, build_title_index

logger = logging.getLogger(__name__)
_surge_lock = threading.Lock()

ROOT      = Path(__file__).parent.parent.parent
SURGE_SRC = ROOT / "repos" / "SurGE" / "src"

# Paths for corpus-based citation metrics
CORPUS_PATH = ROOT / "datasets" / "SurGE" / "corpus.json"
INDEX_PATH  = ROOT / "datasets" / "SurGE" / "title_index.json"

CITATION_METRICS = {"citation_count", "corpus_match_rate", "coverage", "reference_self_cited"}

MAX_JUDGE_TRIES = 5
MAX_RETRY_DELAY = 32  # seconds, cap for exponential backoff


# ── Exceptions ────────────────────────────────────────────────────────────────

class JudgeFailedError(RuntimeError):
    """Raised when the LLM judge exhausts all retries for a single paragraph."""


# ── SurGE path bootstrap ──────────────────────────────────────────────────────

def _add_surge_to_path() -> None:
    path_str = str(SURGE_SRC)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ── Judge client ──────────────────────────────────────────────────────────────

def build_judge_client(config: dict) -> tuple[OpenAI, str]:
    """
    Build an OpenAI-compatible client from the metrics config.
    Expected config keys: judge_model, judge_base_url, judge_api_key_env
    Returns (client, model_name).
    """
    api_key = os.getenv(config["judge_api_key_env"])
    if not api_key:
        raise SystemExit(f"Env var '{config['judge_api_key_env']}' is not set")
    return OpenAI(base_url=config["judge_base_url"], api_key=api_key), config["judge_model"]


# ── Monkey-patch factory ──────────────────────────────────────────────────────

def _make_chat_openai_patched(judge_model: str, log: list, ctx: dict, flush_fn=None, log_fn=None):
    """
    Drop-in replacement for SurGE's chat_openai.
    Appends one log entry per attempt with:
      {metric, paragraph_idx, attempt_idx, prompt_preview, reasoning, raw, score, error}
    Raises JudgeFailedError after MAX_JUDGE_TRIES failed attempts on one paragraph.
    """
    def chat_openai(prompt: str, client: OpenAI, try_number: int):
        metric = ctx.get("metric")
        para   = ctx.get("paragraph_idx", 0)

        if try_number >= MAX_JUDGE_TRIES:
            raise JudgeFailedError(
                f"metric={metric} paragraph={para}: "
                f"failed to get valid score after {MAX_JUDGE_TRIES} tries"
            )

        if log_fn:
            log_fn(f"judge request  metric={metric}  paragraph={para}  attempt={try_number + 1}")

        entry = {
            "metric":         metric,
            "paragraph_idx":  para,
            "attempt_idx":    try_number,
            "prompt_preview": prompt[:100],
            "reasoning":      None,
            "raw":            None,
            "score":          None,
            "error":          None,
        }

        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user",   "content": prompt},
                ],
                stream=False,
                max_tokens=1024,
            )
            entry["reasoning"] = getattr(response.choices[0].message, "reasoning", None)
            raw = response.choices[0].message.content

            if raw is None:
                choice  = response.choices[0]
                finish  = choice.finish_reason
                refusal = getattr(choice.message, "refusal", None)
                entry["error"] = (
                    f"content is None, finish_reason={finish!r}"
                    + (f", refusal={refusal!r}" if refusal else "")
                )
                if log_fn:
                    log_fn(f"judge error    metric={metric}  paragraph={para}  {entry['error']}")
                log.append(entry)
                if flush_fn:
                    flush_fn()
                return chat_openai(prompt, client, try_number + 1)

            content = raw.strip()
            entry["raw"] = content

            if not re.match(r"^[0-5]$", content):
                entry["error"] = f"invalid response: '{content}'"
                if log_fn:
                    log_fn(f"judge error    metric={metric}  paragraph={para}  {entry['error']}")
                log.append(entry)
                if flush_fn:
                    flush_fn()
                return chat_openai(prompt, client, try_number + 1)

            entry["score"] = int(content)
            if log_fn:
                log_fn(f"judge score    metric={metric}  paragraph={para}  score={entry['score']}")
            log.append(entry)
            if flush_fn:
                flush_fn()
            ctx["paragraph_idx"] = ctx.get("paragraph_idx", 0) + 1
            return entry["score"]

        except JudgeFailedError:
            raise
        except Exception as e:
            entry["error"] = str(e)
            if log_fn:
                log_fn(f"judge exception  metric={metric}  paragraph={para}  {type(e).__name__}: {e}")
            log.append(entry)
            if flush_fn:
                flush_fn()
            delay = min(2 ** try_number, MAX_RETRY_DELAY)
            print(f"  [WARN] LLM judge error: {e}")
            print(f"  [WAIT] retry in {delay}s...")
            time.sleep(delay)
            return chat_openai(prompt, client, try_number + 1)

    return chat_openai


# ── Core metric runner ────────────────────────────────────────────────────────

def _run_metrics(
    md_path: Path,
    target_survey: dict,
    eval_list: list[str],
    flag_model,
    judge_client: OpenAI,
    judge_model: str,
    judge_log: list,
    flush_fn=None,
    log_fn=None,
) -> dict:
    """
    Run SurGE metrics on one .md file.
    Returns scores dict: {metric_name: value}.
    Raises JudgeFailedError if the judge fails on any paragraph.
    """
    import structureFuncs
    import informationFuncs
    from markdownParser import parse_markdown

    ctx     = {"metric": None, "paragraph_idx": 0}
    # Guard: fail fast if SurGE renames the function rather than silently using the unpatched version
    if not hasattr(structureFuncs, "chat_openai"):
        raise ImportError("SurGE source not found at structureFuncs.chat_openai — SurGE API may have changed")
    if not hasattr(informationFuncs, "chat_openai"):
        raise ImportError("SurGE source not found at informationFuncs.chat_openai — SurGE API may have changed")

    patched = _make_chat_openai_patched(judge_model, judge_log, ctx, flush_fn=flush_fn, log_fn=log_fn)
    with _surge_lock:
        structureFuncs.chat_openai   = patched
        informationFuncs.chat_openai = patched
        psg_node = parse_markdown(str(md_path))
        scores   = {}

        if "sh_recall" in eval_list:
            if log_fn: log_fn("metric start  sh_recall")
            scores["sh_recall"] = structureFuncs.eval_SHRecall(
                target_survey, psg_node, flag_model
            )
            if log_fn: log_fn(f"metric done   sh_recall = {scores['sh_recall']}")

        if "structure_quality" in eval_list:
            if log_fn: log_fn("metric start  structure_quality")
            ctx["metric"]        = "structure_quality"
            ctx["paragraph_idx"] = 0
            scores["structure_quality"] = structureFuncs.eval_structure_quality_client(
                target_survey, psg_node, judge_client
            )
            if log_fn: log_fn(f"metric done   structure_quality = {scores['structure_quality']}")

        if "logic" in eval_list:
            if log_fn: log_fn("metric start  logic")
            ctx["metric"]        = "logic"
            ctx["paragraph_idx"] = 0
            content_blocks = informationFuncs.get_content_list(psg_node)
            if content_blocks:
                scores["logic"] = informationFuncs.eval_logic_client(psg_node, judge_client)
            else:
                scores["logic"] = None
            if log_fn: log_fn(f"metric done   logic = {scores['logic']}")

    return scores


# ── Evaluator class ───────────────────────────────────────────────────────────

class SurGEEvaluator(BaseEvaluator):
    """
    Evaluates generated surveys using SurGE metrics.

    Config keys (from metrics/surge/config.yaml):
        eval_list         — list of metrics to compute
        judge_model       — model identifier for the LLM judge
        judge_base_url    — OpenAI-compatible API base URL
        judge_api_key_env — env var name holding the API key
        embedding_model   — FlagEmbedding model name (required for sh_recall)
    """
    evaluator_id = "surge"

    def __init__(self, config: dict, dataset: BaseDataset) -> None:
        super().__init__(config, dataset)

        if not SURGE_SRC.exists():
            raise SystemExit(
                f"SurGE source not found at {SURGE_SRC}\n"
                "Clone: git clone https://github.com/oneal2000/SurGE repos/SurGE"
            )
        _add_surge_to_path()

        self.eval_list = config["eval_list"]
        self.judge_client, self.judge_model = build_judge_client(config)

        self.flag_model = None
        if "sh_recall" in self.eval_list:
            try:
                from FlagEmbedding import FlagModel
                import torch
            except ImportError:
                raise SystemExit("FlagEmbedding is required for sh_recall: pip install FlagEmbedding")
            emb_model = config.get("embedding_model", "BAAI/bge-large-en-v1.5")
            print(f"Loading embedding model: {emb_model}")
            self.flag_model = FlagModel(emb_model, use_fp16=torch.cuda.is_available())

        # Citation evaluator (corpus-based)
        self.citation_evaluator = None
        needs_citation = bool(CITATION_METRICS & set(self.eval_list))
        if needs_citation:
            # Build index on first run if not present
            if not INDEX_PATH.exists():
                if not CORPUS_PATH.exists():
                    raise SystemExit(
                        f"corpus.json not found at {CORPUS_PATH}.\n"
                        f"Download from: https://drive.google.com/drive/folders/1ZZPeZvjexFcCmgFqxftKeCPn1vYeBR0Q"
                    )
                build_title_index(CORPUS_PATH, INDEX_PATH)
            self.citation_evaluator = CitationEvaluator(INDEX_PATH)

    def evaluate(self, generation: dict, flush_fn=None, log_fn=None) -> dict:
        """
        Evaluate one unified generation dict.

        Args:
            generation: Dict with "id" and "text" keys
            flush_fn: Optional callback for incremental judge_log persistence
            log_fn: Optional callable(str) for human-readable run logging

        Returns:
            Dict with "scores" and "judge_log" keys

        Raises:
            JudgeFailedError: If the LLM judge fails after max retries
        """
        instance = self.dataset.get_by_id(generation["id"])
        target_survey = instance.meta  # full SurGE survey dict

        judge_log = []

        # SurGE's parser expects a file path — write text to a temp .md file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write(generation.get("text", ""))
            tmp_path = Path(tmp.name)

        try:
            # SurGE text-based metrics (sh_recall, structure_quality, logic)
            surge_eval_list = [m for m in self.eval_list if m not in CITATION_METRICS]
            scores = {}
            if surge_eval_list:
                scores = _run_metrics(
                    tmp_path, target_survey, surge_eval_list,
                    self.flag_model, self.judge_client, self.judge_model,
                    judge_log, flush_fn=flush_fn, log_fn=log_fn,
                )
        finally:
            tmp_path.unlink(missing_ok=True)

        # Corpus-based citation metrics (no LLM judge required)
        if self.citation_evaluator is not None:
            if log_fn: log_fn("metric start  citation (corpus-based)")
            citation_scores = self.citation_evaluator.evaluate(generation, target_survey)
            for metric in CITATION_METRICS:
                if metric in self.eval_list:
                    scores[metric] = citation_scores.get(metric)
                    if log_fn: log_fn(f"metric done   {metric} = {scores[metric]}")

        return {"scores": scores, "judge_log": judge_log}
