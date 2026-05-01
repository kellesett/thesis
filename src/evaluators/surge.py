"""
src/evaluators/surge.py
SurGE evaluator: ROUGE/BLEU, SH-Recall, Structure_Quality, Logic.

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

# Keep third-party tokenizer stacks quiet and fork-safe. Set defaults only:
# users can still override these env vars explicitly before launching.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import re
import sys
import time
import tempfile
import logging
import threading
import io
import json
from contextlib import contextmanager
from contextlib import redirect_stdout
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
RELEVANCE_METRICS = {"relevance_paper", "relevance_section", "relevance_sentence"}
SURGE_TEXT_METRICS = {"rouge_bleu", "sh_recall", "structure_quality", "logic"} | RELEVANCE_METRICS
SUPPORTED_METRICS = SURGE_TEXT_METRICS | CITATION_METRICS

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


@contextmanager
def _flag_embedding_transformers_compat():
    """Translate FlagEmbedding 1.4's dtype kwarg for older transformers 4.x."""
    from transformers import AutoModel

    original_from_pretrained = AutoModel.from_pretrained

    @classmethod
    def from_pretrained_compat(cls, *args, **kwargs):
        if "dtype" in kwargs and "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return original_from_pretrained(*args, **kwargs)

    AutoModel.from_pretrained = from_pretrained_compat
    try:
        yield
    finally:
        AutoModel.from_pretrained = original_from_pretrained


def _resolve_local_or_hf_model(model_name: str) -> str:
    model_path = Path(model_name)
    if not model_path.is_absolute() and (ROOT / model_path).exists():
        return str(ROOT / model_path)
    return model_name


def _quiet_external_metric_logs() -> None:
    """Suppress known non-actionable info/advisory logs from metric libraries."""
    logging.getLogger("absl").setLevel(logging.WARNING)
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.WARNING)
    except Exception:
        pass


def _load_corpus_docs(corpus_path: Path, doc_ids: set[int]) -> dict[int, dict]:
    """Stream corpus.json once and keep only docs needed for relevance metrics."""
    if not doc_ids:
        return {}
    if not corpus_path.exists():
        raise SystemExit(f"corpus.json not found at {corpus_path}")

    import ijson

    wanted = set(doc_ids)
    docs: dict[int, dict] = {}
    print(
        f"Loading relevance corpus entries from {corpus_path} "
        f"({len(wanted):,} doc_ids) ..."
    )
    with open(corpus_path, "rb") as f:
        for paper in ijson.items(f, "item"):
            doc_id = paper.get("doc_id")
            if doc_id is None:
                continue
            doc_id = int(doc_id)
            if doc_id not in wanted:
                continue
            docs[doc_id] = {
                "Title": paper.get("Title") or paper.get("title") or "",
                "Abstract": paper.get("Abstract") or paper.get("abstract") or "",
            }
            if len(docs) == len(wanted):
                break
    missing = len(wanted) - len(docs)
    if missing:
        logger.warning("missing %d relevance corpus docs", missing)
    print(f"  {len(docs):,} relevance corpus entries loaded.")
    return docs


def _reference_docid_map(
    citation_evaluator: CitationEvaluator,
    generation: dict,
) -> dict[int, int | str]:
    references: list[dict] = (generation.get("meta") or {}).get("references", [])
    doc_ids = citation_evaluator.match_references(references)
    out: dict[int, int | str] = {}
    for pos, (ref, doc_id) in enumerate(zip(references, doc_ids), start=1):
        ref_idx = ref.get("idx", pos)
        try:
            ref_idx = int(ref_idx)
        except (TypeError, ValueError):
            continue
        out[ref_idx] = (
            doc_id
            if doc_id is not None
            else (ref.get("canonical_title") or ref.get("title") or "")
        )
    return out


def _collect_relevance_doc_ids(
    citation_evaluator: CitationEvaluator,
    generation_files: list[str],
) -> set[int]:
    doc_ids: set[int] = set()
    for path_str in generation_files:
        path = Path(path_str)
        try:
            generation = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("could not inspect generation %s for relevance doc ids: %s", path, e)
            continue
        for doc_id in citation_evaluator.match_references(
            (generation.get("meta") or {}).get("references", [])
        ):
            if doc_id is not None:
                doc_ids.add(int(doc_id))
    return doc_ids


def _paper_prompt(title: str, abstract: str) -> str:
    return f"There is a paper. Title: '{title}'. Abstract: '{abstract}'"


def _relevance_prompt(title: str, abstract: str, target: str, level: str) -> str:
    if level == "paper":
        return (
            f"The paper titled '{title}' with the given abstract could be cited "
            f"in the paper: '{target}'."
        )
    if level == "section":
        return (
            f"The paper titled '{title}' with the given abstract is relevant "
            f"to the section: '{target}'."
        )
    if level == "sentence":
        return (
            f"The paper titled '{title}' with the given abstract could be cited "
            f"in the sentence: '{target}'."
        )
    raise ValueError(f"unknown relevance level: {level}")


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
    generation: dict,
    target_survey: dict,
    eval_list: list[str],
    flag_model,
    nli_model,
    citation_evaluator: CitationEvaluator | None,
    corpus_docs: dict[int, dict],
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
    unknown = set(eval_list) - SURGE_TEXT_METRICS
    if unknown:
        raise ValueError(f"unsupported SurGE text metric(s): {sorted(unknown)}")

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

        if "rouge_bleu" in eval_list:
            import rougeBleuFuncs

            if log_fn: log_fn("metric start  rouge_bleu")
            r1, r2, rl, bleu = rougeBleuFuncs.eval_rougeBleu(target_survey, psg_node)
            scores.update({
                "rouge_1": float(r1),
                "rouge_2": float(r2),
                "rouge_l": float(rl),
                "bleu":    float(bleu),
            })
            if log_fn:
                log_fn(
                    "metric done   rouge_bleu = "
                    f"rouge_1={scores['rouge_1']} rouge_2={scores['rouge_2']} "
                    f"rouge_l={scores['rouge_l']} bleu={scores['bleu']}"
                )

        if "sh_recall" in eval_list:
            if log_fn: log_fn("metric start  sh_recall")
            scores["sh_recall"] = structureFuncs.eval_SHRecall(
                target_survey, psg_node, flag_model
            )
            if log_fn: log_fn(f"metric done   sh_recall = {scores['sh_recall']}")

        relevance_eval = RELEVANCE_METRICS & set(eval_list)
        if relevance_eval:
            if nli_model is None:
                raise RuntimeError("NLI model is required for relevance metrics")
            if citation_evaluator is None:
                raise RuntimeError("Citation evaluator is required for relevance metrics")

            refid2docid = _reference_docid_map(citation_evaluator, generation)

            if "relevance_paper" in relevance_eval:
                if log_fn: log_fn("metric start  relevance_paper")
                refcontent = {}
                survey_title = target_survey.get("survey_title", "")
                for ref_idx, doc_id in refid2docid.items():
                    doc = corpus_docs.get(doc_id) if isinstance(doc_id, int) else None
                    if doc is None:
                        refcontent[ref_idx] = ("[NOTEXIST]", "[NOTEXIST]")
                        continue
                    title = doc["Title"]
                    abstract = doc["Abstract"]
                    refcontent[ref_idx] = (
                        _paper_prompt(title, abstract),
                        _relevance_prompt(title, abstract, survey_title, "paper"),
                    )
                if refid2docid:
                    with redirect_stdout(io.StringIO()):
                        score = informationFuncs.eval_relevance_paper(
                            target_survey, refid2docid, refcontent, nli_model,
                        )
                else:
                    score = 0
                scores["relevance_paper"] = float(score)
                if log_fn:
                    log_fn(
                        f"metric done   relevance_paper = {scores['relevance_paper']}"
                    )

            if {"relevance_section", "relevance_sentence"} & relevance_eval:
                extracted_cites = informationFuncs.extract_cites_with_subtitle_and_sentence(psg_node)
                nli_pairs_section = []
                nli_pairs_sentence = []
                for ref_num, subtitle, sentence in extracted_cites:
                    doc_id = refid2docid.get(ref_num)
                    doc = corpus_docs.get(doc_id) if isinstance(doc_id, int) else None
                    if doc is None:
                        premise = "[NOTEXIST]"
                        section_hypothesis = "[NOTEXIST]"
                        sentence_hypothesis = "[NOTEXIST]"
                    else:
                        title = doc["Title"]
                        abstract = doc["Abstract"]
                        premise = _paper_prompt(title, abstract)
                        section_hypothesis = _relevance_prompt(
                            title, abstract, subtitle, "section",
                        )
                        sentence_hypothesis = _relevance_prompt(
                            title, abstract, sentence, "sentence",
                        )
                    nli_pairs_section.append((premise, section_hypothesis))
                    nli_pairs_sentence.append((premise, sentence_hypothesis))

                if "relevance_section" in relevance_eval:
                    if log_fn: log_fn("metric start  relevance_section")
                    score = (
                        informationFuncs.eval_relevance_section(nli_pairs_section, nli_model)
                        if extracted_cites else 0
                    )
                    scores["relevance_section"] = float(score)
                    if log_fn: log_fn(
                        f"metric done   relevance_section = {scores['relevance_section']}"
                    )

                if "relevance_sentence" in relevance_eval:
                    if log_fn: log_fn("metric start  relevance_sentence")
                    score = (
                        informationFuncs.eval_relevance_sentence(nli_pairs_sentence, nli_model)
                        if extracted_cites else 0
                    )
                    scores["relevance_sentence"] = float(score)
                    if log_fn: log_fn(
                        f"metric done   relevance_sentence = {scores['relevance_sentence']}"
                    )

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
        nli_model         — CrossEncoder model name (required for relevance_*)
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
        _quiet_external_metric_logs()

        self.eval_list = config["eval_list"]
        unknown = set(self.eval_list) - SUPPORTED_METRICS
        if unknown:
            raise SystemExit(f"Unsupported SurGE metric(s) in eval_list: {sorted(unknown)}")
        self.judge_client, self.judge_model = build_judge_client(config)
        needs_relevance = bool(RELEVANCE_METRICS & set(self.eval_list))

        self.flag_model = None
        if "sh_recall" in self.eval_list:
            try:
                from FlagEmbedding import FlagModel
                import torch
            except ImportError:
                raise SystemExit("FlagEmbedding is required for sh_recall: pip install FlagEmbedding")
            emb_model = config.get("embedding_model", "BAAI/bge-large-en-v1.5")
            print(f"Loading embedding model: {emb_model}")
            with _flag_embedding_transformers_compat():
                self.flag_model = FlagModel(
                    emb_model,
                    query_instruction_for_retrieval=(
                        "Generate a representation for this title to calculate "
                        "the similarity between titles:"
                    ),
                    use_fp16=torch.cuda.is_available(),
                )

        # Citation evaluator (corpus-based)
        self.citation_evaluator = None
        needs_citation = bool(CITATION_METRICS & set(self.eval_list))
        if needs_citation or needs_relevance:
            # Build index on first run if not present
            if not INDEX_PATH.exists():
                if not CORPUS_PATH.exists():
                    raise SystemExit(
                        f"corpus.json not found at {CORPUS_PATH}.\n"
                        f"Download from: https://drive.google.com/drive/folders/1ZZPeZvjexFcCmgFqxftKeCPn1vYeBR0Q"
                    )
                build_title_index(CORPUS_PATH, INDEX_PATH)
            self.citation_evaluator = CitationEvaluator(INDEX_PATH)

        self.nli_model = None
        self.relevance_corpus_docs: dict[int, dict] = {}
        if needs_relevance:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise SystemExit(
                    "sentence-transformers is required for relevance metrics: "
                    "pip install sentence-transformers"
                )
            generation_files = [
                str(p) for p in config.get("_surge_generation_files", [])
            ]
            doc_ids = _collect_relevance_doc_ids(self.citation_evaluator, generation_files)
            self.relevance_corpus_docs = _load_corpus_docs(CORPUS_PATH, doc_ids)

            nli_model_name = _resolve_local_or_hf_model(
                config.get("nli_model", "cross-encoder/nli-deberta-v3-base")
            )
            nli_device = config.get("nli_device") or None
            print(
                f"Loading NLI model: {nli_model_name}"
                + (f" (device={nli_device})" if nli_device else "")
            )
            if nli_device:
                self.nli_model = CrossEncoder(nli_model_name, device=nli_device)
            else:
                self.nli_model = CrossEncoder(nli_model_name)

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
                    tmp_path, generation, target_survey, surge_eval_list,
                    self.flag_model, self.nli_model,
                    self.citation_evaluator, self.relevance_corpus_docs,
                    self.judge_client, self.judge_model,
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
