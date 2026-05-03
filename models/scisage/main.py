#!/usr/bin/env python
"""
models/scisage/main.py
Generation runner for SciSage.

SciSage's upstream code is written around local Qwen model aliases. This
wrapper maps those aliases to one OpenRouter model and then runs the upstream
PaperGenerationPipeline, emitting our canonical generation JSON via BaseModel.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent.parent
SCISAGE_CORE = ROOT / "repos" / "SciSage" / "core"

sys.path.insert(0, str(ROOT))
from src.models.base import BaseModel


_SCISAGE_IMPORTED = False


@contextlib.contextmanager
def _pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _resolve_env(name: str, required: bool = True) -> str:
    value = os.getenv(name, "")
    if required and not value:
        raise SystemExit(f"Env var '{name}' is not set")
    return value


def _patch_configuration(cfg: dict, api_key: str) -> None:
    """Patch SciSage configuration before its main modules are imported."""
    import configuration as scisage_cfg
    from openai import OpenAI

    base_url = cfg["base_url"].rstrip("/")
    openrouter_model = cfg["model"]
    aliases = cfg.get("model_aliases") or [cfg.get("default_model_alias", "Qwen3-8B")]
    default_alias = cfg.get("default_model_alias") or aliases[0]

    def model_config(alias: str):
        return scisage_cfg.ModelConfig(
            url=base_url,
            max_len=int(cfg.get("max_tokens", 12000)),
            model_name=openrouter_model,
            temperature=float(cfg.get("temperature", 0.7)),
            top_p=float(cfg.get("top_p", 0.8)),
            retry_attempts=int(cfg.get("retry_attempts", 3)),
            timeout=int(cfg.get("timeout", 600)),
            think_bool=alias.endswith("-think"),
            openai_client=OpenAI(api_key=api_key, base_url=base_url),
        )

    scisage_cfg.MODEL_CONFIGS.clear()
    scisage_cfg.MODEL_CONFIGS.update({alias: model_config(alias) for alias in aliases})

    # Route every upstream stage through the alias that exists in MODEL_CONFIGS.
    scisage_cfg.DEFAULT_MODEL_FOR_QUERY_INTENT = default_alias
    scisage_cfg.DEFAULT_MODEL_FOR_OUTLINE = default_alias
    scisage_cfg.OUTLINE_GENERAOR_MODELS = [default_alias]
    scisage_cfg.MODEL_GEN_QUERY = default_alias
    scisage_cfg.DEFAULT_MODEL_FOR_SECTION_RETRIVAL = default_alias
    scisage_cfg.DEFAULT_MODEL_FOR_SECTION_WRITER = default_alias
    scisage_cfg.DEFAULT_MODEL_FOR_SECTION_WRITER_IMAGE_EXTRACT = default_alias
    scisage_cfg.DEFAULT_MODEL_FOR_SECTION_WRITER_RERANK = default_alias
    scisage_cfg.SECTION_SUMMARY_MODEL = default_alias
    scisage_cfg.SECTION_REFLECTION_MODEL_LST = [default_alias]
    scisage_cfg.DEFAULT_MODEL_FOR_GLOBAL_REFLECTION = default_alias
    scisage_cfg.MODEL_GEN_ABSTRACT_CONCLUSION = default_alias
    scisage_cfg.DEFAULT_MODEL_FOR_SECTION_NAME_REFLECTION = default_alias
    scisage_cfg.DEFAULT_MODEL_FOR_TRASNLATION = default_alias


def _openrouter_response_text(response) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", None)
    reasoning = getattr(message, "reasoning_content", None)
    return _strip_thinking(content or reasoning or "")


def _chat_completion_from_data(config, data: dict) -> str:
    kwargs = {
        "model": config.model_name,
        "messages": data["messages"],
        "temperature": data.get("temperature", config.temperature),
        "top_p": data.get("top_p", config.top_p),
    }
    max_tokens = data.get("max_tokens") or config.max_len
    if max_tokens:
        kwargs["max_tokens"] = int(max_tokens)
    tools = data.get("tools")
    if tools:
        kwargs["tools"] = tools
    response = config.openai_client.chat.completions.create(**kwargs)
    return _openrouter_response_text(response)


def _patch_request_clients() -> None:
    """
    SciSage has two local-LLM adapters. Patch both to use OpenRouter chat
    completions and to read message.content instead of Qwen reasoning_content.
    """
    import local_model_langchain
    import local_request_v2

    def langchain_make_request(self, url: str, data: dict, timeout: int) -> str | None:
        try:
            config = local_model_langchain.MODEL_CONFIGS[data["model"]]
            return _chat_completion_from_data(config, data)
        except Exception:
            local_model_langchain.logger.error(
                "OpenRouter request failed: %s", traceback.format_exc()
            )
            return None

    def request_v2_make_request(self, url: str, data: dict, timeout: int) -> str | None:
        try:
            config = local_request_v2.MODEL_CONFIGS[data["model"]]
            return _chat_completion_from_data(config, data)
        except Exception:
            local_request_v2.logger.error(
                "OpenRouter request failed: %s", traceback.format_exc()
            )
            return None

    def request_v2_get_from_llm(messages, model_name: str = "Qwen3-8B", **kwargs):
        if model_name not in local_request_v2.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        config = local_request_v2.MODEL_CONFIGS[model_name]
        formatted_messages = local_request_v2.client.format_messages(messages, model_name)
        data = {
            "model": model_name,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", config.temperature),
            "top_p": kwargs.get("top_p", config.top_p),
            "max_tokens": kwargs.get("max_len", config.max_len),
            "tools": [],
        }
        for attempt in range(config.retry_attempts):
            response = local_request_v2.client._make_request(
                config.url, data, config.timeout
            )
            if response:
                return _strip_thinking(response)
            if attempt < config.retry_attempts - 1:
                time.sleep(2)
        return None

    local_model_langchain.RequestsClient.make_request = langchain_make_request
    local_request_v2.LLMClient._make_request = request_v2_make_request
    local_request_v2.get_from_llm = request_v2_get_from_llm


def _load_scisage_pipeline(cfg: dict, api_key: str):
    """Import SciSage after installing our OpenRouter patches."""
    global _SCISAGE_IMPORTED

    if not SCISAGE_CORE.exists():
        raise SystemExit(
            f"SciSage repo not found: {SCISAGE_CORE.parent}\n"
            f"Run: git clone https://github.com/FlagOpen/SciSage.git repos/SciSage"
        )

    if str(SCISAGE_CORE) not in sys.path:
        sys.path.insert(0, str(SCISAGE_CORE))

    _patch_configuration(cfg, api_key)
    _patch_request_clients()

    if not _SCISAGE_IMPORTED:
        _SCISAGE_IMPORTED = True

    from main_workflow_opt_for_paper import PaperGenerationPipeline

    return PaperGenerationPipeline


_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,6})(?:v\d+)?", re.I)


def _as_dict(ref: Any) -> dict:
    if hasattr(ref, "model_dump"):
        return ref.model_dump()
    if hasattr(ref, "dict"):
        return ref.dict()
    if isinstance(ref, dict):
        return ref
    return {}


def _normalize_reference(ref: Any, idx: int) -> dict:
    data = _as_dict(ref)
    url = data.get("url") or data.get("arxivUrl") or ""
    arxiv_id = None
    match = _ARXIV_RE.search(url)
    if match:
        arxiv_id = match.group(1)

    return {
        "idx": idx,
        "title": data.get("title") or None,
        "url": url,
        "arxiv_id": arxiv_id,
        "canonical_title": None,
        "authors": data.get("authors") or None,
        "source": data.get("source") or None,
        "conference": data.get("conference") or data.get("venue") or None,
        "abstract": data.get("abstract") or None,
    }


def _extract_markdown_references(text: str) -> list[dict]:
    refs = []
    in_refs = False
    for line in text.splitlines():
        if re.match(r"^#{1,3}\s+references\s*$", line.strip(), re.I):
            in_refs = True
            continue
        if in_refs and re.match(r"^#{1,3}\s+\S", line):
            break
        if not in_refs:
            continue
        match = re.match(r"^\s*(?:\[(\d+)\]|(\d+)\.)\s*(.+?)\s*(?:URL:\s*(\S+))?\s*$", line)
        if not match:
            continue
        idx = int(match.group(1) or match.group(2))
        title = match.group(3).strip()
        url = match.group(4) or ""
        refs.append(_normalize_reference({"title": title, "url": url}, idx))
    return refs


class SciSage(BaseModel):
    """SciSage pipeline via OpenRouter."""

    def __init__(self) -> None:
        super().__init__(Path(__file__).parent)

        self.api_key = _resolve_env(self.cfg["api_key_env"])
        if self.cfg.get("require_search_api_key", True):
            _resolve_env(self.cfg.get("search_api_key_env", "GOOGLE_SERPER_KEY"))

        self.pipeline_cls = _load_scisage_pipeline(self.cfg, self.api_key)
        print(
            "  [SciSage] OpenRouter patch active: "
            f"{self.cfg['base_url']} -> {self.cfg['model']}"
        )

    def generate(self, instance) -> dict:
        t0 = time.time()
        raw_dir = ROOT / self.cfg.get("raw_output_dir", "tmp/scisage") / str(instance.id)
        raw_dir.mkdir(parents=True, exist_ok=True)

        try:
            kwargs = {
                "outline_max_reflections": self.cfg.get("outline_max_reflections", 1),
                "outline_max_sections": self.cfg.get("outline_max_sections", 3),
                "outline_min_depth": self.cfg.get("outline_min_depth", 1),
                "section_writer_model": self.cfg.get("section_writer_model_alias", "Qwen3-8B"),
                "do_section_reflection": self.cfg.get("do_section_reflection", False),
                "section_reflection_max_turns": self.cfg.get("section_reflection_max_turns", 0),
                "do_global_reflection": self.cfg.get("do_global_reflection", False),
                "global_reflection_max_turns": self.cfg.get("global_reflection_max_turns", 1),
                "global_abstract_conclusion_max_turns": self.cfg.get(
                    "global_abstract_conclusion_max_turns", 1
                ),
                "do_query_understand": self.cfg.get("do_query_understand", False),
                "translate_to_chinese": self.cfg.get("translate_to_chinese", False),
            }

            with _pushd(SCISAGE_CORE):
                pipeline = self.pipeline_cls(
                    user_name="thesis",
                    user_query=instance.query,
                    task_id=f"scisage_{instance.id}",
                    output_dir=str(raw_dir),
                    **kwargs,
                )
                if self.cfg.get("query_domain"):
                    pipeline.state.query_domain = self.cfg["query_domain"]
                results = pipeline.generate_paper()
                # Upstream API is async.
                import asyncio

                results = asyncio.run(results)

            final_paper = results.get("final_paper") or {}
            text = (
                final_paper.get("markdown_content")
                or final_paper.get("markdown_content_en")
                or ""
            ).strip()

            raw_result = raw_dir / "scisage_result.json"
            raw_result.write_text(
                json.dumps(results, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
            if text:
                (raw_dir / "paper.md").write_text(text, encoding="utf-8")

            refs_raw = final_paper.get("reportIndexList") or []
            references = [
                _normalize_reference(ref, i)
                for i, ref in enumerate(refs_raw, 1)
            ]
            if not references and text:
                references = _extract_markdown_references(text)

            error = results.get("error")
            return {
                "text": text,
                "success": bool(text) and not error,
                "meta": {
                    "model": self.cfg["model"],
                    "base_url": self.cfg["base_url"],
                    "latency_sec": round(time.time() - t0, 2),
                    "cost_usd": None,
                    "error": error or (None if text else "No markdown_content in SciSage output"),
                    "references": references,
                    "raw_output_dir": str(raw_dir),
                    "paper_title": final_paper.get("paper_title") or results.get("paper_title"),
                    "outline_max_sections": self.cfg.get("outline_max_sections", 3),
                    "query_domain": self.cfg.get("query_domain"),
                },
            }

        except Exception as e:
            return {
                "text": "",
                "success": False,
                "meta": {
                    "model": self.cfg["model"],
                    "base_url": self.cfg["base_url"],
                    "latency_sec": round(time.time() - t0, 2),
                    "cost_usd": None,
                    "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                    "references": [],
                    "raw_output_dir": str(raw_dir),
                },
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate surveys with SciSage")
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. SurGE)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate only surveys with survey_id <= LIMIT (inclusive, id-based).",
    )
    args = parser.parse_args()

    SciSage().run(args.dataset, limit=args.limit)
