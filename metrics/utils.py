"""Shared utilities for all metric modules.

Consolidates boilerplate that used to be duplicated across every metric:
  - TokenCounter           — thread-safe LLM usage/cost accumulator
  - load_config / make_client — YAML + OpenAI client setup
  - resolve_claims_dir      — Claimify cache guard
  - load_generation_files   — list survey JSONs (filters _raw/_old)
  - check_and_load_cache    — resume-mode helper for process_survey
  - write_summary_csv       — standardized summary.csv writer
  - strip_and_parse_json    — strip markdown code blocks, parse JSON
  - llm_json_call           — LLM call with retry, token tracking, JSON parsing
"""
import csv
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import yaml
from openai import OpenAI

logger = logging.getLogger(__name__)

# ── Repository root (metrics/ is one level below root) ───────────────────────
ROOT = Path(__file__).resolve().parent.parent


# ── Config & client ───────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    """Load YAML configuration from the given path.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is malformed.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_client(cfg: dict) -> OpenAI:
    """Create an OpenAI client from config.

    Expects cfg to contain 'judge_api_key_env' and 'judge_base_url'.

    Args:
        cfg: Configuration dict with judge API settings.

    Returns:
        Configured OpenAI client.

    Raises:
        RuntimeError: If the API key environment variable is not set.
    """
    key_env = cfg.get("judge_api_key_env")
    if not key_env:
        raise RuntimeError("Config missing 'judge_api_key_env'")
    api_key = os.environ.get(key_env, "")
    if not api_key:
        raise RuntimeError(f"API key not set: env var '{key_env}'")
    base_url = cfg.get("judge_base_url")
    if not base_url:
        raise RuntimeError("Config missing 'judge_base_url'")
    return OpenAI(api_key=api_key, base_url=base_url)


# ── Token counter ─────────────────────────────────────────────────────────────

@dataclass
class TokenCounter:
    """Thread-safe accumulator for LLM token usage and cost (fresh API calls only).

    Cache hits should NOT be reported here — only actual API calls count towards
    the totals, otherwise cost accounting becomes misleading.
    """
    in_tokens:  int   = 0
    out_tokens: int   = 0
    cost_usd:   float = 0.0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def add(self, in_tok: int, out_tok: int, cost: float = 0.0) -> None:
        """Add tokens and cost to the counter atomically."""
        with self._lock:
            self.in_tokens  += in_tok
            self.out_tokens += out_tok
            self.cost_usd   += cost

    def fmt(self) -> str:
        """Return a compact human-readable summary, e.g. ``"12k↑ 3k↓ $0.0042"``."""
        def _k(n: int) -> str:
            return f"{n // 1000}k" if n >= 1000 else str(n)
        cost_str = f" ${self.cost_usd:.4f}" if self.cost_usd else ""
        return f"{_k(self.in_tokens)}↑ {_k(self.out_tokens)}↓{cost_str}"


# ── Claimify cache guard ──────────────────────────────────────────────────────

def resolve_claims_dir(dataset: str, model: str) -> Path:
    """Return path to pre-computed Claimify claims directory.

    Fails fast with a clear message if the directory is absent or empty.

    Args:
        dataset: Dataset identifier (e.g. ``"SurGE"``).
        model: Model identifier (e.g. ``"perplexity_dr"``).

    Returns:
        Path to the claims directory.
    """
    claims_dir = ROOT / "results" / "scores" / f"{dataset}_{model}_claims"
    if not claims_dir.exists() or not any(claims_dir.glob("*.json")):
        print(
            f"\n[ERROR] Claimify cache not found at:\n  {claims_dir}\n\n"
            f"Run first:\n"
            f"  make evaluate DATASET={dataset} MODEL={model} METRIC=claimify\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return claims_dir


# ── Generation file discovery ────────────────────────────────────────────────

_RAW_OLD_RE = re.compile(r"_(raw|old)\.json$")


def load_generation_files(gen_dir: Path) -> list[Path]:
    """List survey JSON files in a generation directory, excluding ``_raw``/``_old`` backups.

    Args:
        gen_dir: Path to ``results/generations/<dataset>_<model>/``.

    Returns:
        List of ``.json`` files sorted by numeric ``survey_id`` (filename
        stem). Non-numeric stems (rare) sort lexicographically after the
        numeric bucket. This avoids the classic ``"10.json" < "2.json"``
        trap on sparse/large id sets like SurGE_reference.
    """
    files = sorted(
        gen_dir.glob("*.json"),
        key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem),
    )
    return [f for f in files if not _RAW_OLD_RE.search(f.name)]


def filter_by_limit(files: list[Path], limit: int | None) -> list[Path]:
    """Apply id-based ``--limit`` filter: keep files where ``int(stem) <= limit``.

    ID-based (NOT count-based): ``--limit 10`` on a sparse id set
    {0, 1, 2, 5, 10, 41} returns {0, 1, 2, 5, 10}, NOT the first ten files.
    Non-numeric stems are dropped from the result when ``limit`` is set
    (callers that want them must pass ``limit=None``).

    Args:
        files: List of generation/score JSON paths (filename = ``<sid>.json``).
        limit: Maximum survey id (inclusive), or ``None`` to disable filtering.
    """
    if limit is None:
        return files
    return [f for f in files if f.stem.isdigit() and int(f.stem) <= limit]


# ── Resume-mode cache helper ─────────────────────────────────────────────────

def check_and_load_cache(
    out_file: Path,
    cfg: dict,
    survey_id: str,
    required_keys: Iterable[str] = ("survey_id",),
) -> dict | None:
    """Resume-mode helper: return cached result if valid, else None.

    When ``cfg["resume"]`` is truthy and ``out_file`` exists, load and validate
    that it contains the required keys. Returns the parsed dict on success so
    the caller can ``return`` it immediately. On corrupt / missing keys, logs a
    warning and returns None (caller re-processes).

    Args:
        out_file: Expected cache file path.
        cfg: Metric config; checked for ``cfg.get("resume")``.
        survey_id: Survey identifier, used in log messages.
        required_keys: Keys the cached dict must contain to be considered valid.

    Returns:
        Cached dict (valid cache hit) or None (cache miss / corrupt / resume disabled).
    """
    if not cfg.get("resume") or not out_file.exists():
        return None
    try:
        with open(out_file) as f:
            existing = json.load(f)
    except Exception:
        logger.warning("Corrupt cache for %s at %s, re-processing", survey_id, out_file)
        return None
    for key in required_keys:
        if key not in existing:
            logger.warning(
                "Cache for %s missing required key %r, re-processing",
                survey_id, key,
            )
            return None
    logger.info("[SKIP] %s — already scored", survey_id)
    return existing


# ── Summary CSV writer ───────────────────────────────────────────────────────

def write_summary_csv(
    results: list[dict],
    out_path: Path,
    fields: list[str],
    metric_name: str,
) -> Path:
    """Write a standardized ``summary.csv`` for a metric run.

    Uses ``extrasaction="ignore"`` so extra keys in result dicts are silently
    dropped — this keeps per-metric result structures flexible while the CSV
    stays tidy.

    Args:
        results: List of per-survey result dicts.
        out_path: Output directory (file written as ``out_path/summary.csv``).
        fields: Ordered CSV column names.
        metric_name: Used in the info-log line.

    Returns:
        Path to the written CSV.
    """
    csv_path = out_path / "summary.csv"
    try:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
    except OSError:
        logger.exception("Failed to write summary CSV for %s", metric_name)
        raise
    logger.info("[%s] summary → %s", metric_name, csv_path)
    return csv_path


# ── JSON parsing ─────────────────────────────────────────────────────────────

_MD_FENCE_PREFIX = re.compile(r"^```(?:json)?\s*")
_MD_FENCE_SUFFIX = re.compile(r"\s*```$")


def strip_and_parse_json(content: str) -> dict:
    """Strip markdown code fences and parse JSON.

    LLMs frequently wrap JSON in ``\u0060\u0060\u0060json ... \u0060\u0060\u0060`` despite instructions.
    This helper removes that wrapping and parses the JSON.

    Args:
        content: Raw LLM response text.

    Returns:
        Parsed JSON dict.

    Raises:
        json.JSONDecodeError: If the content is not valid JSON after stripping.
    """
    raw = content.strip()
    raw = _MD_FENCE_PREFIX.sub("", raw)
    raw = _MD_FENCE_SUFFIX.sub("", raw)
    return json.loads(raw)


# ── Unified LLM JSON call ────────────────────────────────────────────────────

def llm_json_call(
    client: OpenAI,
    model: str,
    messages: list[dict],
    *,
    max_retries: int = 3,
    temperature: float = 0.0,
    provider: str | None = None,
    disable_reasoning: bool = False,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    token_counter: TokenCounter | None = None,
    on_failure: Callable[[Exception], dict] | None = None,
) -> dict:
    """Call an LLM and parse the JSON response, with retry + token tracking.

    Consolidates the retry/parse loop that was previously duplicated across
    ``classify_claim``, ``llm_json``, ``llm_json_cached`` and ``llm_call``.

    Args:
        client: OpenAI (or compatible) client.
        model: Model name.
        messages: Chat messages list (``[{"role": ..., "content": ...}, ...]``).
        max_retries: Retry attempts with exponential backoff.
        temperature: Sampling temperature (default 0 for determinism).
        provider: Optional OpenRouter provider override.
        disable_reasoning: Disable thinking tokens (Qwen3, DeepSeek-R1).
        reasoning_effort: ``"low"|"medium"|"high"`` for reasoning-capable
            models. Takes priority over ``disable_reasoning``.
        max_tokens: Optional completion cap.
        token_counter: Optional thread-safe accumulator. Only actual API calls
            are reported (cache hits should be handled by the caller).
        on_failure: Callable invoked with the final exception if all retries
            fail. Its return value is returned from ``llm_json_call``. If
            None, the exception is re-raised.

    Returns:
        Parsed JSON dict from the LLM response.

    Raises:
        The underlying exception on persistent failure if ``on_failure`` is None.
    """
    extra_body: dict = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
    # reasoning_effort (gpt-oss) takes priority over disable_reasoning (Qwen3 / DeepSeek)
    if reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort
    elif disable_reasoning:
        extra_body["reasoning"] = {"enabled": False}

    create_kwargs: dict = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        extra_body=extra_body or None,
    )
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(**create_kwargs)
            content = resp.choices[0].message.content
            if content is None:
                raise RuntimeError(
                    f"LLM returned None content "
                    f"(finish_reason={resp.choices[0].finish_reason})"
                )
            if token_counter is not None and resp.usage is not None:
                cost = (resp.usage.model_extra or {}).get("cost") or 0.0
                token_counter.add(
                    resp.usage.prompt_tokens or 0,
                    resp.usage.completion_tokens or 0,
                    cost=float(cost),
                )
            return strip_and_parse_json(content)
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    # All retries exhausted
    logger.error(
        "LLM call failed after %d retries (model=%s): %s: %s",
        max_retries, model, type(last_exc).__name__, last_exc,
    )
    if on_failure is not None:
        return on_failure(last_exc) if last_exc else on_failure(RuntimeError("unknown"))
    assert last_exc is not None
    raise last_exc
