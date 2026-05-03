"""
app/main.py  —  Thesis results viewer
Run from the repo root:
    streamlit run app/main.py
"""
import json
import logging
import pathlib
import re
import sys
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.log_setup import setup_logging


# Streamlit re-executes the entire script on every widget interaction, which
# without protection fires `setup_logging("app")` dozens of times per session
# and spams the "NEW RUN" banner into results/logs/app.log. `st.cache_resource`
# runs the body exactly once per process — ideal for one-shot side effects
# like logging setup.
@st.cache_resource
def _init_app_logging() -> bool:
    setup_logging("app")
    return True


_init_app_logging()

GENERATIONS_DIR = ROOT / "results" / "generations"
SCORES_DIR      = ROOT / "results" / "scores"
HYPEROPT_DIR    = ROOT / "results" / "hyperopt"
ANALISIS_DIR    = ROOT / "analisis"
SAMPLES_DIR     = ANALISIS_DIR / "samples"
MARKUPS_DIR     = ANALISIS_DIR / "markups"

st.set_page_config(page_title="Thesis Viewer", layout="wide")

# ── Data helpers ───────────────────────────────────────────────────────────────

def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_generation_runs() -> list[tuple[str, pathlib.Path]]:
    if not GENERATIONS_DIR.exists():
        return []
    return sorted(
        (d.name, d) for d in GENERATIONS_DIR.iterdir() if d.is_dir()
    )


def find_score_runs() -> list[tuple[str, pathlib.Path]]:
    runs = []
    if SCORES_DIR.exists():
        for d in SCORES_DIR.iterdir():
            if d.is_dir():
                runs.append((d.name, d))
    # Legacy path
    legacy = ROOT / "results" / "surge_perplexity_surge" / "scores"
    if legacy.exists():
        runs.append(("surge_perplexity_surge [legacy]", legacy))
    return sorted(runs)


def _numeric_stem_key(p: pathlib.Path) -> tuple[int, str]:
    """Sort ``<N>.json`` files numerically; fall back to lexical for non-numeric stems.

    Viewer runs iterate over files whose names are ``<survey_id>.json``. A plain
    ``sorted(glob)`` gives lexical order, so for sparse ids (SurGE_reference =
    0, 1, 2, 3, 5, 6, 8, 10, ...) you get 0, 1, 10, 11, ..., 2, 20, ..., 3.
    Next/previous navigation then jumps unnaturally (1 → 10 instead of 1 → 2).

    The two-element tuple key groups numeric ids first (bucket 0, sorted by
    int) and any non-numeric filename after (bucket 1, sorted by string) so
    both coexist predictably.
    """
    s = p.stem
    return (0, int(s)) if s.isdigit() else (1, s)


def load_generations(run_dir: pathlib.Path) -> list[dict]:
    """Load generation JSON files from a run directory, skipping files that fail to parse."""
    out = []
    for f in sorted(run_dir.glob("*.json"), key=_numeric_stem_key):
        try:
            out.append(load_json(f))
        except Exception:
            logger.debug(f"Failed to load generation file {f}", exc_info=True)
    return out


def normalize_score(data: dict) -> dict:
    """Unify old {topic_id, survey_title, ...} and new {id, ...} formats."""
    return {
        "id":        data.get("id") or data.get("topic_id", "?"),
        "title":     data.get("query") or data.get("survey_title", ""),
        "scores":    data.get("scores", {}),
        "judge_log": data.get("judge_log", []),
    }


def load_scores(run_dir: pathlib.Path) -> list[dict]:
    """Load and normalize score JSON files from a run directory, skipping summary and invalid files."""
    out = []
    for f in sorted(run_dir.glob("*.json"), key=_numeric_stem_key):
        if f.stem == "summary":
            continue
        try:
            out.append(normalize_score(load_json(f)))
        except Exception:
            logger.debug(f"Failed to load score file {f}", exc_info=True)
    return out


@st.cache_data(show_spinner=False)
def _load_full_text_ref_ids(dataset: str, model: str, survey_id: str) -> set[int] | None:
    """Refs with cached full text for one generation file."""
    p = (
        GENERATIONS_DIR / f"{dataset}_{model}"
        / "sources" / f"{survey_id}_sources.json"
    )
    if not p.exists():
        return None
    try:
        data = load_json(p)
    except Exception:
        logger.debug(f"Failed to load sources file {p}", exc_info=True)
        return None

    refs = data.get("refs") if isinstance(data, dict) else None
    if isinstance(refs, dict):
        refs_iter = refs.values()
    elif isinstance(refs, list):
        refs_iter = refs
    else:
        return None

    out: set[int] = set()
    for ref in refs_iter:
        if not isinstance(ref, dict):
            continue
        try:
            idx = int(ref.get("idx"))
        except (TypeError, ValueError):
            continue
        text = ref.get("text") or ref.get("full_text")
        if isinstance(text, str) and text.strip():
            out.add(idx)
    return out


# ── Navigation widget ──────────────────────────────────────────────────────────

def nav_arrows(state_key: str, total: int) -> int:
    """← index/total → . Returns clamped current index from session_state."""
    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    # Clamp: protects against stale index after run/page change
    st.session_state[state_key] = max(0, min(st.session_state[state_key], total - 1))

    col_prev, col_info, col_next = st.columns([1, 6, 1])
    with col_prev:
        if st.button("←", key=f"{state_key}_prev",
                     disabled=st.session_state[state_key] == 0):
            st.session_state[state_key] -= 1
            st.rerun()
    with col_info:
        st.caption(f"Survey {st.session_state[state_key] + 1} of {total}")
    with col_next:
        if st.button("→", key=f"{state_key}_next",
                     disabled=st.session_state[state_key] >= total - 1):
            st.session_state[state_key] += 1
            st.rerun()

    return st.session_state[state_key]


def reset_idx_on_run_change(run_label: str, idx_key: str, run_key: str) -> None:
    if st.session_state.get(run_key) != run_label:
        st.session_state[idx_key] = 0
        st.session_state[run_key] = run_label


# ── Markup helpers ────────────────────────────────────────────────────────────

_STRUCTURAL_MARKUP_CLASSES = {
    "internal_author_contradiction": (
        "внутреннее противоречие автора обзора (настоящий дефект)"
    ),
    "literature_contradiction": (
        "зафиксированное противоречие в литературе с явным указанием источников "
        "(признак критического анализа)"
    ),
    "judge_false_positive": "ложное срабатывание judge (артефакт)",
}


def _sample_items(data) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        items = data.get("samples") or data.get("items") or []
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]
    return []


def _load_existing_markup(sample_file: pathlib.Path) -> dict:
    path = MARKUPS_DIR / f"{sample_file.stem}_markup.json"
    if not path.exists():
        return {}
    try:
        data = load_json(path)
    except Exception:
        logger.debug(f"Failed to load markup file {path}", exc_info=True)
        return {}
    rows = data.get("markups", []) if isinstance(data, dict) else []
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("sample_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("sample_id") is not None
    }


def _render_sample_text(label: str, text: str | None) -> None:
    if not text:
        return
    st.markdown(f"**{label}**")
    st.markdown(str(text))


def _markup_sentence_norm(text: str) -> str:
    text = re.sub(r"\[[^\]]{1,40}\]", "", text or "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _split_markup_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        if len(s.strip()) > 30
    ]


def _find_sentence_window(
    full_text: str,
    paragraph_text: str | None,
    *,
    radius: int = 10,
) -> dict | None:
    if not full_text or not paragraph_text:
        return None

    full_sents = _split_markup_sentences(full_text)
    para_sents = _split_markup_sentences(paragraph_text)
    if not full_sents or not para_sents:
        return None

    full_norm = [_markup_sentence_norm(sent) for sent in full_sents]
    para_norm = [_markup_sentence_norm(sent) for sent in para_sents]
    para_norm = [sent for sent in para_norm if sent]
    if not para_norm:
        return None

    start_idx = None
    width = min(len(para_norm), len(full_norm))
    for i in range(0, len(full_norm) - width + 1):
        if full_norm[i:i + width] == para_norm[:width]:
            start_idx = i
            break

    if start_idx is None:
        first = para_norm[0]
        for i, sent in enumerate(full_norm):
            if first in sent or sent in first:
                start_idx = i
                break

    if start_idx is None:
        para_blob = " ".join(para_norm)
        hits = [i for i, sent in enumerate(full_norm) if sent and sent in para_blob]
        if hits:
            start_idx = hits[0]

    if start_idx is None:
        return None

    end_idx = min(len(full_sents), start_idx + len(para_norm))
    left = max(0, start_idx - radius)
    right = min(len(full_sents), end_idx + radius)
    return {
        "text": " ".join(full_sents[left:right]),
        "window_start_sentence": left + 1,
        "window_end_sentence": right,
        "paragraph_start_sentence": start_idx + 1,
        "paragraph_end_sentence": end_idx,
        "total_sentences": len(full_sents),
        "radius_sentences": radius,
    }


def _dataset_model_from_structural_run(run_name: str | None) -> tuple[str | None, str | None]:
    if not run_name or "_structural" not in run_name:
        return None, None
    prefix = run_name.split("_structural", 1)[0]
    if "_" not in prefix:
        return None, None
    dataset, model = prefix.split("_", 1)
    return dataset or None, model or None


def _load_generation_text_for_sample(sample: dict) -> str | None:
    source_file = sample.get("source_file")
    score_data = {}
    if source_file:
        p = ROOT / source_file
        if not p.exists():
            p = pathlib.Path(source_file)
        if p.exists():
            try:
                score_data = load_json(p)
            except Exception:
                logger.debug(f"Failed to load source score file {p}", exc_info=True)

    dataset = score_data.get("dataset_id")
    model = score_data.get("model_id")
    survey_id = str(score_data.get("survey_id") or sample.get("survey_id") or "")

    if not dataset or not model:
        dataset, model = _dataset_model_from_structural_run(
            sample.get("source_run") or score_data.get("source_run")
        )
    if not dataset or not model or not survey_id:
        return None

    gen_dirs = [GENERATIONS_DIR / f"{dataset}_{model}"]
    if str(model).startswith(f"{dataset}_"):
        gen_dirs.append(GENERATIONS_DIR / str(model))

    gen_file = None
    for gen_dir in gen_dirs:
        candidate = gen_dir / f"{survey_id}.json"
        if candidate.exists():
            gen_file = candidate
            break
    if gen_file is None:
        return None
    try:
        gen = load_json(gen_file)
    except Exception:
        logger.debug(f"Failed to load generation file {gen_file}", exc_info=True)
        return None
    return gen.get("text") or gen.get("markdown_text")


def _context_window_for_sample(sample: dict, side: int) -> dict | None:
    saved = sample.get(f"context_{side}")
    if isinstance(saved, dict) and saved.get("text"):
        return saved

    full_text = _load_generation_text_for_sample(sample)
    paragraph = sample.get(f"paragraph_{side}")
    computed = _find_sentence_window(full_text or "", paragraph)
    if computed:
        return computed

    if saved:
        return {"text": str(saved)}
    return None


def _render_context_position(context: dict) -> None:
    start = context.get("window_start_sentence")
    end = context.get("window_end_sentence")
    total = context.get("total_sentences")
    para_start = context.get("paragraph_start_sentence")
    para_end = context.get("paragraph_end_sentence")
    if start is None or end is None:
        return

    total_part = f" / {total}" if total is not None else ""
    if para_start is not None and para_end is not None:
        st.caption(
            f"Window sentences: {start}-{end}{total_part}; "
            f"paragraph sentences: {para_start}-{para_end}."
        )
    else:
        st.caption(f"Window sentences: {start}-{end}{total_part}.")


def _render_context_window(sample: dict, side: int) -> None:
    context = _context_window_for_sample(sample, side)
    if context:
        _render_sample_text(
            f"Text window {side} (±10 sentences)",
            str(context.get("text") or ""),
        )
        _render_context_position(context)
        return
    _render_sample_text(f"Paragraph {side}", sample.get(f"paragraph_{side}"))


# ── Page: Generations ──────────────────────────────────────────────────────────

def page_generations() -> None:
    """Display generations page: browse generated surveys by dataset and model."""
    st.header("Generations")

    runs = find_generation_runs()
    if not runs:
        st.warning("No generation directories found in results/generations/")
        return

    selected = st.sidebar.selectbox(
        "Dataset × Model", [r[0] for r in runs]
    )
    run_dir = dict(runs)[selected]

    generations = load_generations(run_dir)
    if not generations:
        st.info("No generation files found.")
        return

    reset_idx_on_run_change(selected, "gen_idx", "_gen_run")
    idx = nav_arrows("gen_idx", len(generations))
    gen = generations[idx]

    st.subheader(gen.get("query", "—"))

    # Meta row
    meta = gen.get("meta", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status",   "✅ OK" if gen.get("success") else "❌ Failed")
    c2.metric("Latency",  f"{meta.get('latency_sec', '—')} s")
    c3.metric("Cost",     f"${meta.get('cost_usd', '—')}")
    tok_in  = meta.get("input_tokens",  "—")
    tok_out = meta.get("output_tokens", "—")
    c4.metric("Tokens",   f"{tok_in} in / {tok_out} out")

    st.divider()

    text = gen.get("text", "")
    word_count = len(text.split()) if text else 0
    with st.expander(f"Survey text  ·  {word_count:,} words", expanded=True):
        st.markdown(text or "_No text_")


# ── Page: Evaluations ─────────────────────────────────────────────────────────

def page_evaluations() -> None:
    """Display evaluations page: browse evaluation scores and judge logs for surveys."""
    st.header("Evaluations")

    runs = find_score_runs()
    if not runs:
        st.warning("No score directories found.")
        return

    selected = st.sidebar.selectbox(
        "Evaluation run", [r[0] for r in runs]
    )
    run_dir = dict(runs)[selected]

    # ── Route by run type ──────────────────────────────────────────────────────
    if is_diversity_run(selected):
        page_evaluations_diversity(run_dir, selected)
        return

    if is_factuality_run(selected):
        page_evaluations_factuality(run_dir, selected)
        return

    scores = load_scores(run_dir)
    if not scores:
        st.info("No score files found.")
        return

    # ── Summary table ──────────────────────────────────────────────────────
    st.subheader("Summary")
    rows = [{"id": s["id"], **s["scores"]} for s in scores]
    df = pd.DataFrame(rows).set_index("id")

    numeric_cols = df.select_dtypes("number").columns
    st.dataframe(
        df.style.format("{:.2f}", subset=numeric_cols, na_rep="—"),
        width="stretch",
    )

    st.divider()

    # ── Detail view ────────────────────────────────────────────────────────
    st.subheader("Detail")

    reset_idx_on_run_change(selected, "eval_idx", "_eval_run")
    idx = nav_arrows("eval_idx", len(scores))
    survey = scores[idx]

    title = survey.get("title") or f"Survey {survey['id']}"
    st.markdown(f"**{title}**  ·  ID `{survey['id']}`")

    # Score chips
    metric_scores = survey["scores"]
    if metric_scores:
        cols = st.columns(len(metric_scores))
        for col, (metric, val) in zip(cols, metric_scores.items()):
            # Note: isinstance check relies on Python type runtime checks; consider type metadata for robustness
            formatted = f"{val:.3f}" if isinstance(val, float) else str(val)
            col.metric(metric, formatted)

    st.divider()

    # Judge log grouped by metric → paragraph
    judge_log = survey["judge_log"]
    if not judge_log:
        st.info("No judge log entries.")
        return

    # Preserve metric order from log
    metrics_seen = list(dict.fromkeys(e["metric"] for e in judge_log))

    for metric in metrics_seen:
        entries = [e for e in judge_log if e["metric"] == metric]

        # Group attempts by paragraph_idx
        paragraphs: dict[int, list[dict]] = {}
        for e in entries:
            paragraphs.setdefault(e.get("paragraph_idx", 0), []).append(e)

        n_paragraphs = len(paragraphs)
        score_val = metric_scores.get(metric)
        # Note: isinstance check relies on Python type runtime checks; consider type metadata for robustness
        score_str = f"{score_val:.3f}" if isinstance(score_val, float) else str(score_val)

        with st.expander(
            f"**{metric}**  ·  score {score_str}"
            + (f"  ·  {n_paragraphs} paragraphs" if n_paragraphs > 1 else ""),
            expanded=False,
        ):
            if n_paragraphs == 1:
                # Scalar metric (e.g. structure_quality)
                for attempt in list(paragraphs.values())[0]:
                    _render_attempt(attempt)
            else:
                # List metric (e.g. logic) — mini-table + per-paragraph expanders
                table_rows = []
                for pidx, attempts in sorted(paragraphs.items()):
                    last = attempts[-1]
                    table_rows.append({
                        "paragraph": pidx,
                        "score":     last.get("score"),
                        "attempts":  len(attempts),
                        "error":     "⚠️" if last.get("error") else "",
                    })
                tdf = pd.DataFrame(table_rows).set_index("paragraph")
                st.dataframe(tdf, width="stretch")

                for pidx, attempts in sorted(paragraphs.items()):
                    with st.expander(f"Paragraph {pidx}", expanded=False):
                        for attempt in attempts:
                            _render_attempt(attempt)
                            if len(attempts) > 1:
                                st.divider()


def _render_attempt(entry: dict) -> None:
    """Render a single judge attempt: score, error, reasoning."""
    attempt_idx = entry.get("attempt_idx", 0)
    score       = entry.get("score")
    error       = entry.get("error")
    reasoning   = entry.get("reasoning")
    raw         = entry.get("raw")

    label = f"Attempt {attempt_idx}"
    if score is not None:
        label += f"  ·  score **{score}**"
    if raw and str(raw) != str(score):
        label += f"  (raw: `{raw}`)"

    st.markdown(label)

    if error:
        st.error(f"Error: {error}")

    with st.expander("Reasoning", expanded=False):
        st.markdown(reasoning or "_No reasoning recorded_")


# ── Diversity helpers ─────────────────────────────────────────────────────────

def is_diversity_run(run_name: str) -> bool:
    """Check if a run is a diversity evaluation based on _diversity suffix in name.

    Convention: Diversity runs use _diversity suffix (e.g., SurGE_gpt4_diversity).
    TODO: This should eventually use metadata instead of string suffix matching.
    """
    return run_name.endswith("_diversity") or "_diversity_" in run_name


def load_diversity_scores(run_dir: pathlib.Path) -> list[dict]:
    """Load per-survey diversity score JSONs (skip subdirs like plots/)."""
    out = []
    for f in sorted(run_dir.glob("*.json"), key=_numeric_stem_key):
        try:
            out.append(load_json(f))
        except Exception:
            logger.debug(f"Failed to load diversity score file {f}", exc_info=True)
    return out


def load_dots(run_dir: pathlib.Path, survey_id: str) -> dict | None:
    """Load _dots.json for a given survey_id from the plots/ subdir."""
    p = run_dir / "plots" / f"{survey_id}_dots.json"
    if not p.exists():
        return None
    try:
        return load_json(p)
    except Exception:
        logger.debug(f"Failed to load dots file {p}", exc_info=True)
        return None


def find_diversity_runs_for_dataset(dataset_prefix: str) -> dict[str, pathlib.Path]:
    """Return {model_id: run_dir} for all diversity runs matching the dataset prefix."""
    result = {}
    if not SCORES_DIR.exists():
        return result
    for d in SCORES_DIR.iterdir():
        if not d.is_dir():
            continue
        if is_diversity_run(d.name) and d.name.startswith(dataset_prefix + "_"):
            # Extract model_id: strip dataset prefix and _diversity suffix
            mid = d.name[len(dataset_prefix) + 1:]
            mid = re.sub(r"_diversity.*$", "", mid)
            result[mid] = d
    return result


# ── Factuality helpers ────────────────────────────────────────────────────────

def is_factuality_run(run_name: str) -> bool:
    return "_factuality_" in run_name or run_name.endswith("_factuality")


def load_factuality_scores(run_dir: pathlib.Path) -> list[dict]:
    out = []
    for f in sorted(run_dir.glob("*.json"), key=_numeric_stem_key):
        if f.stem == "summary":
            continue
        try:
            out.append(load_json(f))
        except Exception:
            logger.debug(f"Failed to load factuality file {f}", exc_info=True)
    return out


_CAT_COLORS = {"A": "#2196F3", "B": "#4CAF50", "C": "#FF9800", "D": "#E91E63"}
_CAT_LABELS = {
    "A": "A — General",
    "B": "B — Methodological",
    "C": "C — Quantitative",
    "D": "D — Critical",
}
_CONF_ICON = {"high": "🟢", "medium": "🟡", "low": "🔴"}


def page_evaluations_factuality(run_dir: pathlib.Path, run_name: str) -> None:
    """Factuality evaluation: claim categorisation + optional AlignScore support."""
    st.header(f"Factuality — {run_name}")

    surveys = load_factuality_scores(run_dir)
    if not surveys:
        st.info("No score files found.")
        return

    alignscore_enabled = any(s.get("alignscore_enabled", True) for s in surveys)

    # ── Summary table ──────────────────────────────────────────────────────────
    st.subheader("Summary")

    summary_rows = []
    for s in surveys:
        counts = s.get("category_counts", {})
        row = {
            "survey_id":   s.get("survey_id", "?"),
            "query":       (s.get("query") or "")[:60],
            "n_claims":    s.get("n_claims"),
            "overall":     s.get("cit_correct_overall"),
            "A":           s.get("cit_correct_A"),
            "B":           s.get("cit_correct_B"),
            "C":           s.get("cit_correct_C"),
            "D":           s.get("cit_correct_D"),
            "nA":          (counts.get("A") or {}).get("n", 0),
            "nB":          (counts.get("B") or {}).get("n", 0),
            "nC":          (counts.get("C") or {}).get("n", 0),
            "nD":          (counts.get("D") or {}).get("n", 0),
        }
        summary_rows.append(row)

    sum_df = pd.DataFrame(summary_rows).set_index("survey_id")
    if not alignscore_enabled:
        sum_df = sum_df.drop(columns=["overall", "A", "B", "C", "D"], errors="ignore")
    numeric_cols = sum_df.select_dtypes("number").columns
    st.dataframe(
        sum_df.style.format("{:.3f}", subset=[c for c in ["overall", "A", "B", "C", "D"] if c in numeric_cols], na_rep="—"),
        width="stretch",
    )

    st.divider()

    # ── Detail view ────────────────────────────────────────────────────────────
    st.subheader("Detail")

    reset_idx_on_run_change(run_name, "fact_idx", "_fact_run")
    idx    = nav_arrows("fact_idx", len(surveys))
    survey = surveys[idx]

    query = survey.get("query") or f"Survey {survey.get('survey_id', idx)}"
    st.markdown(f"**{query}**  ·  ID `{survey.get('survey_id', '?')}`")

    # ── Chips ──────────────────────────────────────────────────────────────────
    n_claims = survey.get("n_claims", 0)
    counts   = survey.get("category_counts", {})

    chip_cols = st.columns(9)
    chip_cols[0].metric("Claims", n_claims)
    chip_cols[1].metric("nA", (counts.get("A") or {}).get("n", 0))
    chip_cols[2].metric("nB", (counts.get("B") or {}).get("n", 0))
    chip_cols[3].metric("nC", (counts.get("C") or {}).get("n", 0))
    chip_cols[4].metric("nD", (counts.get("D") or {}).get("n", 0))
    if survey.get("alignscore_enabled", True):
        def _cc(k):
            v = survey.get(f"cit_correct_{k}")
            return f"{v:.3f}" if v is not None else "—"
        chip_cols[5].metric("CitCorr overall", _cc("overall"))
        chip_cols[6].metric("CitCorr A", _cc("A"))
        chip_cols[7].metric("CitCorr B", _cc("B"))
        chip_cols[8].metric("CitCorr C/D", f"{_cc('C')} / {_cc('D')}")
    else:
        chip_cols[5].metric("AlignScore", "disabled")

    st.divider()

    # ── CitCorrect radar ──────────────────────────────────────────────────────
    if survey.get("alignscore_enabled", True):
        radar_labels = [label for _, label in _FACTUALITY_RADAR_METRICS]
        radar_values = [survey.get(key) for key, _ in _FACTUALITY_RADAR_METRICS]
        _render_radar_chart(
            "CitCorrect profile",
            radar_labels,
            [("current survey", radar_values, "#2196F3")],
            height=330,
        )

    # ── Category distribution chart ────────────────────────────────────────────
    categories = ["A", "B", "C", "D"]
    n_vals     = [(counts.get(k) or {}).get("n", 0) for k in categories]
    sup_vals   = [(counts.get(k) or {}).get("n_supported") or 0 for k in categories]

    fig = go.Figure()
    if survey.get("alignscore_enabled", True):
        unsup = [n - s for n, s in zip(n_vals, sup_vals)]
        fig.add_trace(go.Bar(
            name="Supported", x=categories, y=sup_vals,
            marker_color=[_CAT_COLORS[k] for k in categories], opacity=0.9,
        ))
        fig.add_trace(go.Bar(
            name="Unsupported", x=categories, y=unsup,
            marker_color=[_CAT_COLORS[k] for k in categories], opacity=0.35,
        ))
        fig.update_layout(barmode="stack")
    else:
        fig.add_trace(go.Bar(
            x=categories, y=n_vals,
            marker_color=[_CAT_COLORS[k] for k in categories],
            showlegend=False,
        ))

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(tickvals=categories, ticktext=[_CAT_LABELS[k] for k in categories]),
        yaxis_title="# claims",
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig, width="stretch")

    # ── Claims by category ─────────────────────────────────────────────────────
    st.markdown("#### Claims")
    claims = survey.get("claims", [])
    if not claims:
        st.info("No claims recorded.")
        return

    tabs = st.tabs([_CAT_LABELS[k] for k in categories])
    for tab, cat in zip(tabs, categories):
        with tab:
            cat_claims = [c for c in claims if c.get("category") == cat]
            if not cat_claims:
                st.caption("No claims in this category.")
                continue

            rows = []
            for c in cat_claims:
                sup = c.get("supported")
                rows.append({
                    "claim":      c.get("claim", ""),
                    "confidence": f"{_CONF_ICON.get(c.get('confidence', ''), '')} {c.get('confidence', '—')}",
                    "supported":  "✓" if sup is True else ("✗" if sup is False else "—"),
                    "error":      "⚠️" if c.get("error") else "",
                })
            cdf = pd.DataFrame(rows)
            st.dataframe(cdf, width="stretch", hide_index=True)


# ── Page: Diversity view (inside Evaluations) ─────────────────────────────────

_MODEL_COLORS = [
    "#2196F3", "#E91E63", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#FF5722", "#8BC34A",
]


def page_evaluations_diversity(run_dir: pathlib.Path, run_name: str) -> None:
    """Display diversity evaluation page: citation diversity, distribution shift, and PCA visualization."""
    # Derive dataset prefix from run name
    dataset_prefix = run_name.split("_")[0]
    current_model  = re.sub(r"_diversity.*$", "", run_name[len(dataset_prefix) + 1:])

    scores = load_diversity_scores(run_dir)
    if not scores:
        st.info("No diversity score files found.")
        return

    # ── Summary table ──────────────────────────────────────────────────────────
    st.subheader("Summary")

    METRIC_COLS = ["citation_diversity", "distribution_shift",
                   "n_arxiv_refs", "n_reference_cites", "self_cited"]
    rows = []
    for s in scores:
        row = {"id": s["id"], "query": s.get("query", "")[:60]}
        for col in METRIC_COLS:
            row[col] = s.get(col)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("id")
    numeric_cols = df.select_dtypes("number").columns
    st.dataframe(
        df.style.format("{:.4f}", subset=[c for c in numeric_cols
                                          if c not in ("n_arxiv_refs", "n_reference_cites")],
                        na_rep="—")
               .format("{:.0f}",  subset=[c for c in numeric_cols
                                          if c in ("n_arxiv_refs", "n_reference_cites")],
                        na_rep="—"),
        width="stretch",
    )

    if len(scores) > 1:
        agg_cols = ["citation_diversity", "distribution_shift"]
        c1, c2 = st.columns(2)
        for col, container in zip(agg_cols, [c1, c2]):
            vals = [s[col] for s in scores if s.get(col) is not None]
            if vals:
                container.metric(
                    f"avg {col}",
                    f"{sum(vals)/len(vals):.4f}",
                    help=f"min {min(vals):.4f}  ·  max {max(vals):.4f}",
                )

    # ── Per-survey detail ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Detail")
    reset_idx_on_run_change(run_name, "div_idx", "_div_run")
    idx = nav_arrows("div_idx", len(scores))
    s   = scores[idx]

    st.markdown(f"**{s.get('query', '—')}**  ·  ID `{s['id']}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Citation diversity",   f"{s.get('citation_diversity', '—'):.4f}" if s.get('citation_diversity') is not None else "—")
    c2.metric("Distribution shift",   f"{s.get('distribution_shift', '—'):.4f}" if s.get('distribution_shift') is not None else "—")
    c3.metric("arxiv refs (gen)",      str(s.get("n_arxiv_refs", "—")))
    c4.metric("Reference cites",       str(s.get("n_reference_cites", "—")))


# ── PCA chart helper ──────────────────────────────────────────────────────────

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor="rgba(200,200,200,0.3)",
    gridwidth=1,
    zeroline=True,
    zerolinecolor="rgba(150,150,150,0.5)",
    zerolinewidth=1.5,
    showline=True,
    linecolor="rgba(80,80,80,0.8)",
    linewidth=2,
    mirror=True,
    ticks="outside",
    ticklen=5,
    tickcolor="rgba(80,80,80,0.6)",
)


def _render_pca_chart(
    dataset_prefix: str,
    all_diversity_runs: dict,
    survey_options: list[tuple],   # [(id, label), ...]
    default_models: list[str] | None = None,
    key_prefix: str = "pca",
) -> None:
    """Render model-multiselect + survey-selectbox + PCA scatter plot."""
    all_model_ids = sorted(all_diversity_runs.keys())
    if not all_model_ids:
        st.info("No diversity runs found for this dataset.")
        return

    col_sel, col_sur = st.columns([3, 2])
    with col_sel:
        selected_models = st.multiselect(
            "Compare models",
            options=all_model_ids,
            default=default_models if default_models else all_model_ids[:1],
            key=f"{key_prefix}_models",
        )
    with col_sur:
        survey_label = st.selectbox(
            "Survey",
            options=[o[1] for o in survey_options],
            key=f"{key_prefix}_survey",
        )
        survey_id = next(o[0] for o in survey_options if o[1] == survey_label)

    if not selected_models:
        st.info("Select at least one model above.")
        return

    fig = go.Figure()
    ref_plotted = False
    pca_var: list = []
    var_label = ""

    for i, model_id in enumerate(selected_models):
        model_run_dir = all_diversity_runs.get(model_id)
        if model_run_dir is None:
            continue
        dots = load_dots(model_run_dir, survey_id)
        if dots is None:
            st.warning(f"No PCA dots for model `{model_id}` survey `{survey_id}`.")
            continue

        points    = dots.get("points", [])
        color     = _MODEL_COLORS[i % len(_MODEL_COLORS)]
        pca_var   = dots.get("pca_variance", [])
        var_label = (f"PC1 {pca_var[0]*100:.1f}%  ·  PC2 {pca_var[1]*100:.1f}%"
                     if len(pca_var) >= 2 else "")

        ref_pts = [p for p in points if p["group"] == "ref"]
        gen_pts = [p for p in points if p["group"] != "ref"]

        # Reference — grey, plotted once (same for all models)
        if ref_pts and not ref_plotted:
            not_cited  = [p for p in ref_pts if not p.get("self_cite")]
            self_cited = [p for p in ref_pts if p.get("self_cite")]
            if not_cited:
                fig.add_trace(go.Scatter(
                    x=[p["x"] for p in not_cited],
                    y=[p["y"] for p in not_cited],
                    mode="markers",
                    marker=dict(color="rgba(180,180,180,0.5)", size=12, symbol="square",
                                line=dict(color="rgba(110,110,110,0.7)", width=1.2)),
                    name="reference",
                    legendgroup="ref",
                    hovertemplate="ref<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>",
                ))
            if self_cited:
                fig.add_trace(go.Scatter(
                    x=[p["x"] for p in self_cited],
                    y=[p["y"] for p in self_cited],
                    mode="markers",
                    marker=dict(color="rgba(80,80,80,0.8)", size=12, symbol="square",
                                line=dict(color="rgba(40,40,40,0.9)", width=1.5)),
                    name="reference (self-cited)",
                    legendgroup="ref_self",
                    hovertemplate="ref (self-cited)<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>",
                ))
            ref_plotted = True

        # Generated citations for this model
        if gen_pts:
            fig.add_trace(go.Scatter(
                x=[p["x"] for p in gen_pts],
                y=[p["y"] for p in gen_pts],
                mode="markers",
                marker=dict(color=color, size=12, opacity=0.82,
                            line=dict(color="white", width=1)),
                name=model_id,
                legendgroup=model_id,
                hovertemplate=f"{model_id}<br>x=%{{x:.2f}}, y=%{{y:.2f}}<extra></extra>",
            ))

        # Annotate metrics + centroid
        model_scores_file = model_run_dir / f"{survey_id}.json"
        ann = model_id
        if model_scores_file.exists():
            try:
                ms = load_json(model_scores_file)
                cd = ms.get("citation_diversity")
                ds = ms.get("distribution_shift")
                ann = f"{model_id}<br>CD={cd:.3f}  DS={ds:.3f}" if cd is not None else model_id
            except Exception:
                logger.debug(f"Failed to load model scores from {model_scores_file}", exc_info=True)

        if gen_pts:
            cx = sum(p["x"] for p in gen_pts) / len(gen_pts)
            cy = sum(p["y"] for p in gen_pts) / len(gen_pts)
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers+text",
                marker=dict(color=color, size=12, symbol="star", opacity=1.0,
                            line=dict(color="white", width=1.5)),
                text=[model_id],
                textposition="top center",
                textfont=dict(size=11, color=color),
                name=f"{model_id} centroid",
                legendgroup=model_id,
                showlegend=False,
                hovertemplate=f"{ann}<extra></extra>",
            ))

    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(180,180,180,0.6)", borderwidth=1,
            font=dict(size=12),
        ),
        xaxis=dict(
            title=dict(
                text="PC1" + (f" ({pca_var[0]*100:.1f}% variance)" if len(pca_var) >= 2 else ""),
                font=dict(size=13),
            ),
            **_AXIS_STYLE,
        ),
        yaxis=dict(
            title=dict(
                text="PC2" + (f" ({pca_var[1]*100:.1f}% variance)" if len(pca_var) >= 2 else ""),
                font=dict(size=13),
            ),
            **_AXIS_STYLE,
        ),
        hoverlabel=dict(bgcolor="white", bordercolor="rgba(100,100,100,0.4)", font_size=12),
    )
    st.plotly_chart(fig, width="stretch")
    if var_label:
        st.caption(f"PCA fitted on reference citations · {var_label}")


# ── Page: Aggregated Metrics ───────────────────────────────────────────────────

def page_aggregated_metrics() -> None:
    """Display aggregated metrics page: PCA citation space + expert-metric distributions."""
    st.header("Aggregated Metrics")

    datasets = _get_datasets()
    if not datasets:
        st.warning("No datasets found.")
        return

    dataset = st.sidebar.selectbox("Dataset", datasets, key="agg_dataset")
    st.sidebar.divider()

    # ── Citation space (PCA) ─────────────────────────────────────────────────
    all_diversity_runs = find_diversity_runs_for_dataset(dataset)
    survey_ids         = _survey_ids_for_dataset(dataset)
    if all_diversity_runs and survey_ids:
        st.subheader("Citation space (PCA)")
        _render_pca_chart(
            dataset_prefix=dataset,
            all_diversity_runs=all_diversity_runs,
            survey_options=[(sid, sid) for sid in survey_ids],
            default_models=sorted(all_diversity_runs.keys()),
            key_prefix="agg_pca",
        )
    elif not all_diversity_runs:
        st.caption(f"No diversity runs found for dataset **{dataset}**.")
    else:
        st.caption("No surveys found for this dataset.")

    st.divider()

    # ── Expert metrics — per-model distributions ─────────────────────────────
    expert_profile = _render_expert_distributions(dataset)

    st.divider()

    # ── Factuality metrics — per-model distributions ─────────────────────────
    factuality_profile = _render_factuality_distributions(dataset)

    if expert_profile and factuality_profile:
        st.divider()
        _render_aggregate_expert_factuality_profile(
            expert_profile,
            factuality_profile,
        )

    st.divider()

    # ── Structural contradiction metrics — per-model distributions ───────────
    _render_structural_contradiction_distributions(dataset)


# ── Expert distributions renderer ─────────────────────────────────────────────


# Same five metrics the comparison page reports, in the same display order.
# Values are per-survey scalars in [0, 1] except m_mod which is Shannon
# entropy in [0, log2(5)] ≈ [0, 2.32].
_EXPERT_SCALAR_METRICS: list[tuple[str, str]] = [
    ("m_crit",        "Critical (m_crit)"),
    ("m_comp_total",  "Comparative (m_comp_total)"),
    ("m_comp_valid",  "Valid comparisons (m_comp_valid)"),
    ("m_open",        "Open questions (m_open)"),
    ("m_mod",         "Modality entropy (m_mod)"),
]

_EXPERT_RADAR_METRICS: list[tuple[str, str]] = [
    ("m_crit",        "Critical"),
    ("m_comp_total",  "Comparative"),
    ("m_comp_valid",  "Valid comparisons"),
    ("m_open",        "Open questions"),
    ("m_mod",         "Modality entropy"),
]

# Colors for modality categories 1..5. Reds → categorical (assertive),
# blues → explicit uncertainty. Visual continuity helps eyeball skew.
_MODALITY_COLORS = {
    "1": "#b10026",
    "2": "#e31a1c",
    "3": "#fd8d3c",
    "4": "#fecc5c",
    "5": "#2b8cbe",
}
_MODALITY_LABELS = {
    "1": "1 · categorical",
    "2": "2",
    "3": "3 · hedged",
    "4": "4",
    "5": "5 · explicit uncertainty",
}


def _render_expert_distributions(dataset: str) -> dict | None:
    """Per-model boxplot + strip overlay for each expert metric, plus a
    stacked-bar of mean modality_dist.

    Design choices (reflect user decisions 2026-04-18):
      * Single judge selectbox at page level — fair cross-model comparison.
      * Multiselect for models (default: all with a run under chosen judge).
      * Box + all-points overlay on scalar metrics; stacked bar for the
        categorical modality distribution.
    """
    st.subheader("Expert metrics — distributions across surveys")

    all_models = _get_models_for_dataset(dataset)
    models_with_expert = [
        m for m in all_models if _list_score_runs(dataset, m, "expert")
    ]
    if not models_with_expert:
        st.caption(f"No expert runs found for dataset **{dataset}**.")
        return

    # Union of judge-suffixes across all (model, expert) runs, newest-by-mtime
    # first. Lets the user pick one variant that applies uniformly.
    all_runs: list[pathlib.Path] = []
    for m in models_with_expert:
        all_runs.extend(_list_score_runs(dataset, m, "expert"))
    all_runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    suffixes: list[str] = []
    seen: set[str] = set()
    for r in all_runs:
        s = _run_suffix(r, dataset, "expert")
        if s not in seen:
            seen.add(s)
            suffixes.append(s)

    if len(suffixes) > 1:
        judge = st.sidebar.selectbox(
            "Expert judge",
            suffixes,
            index=0,  # newest by mtime
            format_func=lambda s: s.lstrip("_") or "(no suffix)",
            key="agg_expert_judge",
            help="Same judge is applied to every model — keeps the comparison fair.",
        )
    else:
        judge = suffixes[0] if suffixes else ""

    # Only models that actually have a run for the chosen judge are comparable.
    available = [
        m for m in models_with_expert
        if _find_run_with_suffix(dataset, m, "expert", judge) is not None
    ]
    if not available:
        st.info("No models have an expert run with the chosen judge.")
        return

    selected = st.multiselect(
        "Models",
        options=available,
        default=available,
        key="agg_expert_models",
    )
    if not selected:
        st.info("Select at least one model to plot.")
        return

    # Load per-survey rows into a long-format frame: (model, survey_id,
    # metric, value). modality_dist is extracted separately because it's
    # five values per survey, not one.
    rows: list[dict] = []
    mod_rows: list[dict] = []
    for model in selected:
        run_dir = _find_run_with_suffix(dataset, model, "expert", judge)
        if run_dir is None:
            continue
        for f in sorted(run_dir.glob("*.json"), key=_numeric_stem_key):
            if f.stem == "summary":
                continue
            try:
                d = load_json(f)
            except Exception:
                logger.debug(f"Failed to load expert file {f}", exc_info=True)
                continue
            sid = str(d.get("survey_id") or d.get("id") or f.stem)
            for metric_key, _ in _EXPERT_SCALAR_METRICS:
                v = d.get(metric_key)
                if isinstance(v, (int, float)):
                    rows.append({
                        "model":     model,
                        "survey_id": sid,
                        "metric":    metric_key,
                        "value":     float(v),
                    })
            mdist = d.get("modality_dist")
            if isinstance(mdist, dict):
                total = sum(int(v) for v in mdist.values() if isinstance(v, (int, float)))
                if total > 0:
                    mod_rows.append({
                        "model":     model,
                        "survey_id": sid,
                        **{str(c): mdist.get(str(c), 0) / total for c in range(1, 6)},
                    })

    if not rows:
        st.info("No expert-metric data loaded for the selected models / judge.")
        return

    df = pd.DataFrame(rows)

    # Scalar metrics — 2 columns × 3 rows. Each plot is a box with all points
    # overlaid (user's choice of 'box + strip overlay' — see answers above).
    st.markdown("#### Scalar metrics")
    cols_per_row = 2
    for i in range(0, len(_EXPERT_SCALAR_METRICS), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j >= len(_EXPERT_SCALAR_METRICS):
                break
            metric, label = _EXPERT_SCALAR_METRICS[i + j]
            sub = df[df["metric"] == metric]
            with col:
                if sub.empty:
                    st.caption(f"**{label}** — no data")
                    continue
                fig = go.Figure()
                for model in selected:
                    mdata = sub[sub["model"] == model]
                    if mdata.empty:
                        continue
                    fig.add_trace(go.Box(
                        y=mdata["value"],
                        name=model,
                        boxpoints="all",       # show every survey as a dot
                        jitter=0.35,
                        pointpos=0,
                        marker=dict(size=4, opacity=0.6),
                        line=dict(width=1),
                        hovertemplate="<b>%{fullData.name}</b><br>value: %{y:.4f}<extra></extra>",
                    ))
                fig.update_layout(
                    title=label,
                    yaxis_title="value",
                    showlegend=False,
                    height=320,
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig, width="stretch")

    # Modality category distribution (mean per model). Stacked bar summing
    # to 1.0 per model — skew-to-1 on the left reads "assertive", heavier
    # tail on the right reads "more uncertainty acknowledged".
    if mod_rows:
        st.markdown("#### Modality category distribution (mean across surveys)")
        mdf = pd.DataFrame(mod_rows)
        means = (
            mdf.groupby("model")[["1", "2", "3", "4", "5"]]
               .mean()
               .reindex(selected)                # keep the user's order
               .dropna(how="all")                # drop models with no dist
        )
        mod_labels = [_MODALITY_LABELS[str(c)] for c in range(1, 6)]
        mod_series = []
        for i, model in enumerate(means.index):
            mod_series.append((
                model,
                [float(means.loc[model, str(c)]) for c in range(1, 6)],
                _RADAR_COLORS[i % len(_RADAR_COLORS)],
            ))
        _render_radar_chart(
            "Average modality mix",
            mod_labels,
            mod_series,
            height=430,
        )

        fig = go.Figure()
        for cat in ("1", "2", "3", "4", "5"):
            fig.add_trace(go.Bar(
                x=means.index,
                y=means[cat],
                name=_MODALITY_LABELS[cat],
                marker_color=_MODALITY_COLORS[cat],
                hovertemplate=f"%{{x}}<br>{_MODALITY_LABELS[cat]}: %{{y:.3f}}<extra></extra>",
            ))
        fig.update_layout(
            barmode="stack",
            yaxis_title="mean share",
            yaxis=dict(range=[0, 1.02]),
            legend_title="modality",
            height=400,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, width="stretch")

    # ── Pairwise delta (M1 − M2) ──────────────────────────────────────────────
    # Collapsed by default — user asked for it under a `расхлоп`. Scoped to
    # the already-``selected`` set so the two selectboxes only show models the
    # user opted into at the top of the page; keeps the flow consistent.
    with st.expander("Pairwise delta (M1 − M2) across surveys", expanded=False):
        if len(selected) < 2:
            st.info("Pick at least 2 models in the multiselect above to compute a pairwise delta.")
        else:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                m1 = st.selectbox(
                    "M1 (minuend)",
                    options=selected,
                    index=0,
                    key="agg_delta_m1",
                )
            with col_m2:
                m2 = st.selectbox(
                    "M2 (subtrahend)",
                    options=selected,
                    index=min(1, len(selected) - 1),
                    key="agg_delta_m2",
                )
            if m1 == m2:
                st.info("Pick two different models.")
            else:
                # Wide frames indexed by (survey_id, metric) — merging on
                # those two keys gives us exactly the intersection of
                # surveys × metrics where BOTH models have a value.
                left  = (df[df["model"] == m1]
                         [["survey_id", "metric", "value"]]
                         .rename(columns={"value": "v1"}))
                right = (df[df["model"] == m2]
                         [["survey_id", "metric", "value"]]
                         .rename(columns={"value": "v2"}))
                merged = left.merge(right, on=["survey_id", "metric"], how="inner")
                merged["delta"] = merged["v1"] - merged["v2"]

                if merged.empty:
                    st.info(
                        "No overlapping (survey_id × metric) pairs between "
                        f"**{m1}** and **{m2}**."
                    )
                else:
                    st.caption(f"**{m1} − {m2}** · paired survey deltas")

                    # One figure per metric, all in one row. Each col gets an
                    # equal share via ``st.columns(N)``; y-axis is per-plot
                    # (metric-local), so magnitudes don't collapse around a
                    # shared scale. Dashed line at Δ = 0 on every plot.
                    non_empty = [
                        (k, lbl) for k, lbl in _EXPERT_SCALAR_METRICS
                        if not merged[merged["metric"] == k].empty
                    ]
                    if non_empty:
                        cols = st.columns(len(non_empty))
                        for col, (metric_key, label) in zip(cols, non_empty):
                            sub = merged[merged["metric"] == metric_key]
                            with col:
                                fig = go.Figure()
                                fig.add_trace(go.Box(
                                    y=sub["delta"],
                                    name=label,
                                    boxpoints="all",
                                    jitter=0.35,
                                    pointpos=0,
                                    marker=dict(size=5, opacity=0.65),
                                    line=dict(width=1),
                                    hovertemplate=(
                                        "Δ: %{y:.4f}<br>"
                                        "survey: %{customdata}<extra></extra>"
                                    ),
                                    customdata=sub["survey_id"],
                                ))
                                fig.add_hline(
                                    y=0, line_dash="dash", line_color="gray",
                                )
                                fig.update_layout(
                                    title=dict(text=label, font=dict(size=13)),
                                    yaxis_title="Δ",
                                    showlegend=False,
                                    height=380,
                                    margin=dict(l=40, r=10, t=40, b=20),
                                    xaxis=dict(showticklabels=False),
                                )
                                st.plotly_chart(fig, width="stretch")

                    # Full-quantile summary: count, mean, std, min,
                    # 10/25/50/75/90%, max, plus win-rate. Gives a richer
                    # shape than mean/median alone — useful for spotting
                    # heavy tails or one-sided skew.
                    summary_rows = []
                    for metric_key, label in _EXPERT_SCALAR_METRICS:
                        sub = merged[merged["metric"] == metric_key]
                        if sub.empty:
                            continue
                        d = sub["delta"]
                        summary_rows.append({
                            "metric":      label,
                            "n":           len(d),
                            "mean Δ":      d.mean(),
                            "std":         d.std(ddof=0),
                            "min":         d.min(),
                            "q10":         d.quantile(0.10),
                            "q25":         d.quantile(0.25),
                            "median":      d.median(),
                            "q75":         d.quantile(0.75),
                            "q90":         d.quantile(0.90),
                            "max":         d.max(),
                            "M1 > M2 (%)": 100.0 * (d > 0).mean(),
                        })
                    if summary_rows:
                        sum_df = pd.DataFrame(summary_rows).set_index("metric")
                        st.dataframe(
                            sum_df.style.format({
                                "n":           "{:.0f}",
                                "mean Δ":      "{:+.4f}",
                                "std":         "{:.4f}",
                                "min":         "{:+.4f}",
                                "q10":         "{:+.4f}",
                                "q25":         "{:+.4f}",
                                "median":      "{:+.4f}",
                                "q75":         "{:+.4f}",
                                "q90":         "{:+.4f}",
                                "max":         "{:+.4f}",
                                "M1 > M2 (%)": "{:.0f}%",
                            }),
                            width="stretch",
                        )

    return {
        "selected": selected,
        "df": df,
    }


# ── Factuality distributions renderer ─────────────────────────────────────────

# Per-category "supported-share" (CitCorrect_k), plus overall. Values are in
# [0, 1]; per-survey scalars live in the factuality result JSON alongside
# `category_counts` (used for the distribution bar below).
_FACTUALITY_SCALAR_METRICS: list[tuple[str, str]] = [
    ("cit_correct_overall", "Overall (cit_correct)"),
    ("cit_correct_A",       "A — topical"),
    ("cit_correct_B",       "B — methodological"),
    ("cit_correct_C",       "C — quantitative"),
    ("cit_correct_D",       "D — critical / comparative"),
]

# Colors for category shares in the distribution bar. Warmer for general
# (abstract) content, cooler for methods/discussion — helps eyeball skew.
_FACTUALITY_CATEGORY_COLORS = {
    "A": "#fecc5c",   # topical — amber
    "B": "#41b6c4",   # methodological — teal
    "C": "#7fcdbb",   # quantitative — sea
    "D": "#b10026",   # critical/comparative — red
}
_FACTUALITY_CATEGORY_LABELS = {
    "A": "A · topical",
    "B": "B · methodological",
    "C": "C · quantitative",
    "D": "D · critical",
}

_FACTUALITY_RADAR_METRICS: list[tuple[str, str]] = [
    ("cit_correct_overall", "Overall"),
    ("cit_correct_A",       "A"),
    ("cit_correct_B",       "B"),
    ("cit_correct_C",       "C"),
    ("cit_correct_D",       "D"),
]

_RADAR_COLORS = [
    "#2196F3", "#E91E63", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#FF5722", "#8BC34A",
]


def _claim_scope_refs(claim: dict) -> list[int]:
    refs = claim.get("scope_citations")
    if not isinstance(refs, list):
        return []
    out: list[int] = []
    for ref in refs:
        try:
            out.append(int(ref))
        except (TypeError, ValueError):
            continue
    return out


def _claim_has_full_text_for_scope(claim: dict, full_text_ref_ids: set[int]) -> bool:
    refs = _claim_scope_refs(claim)
    return bool(refs) and all(ref in full_text_ref_ids for ref in refs)


def _cit_correct_profile_from_claims(claims: list[dict]) -> dict[str, float | None]:
    def _rate(subset: list[dict]) -> float | None:
        if not subset:
            return None
        n_supported = sum(1 for claim in subset if claim.get("supported") is True)
        return n_supported / len(subset)

    profile = {
        "cit_correct_overall": _rate(claims),
    }
    for cat in ("A", "B", "C", "D"):
        subset = [claim for claim in claims if claim.get("category") == cat]
        profile[f"cit_correct_{cat}"] = _rate(subset)
    return profile


def _render_radar_chart(
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float | None], str]],
    *,
    height: int = 360,
    axis_maxima: list[float] | None = None,
    show_axis_maxima: bool = False,
) -> None:
    """Render a compact Plotly radar chart.

    If ``axis_maxima`` is passed, each axis is normalized independently:
    displayed r = value / axis_max. This keeps small-scale expert metrics
    readable while preserving raw values in hover text.
    """
    fig = go.Figure()
    if axis_maxima is not None and len(axis_maxima) != len(labels):
        axis_maxima = None

    axis_labels = labels
    if axis_maxima is not None and show_axis_maxima:
        axis_labels = [
            f"{label}<br>max {max_v:.3g}"
            for label, max_v in zip(labels, axis_maxima)
        ]
    theta = axis_labels + [axis_labels[0]]

    for name, values, color in series:
        if not any(v is not None for v in values):
            continue
        if axis_maxima is None:
            closed = [
                float(v) if isinstance(v, (int, float)) else None
                for v in values + [values[0]]
            ]
            customdata = None
            hovertemplate = "<b>%{fullData.name}</b><br>%{theta}: %{r:.3f}<extra></extra>"
        else:
            scaled = []
            raw = []
            for v, max_v in zip(values, axis_maxima):
                raw_v = float(v) if isinstance(v, (int, float)) else None
                axis_max = float(max_v) if isinstance(max_v, (int, float)) and max_v > 0 else 1.0
                scaled.append(raw_v / axis_max if raw_v is not None else None)
                raw.append([raw_v, axis_max])
            closed = scaled + [scaled[0]]
            customdata = raw + [raw[0]]
            hovertemplate = (
                "<b>%{fullData.name}</b><br>%{theta}<br>"
                "value: %{customdata[0]:.3f}<br>"
                "axis max: %{customdata[1]:.3f}<br>"
                "scaled: %{r:.3f}<extra></extra>"
            )
        fig.add_trace(go.Scatterpolar(
            r=closed,
            theta=theta,
            name=name,
            fill="toself",
            opacity=0.72,
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            connectgaps=True,
            customdata=customdata,
            hovertemplate=hovertemplate,
        ))

    if not fig.data:
        st.caption(f"**{title}** — no radar data")
        return

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=30, r=30, t=50, b=30),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".1f",
                gridcolor="rgba(150,150,150,0.25)",
            ),
            angularaxis=dict(gridcolor="rgba(150,150,150,0.25)"),
        ),
        legend=dict(orientation="h", y=-0.12),
    )
    st.plotly_chart(fig, width="stretch")


def _radar_axis_maxima(
    series: list[tuple[str, list[float | None], str]],
    width: int,
) -> list[float]:
    maxima: list[float] = []
    for idx in range(width):
        vals = [
            float(values[idx])
            for _, values, _ in series
            if idx < len(values) and isinstance(values[idx], (int, float))
        ]
        max_v = max(vals) if vals else 1.0
        maxima.append(max_v if max_v > 0 else 1.0)
    return maxima


def _render_expert_factuality_radar(
    title: str,
    series: list[tuple[str, list[float | None], str]],
    *,
    height: int,
) -> None:
    expert_labels = [f"Expert · {label}" for _, label in _EXPERT_RADAR_METRICS]
    fact_labels = [f"CitCorrect · {label}" for _, label in _FACTUALITY_RADAR_METRICS]
    expert_width = len(_EXPERT_RADAR_METRICS)
    labels = expert_labels + fact_labels
    axis_maxima = (
        _radar_axis_maxima(series, expert_width)
        + [1.0] * len(_FACTUALITY_RADAR_METRICS)
    )
    _render_radar_chart(
        title,
        labels,
        series,
        height=height,
        axis_maxima=axis_maxima,
        show_axis_maxima=True,
    )


def _mean_metric(df: pd.DataFrame, model: str, metric_key: str) -> float | None:
    sub = df[(df["model"] == model) & (df["metric"] == metric_key)]
    return float(sub["value"].mean()) if not sub.empty else None


def _render_aggregate_expert_factuality_profile(
    expert_profile: dict,
    factuality_profile: dict,
) -> None:
    expert_df = expert_profile["df"]
    factuality_df = factuality_profile["df"]
    factuality_models = set(factuality_profile["selected"])
    selected = [
        model for model in expert_profile["selected"]
        if model in factuality_models
    ]
    if not selected:
        st.caption("No overlapping models for combined expert + factuality profile.")
        return

    series = []
    for i, model in enumerate(selected):
        values = (
            [_mean_metric(expert_df, model, key) for key, _ in _EXPERT_RADAR_METRICS]
            + [_mean_metric(factuality_df, model, key) for key, _ in _FACTUALITY_RADAR_METRICS]
        )
        if any(v is not None for v in values):
            series.append((model, values, _RADAR_COLORS[i % len(_RADAR_COLORS)]))

    if not series:
        st.caption("No data for combined expert + factuality profile.")
        return

    st.subheader("Expert + CitCorrect profile")
    st.caption(
        "Expert axes are scaled by the maximum among selected models; "
        "CitCorrect axes use max 1.0."
    )
    _render_expert_factuality_radar(
        "Average across surveys",
        series,
        height=540,
    )


def _render_factuality_distributions(dataset: str) -> dict | None:
    """Per-model distributions for factuality metric.

    Mirrors :func:`_render_expert_distributions` but targets
    ``CitCorrect_k`` scalars and per-category claim-share distribution:

      * **Scalar metrics** (boxplots): overall citation correctness plus
        the four per-category rates (A / B / C / D). Each plot shows every
        survey as a point so skew and outliers are visible.
      * **Category distribution** (stacked bar): share of each category in
        each model's claim pool, averaged across surveys.
      * **Pairwise delta** (expander): M1 − M2 across paired surveys for
        each scalar metric, with a full-quantile summary table.
    """
    st.subheader("Factuality metrics — distributions across surveys")

    all_models = _get_models_for_dataset(dataset)
    models_with_fact = [
        m for m in all_models if _list_score_runs(dataset, m, "factuality")
    ]
    if not models_with_fact:
        st.caption(f"No factuality runs found for dataset **{dataset}**.")
        return

    # Gather all (model × factuality) runs, pull out suffixes. Factuality
    # suffixes carry the full variant (judge + comment + scope + src + agg),
    # so the selector is the natural way to pin one combination across
    # models for a fair comparison.
    all_runs: list[pathlib.Path] = []
    for m in models_with_fact:
        all_runs.extend(_list_score_runs(dataset, m, "factuality"))
    all_runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    suffixes: list[str] = []
    seen: set[str] = set()
    for r in all_runs:
        s = _run_suffix(r, dataset, "factuality")
        if s not in seen:
            seen.add(s)
            suffixes.append(s)

    if len(suffixes) > 1:
        variant = st.sidebar.selectbox(
            "Factuality variant",
            suffixes,
            index=0,
            format_func=lambda s: s.lstrip("_") or "(no suffix)",
            key="agg_factuality_variant",
            help=(
                "Suffix after `_factuality` in the run folder. Encodes judge, "
                "comment, claim_scope, evidence_source, and aggregation — same "
                "variant is applied across all models for consistency."
            ),
        )
    else:
        variant = suffixes[0] if suffixes else ""

    available = [
        m for m in models_with_fact
        if _find_run_with_suffix(dataset, m, "factuality", variant) is not None
    ]
    if not available:
        st.info("No models have a factuality run with the chosen variant.")
        return

    selected = st.multiselect(
        "Models",
        options=available,
        default=available,
        key="agg_factuality_models",
    )
    if not selected:
        st.info("Select at least one model to plot.")
        return

    # Long-format scalar rows + wide-format category-distribution rows.
    rows:           list[dict] = []
    cat_rows:       list[dict] = []
    full_text_rows: list[dict] = []
    for model in selected:
        run_dir = _find_run_with_suffix(dataset, model, "factuality", variant)
        if run_dir is None:
            continue
        for f in sorted(run_dir.glob("*.json"), key=_numeric_stem_key):
            if f.stem == "summary":
                continue
            try:
                d = load_json(f)
            except Exception:
                logger.debug(f"Failed to load factuality file {f}", exc_info=True)
                continue
            sid = str(d.get("survey_id") or d.get("id") or f.stem)

            for metric_key, _ in _FACTUALITY_SCALAR_METRICS:
                v = d.get(metric_key)
                if isinstance(v, (int, float)):
                    rows.append({
                        "model":     model,
                        "survey_id": sid,
                        "metric":    metric_key,
                        "value":     float(v),
                    })

            full_text_ref_ids = _load_full_text_ref_ids(dataset, model, sid)
            claims = d.get("claims")
            if full_text_ref_ids and isinstance(claims, list):
                full_text_claims = [
                    claim for claim in claims
                    if isinstance(claim, dict)
                    and _claim_has_full_text_for_scope(claim, full_text_ref_ids)
                ]
                profile = _cit_correct_profile_from_claims(full_text_claims)
                for metric_key, _ in _FACTUALITY_RADAR_METRICS:
                    v = profile.get(metric_key)
                    if isinstance(v, (int, float)):
                        full_text_rows.append({
                            "model":     model,
                            "survey_id": sid,
                            "metric":    metric_key,
                            "value":     float(v),
                            "n_claims":  len(full_text_claims),
                        })

            # category_counts shape:
            #   {"A": {"n": int, "n_supported": int}, "B": ..., "C": ..., "D": ...}
            cc = d.get("category_counts")
            if isinstance(cc, dict):
                total = sum(
                    int(sub.get("n", 0))
                    for sub in cc.values() if isinstance(sub, dict)
                )
                if total > 0:
                    cat_rows.append({
                        "model":     model,
                        "survey_id": sid,
                        **{
                            c: (cc.get(c, {}).get("n", 0) / total
                                if isinstance(cc.get(c), dict) else 0.0)
                            for c in ("A", "B", "C", "D")
                        },
                    })

    if not rows:
        st.info("No factuality-metric data loaded for the selected models / variant.")
        return

    df = pd.DataFrame(rows)

    # ── Scalar metrics — 5 plots in a 2-col grid (same layout as expert) ──
    st.markdown("#### Scalar metrics (CitCorrect_k)")
    cols_per_row = 2
    for i in range(0, len(_FACTUALITY_SCALAR_METRICS), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j >= len(_FACTUALITY_SCALAR_METRICS):
                break
            metric, label = _FACTUALITY_SCALAR_METRICS[i + j]
            sub = df[df["metric"] == metric]
            with col:
                if sub.empty:
                    st.caption(f"**{label}** — no data")
                    continue
                fig = go.Figure()
                for model in selected:
                    mdata = sub[sub["model"] == model]
                    if mdata.empty:
                        continue
                    fig.add_trace(go.Box(
                        y=mdata["value"],
                        name=model,
                        boxpoints="all",
                        jitter=0.35,
                        pointpos=0,
                        marker=dict(size=4, opacity=0.6),
                        line=dict(width=1),
                        hovertemplate="<b>%{fullData.name}</b><br>value: %{y:.4f}<extra></extra>",
                    ))
                fig.update_layout(
                    title=label,
                    yaxis_title="value",
                    yaxis=dict(range=[0, 1.02]),
                    showlegend=False,
                    height=320,
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig, width="stretch")

    # ── Full-text-covered scope profile — extra spider chart ─────────────
    # Keeps only claims where every citation in `scope_citations` maps to a
    # source ref with cached full text. This is viewer-side recomputation
    # from per-claim support labels; original score files stay unchanged.
    if full_text_rows:
        st.markdown("#### CitCorrect profile (full-text-covered scopes only)")
        ftdf = pd.DataFrame(full_text_rows)
        radar_labels = [label for _, label in _FACTUALITY_RADAR_METRICS]
        radar_series = []
        kept_parts = []
        for i, model in enumerate(selected):
            mdf = ftdf[ftdf["model"] == model]
            if mdf.empty:
                continue
            values = []
            for metric_key, _ in _FACTUALITY_RADAR_METRICS:
                sub = mdf[mdf["metric"] == metric_key]
                values.append(float(sub["value"].mean()) if not sub.empty else None)
            radar_series.append((model, values, _RADAR_COLORS[i % len(_RADAR_COLORS)]))

            per_survey_n = (
                mdf[["survey_id", "n_claims"]]
                .drop_duplicates()
                .set_index("survey_id")["n_claims"]
            )
            if not per_survey_n.empty:
                kept_parts.append(
                    f"{model}: {per_survey_n.mean():.1f} claims/survey"
                )

        if radar_series:
            _render_radar_chart(
                "Average across surveys",
                radar_labels,
                radar_series,
                height=430,
            )
            if kept_parts:
                st.caption("Filtered subset size — " + "  ·  ".join(kept_parts))

    # ── Category distribution (mean per model) — stacked bar ──────────────
    # Shows how claim categories are split across a model's surveys. A
    # skew toward A means the model mostly cites abstract-level stuff; a
    # heavy D share means more critical / comparative claims.
    if cat_rows:
        st.markdown("#### Category distribution (mean across surveys)")
        cdf = pd.DataFrame(cat_rows)
        means = (
            cdf.groupby("model")[["A", "B", "C", "D"]]
               .mean()
               .reindex(selected)
               .dropna(how="all")
        )
        cat_labels = [_FACTUALITY_CATEGORY_LABELS[c] for c in ("A", "B", "C", "D")]
        cat_series = []
        for i, model in enumerate(means.index):
            cat_series.append((
                model,
                [float(means.loc[model, c]) for c in ("A", "B", "C", "D")],
                _RADAR_COLORS[i % len(_RADAR_COLORS)],
            ))
        _render_radar_chart(
            "Average claim-category mix",
            cat_labels,
            cat_series,
            height=430,
        )

        fig = go.Figure()
        for cat in ("A", "B", "C", "D"):
            fig.add_trace(go.Bar(
                x=means.index,
                y=means[cat],
                name=_FACTUALITY_CATEGORY_LABELS[cat],
                marker_color=_FACTUALITY_CATEGORY_COLORS[cat],
                hovertemplate=(
                    f"%{{x}}<br>{_FACTUALITY_CATEGORY_LABELS[cat]}: "
                    "%{y:.3f}<extra></extra>"
                ),
            ))
        fig.update_layout(
            barmode="stack",
            yaxis_title="mean share",
            yaxis=dict(range=[0, 1.02]),
            legend_title="category",
            height=400,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, width="stretch")

    # ── Pairwise delta (M1 − M2) ──────────────────────────────────────────
    with st.expander("Pairwise delta (M1 − M2) across surveys", expanded=False):
        if len(selected) < 2:
            st.info(
                "Pick at least 2 models in the multiselect above to compute "
                "a pairwise delta."
            )
        else:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                m1 = st.selectbox(
                    "M1 (minuend)",
                    options=selected,
                    index=0,
                    key="agg_fact_delta_m1",
                )
            with col_m2:
                m2 = st.selectbox(
                    "M2 (subtrahend)",
                    options=selected,
                    index=min(1, len(selected) - 1),
                    key="agg_fact_delta_m2",
                )
            if m1 == m2:
                st.info("Pick two different models.")
            else:
                left  = (df[df["model"] == m1]
                         [["survey_id", "metric", "value"]]
                         .rename(columns={"value": "v1"}))
                right = (df[df["model"] == m2]
                         [["survey_id", "metric", "value"]]
                         .rename(columns={"value": "v2"}))
                merged = left.merge(right, on=["survey_id", "metric"], how="inner")
                merged["delta"] = merged["v1"] - merged["v2"]

                if merged.empty:
                    st.info(
                        "No overlapping (survey_id × metric) pairs between "
                        f"**{m1}** and **{m2}**."
                    )
                else:
                    st.caption(f"**{m1} − {m2}** · paired survey deltas")

                    non_empty = [
                        (k, lbl) for k, lbl in _FACTUALITY_SCALAR_METRICS
                        if not merged[merged["metric"] == k].empty
                    ]
                    if non_empty:
                        cols = st.columns(len(non_empty))
                        for col, (metric_key, label) in zip(cols, non_empty):
                            sub = merged[merged["metric"] == metric_key]
                            with col:
                                fig = go.Figure()
                                fig.add_trace(go.Box(
                                    y=sub["delta"],
                                    name=label,
                                    boxpoints="all",
                                    jitter=0.35,
                                    pointpos=0,
                                    marker=dict(size=5, opacity=0.65),
                                    line=dict(width=1),
                                    hovertemplate=(
                                        "Δ: %{y:.4f}<br>"
                                        "survey: %{customdata}<extra></extra>"
                                    ),
                                    customdata=sub["survey_id"],
                                ))
                                fig.add_hline(
                                    y=0, line_dash="dash", line_color="gray",
                                )
                                fig.update_layout(
                                    title=dict(text=label, font=dict(size=13)),
                                    yaxis_title="Δ",
                                    showlegend=False,
                                    height=380,
                                    margin=dict(l=40, r=10, t=40, b=20),
                                    xaxis=dict(showticklabels=False),
                                )
                                st.plotly_chart(fig, width="stretch")

                    summary_rows = []
                    for metric_key, label in _FACTUALITY_SCALAR_METRICS:
                        sub = merged[merged["metric"] == metric_key]
                        if sub.empty:
                            continue
                        d = sub["delta"]
                        summary_rows.append({
                            "metric":      label,
                            "n":           len(d),
                            "mean Δ":      d.mean(),
                            "std":         d.std(ddof=0),
                            "min":         d.min(),
                            "q10":         d.quantile(0.10),
                            "q25":         d.quantile(0.25),
                            "median":      d.median(),
                            "q75":         d.quantile(0.75),
                            "q90":         d.quantile(0.90),
                            "max":         d.max(),
                            "M1 > M2 (%)": 100.0 * (d > 0).mean(),
                        })
                    if summary_rows:
                        sum_df = pd.DataFrame(summary_rows).set_index("metric")
                        st.dataframe(
                            sum_df.style.format({
                                "n":           "{:.0f}",
                                "mean Δ":      "{:+.4f}",
                                "std":         "{:.4f}",
                                "min":         "{:+.4f}",
                                "q10":         "{:+.4f}",
                                "q25":         "{:+.4f}",
                                "median":      "{:+.4f}",
                                "q75":         "{:+.4f}",
                                "q90":         "{:+.4f}",
                                "max":         "{:+.4f}",
                                "M1 > M2 (%)": "{:.0f}%",
                            }),
                            width="stretch",
                        )

    return {
        "selected": selected,
        "df": df,
    }


# ── Structural contradiction distributions renderer ──────────────────────────

_STRUCTURAL_CONTR_METRICS: list[tuple[str, str]] = [
    ("contr_m_contr",                 "M_contr"),
    ("contr_n_contradictions",        "# contradictions"),
    ("contr_n_after_topic_filter",    "# after topic filter"),
    ("contr_n_candidates_stage1",     "# SPECTER candidates"),
    ("contr_n_failed",                "# failed checks"),
]


def _render_structural_contradiction_distributions(dataset: str) -> None:
    st.subheader("Structural contradiction — distributions across surveys")

    all_models = _get_models_for_dataset(dataset)
    models_with_structural = [
        m for m in all_models if _list_score_runs(dataset, m, "structural")
    ]
    if not models_with_structural:
        st.caption(f"No structural runs found for dataset **{dataset}**.")
        return

    all_runs: list[pathlib.Path] = []
    for model in models_with_structural:
        all_runs.extend(_list_score_runs(dataset, model, "structural"))
    all_runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    suffixes: list[str] = []
    seen: set[str] = set()
    for run_dir in all_runs:
        suffix = _run_suffix(run_dir, dataset, "structural")
        if suffix not in seen:
            seen.add(suffix)
            suffixes.append(suffix)

    if len(suffixes) > 1:
        variant = st.sidebar.selectbox(
            "Structural variant",
            suffixes,
            index=0,
            format_func=lambda s: s.lstrip("_") or "(no suffix)",
            key="agg_structural_variant",
            help="Same structural run variant is applied across all models.",
        )
    else:
        variant = suffixes[0] if suffixes else ""

    available = [
        model for model in models_with_structural
        if _find_run_with_suffix(dataset, model, "structural", variant) is not None
    ]
    if not available:
        st.info("No models have a structural run with the chosen variant.")
        return

    selected = st.multiselect(
        "Models",
        options=available,
        default=available,
        key="agg_structural_models",
    )
    if not selected:
        st.info("Select at least one model to plot.")
        return

    rows: list[dict] = []
    for model in selected:
        run_dir = _find_run_with_suffix(dataset, model, "structural", variant)
        if run_dir is None:
            continue
        for f in sorted(run_dir.glob("*.json"), key=_numeric_stem_key):
            if f.stem in {"summary", "aggregate"}:
                continue
            try:
                d = load_json(f)
            except Exception:
                logger.debug(f"Failed to load structural file {f}", exc_info=True)
                continue
            sid = str(d.get("survey_id") or d.get("id") or f.stem)
            for metric_key, _ in _STRUCTURAL_CONTR_METRICS:
                v = d.get(metric_key)
                if isinstance(v, (int, float)):
                    rows.append({
                        "model":     model,
                        "survey_id": sid,
                        "metric":    metric_key,
                        "value":     float(v),
                    })

    if not rows:
        st.info("No contradiction data loaded for the selected models / variant.")
        return

    df = pd.DataFrame(rows)

    st.markdown("#### Contradiction metrics")
    cols_per_row = 2
    for i in range(0, len(_STRUCTURAL_CONTR_METRICS), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j >= len(_STRUCTURAL_CONTR_METRICS):
                break
            metric, label = _STRUCTURAL_CONTR_METRICS[i + j]
            sub = df[df["metric"] == metric]
            with col:
                if sub.empty:
                    st.caption(f"**{label}** — no data")
                    continue
                fig = go.Figure()
                for model in selected:
                    mdata = sub[sub["model"] == model]
                    if mdata.empty:
                        continue
                    fig.add_trace(go.Box(
                        y=mdata["value"],
                        name=model,
                        boxpoints="all",
                        jitter=0.35,
                        pointpos=0,
                        marker=dict(size=4, opacity=0.6),
                        line=dict(width=1),
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "value: %{y:.4f}<br>"
                            "survey: %{customdata}<extra></extra>"
                        ),
                        customdata=mdata["survey_id"],
                    ))
                fig.update_layout(
                    title=label,
                    yaxis_title="value",
                    showlegend=False,
                    height=320,
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig, width="stretch")

    with st.expander("Pairwise delta (M1 − M2) across surveys", expanded=False):
        if len(selected) < 2:
            st.info("Pick at least 2 models in the multiselect above to compute a pairwise delta.")
        else:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                m1 = st.selectbox(
                    "M1 (minuend)",
                    options=selected,
                    index=0,
                    key="agg_struct_delta_m1",
                )
            with col_m2:
                m2 = st.selectbox(
                    "M2 (subtrahend)",
                    options=selected,
                    index=min(1, len(selected) - 1),
                    key="agg_struct_delta_m2",
                )
            if m1 == m2:
                st.info("Pick two different models.")
            else:
                left = (
                    df[df["model"] == m1]
                    [["survey_id", "metric", "value"]]
                    .rename(columns={"value": "v1"})
                )
                right = (
                    df[df["model"] == m2]
                    [["survey_id", "metric", "value"]]
                    .rename(columns={"value": "v2"})
                )
                merged = left.merge(right, on=["survey_id", "metric"], how="inner")
                merged["delta"] = merged["v1"] - merged["v2"]

                if merged.empty:
                    st.info(
                        "No overlapping (survey_id × metric) pairs between "
                        f"**{m1}** and **{m2}**."
                    )
                else:
                    st.caption(f"**{m1} − {m2}** · paired survey deltas")
                    non_empty = [
                        (key, label) for key, label in _STRUCTURAL_CONTR_METRICS
                        if not merged[merged["metric"] == key].empty
                    ]
                    if non_empty:
                        cols = st.columns(len(non_empty))
                        for col, (metric_key, label) in zip(cols, non_empty):
                            sub = merged[merged["metric"] == metric_key]
                            with col:
                                fig = go.Figure()
                                fig.add_trace(go.Box(
                                    y=sub["delta"],
                                    name=label,
                                    boxpoints="all",
                                    jitter=0.35,
                                    pointpos=0,
                                    marker=dict(size=5, opacity=0.65),
                                    line=dict(width=1),
                                    hovertemplate=(
                                        "Δ: %{y:.4f}<br>"
                                        "survey: %{customdata}<extra></extra>"
                                    ),
                                    customdata=sub["survey_id"],
                                ))
                                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                fig.update_layout(
                                    title=dict(text=label, font=dict(size=13)),
                                    yaxis_title="Δ",
                                    showlegend=False,
                                    height=380,
                                    margin=dict(l=40, r=10, t=40, b=20),
                                    xaxis=dict(showticklabels=False),
                                )
                                st.plotly_chart(fig, width="stretch")

                    summary_rows = []
                    for metric_key, label in _STRUCTURAL_CONTR_METRICS:
                        sub = merged[merged["metric"] == metric_key]
                        if sub.empty:
                            continue
                        d = sub["delta"]
                        summary_rows.append({
                            "metric":       label,
                            "n":            len(d),
                            "mean Δ":       d.mean(),
                            "std":          d.std(ddof=0),
                            "min":          d.min(),
                            "q10":          d.quantile(0.10),
                            "q25":          d.quantile(0.25),
                            "median":       d.median(),
                            "q75":          d.quantile(0.75),
                            "q90":          d.quantile(0.90),
                            "max":          d.max(),
                            "M1 lower (%)": 100.0 * (d < 0).mean(),
                        })
                    if summary_rows:
                        sum_df = pd.DataFrame(summary_rows).set_index("metric")
                        st.dataframe(
                            sum_df.style.format({
                                "n":            "{:.0f}",
                                "mean Δ":       "{:+.4f}",
                                "std":          "{:.4f}",
                                "min":          "{:+.4f}",
                                "q10":          "{:+.4f}",
                                "q25":          "{:+.4f}",
                                "median":       "{:+.4f}",
                                "q75":          "{:+.4f}",
                                "q90":          "{:+.4f}",
                                "max":          "{:+.4f}",
                                "M1 lower (%)": "{:.0f}%",
                            }),
                            width="stretch",
                        )


# ── Comparison helpers ─────────────────────────────────────────────────────────

def _get_datasets() -> list[str]:
    """Unique dataset prefixes from SCORES_DIR and GENERATIONS_DIR."""
    prefixes: set[str] = set()
    for base in (SCORES_DIR, GENERATIONS_DIR):
        if base.exists():
            for d in base.iterdir():
                if d.is_dir():
                    prefixes.add(d.name.split("_")[0])
    return sorted(prefixes)


def _get_models_for_dataset(dataset: str) -> list[str]:
    """All model IDs that have any score or generation run for this dataset.

    Deduplicates case-insensitively: if both ``SurveyForge`` and
    ``surveyforge`` directories exist on disk, they collapse into one entry.
    Canonical capitalization is chosen in this order:
      1. The form that appears under ``results/generations/`` (that's what
         ``BaseModel`` writes with; usually matches the ``MODEL=...`` CLI arg).
      2. First-seen form under ``results/scores/`` as a fallback.
    """
    canonical: dict[str, str] = {}  # lower_key -> preferred original form
    prefix = dataset + "_"

    def _extract_model(d: pathlib.Path) -> str | None:
        if not d.is_dir() or not d.name.startswith(prefix):
            return None
        tail = d.name[len(prefix):]
        m = re.sub(r"_(diversity|claims|expert|factuality|structural|surge).*$", "", tail)
        return m or None

    # Pass 1: generation dirs produce the canonical form (MODEL= arg spelling).
    if GENERATIONS_DIR.exists():
        for d in GENERATIONS_DIR.iterdir():
            model = _extract_model(d)
            if model:
                canonical.setdefault(model.lower(), model)

    # Pass 2: score dirs fill in models that have no generation dir yet.
    if SCORES_DIR.exists():
        for d in SCORES_DIR.iterdir():
            model = _extract_model(d)
            if model:
                canonical.setdefault(model.lower(), model)

    return sorted(canonical.values(), key=str.lower)


def _list_score_runs(dataset: str, model: str, kind: str) -> list[pathlib.Path]:
    """All score directories matching (dataset, model, kind), newest first.

    Matching is **case-insensitive** on dataset and model so a dir called
    ``SurGE_surveyforge_expert_run1`` is picked up when the UI passes
    ``model="SurveyForge"``. Sorted by filesystem ``mtime`` descending —
    the first element is what most callers want as "the latest run" (the
    previous lexical ``sorted()[-1]`` was a bug: ``qwen3`` wins over the
    newer ``gemma-4-31b`` alphabetically).
    """
    if not SCORES_DIR.exists():
        return []
    prefix_lower = f"{dataset}_{model}_{kind}".lower()
    matches = [
        d for d in SCORES_DIR.iterdir()
        if d.is_dir() and d.name.lower().startswith(prefix_lower)
    ]
    matches.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return matches


def _find_score_run(dataset: str, model: str, kind: str) -> pathlib.Path | None:
    """Most-recent score directory for (dataset, model, kind). Case-insensitive.

    Thin wrapper over :func:`_list_score_runs` for callers that only need
    one directory and are happy with "newest by mtime".
    """
    runs = _list_score_runs(dataset, model, kind)
    return runs[0] if runs else None


def _run_suffix(run_dir: pathlib.Path, dataset: str, kind: str) -> str:
    """Extract the judge/comment suffix after the ``..._{kind}`` portion.

    For ``SurGE_SurveyForge_expert_gemma-4-31b_run1`` with kind=``expert``
    returns ``"_gemma-4-31b_run1"``. For a bare ``SurGE_SurveyForge_expert``
    returns ``""``. The match is case-insensitive; returned suffix is
    taken from the dir name verbatim (preserves the on-disk capitalization
    so it round-trips with :func:`_find_run_with_suffix`).
    """
    pattern = re.compile(
        rf"^{re.escape(dataset)}_.+?_{re.escape(kind)}(.*)$",
        re.IGNORECASE,
    )
    m = pattern.match(run_dir.name)
    return m.group(1) if m else ""


def _find_run_with_suffix(
    dataset: str, model: str, kind: str, suffix: str,
) -> pathlib.Path | None:
    """Locate the score dir with the exact given suffix (case-insensitive).

    Used by the comparison page after the user picks a judge variant — we
    need a specific run (e.g. ``_gemma-4-31b_run1``), not just the newest.
    """
    if not SCORES_DIR.exists():
        return None
    target_lower = f"{dataset}_{model}_{kind}{suffix or ''}".lower()
    for d in SCORES_DIR.iterdir():
        if d.is_dir() and d.name.lower() == target_lower:
            return d
    return None


def _survey_ids_for_dataset(dataset: str) -> list[str]:
    """
    Canonical survey ID list from the first available generation dir
    or any score dir for this dataset.

    Ordering uses :func:`_numeric_stem_key` so sparse numeric ids (e.g.
    SurGE_reference = 0, 1, 2, 3, 5, 6, 8, 10, ...) navigate naturally on
    the comparison page; plain ``sorted()`` would give 0, 1, 10, 11, ..., 2.
    """
    if GENERATIONS_DIR.exists():
        for d in sorted(GENERATIONS_DIR.iterdir()):
            if d.is_dir() and d.name.startswith(dataset + "_"):
                files = sorted(d.glob("*.json"), key=_numeric_stem_key)
                ids = [f.stem for f in files]
                if ids:
                    return ids
    if SCORES_DIR.exists():
        for d in sorted(SCORES_DIR.iterdir()):
            if d.is_dir() and d.name.startswith(dataset + "_"):
                files = sorted(
                    (f for f in d.glob("*.json") if f.stem != "summary"),
                    key=_numeric_stem_key,
                )
                ids = [f.stem for f in files]
                if ids:
                    return ids
    return []


def _load_survey_json(run_dir: pathlib.Path | None, survey_id: str) -> dict | None:
    """Load a survey JSON file from a run directory, returning None if not found or on parse error."""
    if run_dir is None:
        return None
    p = run_dir / f"{survey_id}.json"
    if not p.exists():
        return None
    try:
        return load_json(p)
    except Exception:
        logger.debug(f"Failed to load survey JSON {p}", exc_info=True)
        return None


def _query_for_survey(dataset: str, survey_id: str) -> str:
    """Best-effort: get survey query from any generation or score file."""
    if GENERATIONS_DIR.exists():
        for d in GENERATIONS_DIR.iterdir():
            if d.is_dir() and d.name.startswith(dataset + "_"):
                p = d / f"{survey_id}.json"
                if p.exists():
                    try:
                        return load_json(p).get("query", "")
                    except Exception:
                        logger.debug(f"Failed to load query from {p}", exc_info=True)
    return ""


def _fmt(val, decimals: int = 4) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _delta_str(a, b) -> str:
    if a is None or b is None:
        return "—"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = b - a
        return f"{d:+.4f}" if isinstance(a, float) else f"{d:+d}"
    return "—"


def _comparison_df(
    rows: list[tuple[str, str, object, object]]
) -> pd.DataFrame:
    """
    Build a styled comparison DataFrame.
    rows = [(display_name, key, val_a, val_b), ...]
    """
    data = []
    for name, _, a, b in rows:
        data.append({
            "Metric":  name,
            "Model A": _fmt(a),
            "Model B": _fmt(b),
            "Δ (B−A)": _delta_str(a, b),
        })
    return pd.DataFrame(data).set_index("Metric")


# ── Page: Comparison ───────────────────────────────────────────────────────────

_DIVERSITY_METRICS = [
    ("citation_diversity",  "Citation diversity"),
    ("distribution_shift",  "Distribution shift"),
    ("n_arxiv_refs",        "arXiv refs (gen)"),
    ("n_reference_cites",   "Reference cites"),
    ("self_cited",          "Self-cited"),
]

_EXPERT_METRICS = [
    ("n_claims",      "# claims"),
    ("m_crit",        "Critical (m_crit)"),
    ("m_comp_total",  "Comparative (m_comp_total)"),
    ("m_comp_valid",  "Valid comparisons (m_comp_valid)"),
    ("m_open",        "Open questions (m_open)"),
    ("m_mod",         "Modality entropy (m_mod)"),
]

_FACTUALITY_METRICS = [
    ("n_claims",            "# claims"),
    ("cit_correct_overall", "CitCorrect overall"),
    ("cit_correct_A",       "CitCorrect A — General"),
    ("cit_correct_B",       "CitCorrect B — Methodological"),
    ("cit_correct_C",       "CitCorrect C — Quantitative"),
    ("cit_correct_D",       "CitCorrect D — Critical"),
]


def page_comparison() -> None:
    """Display model comparison page: side-by-side diversity and expert evaluation metrics."""
    st.header("Evaluation: Model Comparison")

    datasets = _get_datasets()
    if not datasets:
        st.warning("No datasets found.")
        return

    # ── Sidebar controls ───────────────────────────────────────────────────────
    dataset = st.sidebar.selectbox("Dataset", datasets)
    models  = _get_models_for_dataset(dataset)
    if not models:
        st.sidebar.warning("No models found for this dataset.")
        return

    default_b = models[1] if len(models) > 1 else models[0]
    model_a = st.sidebar.selectbox("Model A", models, index=0)
    model_b = st.sidebar.selectbox("Model B", models,
                                   index=models.index(default_b))
    st.sidebar.divider()

    # ── Judge/run-variant picker per metric ────────────────────────────────────
    # When multiple runs exist for (model, kind) — e.g. expert scored by
    # gemma/qwen/gpt-oss/llama — let the user pick which one to compare.
    # Default = the most recent by mtime from the UNION across model A + B
    # (not lexical; the old ``sorted()[-1]`` logic picked ``qwen3`` over a
    # newer ``gemma-4-31b`` run). Same suffix is applied to both models —
    # if only one side has that judge, the other shows as empty.
    def _variants_for(kind: str) -> list[str]:
        runs = (
            _list_score_runs(dataset, model_a, kind)
            + _list_score_runs(dataset, model_b, kind)
        )
        # Re-sort the union by mtime desc (each half is already sorted but
        # concatenation breaks the order).
        runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        out: list[str] = []
        seen: set[str] = set()
        for r in runs:
            s = _run_suffix(r, dataset, kind)
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    div_variants = _variants_for("diversity")
    exp_variants = _variants_for("expert")
    fac_variants = _variants_for("factuality")

    def _pick_variant(label: str, key: str, variants: list[str]) -> str | None:
        """Show a selectbox in the sidebar when >1 judge variant exists."""
        if not variants:
            return None
        if len(variants) == 1:
            return variants[0]
        return st.sidebar.selectbox(
            label,
            variants,
            index=0,  # newest-by-mtime
            format_func=lambda s: s.lstrip("_") or "(no suffix)",
            key=key,
            help="Multiple judge/run variants exist on disk. Newest by mtime is default.",
        )

    if exp_variants or fac_variants or (div_variants and len(div_variants) > 1):
        st.sidebar.markdown("**Judge / run variant**")
    div_pick = _pick_variant("Diversity run",     "cmp_div_variant", div_variants)
    exp_pick = _pick_variant("Expert judge",      "cmp_exp_variant", exp_variants)
    fac_pick = _pick_variant("Factuality judge",  "cmp_fac_variant", fac_variants)
    st.sidebar.divider()

    # ── Load run dirs for chosen variants ──────────────────────────────────────
    div_a = _find_run_with_suffix(dataset, model_a, "diversity",  div_pick)  if div_pick is not None else None
    div_b = _find_run_with_suffix(dataset, model_b, "diversity",  div_pick)  if div_pick is not None else None
    exp_a = _find_run_with_suffix(dataset, model_a, "expert",     exp_pick)  if exp_pick is not None else None
    exp_b = _find_run_with_suffix(dataset, model_b, "expert",     exp_pick)  if exp_pick is not None else None
    fac_a = _find_run_with_suffix(dataset, model_a, "factuality", fac_pick)  if fac_pick is not None else None
    fac_b = _find_run_with_suffix(dataset, model_b, "factuality", fac_pick)  if fac_pick is not None else None

    have_diversity  = div_a is not None or div_b is not None
    have_expert     = exp_a is not None or exp_b is not None
    have_factuality = fac_a is not None or fac_b is not None

    if not have_diversity and not have_expert and not have_factuality:
        st.info("No diversity, expert, or factuality scores found for these models.")
        return

    # ── Survey navigation ──────────────────────────────────────────────────────
    survey_ids = _survey_ids_for_dataset(dataset)
    if not survey_ids:
        st.info("No surveys found.")
        return

    cmp_state = f"cmp_{dataset}_{model_a}_{model_b}"
    reset_idx_on_run_change(cmp_state, "cmp_idx", "_cmp_run")
    idx        = nav_arrows("cmp_idx", len(survey_ids))
    survey_id  = survey_ids[idx]

    query = _query_for_survey(dataset, survey_id)
    st.subheader(query or f"Survey {survey_id}")

    col_a, col_b = st.columns(2)
    col_a.caption(f"🔵 Model A: **{model_a}**")
    col_b.caption(f"🔴 Model B: **{model_b}**")

    st.divider()

    # Small helper: show which on-disk directory is actually being read for
    # each side so there's no "why is B empty?" surprise when only one side
    # has that judge variant.
    def _source_caption(da_: pathlib.Path | None, db_: pathlib.Path | None) -> str:
        a = da_.name if da_ else "—"
        b = db_.name if db_ else "—"
        return f"<small>🔵 {a}  ·  🔴 {b}</small>"

    ea = _load_survey_json(exp_a, survey_id) if have_expert else None
    eb = _load_survey_json(exp_b, survey_id) if have_expert else None
    fa = _load_survey_json(fac_a, survey_id) if have_factuality else None
    fb = _load_survey_json(fac_b, survey_id) if have_factuality else None

    if (ea or eb) and (fa or fb):
        st.markdown("### Expert + Factuality")
        st.markdown(
            _source_caption(exp_a, exp_b) + "<br>" + _source_caption(fac_a, fac_b),
            unsafe_allow_html=True,
        )
        combined_series = []
        if ea or fa:
            combined_series.append((
                model_a,
                [ea.get(key) if ea else None for key, _ in _EXPERT_RADAR_METRICS]
                + [fa.get(key) if fa else None for key, _ in _FACTUALITY_RADAR_METRICS],
                "#2196F3",
            ))
        if eb or fb:
            combined_series.append((
                model_b,
                [eb.get(key) if eb else None for key, _ in _EXPERT_RADAR_METRICS]
                + [fb.get(key) if fb else None for key, _ in _FACTUALITY_RADAR_METRICS],
                "#E91E63",
            ))
        st.caption(
            "Expert axes are scaled by the maximum among the visible models; "
            "CitCorrect axes use max 1.0."
        )
        _render_expert_factuality_radar(
            "Expert + CitCorrect profile",
            combined_series,
            height=420,
        )
        st.divider()

    # ── Diversity table ────────────────────────────────────────────────────────
    if have_diversity:
        st.markdown("### Diversity")
        st.markdown(_source_caption(div_a, div_b), unsafe_allow_html=True)
        da = _load_survey_json(div_a, survey_id)
        db = _load_survey_json(div_b, survey_id)

        if da is None and db is None:
            st.info("No diversity data for this survey.")
        else:
            rows = [
                (label, key, da.get(key) if da else None, db.get(key) if db else None)
                for key, label in _DIVERSITY_METRICS
            ]
            st.dataframe(_comparison_df(rows), width="stretch")

    # ── Expert table ───────────────────────────────────────────────────────────
    if have_expert:
        st.markdown("### Expert")
        st.markdown(_source_caption(exp_a, exp_b), unsafe_allow_html=True)

        if ea is None and eb is None:
            st.info("No expert data for this survey.")
        else:
            rows = [
                (label, key, ea.get(key) if ea else None, eb.get(key) if eb else None)
                for key, label in _EXPERT_METRICS
            ]
            st.dataframe(_comparison_df(rows), width="stretch")

            # Modality distribution side-by-side
            mod_a = ea.get("modality_dist") if ea else None
            mod_b = eb.get("modality_dist") if eb else None
            if mod_a or mod_b:
                st.markdown("**Modality distribution**")
                levels = ["1", "2", "3", "4", "5"]
                mod_labels = [_MODALITY_LABELS[l] for l in levels]
                mod_radar_series = []

                def _norm_modality(mdist):
                    if not isinstance(mdist, dict):
                        return None
                    total = sum(
                        float(mdist.get(l, 0))
                        for l in levels
                        if isinstance(mdist.get(l, 0), (int, float))
                    )
                    if total <= 0:
                        return None
                    return [float(mdist.get(l, 0)) / total for l in levels]

                mod_vals_a = _norm_modality(mod_a)
                mod_vals_b = _norm_modality(mod_b)
                if mod_vals_a:
                    mod_radar_series.append((model_a, mod_vals_a, "#2196F3"))
                if mod_vals_b:
                    mod_radar_series.append((model_b, mod_vals_b, "#E91E63"))
                if mod_radar_series:
                    _render_radar_chart(
                        "Modality category mix",
                        mod_labels,
                        mod_radar_series,
                        height=340,
                    )

                fig = go.Figure()
                if mod_a:
                    fig.add_trace(go.Bar(
                        name=model_a,
                        x=levels,
                        y=[mod_a.get(l, 0) for l in levels],
                        marker_color="#2196F3",
                    ))
                if mod_b:
                    fig.add_trace(go.Bar(
                        name=model_b,
                        x=levels,
                        y=[mod_b.get(l, 0) for l in levels],
                        marker_color="#E91E63",
                    ))
                fig.update_layout(
                    barmode="group",
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=30),
                    xaxis_title="Modality level",
                    yaxis_title="# claims",
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(fig, width="stretch")

    # ── Factuality table ───────────────────────────────────────────────────────
    if have_factuality:
        st.markdown("### Factuality")
        st.markdown(_source_caption(fac_a, fac_b), unsafe_allow_html=True)

        if fa is None and fb is None:
            st.info("No factuality data for this survey.")
        else:
            def _cc(d):
                """Extract category_counts dict safely."""
                return (d or {}).get("category_counts", {})

            def _n(d, cat):
                return (_cc(d).get(cat) or {}).get("n")

            def _sup(d, cat):
                return (_cc(d).get(cat) or {}).get("n_supported")

            _CAT_NAMES = [
                ("A", "General topical"),
                ("B", "Methodological"),
                ("C", "Quantitative"),
                ("D", "Critical & comparative"),
            ]

            # ── Table 1: Claim distribution (always) ──────────────────────────
            st.markdown("**Claim distribution**")
            cat_labels = [name for _, name in _CAT_NAMES]
            cat_radar_series = []

            def _claim_category_mix(d):
                total = d.get("n_claims") if d else None
                if not isinstance(total, (int, float)) or total <= 0:
                    return None
                return [
                    ((_cc(d).get(k) or {}).get("n", 0) / total)
                    for k, _ in _CAT_NAMES
                ]

            cat_vals_a = _claim_category_mix(fa)
            cat_vals_b = _claim_category_mix(fb)
            if cat_vals_a:
                cat_radar_series.append((model_a, cat_vals_a, "#2196F3"))
            if cat_vals_b:
                cat_radar_series.append((model_b, cat_vals_b, "#E91E63"))
            if cat_radar_series:
                _render_radar_chart(
                    "Claim-category mix",
                    cat_labels,
                    cat_radar_series,
                    height=340,
                )

            dist_rows = [
                (name, None, _n(fa, k), _n(fb, k))
                for k, name in _CAT_NAMES
            ]
            dist_rows.append((
                "**Total**", None,
                (fa or {}).get("n_claims"),
                (fb or {}).get("n_claims"),
            ))
            st.dataframe(_comparison_df(dist_rows), width="stretch")

            # ── Table 2: Supported claims (only if AlignScore ran) ─────────────
            alignscore_on = (fa or {}).get("alignscore_enabled", True) or \
                            (fb or {}).get("alignscore_enabled", True)
            if alignscore_on:
                st.markdown("**Fraction of factually supported claims**")
                sup_rows = [
                    (name, None,
                     (fa or {}).get(f"cit_correct_{k}"),
                     (fb or {}).get(f"cit_correct_{k}"))
                    for k, name in _CAT_NAMES
                ]
                sup_rows.append((
                    "**Overall**", None,
                    (fa or {}).get("cit_correct_overall"),
                    (fb or {}).get("cit_correct_overall"),
                ))
                st.dataframe(_comparison_df(sup_rows), width="stretch")

                n_sup_a = (fa or {}).get("n_supported")
                n_sup_b = (fb or {}).get("n_supported")
                n_tot_a = (fa or {}).get("n_claims")
                n_tot_b = (fb or {}).get("n_claims")
                parts = []
                if n_sup_a is not None:
                    parts.append(f"{model_a}: **{n_sup_a}** / {n_tot_a}")
                if n_sup_b is not None:
                    parts.append(f"{model_b}: **{n_sup_b}** / {n_tot_b}")
                if parts:
                    st.caption("Supported claims — " + "  ·  ".join(parts))
            else:
                st.caption("AlignScore disabled — citation correctness not available")


# ── Page: HyperOpt ────────────────────────────────────────────────────────────

def _sim_badge(sim: float) -> str:
    """Colored HTML badge for a similarity value (red→green gradient)."""
    # interpolate red(0.5) → yellow(0.7) → green(0.9)
    t = max(0.0, min(1.0, (sim - 0.5) / 0.4))
    if t < 0.5:
        r, g = 239, int(83 + t * 2 * (238 - 83))
        b = 50
    else:
        r, g = int(239 - (t - 0.5) * 2 * (239 - 102)), 195
        b = int(50 + (t - 0.5) * 2 * (58 - 50))
    color = f"rgb({r},{g},{b})"
    return (
        f'<span style="background:{color};color:#111;padding:2px 8px;'
        f'border-radius:4px;font-weight:600;font-size:0.9em">{sim:.3f}</span>'
    )


def _kde(values: list[float], x: "np.ndarray", bandwidth: float = 0.025) -> "np.ndarray":
    """Gaussian KDE via numpy — no scipy/seaborn dependency."""
    import numpy as np
    arr = np.array(values)
    return np.array([
        float(np.sum(np.exp(-0.5 * ((xi - arr) / bandwidth) ** 2))
              / (len(arr) * bandwidth * (2 * 3.14159265358979) ** 0.5))
        for xi in x
    ])


def page_hyperopt_specter_thr() -> None:
    import numpy as np

    path = HYPEROPT_DIR / "specter_thr.json"
    if not path.exists():
        st.info(
            "No data yet. Run the helper script first:\n\n"
            "```bash\n"
            "python scripts/structural_specter_thr.py \\\n"
            "    --gen-dir results/generations/<dataset>_<model> \\\n"
            "    --specter models_cache/sentence-transformers--allenai-specter\n"
            "```"
        )
        return

    try:
        all_pairs: list[dict] = json.loads(path.read_text())
    except Exception as e:
        st.error(f"Failed to load specter_thr.json: {e}")
        return

    if not all_pairs:
        st.warning("specter_thr.json is empty.")
        return

    sims    = [p["similarity"] for p in all_pairs]
    surveys = sorted({p["survey_id"] for p in all_pairs})

    # ── KDE plot (numpy + plotly, no extra deps) ───────────────────────────────
    x    = np.linspace(0.45, 0.95, 300)
    dens = _kde(sims, x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=dens,
        mode="lines",
        line=dict(color="#42a5f5", width=2),
        fill="tozeroy",
        fillcolor="rgba(66,165,245,0.2)",
        name="KDE",
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=40),
        xaxis=dict(title="Cosine similarity", range=[0.45, 0.95]),
        yaxis=dict(title="Density"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        f"**{len(all_pairs)} pairs** · **{len(surveys)} surveys** · "
        f"range {min(sims):.3f} – {max(sims):.3f}"
    )

    st.divider()

    # ── Sorted expandable pair list ────────────────────────────────────────────
    for i, pair in enumerate(all_pairs):  # already sorted ascending by script
        sim    = pair["similarity"]
        sec_i  = pair.get("section_i", "?")
        sec_j  = pair.get("section_j", "?")
        survey = pair.get("survey_id", "—")
        s1     = pair.get("s1", "")
        s2     = pair.get("s2", "")

        label = (
            f"{_sim_badge(sim)} &nbsp;"
            f"<span style='color:#888;font-size:0.82em'>"
            f"{sec_i} → {sec_j} &nbsp;·&nbsp; <code>{survey}</code>"
            f"</span>"
        )

        with st.expander(f"sim={sim:.3f}  ·  {sec_i} → {sec_j}", expanded=False):
            st.markdown(label, unsafe_allow_html=True)
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(
                    f"<div style='background:#1e1e2e;border-radius:6px;padding:10px 14px'>"
                    f"<div style='font-size:0.75em;color:#aaa;margin-bottom:4px'>📂 {sec_i}</div>"
                    f"<div style='font-size:0.93em;line-height:1.5'>{s1}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_r:
                st.markdown(
                    f"<div style='background:#1e2e1e;border-radius:6px;padding:10px 14px'>"
                    f"<div style='font-size:0.75em;color:#aaa;margin-bottom:4px'>📂 {sec_j}</div>"
                    f"<div style='font-size:0.93em;line-height:1.5'>{s2}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


def page_hyperopt() -> None:
    st.header("HyperOpt")

    tools = ["specter_thr"]
    tool  = st.sidebar.selectbox("Tool", tools, key="hp_tool")
    st.sidebar.divider()

    if tool == "specter_thr":
        st.subheader("SPECTER similarity threshold calibration")
        st.markdown(
            "Browse sampled sentence pairs at each similarity bucket to calibrate "
            "`similarity_threshold` in `metrics/structural/config.yaml`."
        )
        page_hyperopt_specter_thr()


# ── Page: Markup ──────────────────────────────────────────────────────────────

def page_markup() -> None:
    st.header("Разметка")

    metric = st.sidebar.selectbox("Metric", ["Structural"], key="markup_metric")
    st.sidebar.divider()

    sample_files = sorted(SAMPLES_DIR.glob("*.json"))
    if not sample_files:
        st.info(
            "No sample files found. Generate one into "
            "`analisis/samples/` from the analysis notebook first."
        )
        return

    sample_file = st.sidebar.selectbox(
        "Sample file",
        sample_files,
        format_func=lambda p: p.name,
        key="markup_sample_file",
    )

    try:
        data = load_json(sample_file)
    except Exception as e:
        st.error(f"Failed to load sample file: {e}")
        return

    samples = _sample_items(data)
    if not samples:
        st.warning("Selected sample file has no samples/items.")
        return

    existing = _load_existing_markup(sample_file)
    options = list(_STRUCTURAL_MARKUP_CLASSES.keys())

    if isinstance(data, dict):
        st.caption(
            f"metric={metric} · source={data.get('source_run', '—')} · "
            f"sample_file={sample_file.name} · n={len(samples)}"
        )
    else:
        st.caption(f"metric={metric} · sample_file={sample_file.name} · n={len(samples)}")

    answers = []
    with st.form("markup_form"):
        for i, sample in enumerate(samples, 1):
            sample_id = str(sample.get("sample_id") or f"{sample_file.stem}:{i}")
            saved = existing.get(sample_id, {})
            saved_class = saved.get("class")
            saved_comment = saved.get("comment") or ""
            default_index = options.index(saved_class) if saved_class in options else None

            st.markdown("---")
            st.markdown(f"### {i}. Survey `{sample.get('survey_id', '—')}`")
            st.caption(
                f"id={sample_id} · type={sample.get('contradiction_type', '—')} · "
                f"similarity={sample.get('similarity', '—')}"
            )
            if sample.get("query"):
                st.markdown(f"**{sample.get('query')}**")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Section 1:** {sample.get('section_1', '—')}")
                _render_sample_text("Statement 1", sample.get("statement_1"))
            with col_b:
                st.markdown(f"**Section 2:** {sample.get('section_2', '—')}")
                _render_sample_text("Statement 2", sample.get("statement_2"))

            if sample.get("reasoning"):
                with st.expander("Judge reasoning", expanded=False):
                    st.markdown(sample["reasoning"])

            with st.expander("Paragraph context", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    _render_context_window(sample, 1)
                with c2:
                    _render_context_window(sample, 2)

            cls = st.radio(
                "Класс",
                options,
                index=default_index,
                format_func=lambda x: _STRUCTURAL_MARKUP_CLASSES[x],
                key=f"markup_class_{sample_file.stem}_{i}",
            )
            comment = st.text_area(
                "Комментарий",
                value=saved_comment,
                key=f"markup_comment_{sample_file.stem}_{i}",
            )
            answers.append({
                "sample_id": sample_id,
                "class": cls,
                "comment": comment.strip(),
                "sample": sample,
            })

        submitted = st.form_submit_button("Submit")

    if not submitted:
        return

    missing = [row["sample_id"] for row in answers if row["class"] is None]
    if missing:
        st.error(
            "Not all samples are marked: " + ", ".join(missing[:5])
            + (" ..." if len(missing) > 5 else "")
        )
        return

    MARKUPS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = MARKUPS_DIR / f"{sample_file.stem}_markup.json"
    payload = {
        "metric": metric,
        "sample_file": str(sample_file.relative_to(ROOT)),
        "source_run": data.get("source_run") if isinstance(data, dict) else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(answers),
        "classes": _STRUCTURAL_MARKUP_CLASSES,
        "markups": answers,
    }
    out_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    st.success(f"Saved markup → {out_file.relative_to(ROOT)}")


# ── Routing ────────────────────────────────────────────────────────────────────

# Reset nav indices when switching pages
def _on_page_change() -> None:
    st.session_state.pop("gen_idx",  None)
    st.session_state.pop("eval_idx", None)
    st.session_state.pop("cmp_idx",  None)
    st.session_state.pop("_gen_run", None)
    st.session_state.pop("_eval_run",None)
    st.session_state.pop("_cmp_run", None)


page = st.sidebar.radio(
    "View",
    [
        "Generations",
        "Evaluations",
        "PointsMetrics",
        "AggregatedMetrics",
        "Разметка",
        "HyperOpt",
    ],
    index=0, on_change=_on_page_change
)
st.sidebar.divider()

if page == "Generations":
    page_generations()
elif page == "PointsMetrics":
    page_comparison()
elif page == "AggregatedMetrics":
    page_aggregated_metrics()
elif page == "Разметка":
    page_markup()
elif page == "HyperOpt":
    page_hyperopt()
else:
    page_evaluations()
