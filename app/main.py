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

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.log_setup import setup_logging

setup_logging("app")

GENERATIONS_DIR = ROOT / "results" / "generations"
SCORES_DIR      = ROOT / "results" / "scores"
HYPEROPT_DIR    = ROOT / "results" / "hyperopt"

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
    _render_expert_distributions(dataset)


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


def _render_expert_distributions(dataset: str) -> None:
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
                st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

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
                                st.plotly_chart(fig, use_container_width=True)

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
        ea = _load_survey_json(exp_a, survey_id)
        eb = _load_survey_json(exp_b, survey_id)

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
        fa = _load_survey_json(fac_a, survey_id)
        fb = _load_survey_json(fac_b, survey_id)

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
    ["Generations", "Evaluations", "PointsMetrics", "AggregatedMetrics", "HyperOpt"],
    index=0, on_change=_on_page_change
)
st.sidebar.divider()

if page == "Generations":
    page_generations()
elif page == "PointsMetrics":
    page_comparison()
elif page == "AggregatedMetrics":
    page_aggregated_metrics()
elif page == "HyperOpt":
    page_hyperopt()
else:
    page_evaluations()
