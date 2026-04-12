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


def load_generations(run_dir: pathlib.Path) -> list[dict]:
    """Load generation JSON files from a run directory, skipping files that fail to parse."""
    out = []
    for f in sorted(run_dir.glob("*.json")):
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
    for f in sorted(run_dir.glob("*.json")):
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
    # Routes to diversity view if run name contains _diversity suffix
    if is_diversity_run(selected):
        page_evaluations_diversity(run_dir, selected)
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
    for f in sorted(run_dir.glob("*.json")):
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
    """Display aggregated metrics page: PCA citation space visualization for all models and datasets."""
    st.header("Aggregated Metrics")

    datasets = _get_datasets()
    if not datasets:
        st.warning("No datasets found.")
        return

    dataset = st.sidebar.selectbox("Dataset", datasets, key="agg_dataset")
    st.sidebar.divider()

    all_diversity_runs = find_diversity_runs_for_dataset(dataset)
    if not all_diversity_runs:
        st.info(f"No diversity runs found for dataset **{dataset}**.")
        return

    st.subheader("Citation space (PCA)")

    survey_ids = _survey_ids_for_dataset(dataset)
    if not survey_ids:
        st.info("No surveys found for this dataset.")
        return

    survey_options = [(sid, sid) for sid in survey_ids]
    _render_pca_chart(
        dataset_prefix=dataset,
        all_diversity_runs=all_diversity_runs,
        survey_options=survey_options,
        default_models=sorted(all_diversity_runs.keys()),
        key_prefix="agg_pca",
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
    """All model IDs that have any score or generation run for this dataset."""
    models: set[str] = set()
    prefix = dataset + "_"

    for base in (SCORES_DIR, GENERATIONS_DIR):
        if not base.exists():
            continue
        for d in base.iterdir():
            if not d.is_dir() or not d.name.startswith(prefix):
                continue
            tail = d.name[len(prefix):]
            # Strip known suffixes to get the model_id
            model = re.sub(r"_(diversity|claims|expert|factuality|structural|surge).*$", "", tail)
            if model:
                models.add(model)

    return sorted(models)


def _find_score_run(dataset: str, model: str, kind: str) -> pathlib.Path | None:
    """
    Find the score directory for (dataset, model, kind).
    kind = 'diversity' | 'expert' | 'claims'
    Returns the last alphabetically (latest run) if multiple exist.
    """
    if not SCORES_DIR.exists():
        return None
    prefix = f"{dataset}_{model}_{kind}"
    matches = sorted(
        d for d in SCORES_DIR.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    )
    return matches[-1] if matches else None


def _survey_ids_for_dataset(dataset: str) -> list[str]:
    """
    Canonical survey ID list from the first available generation dir
    or any score dir for this dataset.
    """
    if GENERATIONS_DIR.exists():
        for d in sorted(GENERATIONS_DIR.iterdir()):
            if d.is_dir() and d.name.startswith(dataset + "_"):
                ids = sorted(f.stem for f in d.glob("*.json"))
                if ids:
                    return ids
    if SCORES_DIR.exists():
        for d in sorted(SCORES_DIR.iterdir()):
            if d.is_dir() and d.name.startswith(dataset + "_"):
                ids = sorted(f.stem for f in d.glob("*.json") if f.stem != "summary")
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

    # ── Load run dirs ──────────────────────────────────────────────────────────
    div_a  = _find_score_run(dataset, model_a, "diversity")
    div_b  = _find_score_run(dataset, model_b, "diversity")
    exp_a  = _find_score_run(dataset, model_a, "expert")
    exp_b  = _find_score_run(dataset, model_b, "expert")

    have_diversity = div_a is not None or div_b is not None
    have_expert    = exp_a is not None or exp_b is not None

    if not have_diversity and not have_expert:
        st.info("No diversity or expert scores found for these models.")
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

    # ── Diversity table ────────────────────────────────────────────────────────
    if have_diversity:
        st.markdown("### Diversity")
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
    "View", ["Generations", "Evaluations", "PointsMetrics", "AggregatedMetrics"],
    index=0, on_change=_on_page_change
)
st.sidebar.divider()

if page == "Generations":
    page_generations()
elif page == "PointsMetrics":
    page_comparison()
elif page == "AggregatedMetrics":
    page_aggregated_metrics()
else:
    page_evaluations()
