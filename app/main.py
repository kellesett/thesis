"""
app/main.py  —  Thesis results viewer
Run from the repo root:
    streamlit run app/main.py
"""
import json
import pathlib
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = pathlib.Path(__file__).parent.parent
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
    out = []
    for f in sorted(run_dir.glob("*.json")):
        try:
            out.append(load_json(f))
        except Exception:
            pass
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
    out = []
    for f in sorted(run_dir.glob("*.json")):
        if f.stem == "summary":
            continue
        try:
            out.append(normalize_score(load_json(f)))
        except Exception:
            pass
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
    return run_name.endswith("_diversity") or "_diversity_" in run_name


def load_diversity_scores(run_dir: pathlib.Path) -> list[dict]:
    """Load per-survey diversity score JSONs (skip subdirs like plots/)."""
    out = []
    for f in sorted(run_dir.glob("*.json")):
        try:
            out.append(load_json(f))
        except Exception:
            pass
    return out


def load_dots(run_dir: pathlib.Path, survey_id: str) -> dict | None:
    """Load _dots.json for a given survey_id from the plots/ subdir."""
    p = run_dir / "plots" / f"{survey_id}_dots.json"
    if not p.exists():
        return None
    try:
        return load_json(p)
    except Exception:
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

    st.divider()

    # ── PCA scatter ────────────────────────────────────────────────────────────
    st.subheader("Citation space (PCA)")

    all_diversity_runs = find_diversity_runs_for_dataset(dataset_prefix)
    all_model_ids = sorted(all_diversity_runs.keys())

    col_sel, col_sur = st.columns([3, 2])
    with col_sel:
        selected_models = st.multiselect(
            "Compare models",
            options=all_model_ids,
            default=[current_model] if current_model in all_model_ids else all_model_ids[:1],
        )
    with col_sur:
        survey_options = [(s["id"], s.get("query", f"Survey {s['id']}")[:60]) for s in scores]
        survey_label   = st.selectbox(
            "Survey",
            options=[o[1] for o in survey_options],
        )
        survey_id = next(o[0] for o in survey_options if o[1] == survey_label)

    if not selected_models:
        st.info("Select at least one model above.")
        return

    fig = go.Figure()
    ref_plotted = False

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
            self_cited = [p for p in ref_pts if p.get("self_cite")]
            not_cited  = [p for p in ref_pts if not p.get("self_cite")]

            if not_cited:
                fig.add_trace(go.Scatter(
                    x=[p["x"] for p in not_cited],
                    y=[p["y"] for p in not_cited],
                    mode="markers",
                    marker=dict(
                        color="rgba(180,180,180,0.5)",
                        size=12,
                        symbol="square",
                        line=dict(color="rgba(110,110,110,0.7)", width=1.2),
                    ),
                    name="reference",
                    legendgroup="ref",
                    hovertemplate="ref<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>",
                ))
            if self_cited:
                fig.add_trace(go.Scatter(
                    x=[p["x"] for p in self_cited],
                    y=[p["y"] for p in self_cited],
                    mode="markers",
                    marker=dict(
                        color="rgba(80,80,80,0.8)",
                        size=12,
                        symbol="square",
                        line=dict(color="rgba(40,40,40,0.9)", width=1.5),
                    ),
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
                marker=dict(
                    color=color,
                    size=12,
                    opacity=0.82,
                    line=dict(color="white", width=1),
                ),
                name=model_id,
                legendgroup=model_id,
                hovertemplate=f"{model_id}<br>x=%{{x:.2f}}, y=%{{y:.2f}}<extra></extra>",
            ))

        # Annotate metrics for this model
        model_scores_file = model_run_dir / f"{survey_id}.json"
        if model_scores_file.exists():
            try:
                ms  = load_json(model_scores_file)
                cd  = ms.get("citation_diversity")
                ds  = ms.get("distribution_shift")
                ann = f"{model_id}<br>CD={cd:.3f}  DS={ds:.3f}" if cd is not None else model_id
            except Exception:
                ann = model_id
        else:
            ann = model_id

        # Centroid marker per model — star, same size as points, fully opaque
        if gen_pts:
            cx = sum(p["x"] for p in gen_pts) / len(gen_pts)
            cy = sum(p["y"] for p in gen_pts) / len(gen_pts)
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers+text",
                marker=dict(
                    color=color,
                    size=12,
                    symbol="star",
                    opacity=1.0,
                    line=dict(color="white", width=1.5),
                ),
                text=[model_id],
                textposition="top center",
                textfont=dict(size=11, color=color),
                name=f"{model_id} centroid",
                legendgroup=model_id,
                showlegend=False,
                hovertemplate=f"{ann}<extra></extra>",
            ))

    # pca_var / var_label come from the last successfully loaded dots — may be unset
    _pca_var   = locals().get("pca_var", [])
    _var_label = locals().get("var_label", "")

    _axis_style = dict(
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

    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left",   x=0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(180,180,180,0.6)",
            borderwidth=1,
            font=dict(size=12),
        ),
        xaxis=dict(
            title=dict(
                text="PC1" + (f" ({_pca_var[0]*100:.1f}% variance)" if len(_pca_var) >= 2 else ""),
                font=dict(size=13),
            ),
            **_axis_style,
        ),
        yaxis=dict(
            title=dict(
                text="PC2" + (f" ({_pca_var[1]*100:.1f}% variance)" if len(_pca_var) >= 2 else ""),
                font=dict(size=13),
            ),
            **_axis_style,
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="rgba(100,100,100,0.4)",
            font_size=12,
        ),
    )
    st.plotly_chart(fig, width="stretch")
    if _var_label:
        st.caption(f"PCA fitted on reference citations · {_var_label}")

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


# ── Routing ────────────────────────────────────────────────────────────────────

# Reset nav indices when switching pages
def _on_page_change() -> None:
    st.session_state.pop("gen_idx", None)
    st.session_state.pop("eval_idx", None)
    st.session_state.pop("_gen_run", None)
    st.session_state.pop("_eval_run", None)


page = st.sidebar.radio(
    "View", ["Generations", "Evaluations"], index=0, on_change=_on_page_change
)
st.sidebar.divider()

if page == "Generations":
    page_generations()
else:
    page_evaluations()
