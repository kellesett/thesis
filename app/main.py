"""
app/main.py  —  Thesis results viewer
Run from the repo root:
    streamlit run app/main.py
"""
import json
import pathlib

import pandas as pd
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
