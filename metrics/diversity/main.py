#!/usr/bin/env python
"""
metrics/diversity/main.py
Citation diversity evaluation for generated surveys.

Computes two metrics per survey:
  citation_diversity    — mean pairwise cosine distance among embeddings of
                          the generated survey's arxiv citations. Measures how
                          broadly the model covers sub-topics (0 = all same
                          topic, 1 = maximally diverse).
  distribution_shift    — cosine distance between the centroid of generated
                          citations and the centroid of reference citations in
                          SPECTER2 embedding space. Measures how far the
                          model's citation cloud drifts from the ground truth
                          topic distribution (0 = identical, 1 = orthogonal).

Additionally saves a PCA scatter plot (PNG) for each survey:
  results/scores/<dataset_id>_<model_id>_diversity/plots/<id>.png

All numerical scores are saved to:
  results/scores/<dataset_id>_<model_id>_diversity/<id>.json
  results/scores/<dataset_id>_<model_id>_diversity/summary.csv

Embeddings:
  Generated citations  — arxiv only (refs with arxiv_id); abstract fetched
                         via arxiv API (title + SEP + abstract → SPECTER2).
  Reference citations  — all_cites doc_ids from SurGE corpus; abstract loaded
                         by streaming corpus.json once at startup.

Usage (inside Docker):
    python metrics/diversity/main.py --dataset SurGE --model perplexity_dr
"""
import argparse
import csv
import json
import logging
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ROOT   = Path(__file__).parent.parent.parent
CONFIG = Path(__file__).parent / "config.yaml"
sys.path.insert(0, str(ROOT))

from src.log_setup import setup_logging
from src.datasets import load_dataset as load_dataset_cls


# ── Config & registry ─────────────────────────────────────────────────────────

def load_registry(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    return {e["id"]: e["path"] for e in reg["datasets"]}


# ── Corpus streaming ──────────────────────────────────────────────────────────

def load_corpus_abstracts(
    corpus_path: Path,
    needed_doc_ids: set,
) -> dict[int, tuple[str, str]]:
    """
    Stream corpus.json and return {doc_id: (title, abstract)} for needed ids.

    Uses ijson for memory-efficient streaming of the 1.6 GB corpus.
    Stops early once all needed ids are found.
    """
    import ijson

    results: dict[int, tuple[str, str]] = {}
    needed = set(needed_doc_ids)
    print(f"  [corpus] streaming {corpus_path.name} for {len(needed)} doc_ids …")

    with open(corpus_path, "rb") as f:
        for obj in ijson.items(f, "item"):
            doc_id = obj.get("doc_id")
            if doc_id in needed:
                results[doc_id] = (
                    obj.get("Title", ""),
                    obj.get("Abstract", ""),
                )
                needed.discard(doc_id)
                if not needed:
                    break  # found everything

    print(f"  [corpus] found {len(results)}/{len(needed_doc_ids)} abstracts")
    return results


# ── arxiv API ─────────────────────────────────────────────────────────────────

def _fetch_arxiv_batch(
    arxiv_ids: list[str],
    max_retries: int,
    retry_delays: list[int],
) -> dict[str, tuple[str, str]]:
    """Single batch request (up to ARXIV_BATCH_SIZE ids) with retry logic."""
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    ids_str = ",".join(arxiv_ids)
    url = (
        f"https://export.arxiv.org/api/query"
        f"?id_list={ids_str}&max_results={len(arxiv_ids)}"
    )

    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "thesis-diversity-metric/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_data = resp.read()

            root = ET.fromstring(xml_data)
            results: dict[str, tuple[str, str]] = {}
            for entry in root.findall("atom:entry", ns):
                id_el    = entry.find("atom:id", ns)
                title_el = entry.find("atom:title", ns)
                abs_el   = entry.find("atom:summary", ns)
                if id_el is None:
                    continue
                raw_id   = id_el.text.strip().split("/")[-1]
                bare_id  = re.sub(r"v\d+$", "", raw_id)
                title    = " ".join(title_el.text.split()) if title_el is not None else ""
                abstract = " ".join(abs_el.text.split())   if abs_el  is not None else ""
                results[bare_id] = (title, abstract)
            return results

        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = retry_delays[min(attempt, len(retry_delays) - 1)]
                print(f"  [arxiv] 429 rate-limit, attempt {attempt+1}/{max_retries+1}, "
                      f"waiting {wait}s …")
                if attempt < max_retries:
                    time.sleep(wait)
                else:
                    print("  [arxiv] giving up after max retries")
                    return {}
            else:
                print(f"  [arxiv] HTTP error {e.code}: {e}")
                return {}
        except urllib.error.URLError as e:
            print(f"  [arxiv] URL error (timeout?): {e}")
            return {}
        except ET.ParseError as e:
            print(f"  [arxiv] XML parse error: {e}")
            return {}

    return {}


ARXIV_BATCH_SIZE = 10  # arxiv API ignores max_results for large id_lists


def fetch_arxiv_metadata(
    arxiv_ids: list[str],
    max_retries: int = 2,
    retry_delays: list[int] = None,
) -> dict[str, tuple[str, str]]:
    """Fetch {arxiv_id: (title, abstract)} from arxiv export API.

    Splits request into batches by ARXIV_BATCH_SIZE because arxiv API
    unreliably handles max_results with large id_list and silently returns
    only 10 results instead of respecting the requested count.
    """
    if retry_delays is None:
        retry_delays = [5, 10]

    if not arxiv_ids:
        return {}

    results: dict[str, tuple[str, str]] = {}
    for i in range(0, len(arxiv_ids), ARXIV_BATCH_SIZE):
        batch = arxiv_ids[i : i + ARXIV_BATCH_SIZE]
        batch_results = _fetch_arxiv_batch(batch, max_retries, retry_delays)
        results.update(batch_results)
        if i + ARXIV_BATCH_SIZE < len(arxiv_ids):
            time.sleep(1)  # courteous delay between batches

    return results


# ── SPECTER2 embedding ────────────────────────────────────────────────────────

def load_specter2(model_name: str, adapter_name: str, device: str):
    """Load SPECTER2 base model + proximity adapter.

    Args:
        model_name: HuggingFace model ID for base SPECTER2 model.
        adapter_name: HuggingFace adapter name for proximity task.
        device: torch device (e.g., "cuda" or "cpu").

    Returns:
        Tuple of (tokenizer, model) ready for embedding text.
    """
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    print(f"  [emb] loading tokenizer: {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"  [emb] loading model …")
    model = AutoAdapterModel.from_pretrained(model_name)

    print(f"  [emb] loading adapter: {adapter_name} …")
    model.load_adapter(adapter_name, source="hf", load_as="specter2", set_active=True)
    model.eval()
    model.to(device)
    print(f"  [emb] SPECTER2 ready on {device}")
    return tokenizer, model


def embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Embed a list of strings with SPECTER2.

    Expected format: "Title [SEP] Abstract"

    Args:
        texts: List of text strings to embed.
        tokenizer: Tokenizer instance from load_specter2.
        model: Model instance from load_specter2.
        device: torch device (e.g., "cuda" or "cpu").
        batch_size: Processing batch size.

    Returns:
        Float32 array of shape (N, 768) with embeddings.
    """
    import torch

    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token embedding
        embs = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()
        all_embs.append(embs)

    return np.vstack(all_embs)


def make_specter2_texts(pairs: list[tuple[str, str]], sep_token: str) -> list[str]:
    """Format (title, abstract) pairs as 'title SEP abstract' for SPECTER2."""
    return [f"{title}{sep_token}{abstract}" for title, abstract in pairs]


# ── Numerical metrics ─────────────────────────────────────────────────────────

def compute_citation_diversity(embeddings: np.ndarray) -> float | None:
    """
    Mean pairwise cosine distance among citation embeddings.
    Requires at least 2 embeddings; returns None otherwise.
    """
    n = len(embeddings)
    if n < 2:
        return None

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_n = embeddings / np.maximum(norms, 1e-8)
    sim   = emb_n @ emb_n.T                    # (N, N) cosine similarities
    idx   = np.triu_indices(n, k=1)            # upper triangle
    dists = 1.0 - sim[idx]                     # cosine distances
    return float(np.mean(dists))


def compute_distribution_shift(
    emb_gen: np.ndarray,
    emb_ref: np.ndarray,
) -> float | None:
    """
    Cosine distance between the centroid of generated citations and the
    centroid of reference citations.  Returns None if either set is empty.
    """
    if len(emb_gen) == 0 or len(emb_ref) == 0:
        return None

    c_gen = emb_gen.mean(axis=0)
    c_ref = emb_ref.mean(axis=0)

    norm_gen = np.linalg.norm(c_gen)
    norm_ref = np.linalg.norm(c_ref)
    if norm_gen < 1e-8 or norm_ref < 1e-8:
        return None

    cosine_sim = np.dot(c_gen, c_ref) / (norm_gen * norm_ref)
    return float(1.0 - cosine_sim)


# ── Title similarity helper ───────────────────────────────────────────────────

def _normalize_title(title: str) -> str:
    """Lowercase, strip LaTeX escapes and punctuation for exact comparison."""
    title = re.sub(r"\\+", " ", title)               # remove LaTeX backslashes
    title = re.sub(r"[^\w\s]", " ", title.lower())   # strip punctuation
    return " ".join(title.split())                    # collapse whitespace


# ── PCA visualization ─────────────────────────────────────────────────────────

def save_pca_plot(
    emb_gen: np.ndarray,
    emb_ref: np.ndarray,
    survey_title: str,
    metrics: dict,
    save_path: Path,
    dpi: int = 150,
    self_cite_gen_idx: int | None = None,
    dots_save_path: Path | None = None,
    survey_id: str | None = None,
    model_id: str | None = None,
    dataset_id: str | None = None,
) -> None:
    """
    Project embeddings to 2D with PCA and save a matplotlib scatter plot.

    Blue   = reference citations (ground truth)
    Coral  = generated citations (arxiv only)
    Stars  = centroids of each cloud
    Square = reference survey paper itself, if it appears in generated citations

    If dots_save_path is given, also saves the projected 2D points as JSON
    (for later multi-model comparison plots).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    has_gen = len(emb_gen) > 0
    has_ref = len(emb_ref) > 0

    if not has_gen and not has_ref:
        return

    # Fit PCA on reference citations only — this anchors the coordinate space
    # to the ground-truth distribution, which is the same for every model.
    # Generated citations are then projected with pca.transform(), so all
    # models' dots live in the same 2D space and can be compared directly.
    if has_ref:
        n_components = min(2, emb_ref.shape[0], emb_ref.shape[1])
        pca      = PCA(n_components=n_components)
        proj_ref = pca.fit_transform(emb_ref)
        proj_gen = pca.transform(emb_gen) if has_gen else np.zeros((0, 2))
    else:
        # No reference embeddings: fall back to fitting on generated only
        n_components = min(2, emb_gen.shape[0], emb_gen.shape[1])
        pca      = PCA(n_components=n_components)
        proj_ref = np.zeros((0, 2))
        proj_gen = pca.fit_transform(emb_gen)

    # ── Save 2D dots for multi-model comparison ───────────────────────────────
    if dots_save_path is not None:
        points = []
        for i, (x, y) in enumerate(proj_ref.tolist()):
            points.append({"x": x, "y": y, "group": "ref", "self_cite": False})
        for i, (x, y) in enumerate(proj_gen.tolist()):
            points.append({"x": x, "y": y, "group": "gen",
                           "self_cite": i == self_cite_gen_idx})
        dots = {
            "survey_id":    survey_id,
            "survey_title": survey_title,
            "model_id":     model_id,
            "dataset_id":   dataset_id,
            "pca_variance": pca.explained_variance_ratio_.tolist(),
            "metrics":      {k: v for k, v in metrics.items()
                             if k in ("citation_diversity", "distribution_shift")},
            "points":       points,
        }
        dots_save_path.parent.mkdir(parents=True, exist_ok=True)
        dots_save_path.write_text(
            json.dumps(dots, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    fig, ax = plt.subplots(figsize=(8, 6))

    if has_ref:
        ax.scatter(
            proj_ref[:, 0], proj_ref[:, 1],
            alpha=0.55, s=35, color="steelblue",
            label=f"Reference citations (n={len(emb_ref)})",
            zorder=2,
        )
        c_ref = proj_ref.mean(axis=0)
        ax.scatter(*c_ref, marker="*", s=220, color="navy", zorder=5, label="Ref centroid")

    if has_gen:
        ax.scatter(
            proj_gen[:, 0], proj_gen[:, 1],
            alpha=0.55, s=35, color="coral",
            label=f"Generated citations / arxiv (n={len(emb_gen)})",
            zorder=2,
        )
        c_gen = proj_gen.mean(axis=0)
        ax.scatter(*c_gen, marker="*", s=220, color="darkred", zorder=5, label="Gen centroid")

    # Self-citation marker: reference survey paper cited by the model
    if self_cite_gen_idx is not None and has_gen and self_cite_gen_idx < len(proj_gen):
        sc = proj_gen[self_cite_gen_idx]
        ax.scatter(
            sc[0], sc[1],
            marker="s", s=120, facecolors="none",
            edgecolors="black", linewidths=1.8,
            zorder=6, label="Reference survey itself (self-cite)",
        )

    # Metrics annotation
    lines = []
    if metrics.get("citation_diversity") is not None:
        lines.append(f"diversity = {metrics['citation_diversity']:.4f}")
    if metrics.get("distribution_shift") is not None:
        lines.append(f"shift     = {metrics['distribution_shift']:.4f}")
    if lines:
        ax.text(
            0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, verticalalignment="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.6),
        )

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)" if len(var) > 0 else "PC1")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)" if len(var) > 1 else "PC2")
    ax.set_title(
        (survey_title[:70] + "…") if survey_title and len(survey_title) > 70
        else (survey_title or "Citation distribution"),
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)


# ── Per-survey evaluation ─────────────────────────────────────────────────────

def evaluate_survey(
    generation: dict,
    ref_all_cites: list[int] | None,
    corpus_abstracts: dict[int, tuple[str, str]],
    tokenizer,
    model,
    device: str,
    cfg: dict,
    scores_dir: Path,
) -> dict:
    """Full evaluation pipeline for one survey.

    Extracts citations from generation, fetches arxiv metadata, embeds both
    generated and reference citations, computes diversity metrics.

    Args:
        generation: Generation dict with references and metadata.
        ref_all_cites: Reference doc_ids from ground truth survey.
        corpus_abstracts: Mapping of doc_id to (title, abstract).
        tokenizer: SPECTER2 tokenizer.
        model: SPECTER2 model.
        device: torch device.
        cfg: Config dict with arxiv/embedding settings.
        scores_dir: Output directory for PCA plots.

    Returns:
        Score dict with citation_diversity, distribution_shift, and metadata.
    """
    sep = tokenizer.sep_token or "[SEP]"

    # ── Generated citations: arxiv only ──────────────────────────────────────
    references = generation.get("meta", {}).get("references") or []
    arxiv_refs = [
        r for r in references
        if r.get("arxiv_id")
    ]
    # Strip version suffix (e.g. "1609.01967v1" → "1609.01967") so that
    # keys match what the arxiv API returns as bare IDs in responses.
    arxiv_ids = list({re.sub(r"v\d+$", "", r["arxiv_id"]) for r in arxiv_refs})

    print(f"    arxiv refs: {len(arxiv_ids)}")

    arxiv_meta: dict[str, tuple[str, str]] = {}
    if arxiv_ids:
        arxiv_meta = fetch_arxiv_metadata(
            arxiv_ids,
            max_retries=cfg.get("arxiv_max_retries", 2),
            retry_delays=cfg.get("arxiv_retry_delays", [5, 10]),
        )
        print(f"    fetched abstracts: {len(arxiv_meta)}/{len(arxiv_ids)}")

    # Keep ordered list of (aid, title, abstract) for the papers we have metadata for
    gen_items = [
        (aid, arxiv_meta[aid][0], arxiv_meta[aid][1])
        for aid in arxiv_ids
        if aid in arxiv_meta
    ]
    gen_pairs = [(title, abstract) for _, title, abstract in gen_items]
    gen_texts = make_specter2_texts(gen_pairs, sep)

    # Detect self-citation: check if the reference survey paper itself was cited
    survey_title = generation.get("query", "")
    self_cite_gen_idx: int | None = None
    if survey_title:
        norm_survey = _normalize_title(survey_title)
        for i, (_, title, _) in enumerate(gen_items):
            if _normalize_title(title) == norm_survey:
                self_cite_gen_idx = i
                print(f"    [self-cite] survey paper found in generated refs at idx={i}: {title[:80]}")
                break

    # ── Reference citations: from SurGE corpus ────────────────────────────────
    ref_pairs: list[tuple[str, str]] = []
    if ref_all_cites:
        ref_pairs = [
            corpus_abstracts[doc_id]
            for doc_id in ref_all_cites
            if doc_id in corpus_abstracts
        ]
    ref_texts = make_specter2_texts(ref_pairs, sep)

    print(f"    gen texts={len(gen_texts)}  ref texts={len(ref_texts)}")

    # ── Embed ─────────────────────────────────────────────────────────────────
    batch_size = cfg.get("embedding_batch_size", 16)
    emb_gen = embed_texts(gen_texts, tokenizer, model, device, batch_size) if gen_texts else np.zeros((0, 768), dtype=np.float32)
    emb_ref = embed_texts(ref_texts, tokenizer, model, device, batch_size) if ref_texts else np.zeros((0, 768), dtype=np.float32)

    # ── Metrics ───────────────────────────────────────────────────────────────
    diversity = compute_citation_diversity(emb_gen)
    shift     = compute_distribution_shift(emb_gen, emb_ref)

    scores = {
        "citation_diversity":      diversity,
        "distribution_shift":      shift,
        "n_arxiv_refs":            len(arxiv_ids),
        "n_embedded_gen":          len(emb_gen),
        "n_reference_cites":       len(ref_all_cites) if ref_all_cites else None,
        "n_embedded_ref":          len(emb_ref),
        "self_cited":              self_cite_gen_idx is not None,
    }

    # ── PCA plot ──────────────────────────────────────────────────────────────
    gid       = str(generation["id"])
    plot_path = scores_dir / "plots" / f"{gid}.png"
    dots_path = scores_dir / "plots" / f"{gid}_dots.json"
    try:
        save_pca_plot(
            emb_gen, emb_ref,
            survey_title, scores,
            plot_path,
            dpi=cfg.get("plot_dpi", 150),
            self_cite_gen_idx=self_cite_gen_idx,
            dots_save_path=dots_path,
            survey_id=gid,
            model_id=generation.get("model_id"),
            dataset_id=generation.get("dataset_id"),
        )
        scores["plot"] = str(plot_path.relative_to(ROOT / "results"))
    except Exception as e:
        print(f"    [WARN] plot failed: {e}")
        scores["plot"] = None

    return scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging("diversity")
    parser = argparse.ArgumentParser(description="Citation diversity evaluation")
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. SurGE)")
    parser.add_argument("--model",   required=True, help="Model id (e.g. perplexity_dr)")
    args = parser.parse_args()

    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    resume      = cfg.get("resume", True)
    corpus_path = Path(cfg["corpus_path"])

    # ── Output directory ──────────────────────────────────────────────────────
    scores_name = f"{args.dataset}_{args.model}_diversity"
    scores_dir  = ROOT / "results" / "scores" / scores_name
    scores_dir.mkdir(parents=True, exist_ok=True)
    (scores_dir / "plots").mkdir(exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    registry = load_registry(ROOT / "datasets" / "registry.yaml")
    if args.dataset not in registry:
        raise SystemExit(f"Dataset '{args.dataset}' not in registry")
    dataset = load_dataset_cls(args.dataset, registry[args.dataset])

    # ── Load generations ──────────────────────────────────────────────────────
    gen_dir = ROOT / "results" / "generations" / f"{args.dataset}_{args.model}"
    if not gen_dir.exists():
        raise SystemExit(f"Generations not found: {gen_dir}")

    gen_files = sorted(
        f for f in gen_dir.iterdir()
        if re.fullmatch(r"\d+\.json", f.name)
    )
    print(f"Dataset   : {args.dataset}")
    print(f"Model     : {args.model}")
    print(f"Scores    : {scores_dir}")
    print(f"Generations found: {len(gen_files)}\n")

    # ── Collect needed corpus doc_ids ─────────────────────────────────────────
    # We need to stream corpus.json once → gather all needed doc_ids upfront
    needed_doc_ids: set[int] = set()
    for gf in gen_files:
        gid = gf.stem
        instance = dataset.get_by_id(gid)
        if instance and hasattr(instance, "meta"):
            for doc_id in (instance.meta.get("all_cites") or []):
                needed_doc_ids.add(doc_id)

    # ── Stream corpus ─────────────────────────────────────────────────────────
    corpus_abstracts: dict[int, tuple[str, str]] = {}
    if needed_doc_ids and corpus_path.exists():
        corpus_abstracts = load_corpus_abstracts(corpus_path, needed_doc_ids)
    elif not corpus_path.exists():
        logger.warning(f"Corpus not found at {corpus_path} — distribution_shift will be None")

    # ── Load SPECTER2 ─────────────────────────────────────────────────────────
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice    : {device}")

    tokenizer, model = load_specter2(
        cfg["embedding_model"],
        cfg["embedding_adapter"],
        device,
    )
    print()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = []
    for gf in gen_files:
        gid      = gf.stem
        out_file = scores_dir / f"{gid}.json"

        if resume and out_file.exists():
            try:
                existing = json.loads(out_file.read_text(encoding="utf-8"))
                if existing.get("citation_diversity") is not None:
                    print(f"  [SKIP] {gid}")
                    results.append(existing)
                    continue
            except Exception:
                pass

        try:
            generation = json.loads(gf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [ERR]  {gid} — failed to load: {e}")
            continue

        instance = dataset.get_by_id(gid)
        ref_all_cites = None
        if instance and hasattr(instance, "meta"):
            ref_all_cites = instance.meta.get("all_cites")

        print(f"  [EVAL] {gid} | {generation.get('query','')[:60]}")

        t0 = time.time()
        try:
            scores = evaluate_survey(
                generation, ref_all_cites, corpus_abstracts,
                tokenizer, model, device, cfg, scores_dir,
            )
        except Exception as e:
            print(f"    [ERR] {e}\n{traceback.format_exc()}")
            scores = {
                "citation_diversity": None,
                "distribution_shift": None,
                "error": str(e),
            }

        record = {
            "id":         gid,
            "dataset_id": args.dataset,
            "model_id":   args.model,
            "query":      generation.get("query", ""),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec":  round(time.time() - t0, 1),
            **scores,
        }
        out_file.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results.append(record)

        div = scores.get("citation_diversity")
        sft = scores.get("distribution_shift")
        print(
            f"    diversity={div:.4f}" if div is not None else "    diversity=None",
            f"  shift={sft:.4f}" if sft is not None else "  shift=None",
        )

    # ── Summary CSV ───────────────────────────────────────────────────────────
    csv_path = scores_dir / "summary.csv"
    fields = [
        "id", "dataset_id", "model_id", "query",
        "citation_diversity", "distribution_shift",
        "n_arxiv_refs", "n_embedded_gen",
        "n_reference_cites", "n_embedded_ref",
        "self_cited", "plot",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # ── Aggregate stats ───────────────────────────────────────────────────────
    divs  = [r["citation_diversity"] for r in results if r.get("citation_diversity") is not None]
    shifts = [r["distribution_shift"] for r in results if r.get("distribution_shift") is not None]

    print(f"\n{'─'*55}")
    print(f"  Evaluated : {len(results)} surveys")
    if divs:
        print(f"  citation_diversity : mean={np.mean(divs):.4f}  std={np.std(divs):.4f}")
    if shifts:
        print(f"  distribution_shift : mean={np.mean(shifts):.4f}  std={np.std(shifts):.4f}")
    print(f"  Scores    → {scores_dir}")
    print(f"  Plots     → {scores_dir}/plots/")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()
