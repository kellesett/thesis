# metrics/structural/contradiction/candidates.py
# Stage 1 — similarity filter: encode all sentences, return cross-section pairs
# above threshold sorted by similarity descending.

import numpy as np


def generate_candidates(
    sections: list[dict],
    embedder,
    threshold: float = 0.6,
    top_k_per_sentence: int | None = None,
    batch_size: int = 32,
    pbar=None,
) -> list[dict]:
    """Encode all sentences and return cross-section pairs selected by similarity.

    Args:
        sections:   List of {"title": str, "sentences": list[str]}.
        embedder:   SentenceTransformer model (already loaded, reused from M_rep).
        threshold:  Minimum cosine similarity for candidate selection.
        top_k_per_sentence:
                    If positive, keep only the top-k cross-section neighbours
                    for each sentence after applying the threshold. If unset,
                    keep all cross-section pairs above the threshold.
        batch_size: Encoding batch size.
        pbar:       Optional tqdm bar. Must be reset to the correct total before calling.
                    Updated once per pair in threshold-only mode or once per
                    sentence in top-k mode; postfix set on completion.

    Returns:
        List of candidate dicts sorted by similarity descending:
        {s1, t1, s2, t2, similarity}
    """
    # Flatten all sentences with section metadata
    flat: list[dict] = []
    for sec in sections:
        for sent in sec["sentences"]:
            flat.append({"text": sent, "section": sec["title"]})

    if len(flat) < 2:
        return []

    texts = [s["text"] for s in flat]
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # unit vectors → dot product = cosine similarity
        show_progress_bar=False,
    )

    # Full cosine similarity matrix via dot product (embeddings already normalized)
    sims = np.dot(embeddings, embeddings.T)

    candidates = []
    n = len(flat)

    if top_k_per_sentence is not None and top_k_per_sentence > 0:
        selected_pairs: set[tuple[int, int]] = set()
        sections_by_idx = np.array([s["section"] for s in flat])

        for i in range(n):
            if pbar is not None:
                pbar.update(1)

            row = sims[i].copy()
            row[i] = -np.inf
            row[sections_by_idx == flat[i]["section"]] = -np.inf

            valid = np.flatnonzero(row >= threshold)
            if len(valid) == 0:
                continue

            top = valid[np.argsort(row[valid])[::-1][:top_k_per_sentence]]
            for j in top:
                a, b = sorted((i, int(j)))
                pair_key = (a, b)
                if pair_key in selected_pairs:
                    continue
                selected_pairs.add(pair_key)
                sim = float(sims[a, b])
                candidates.append({
                    "s1":         flat[a]["text"],
                    "t1":         flat[a]["section"],
                    "s2":         flat[b]["text"],
                    "t2":         flat[b]["section"],
                    "similarity": round(sim, 4),
                })

        if pbar is not None:
            pbar.set_postfix_str(
                f"top_k={top_k_per_sentence} → {len(candidates)} sel"
            )

        return sorted(candidates, key=lambda x: x["similarity"], reverse=True)

    for i in range(n):
        for j in range(i + 1, n):
            if pbar is not None:
                pbar.update(1)
            if flat[i]["section"] == flat[j]["section"]:
                continue  # same section pairs excluded
            sim = float(sims[i, j])
            if sim >= threshold:
                candidates.append({
                    "s1":         flat[i]["text"],
                    "t1":         flat[i]["section"],
                    "s2":         flat[j]["text"],
                    "t2":         flat[j]["section"],
                    "similarity": round(sim, 4),
                })

    if pbar is not None:
        pbar.set_postfix_str(f"→ {len(candidates)} sel")

    return sorted(candidates, key=lambda x: x["similarity"], reverse=True)


def _flatten_paragraphs(sections: list[dict]) -> list[dict]:
    flat: list[dict] = []
    for sec in sections:
        title = sec["title"]
        paragraphs = sec.get("paragraphs") or []
        if paragraphs:
            for idx, par in enumerate(paragraphs):
                sentences = par.get("sentences") or []
                text = par.get("text") or " ".join(sentences)
                text = text.strip()
                if text:
                    flat.append({
                        "text": text,
                        "section": title,
                        "paragraph_index": idx,
                        "sentences": sentences,
                    })
        else:
            for idx, sent in enumerate(sec.get("sentences", [])):
                flat.append({
                    "text": sent,
                    "section": title,
                    "paragraph_index": idx,
                    "sentences": [sent],
                })
    return flat


def generate_paragraph_candidates(
    sections: list[dict],
    embedder,
    threshold: float = 0.6,
    top_k_per_paragraph: int | None = None,
    batch_size: int = 32,
    pbar=None,
) -> list[dict]:
    """Encode paragraphs and return cross-section paragraph pairs."""
    flat = _flatten_paragraphs(sections)
    if len(flat) < 2:
        return []

    texts = [p["text"] for p in flat]
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    sims = np.dot(embeddings, embeddings.T)

    def make_candidate(a: int, b: int) -> dict:
        sim = float(sims[a, b])
        return {
            "unit": "paragraph",
            "s1": flat[a]["text"],
            "t1": flat[a]["section"],
            "paragraph_i": flat[a]["paragraph_index"],
            "sentences1": flat[a]["sentences"],
            "s2": flat[b]["text"],
            "t2": flat[b]["section"],
            "paragraph_j": flat[b]["paragraph_index"],
            "sentences2": flat[b]["sentences"],
            "similarity": round(sim, 4),
        }

    candidates = []
    n = len(flat)

    if top_k_per_paragraph is not None and top_k_per_paragraph > 0:
        selected_pairs: set[tuple[int, int]] = set()
        sections_by_idx = np.array([p["section"] for p in flat])

        for i in range(n):
            if pbar is not None:
                pbar.update(1)

            row = sims[i].copy()
            row[i] = -np.inf
            row[sections_by_idx == flat[i]["section"]] = -np.inf

            valid = np.flatnonzero(row >= threshold)
            if len(valid) == 0:
                continue

            top = valid[np.argsort(row[valid])[::-1][:top_k_per_paragraph]]
            for j in top:
                a, b = sorted((i, int(j)))
                pair_key = (a, b)
                if pair_key in selected_pairs:
                    continue
                selected_pairs.add(pair_key)
                candidates.append(make_candidate(a, b))

        if pbar is not None:
            pbar.set_postfix_str(
                f"para_top_k={top_k_per_paragraph} → {len(candidates)} sel"
            )

        return sorted(candidates, key=lambda x: x["similarity"], reverse=True)

    for i in range(n):
        for j in range(i + 1, n):
            if pbar is not None:
                pbar.update(1)
            if flat[i]["section"] == flat[j]["section"]:
                continue
            if sims[i, j] >= threshold:
                candidates.append(make_candidate(i, j))

    if pbar is not None:
        pbar.set_postfix_str(f"→ {len(candidates)} sel")

    return sorted(candidates, key=lambda x: x["similarity"], reverse=True)
