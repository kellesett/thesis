# metrics/structural/contradiction/candidates.py
# Stage 1 — similarity filter: encode all sentences, return cross-section pairs
# above threshold sorted by similarity descending.

import numpy as np
from tqdm import tqdm


def generate_candidates(
    sections: list[dict],
    embedder,
    threshold: float = 0.6,
    batch_size: int = 32,
) -> list[dict]:
    """Encode all sentences and return cross-section pairs above similarity threshold.

    Args:
        sections: List of {"title": str, "sentences": list[str]}.
        embedder: SentenceTransformer model (already loaded, reused from M_rep).
        threshold: Cosine similarity threshold for candidate selection.
        batch_size: Encoding batch size.

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
    n_pairs = n * (n - 1) // 2
    with tqdm(total=n_pairs, desc="  1/5 SPECTER pairs", leave=False, unit="pair") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
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

    return sorted(candidates, key=lambda x: x["similarity"], reverse=True)
