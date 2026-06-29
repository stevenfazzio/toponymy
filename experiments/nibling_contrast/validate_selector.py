"""
Deterministic validation of the *geometric contrast-set selector* on the committed
arXiv testbed -- NO LLM, NO API cost.

Goal of this script (Gate-0, before any naming):
  1. Prove the selector is well-formed (layer-0 empty; never returns the cluster's
     own children; in-range; sorted nearest-first).
  2. Let us *eyeball* whether "nearby but separate" finer-layer clusters actually
     look like sensible contrasts, using each cluster's nearest-to-centroid paper
     title as a human-readable proxy for its (not-yet-generated) name.

The contrast set is computed purely from structures Toponymy already produces at
fit time: cluster_tree_ (parent->children) and per-layer centroid_vectors. Selection
is geometric -- the k nearest layer-(L-1) clusters by embedding-centroid cosine
distance, excluding the cluster's own children. This subsumes the genealogical
"children of my siblings" notion and handles the top-layer / synthetic-root edge
case for free (we never look up a parent).

Run:
  uv run --project /Users/stevenfazzio/repos/toponymy \
      python experiments/nibling_contrast/validate_selector.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import norm

from toponymy.clustering import ToponymyClusterer
from toponymy.cluster_layer import ClusterLayerText

REPO = Path("/Users/stevenfazzio/repos/toponymy")
EXAMPLES = REPO / "examples"
K = 5  # contrast-set size

ClusterTree = Dict[Tuple[int, int], List[Tuple[int, int]]]


# --------------------------------------------------------------------------- #
# Selector (will be extracted into selector.py once validated)
# --------------------------------------------------------------------------- #
def n_clusters(layer: Any) -> int:
    return int(layer.centroid_vectors.shape[0])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (norm(v, axis=-1, keepdims=True) + 1e-12)


def own_children(tree: ClusterTree, layer_id: int, idx: int, child_layer: int) -> set:
    """Indices of (layer_id, idx)'s direct children that live at `child_layer`."""
    return {j for (cl, j) in tree.get((layer_id, idx), []) if cl == child_layer}


def contrast_set(
    cluster_layers: List[Any],
    tree: ClusterTree,
    layer_id: int,
    idx: int,
    k: int = K,
    max_dist: Optional[float] = None,
) -> List[Tuple[int, float]]:
    """Nearest finer-layer (layer_id-1) clusters that are NOT this cluster's children.

    Returns up to k (finer_cluster_idx, cosine_distance) pairs, nearest-first.
    Empty for layer 0 (no finer layer to contrast against).

    NOTE: distance is centroid-to-centroid. A future refinement could use the min
    distance from any of this cluster's *children* centroids to the candidate, which
    better captures "adjacent to my boundary". Centroid-to-centroid is the v1 choice.
    """
    if layer_id <= 0:
        return []
    finer = layer_id - 1
    finer_layer = cluster_layers[finer]
    excluded = own_children(tree, layer_id, idx, finer)
    cand = np.array([j for j in range(n_clusters(finer_layer)) if j not in excluded], dtype=int)
    if cand.size == 0:
        return []
    c = _unit(cluster_layers[layer_id].centroid_vectors[idx])
    B = _unit(finer_layer.centroid_vectors[cand])
    dist = 1.0 - B @ c
    order = np.argsort(dist, kind="stable")
    pairs = [(int(cand[o]), float(dist[o])) for o in order]
    if max_dist is not None:
        pairs = [(j, d) for j, d in pairs if d <= max_dist]
    return pairs[:k]


# --------------------------------------------------------------------------- #
# Data + proxies
# --------------------------------------------------------------------------- #
def load():
    emb = np.load(EXAMPLES / "ai_arxiv_vectors.npy").astype(np.float32)
    coords = np.load(EXAMPLES / "ai_arxiv_coordinates.npz.npy").astype(np.float32)
    with zipfile.ZipFile(EXAMPLES / "ai_arxiv_papers.zip") as z:
        with z.open("ai_arxiv_papers") as fh:
            papers = pd.read_csv(fh)
    titles = (
        papers["title"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().tolist()
    )
    assert emb.shape[0] == coords.shape[0] == len(titles), (emb.shape, coords.shape, len(titles))
    return emb, coords, titles, list(papers.columns)


def layer_proxies(layer: Any, emb_unit: np.ndarray, titles: List[str]) -> List[str]:
    """Per-cluster proxy = title of the member paper nearest the cluster centroid."""
    labels = layer.cluster_labels
    cents = layer.centroid_vectors
    out = []
    for i in range(cents.shape[0]):
        members = np.where(labels == i)[0]
        if members.size == 0:
            out.append("(empty)")
            continue
        c = cents[i] / (norm(cents[i]) + 1e-12)
        sims = emb_unit[members] @ c
        out.append(titles[int(members[int(np.argmax(sims))])])
    return out


# --------------------------------------------------------------------------- #
def main():
    emb, coords, titles, cols = load()
    print(f"Loaded arXiv: emb={emb.shape}, coords={coords.shape}, n_titles={len(titles)}")
    print(f"CSV columns: {cols}\n")
    emb_unit = _unit(emb)

    clusterer = ToponymyClusterer(min_clusters=4, base_min_cluster_size=25, verbose=False)
    cluster_layers, tree = clusterer.fit_predict(coords, emb, ClusterLayerText)
    n_layers = len(cluster_layers)
    print(
        f"Fitted {n_layers} layers; clusters per layer (finest->coarsest): "
        f"{[n_clusters(l) for l in cluster_layers]}\n"
    )

    proxies = [layer_proxies(l, emb_unit, titles) for l in cluster_layers]

    # --- correctness assertions -------------------------------------------- #
    problems = 0
    for L in range(n_layers):
        for i in range(n_clusters(cluster_layers[L])):
            cs = contrast_set(cluster_layers, tree, L, i)
            if L == 0:
                if cs:
                    print(f"  !! layer0 ({L},{i}) returned a non-empty contrast set")
                    problems += 1
                continue
            kids = own_children(tree, L, i, L - 1)
            idxs = [j for j, _ in cs]
            dists = [d for _, d in cs]
            if set(idxs) & kids:
                print(f"  !! ({L},{i}) contrast overlaps own children")
                problems += 1
            if any(j < 0 or j >= n_clusters(cluster_layers[L - 1]) for j in idxs):
                print(f"  !! ({L},{i}) out-of-range index")
                problems += 1
            if dists != sorted(dists):
                print(f"  !! ({L},{i}) distances not sorted")
                problems += 1
    print(f"Correctness assertions complete. Problems found: {problems}\n")

    # --- contrast-set size + nearest-distance distribution ----------------- #
    for L in range(1, n_layers):
        css = [contrast_set(cluster_layers, tree, L, i) for i in range(n_clusters(cluster_layers[L]))]
        sizes = [len(c) for c in css]
        nearest = [c[0][1] for c in css if c]
        print(
            f"layer {L}: {n_clusters(cluster_layers[L])} clusters | "
            f"contrast size min/med/max = {min(sizes)}/{int(np.median(sizes))}/{max(sizes)} | "
            f"nearest-contrast cos-dist min/med/max = "
            f"{min(nearest):.3f}/{np.median(nearest):.3f}/{max(nearest):.3f}"
        )
    print()

    # --- qualitative eyeball ----------------------------------------------- #
    for L in range(1, n_layers):
        ncl = n_clusters(cluster_layers[L])
        sample = sorted(set(np.linspace(0, ncl - 1, num=min(6, ncl)).astype(int)))
        print(f"\n===== LAYER {L}  ({ncl} clusters; showing {len(sample)}) =====")
        for i in sample:
            cs = contrast_set(cluster_layers, tree, L, i)
            kids = sorted(own_children(tree, L, i, L - 1))
            print(f"\n[cluster ({L},{i})]  proxy: {proxies[L][i][:90]}")
            print(f"   own children @L-1: {kids[:8]}{' ...' if len(kids) > 8 else ''}")
            print(f"   nearest SEPARATE @L-1 (contrast):")
            for j, d in cs:
                print(f"      d={d:.3f}  (L-1,{j:>3})  {proxies[L - 1][j][:78]}")


if __name__ == "__main__":
    main()
