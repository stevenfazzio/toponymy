"""
Geometric contrast-set selection for Toponymy cluster naming (validated by
validate_selector.py).

When naming a cluster at layer L, surface a small set of *nearby but separate*
already-named clusters from the finer layer L-1, so the LLM can pick a name that
distinguishes this region from its neighbours (e.g. "placental mammals" instead of
"mammals") rather than naming from positive content alone.

Computed purely from structures Toponymy already produces at fit time
(cluster_tree_, per-layer centroid_vectors) -- no extra LLM calls. Selection is
geometric: the k nearest layer-(L-1) clusters by embedding-centroid cosine
distance, excluding the cluster's own children. This subsumes the genealogical
"children of my siblings" notion and handles the top-layer / synthetic-root edge
case for free (we never look up a parent).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import norm

ClusterTree = Dict[Tuple[int, int], List[Tuple[int, int]]]


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
    k: int = 5,
    max_dist: Optional[float] = None,
) -> List[Tuple[int, float]]:
    """Nearest finer-layer (layer_id-1) clusters that are NOT this cluster's children.

    Returns up to k (finer_cluster_idx, cosine_distance) pairs, nearest-first.
    Empty for layer 0 (no finer layer to contrast against).
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


def contrast_names(
    all_topic_names: List[List[str]],
    cluster_layers: List[Any],
    tree: ClusterTree,
    layer_id: int,
    idx: int,
    k: int = 5,
    max_dist: Optional[float] = None,
) -> List[str]:
    """The names of the contrast set (deduped; skips empty / 'Unlabelled')."""
    finer = layer_id - 1
    if finer < 0 or finer >= len(all_topic_names):
        return []
    names_finer = all_topic_names[finer]
    out, seen = [], set()
    for j, _ in contrast_set(cluster_layers, tree, layer_id, idx, k=k, max_dist=max_dist):
        if j < len(names_finer):
            nm = names_finer[j]
            if nm and nm != "Unlabelled" and nm not in seen:
                out.append(nm)
                seen.add(nm)
    return out


CONTRAST_PREAMBLE = (
    "\nThe following are nearby but SEPARATE groups (NOT part of this group). "
    "Use them only to choose a name that clearly distinguishes THIS group from them. "
    "Do not include them in the name, and do not name the group by what it excludes:\n"
)

# v2: tuned to counter the observed failure mode (verbose over-qualification / enumeration).
CONTRAST_PREAMBLE_V2 = (
    "\nFor context only -- a few SEPARATE neighbouring groups that are NOT part of this group:\n"
)
CONTRAST_SUFFIX_V2 = (
    "\nKeep the name concise and natural. Do NOT mention, list, or enumerate these "
    "neighbouring groups, and do not lengthen the name to contrast with them. Only if your "
    "name would otherwise be too generic to tell this group apart from a neighbour, make it "
    "slightly more specific -- but keep it short.\n"
)


def contrast_block(names: List[str], version: str = "v1") -> str:
    if not names:
        return ""
    bullets = "\n".join(f"      * {n}" for n in names)
    if version == "v2":
        return CONTRAST_PREAMBLE_V2 + bullets + CONTRAST_SUFFIX_V2
    return CONTRAST_PREAMBLE + bullets + "\n"


def append_contrast(prompt, block: str):
    """Append a contrast block to a prompt (str=combined, or {'system','user'} dict)."""
    if not block:
        return prompt
    if isinstance(prompt, dict):
        out = dict(prompt)
        out["user"] = out.get("user", "") + block
        return out
    return prompt + block
