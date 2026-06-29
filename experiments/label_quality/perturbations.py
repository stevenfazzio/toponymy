"""
Phase 0 . leg 1 -- perturbation / intrusion battery for label-quality evaluation.

Given a fitted Toponymy hierarchy (cluster_layers_, cluster_tree_) and a gold
labeling (topic_names_, finest->coarsest), build for each named cluster a set of
KNOWN-BAD label variants. Any candidate label-quality metric must rank the gold
label above every degraded variant -- a Chang et al. (2009) intrusion test made
Toponymy-native via the cluster tree.

Degradations (each targets a failure the metric must catch):
  ancestor : label of the cluster's PARENT (coarser) region          -- over-generalization
  sibling  : label of a SIBLING region (shares a parent), same layer  -- plausible wrong neighbour
  distant  : label of the farthest same-layer region (centroid cos)   -- off-topic
  generic  : a contentless placeholder ("Various topics", ...)        -- the genericness trap
  verbose  : gold over-qualified with parent + sibling material       -- the over-hedged failure mode

Only `generic` and `verbose` are synthesised; the rest are real labels lifted from
elsewhere in the same hierarchy. Clusters that are `Unlabelled` or that lack the
needed relative are skipped for that degradation.

Structure-only check (no LLM; placeholder names) on the real 20NG tree:
  uv run python experiments/label_quality/perturbations.py --structure-only --subsample 2000
Real battery from a cached gold labeling (topic_names_ JSON, finest->coarsest):
  uv run python experiments/label_quality/perturbations.py --labels data/labels_20ng_haiku.json

NOTE: clusterer params (base_min_cluster_size, min_clusters) and subsample MUST match
the run that produced the gold labels, so cluster indices line up (asserted on counts).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import norm

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))

Node = Tuple[int, int]
ClusterTree = Dict[Node, List[Node]]

GENERIC = ["Various topics", "Miscellaneous discussions", "General content",
           "Assorted posts", "A range of subjects"]


def invert_tree(tree: ClusterTree) -> Dict[Node, Node]:
    """child (layer, idx) -> its single parent (layer+1, idx)."""
    parent: Dict[Node, Node] = {}
    for p, kids in tree.items():
        for c in kids:
            parent[c] = p
    return parent


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (norm(v, axis=-1, keepdims=True) + 1e-12)


def farthest_same_layer(cluster_layers, layer: int, idx: int) -> Optional[int]:
    """Same-layer cluster with max centroid cosine distance to idx (the off-topic pick)."""
    C = cluster_layers[layer].centroid_vectors
    if C.shape[0] < 2:
        return None
    d = 1.0 - _unit(C) @ _unit(C[idx])
    d[idx] = -np.inf
    return int(np.argmax(d))


def name_at(topic_names: List[List[str]], layer: int, idx: int) -> Optional[str]:
    if 0 <= layer < len(topic_names) and 0 <= idx < len(topic_names[layer]):
        nm = topic_names[layer][idx]
        if nm and nm != "Unlabelled":
            return nm
    return None


def build_battery(cluster_layers, tree: ClusterTree, topic_names: List[List[str]]) -> List[dict]:
    """For every named cluster, the gold label + a dict of degraded variants."""
    parent = invert_tree(tree)
    items: List[dict] = []
    for L in range(len(cluster_layers)):
        n = int(cluster_layers[L].centroid_vectors.shape[0])
        for i in range(n):
            gold = name_at(topic_names, L, i)
            if gold is None:
                continue
            variants: Dict[str, str] = {}

            # ancestor: the parent's (coarser) label -- over-generalization
            par = parent.get((L, i))
            par_name = name_at(topic_names, *par) if par else None
            if par_name and par_name != gold:
                variants["ancestor"] = par_name

            # sibling: another child of the same parent at this layer
            sib_name = None
            if par:
                for (cl, cj) in tree.get(par, []):
                    if cl == L and cj != i:
                        cand = name_at(topic_names, cl, cj)
                        if cand and cand != gold:
                            sib_name = cand
                            break
            if sib_name:
                variants["sibling"] = sib_name

            # distant: farthest same-layer label -- off-topic
            fj = farthest_same_layer(cluster_layers, L, i)
            dist_name = name_at(topic_names, L, fj) if fj is not None else None
            if dist_name and dist_name != gold:
                variants["distant"] = dist_name

            # generic: deterministic contentless placeholder
            variants["generic"] = GENERIC[(L * 1009 + i) % len(GENERIC)]

            # verbose: over-qualified with parent + sibling material
            extra = ", ".join(x for x in (par_name, sib_name) if x)
            if extra:
                variants["verbose"] = (
                    f"{gold}, including various related discussions of {extra}, "
                    f"among other assorted and loosely connected topics"
                )

            items.append(dict(layer=L, idx=i, gold=gold, variants=variants))
    return items


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_fit(dataset: str, subsample: Optional[int], base_min_cluster_size: int, min_clusters: int):
    from ab_harness import load_dataset  # reuse the nibling 20NG/arxiv loader

    from toponymy.clustering import ToponymyClusterer
    from toponymy.cluster_layer import ClusterLayerText

    objects, emb, coords, meta = load_dataset(dataset, subsample)
    cl = ToponymyClusterer(min_clusters=min_clusters,
                           base_min_cluster_size=base_min_cluster_size, verbose=False)
    cl.fit_predict(coords, emb, ClusterLayerText)
    return cl, objects, emb, coords, meta


def placeholder_names(cluster_layers) -> List[List[str]]:
    return [[f"L{L}#{i}" for i in range(int(layer.centroid_vectors.shape[0]))]
            for L, layer in enumerate(cluster_layers)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="20ng")
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--base-min-cluster-size", type=int, default=25)
    ap.add_argument("--min-clusters", type=int, default=4)
    ap.add_argument("--labels", type=str, default=None,
                    help="JSON of gold topic_names_ (finest->coarsest)")
    ap.add_argument("--structure-only", action="store_true",
                    help="placeholder names to verify tree navigation without an LLM run")
    args = ap.parse_args()

    cl, *_ = load_fit(args.dataset, args.subsample, args.base_min_cluster_size, args.min_clusters)
    counts = [int(l.centroid_vectors.shape[0]) for l in cl.cluster_layers_]
    print(f"[{args.dataset}] clusters per layer (finest->coarsest): {counts}")

    if args.structure_only:
        names = placeholder_names(cl.cluster_layers_)
    elif args.labels:
        p = Path(args.labels)
        names = json.loads((p if p.is_absolute() else HERE / p).read_text())
        assert [len(x) for x in names] == counts, \
            f"labels/cluster mismatch: {[len(x) for x in names]} vs {counts} -- params must match the naming run"
    else:
        raise SystemExit("provide --labels <json> or pass --structure-only")

    battery = build_battery(cl.cluster_layers_, cl.cluster_tree_, names)
    cov = Counter(k for it in battery for k in it["variants"])
    print(f"named clusters: {len(battery)}  |  variant coverage: {dict(cov)}")

    for it in sorted(battery, key=lambda it: -len(it["variants"]))[:3]:
        print(f"\n  (L{it['layer']},{it['idx']}) gold: {it['gold']!r}")
        for k, v in it["variants"].items():
            print(f"      {k:<9} {v!r}")

    out = HERE / "data" / f"battery_{args.dataset}{'_structure' if args.structure_only else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(battery, indent=2))
    print(f"\nsaved battery ({len(battery)} clusters) -> {out}")


if __name__ == "__main__":
    main()
