"""
Phase 2b -- hierarchy / generality check on Toponymy's own tree.

Applies the HyperLex-learned generality axis (phase2_generality.py) to Toponymy region labels and
asks, for every REAL parent->child edge in cluster_tree_: is the PARENT label more general than the
CHILD label? A well-formed hierarchy should have G(parent) > G(child). Edges where it doesn't are
UNDER-GENERALIZED coarse labels (e.g. a coarse region that inherited a child's exact name).

Even though the axis is weak on HyperLex (rho~0.45), Toponymy parent->child generality gaps may be
clearer than HyperLex's subtle pairs -- this measures whether it's useful in practice. Length and
mean word-frequency are included as cheap baselines (a coarse label is often just shorter).

  uv run --with wordfreq python experiments/label_quality/phase2b_hierarchy.py --labels data/labels_20ng_haiku.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from numpy.linalg import norm

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))

EMB_MODEL = "all-MiniLM-L6-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="20ng")
    ap.add_argument("--labels", default="data/labels_20ng_haiku.json")
    ap.add_argument("--base-min-cluster-size", type=int, default=25)
    ap.add_argument("--min-clusters", type=int, default=4)
    args = ap.parse_args()

    from perturbations import load_fit

    cl, *_ = load_fit(args.dataset, None, args.base_min_cluster_size, args.min_clusters)
    labels = json.loads((HERE / args.labels).read_text())
    counts = [int(l.centroid_vectors.shape[0]) for l in cl.cluster_layers_]
    assert [len(x) for x in labels] == counts, "labels/cluster mismatch"
    axis = np.load(HERE / "data" / "generality_axis.npy")
    tree, nL = cl.cluster_tree_, len(labels)

    def lab(node):
        L, i = node
        if L < nL and i < counts[L] and labels[L][i] and labels[L][i] != "Unlabelled":
            return labels[L][i]
        return None

    uniq = sorted({lab((L, i)) for L in range(nL) for i in range(counts[L]) if lab((L, i))})
    from sentence_transformers import SentenceTransformer
    raw = SentenceTransformer(EMB_MODEL, device="cpu").encode(uniq, normalize_embeddings=False,
                                                              convert_to_numpy=True).astype(np.float64)
    U = {w: raw[i] / (norm(raw[i]) + 1e-12) for i, w in enumerate(uniq)}
    Gax = {w: float(U[w] @ axis) for w in uniq}

    from wordfreq import zipf_frequency

    def freq(s):
        ws = re.findall(r"[a-zA-Z]+", s.lower())
        return float(np.mean([zipf_frequency(w, "en") for w in ws])) if ws else 0.0

    edges = []
    for parent, kids in tree.items():
        pL = lab(parent)
        if pL is None:
            continue
        for child in kids:
            cL = lab(child)
            if cL is None:
                continue
            edges.append(dict(p=pL, c=cL, pl=parent[0], clr=child[0],
                              d_axis=Gax[pL] - Gax[cL], d_len=len(cL) - len(pL),
                              d_freq=freq(pL) - freq(cL), cos=float(U[pL] @ U[cL]), same=pL == cL))

    da = np.array([e["d_axis"] for e in edges])
    print(f"scored {len(edges)} real parent->child edges (labels both named)")
    print("\ndirection accuracy -- 'parent label is more general than child' (chance = 50%)")
    print(f"  emb-axis    {100*np.mean(da > 0):5.1f}%   mean Δ = {da.mean():+.3f}")
    print(f"  length      {100*np.mean([e['d_len'] > 0 for e in edges]):5.1f}%   (coarse label shorter)")
    print(f"  frequency   {100*np.mean([e['d_freq'] > 0 for e in edges]):5.1f}%   (coarse label more frequent)")
    same = sum(e["same"] for e in edges)
    print(f"\nname propagation (parent label == child label): {same}/{len(edges)} "
          f"edges -- these are exactly Δ=0 under-generalizations")

    print("\nmost UNDER-GENERALIZED edges (axis says parent NOT more general than child):")
    for e in sorted(edges, key=lambda e: e["d_axis"])[:8]:
        flag = " [identical name]" if e["same"] else ""
        print(f"  Δaxis={e['d_axis']:+.3f} cos={e['cos']:.2f}{flag}")
        print(f"     parent L{e['pl']}: {e['p'][:60]!r}")
        print(f"     child  L{e['clr']}: {e['c'][:60]!r}")

    out = HERE / "data" / "hierarchy_20ng.json"
    out.write_text(json.dumps(edges, indent=2))
    print(f"\nsaved {len(edges)} edges -> {out}")


if __name__ == "__main__":
    main()
