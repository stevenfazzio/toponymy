"""
Phase 1 -- fidelity bake-off + judge-free intrusion gate.

For each named cluster, score a candidate label by cosine similarity to one of four
REFERENCE POINTS, then ask the judge-free question: does the gold label out-score each
KNOWN-BAD variant from the battery?

Reference points (what the label embedding is compared against):
  centroid : the cluster's mean embedding (raw)          -- the prior baseline (analyze_cosine.py)
  whitened : centroid in a whitened/isotropic space      -- Su 2021 / Mu & Viswanath fix for the
                                                            central/verbose bias
  medoid   : the most-central real member document        -- a real in-distribution point (keeps specificity)
  exemplar : Toponymy's top facility-location exemplar    -- the evidence the namer actually saw

Labels (gold + variants) are embedded with the SAME model the documents use. Whitening is
fit on the document embeddings and applied to centroid + label.

Reports, per (metric x degradation): INTRUSION ACCURACY = fraction of clusters whose gold
label out-scores that degradation (pairwise chance = 0.5), plus a strict TOP-1 rate (gold
beats ALL its variants at once) and the mean score margin (gold - variant).

  uv run python experiments/label_quality/metrics.py --labels data/labels_20ng_haiku.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.linalg import norm

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))

DEGRADATIONS = ["ancestor", "sibling", "distant", "generic", "verbose"]
METRICS = ["centroid", "whitened", "medoid", "exemplar"]


def unit(v: np.ndarray) -> np.ndarray:
    return v / (norm(v, axis=-1, keepdims=True) + 1e-12)


def fit_whitening(X: np.ndarray):
    """Su 2021 whitening: returns (mu, W) with (x-mu)@W having identity covariance."""
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (len(X) - 1)
    U, S, _ = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S + 1e-8))
    return mu, W


def medoid_vector(memb_emb: np.ndarray, cap: int = 1500) -> np.ndarray:
    """Member maximizing mean cosine similarity to the (sub-sampled) membership."""
    U = unit(memb_emb)
    if len(U) > cap:
        idx = np.random.default_rng(0).choice(len(U), cap, replace=False)
        ref = U[idx]
    else:
        ref = U
    return memb_emb[int(np.argmax((U @ ref.T).mean(1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="20ng")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--base-min-cluster-size", type=int, default=25)
    ap.add_argument("--min-clusters", type=int, default=4)
    args = ap.parse_args()

    from ab_harness import load_dataset, make_embedder

    from perturbations import build_battery, load_fit

    cl, objects, emb, coords, meta = load_fit(args.dataset, None,
                                              args.base_min_cluster_size, args.min_clusters)
    # Populate the real Toponymy exemplars (facility-location; no LLM) so the "exemplar"
    # reference is the evidence the namer actually saw -- deterministic given the same fit.
    for layer in cl.cluster_layers_:
        layer.make_exemplar_texts(objects, emb)
    emb = emb.astype(np.float64)
    names = json.loads((HERE / args.labels).read_text())
    counts = [int(l.centroid_vectors.shape[0]) for l in cl.cluster_layers_]
    assert [len(x) for x in names] == counts, f"labels/cluster mismatch {[len(x) for x in names]} vs {counts}"
    battery = build_battery(cl.cluster_layers_, cl.cluster_tree_, names)
    print(f"[{args.dataset}] {len(battery)} clusters; layers {counts}")

    # whitening fit on the document embeddings
    mu, W = fit_whitening(emb)
    whiten = lambda x: (np.atleast_2d(x) - mu) @ W

    # embed every candidate label once
    embedder = make_embedder(meta["emb_model"])
    all_labels = sorted({c for it in battery for c in ([it["gold"]] + list(it["variants"].values()))})
    LE = dict(zip(all_labels,
                  embedder.encode(all_labels, convert_to_numpy=True, batch_size=64).astype(np.float64)))
    print(f"embedded {len(all_labels)} unique candidate labels")

    acc = {m: defaultdict(list) for m in METRICS}
    margin = {m: defaultdict(list) for m in METRICS}
    top1 = {m: [] for m in METRICS}
    per_candidate = []
    ex_fallback = 0

    for it in battery:
        layer = cl.cluster_layers_[it["layer"]]
        i = it["idx"]
        centroid = layer.centroid_vectors[i].astype(np.float64)
        members = np.where(layer.cluster_labels == i)[0]
        memb_emb = emb[members] if len(members) else centroid[None]
        medoid = medoid_vector(memb_emb)
        ex_idx = layer.exemplar_indices[i] if i < len(layer.exemplar_indices) else []
        if len(ex_idx) and ex_idx[0] < len(emb):
            exemplar = emb[ex_idx[0]]
        else:
            exemplar, _ = medoid, (ex_fallback := ex_fallback + 1)

        cands = {"gold": it["gold"], **it["variants"]}
        scores = {m: {} for m in METRICS}
        rc, rm, re = unit(centroid), unit(medoid), unit(exemplar)
        rw = unit(whiten(centroid))[0]
        for nm, lab in cands.items():
            e = LE[lab]
            ue = unit(e)
            scores["centroid"][nm] = float(ue @ rc)
            scores["medoid"][nm] = float(ue @ rm)
            scores["exemplar"][nm] = float(ue @ re)
            scores["whitened"][nm] = float(unit(whiten(e))[0] @ rw)

        for m in METRICS:
            g = scores[m]["gold"]
            for d in it["variants"]:
                acc[m][d].append(1.0 if g > scores[m][d] else 0.0)
                margin[m][d].append(g - scores[m][d])
            top1[m].append(1.0 if g > max(scores[m][d] for d in it["variants"]) else 0.0)
        for nm in cands:
            per_candidate.append(dict(layer=it["layer"], idx=it["idx"], type=nm, label=cands[nm],
                                      **{m: scores[m][nm] for m in METRICS}))

    if ex_fallback:
        print(f"(exemplar fell back to medoid for {ex_fallback} clusters)")
    n_per = {d: len(acc["centroid"][d]) for d in DEGRADATIONS}
    print("\ncoverage per degradation:", n_per, f" | total clusters {len(battery)}")
    print("\nINTRUSION ACCURACY  (gold out-scores the variant; pairwise chance = 0.50)")
    hdr = "  ".join(f"{d:>8}" for d in DEGRADATIONS)
    print(f"{'metric':<10} {hdr}   {'TOP-1':>7}")
    for m in METRICS:
        cells = "  ".join(f"{np.mean(acc[m][d])*100:7.1f}%" if acc[m][d] else f"{'-':>8}" for d in DEGRADATIONS)
        print(f"{m:<10} {cells}   {np.mean(top1[m])*100:6.1f}%")

    print("\nMEAN MARGIN  (gold - variant cosine; >0 means gold preferred)")
    for m in METRICS:
        cells = "  ".join(f"{np.mean(margin[m][d]):+7.3f} " if margin[m][d] else f"{'-':>8}" for d in DEGRADATIONS)
        print(f"{m:<10} {cells}")

    (HERE / "data" / "metric_scores_20ng.json").write_text(json.dumps(per_candidate, indent=2))
    out = HERE / "data" / "metrics_20ng.json"
    out.write_text(json.dumps({
        "n_per_degradation": n_per,
        "intrusion_accuracy": {m: {d: float(np.mean(acc[m][d])) if acc[m][d] else None
                                   for d in DEGRADATIONS} for m in METRICS},
        "top1": {m: float(np.mean(top1[m])) for m in METRICS},
        "mean_margin": {m: {d: float(np.mean(margin[m][d])) if margin[m][d] else None
                            for d in DEGRADATIONS} for m in METRICS},
    }, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
