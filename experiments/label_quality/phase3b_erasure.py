"""
Phase 3b -- does erasing the label<->document offset actually HELP the metric?

The rank diagnostic showed the label/doc gap is a low-rank LINEAR mean offset (erasable -- the gate
PASSED). This tests whether removing it improves label-quality scoring: erase the rank-1 label-vs-doc
direction from label embeddings AND cluster centroids, recompute label<->centroid cosine, and compare
to raw centroid and whitened on (a) the gate-b judge correlation and (b) the verbose intrusion.

Prediction: little help -- the offset is class-uniform (every label shares it, so removing it can't
create fine-discrimination signal), and whitening (which removes this direction among others) already
handled the coarse verbose confound. This nails down whether idea A adds anything beyond whitening.

  uv run python experiments/label_quality/phase3b_erasure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from scipy.stats import spearmanr

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))

from metrics import fit_whitening, unit  # noqa: E402


def main():
    from ab_harness import make_embedder

    from perturbations import build_battery, load_fit

    cl, objects, emb, coords, meta = load_fit("20ng", None, 25, 4)
    emb = emb.astype(np.float64)
    labels = json.loads((HERE / "data" / "labels_20ng_haiku.json").read_text())
    battery = build_battery(cl.cluster_layers_, cl.cluster_tree_, labels)

    embedder = make_embedder(meta["emb_model"])
    gold = sorted({labels[L][i] for L in range(len(labels)) for i in range(len(labels[L]))
                   if labels[L][i] and labels[L][i] != "Unlabelled"})
    GL = embedder.encode(gold, convert_to_numpy=True).astype(np.float64)
    d1 = GL.mean(0) - emb.mean(0)
    d1 /= norm(d1)                                   # rank-1 label<->doc offset direction
    def erase(x):
        x = np.atleast_2d(x)
        return x - np.outer(x @ d1, d1)
    mu, W = fit_whitening(emb)
    def whiten(x):
        return (np.atleast_2d(x) - mu) @ W

    cand = sorted({c for it in battery for c in [it["gold"]] + list(it["variants"].values())})
    CE = dict(zip(cand, embedder.encode(cand, convert_to_numpy=True).astype(np.float64)))

    S = {}
    for it in battery:
        L, i = it["layer"], it["idx"]
        c = cl.cluster_layers_[L].centroid_vectors[i].astype(np.float64)
        rc, rw, re_ = unit(c), unit(whiten(c))[0], unit(erase(c))[0]
        for typ, lab in {"gold": it["gold"], **it["variants"]}.items():
            e = CE[lab]
            S[(L, i, typ)] = dict(centroid=float(unit(e) @ rc),
                                  whitened=float(unit(whiten(e))[0] @ rw),
                                  erased=float(unit(erase(e))[0] @ re_))

    J = {(r["layer"], r["idx"], r["type"]): r
         for r in json.loads((HERE / "data" / "judge_ratings_20ng_sonnet.json").read_text())}
    keys = [k for k in J if k in S and J[k].get("overall") is not None]
    jv = np.array([J[k]["overall"] for k in keys])
    print(f"rank-1 label/doc offset erased; {len(keys)} judged candidates\n")
    print("Spearman(metric, judge overall):   [raw=0.740 whitened=0.708 from gate b]")
    for m in ["centroid", "whitened", "erased"]:
        print(f"  {m:<9} {spearmanr([S[k][m] for k in keys], jv)[0]:+.3f}")
    print("\nverbose intrusion accuracy (gold out-scores the verbose pad):   [raw=77.6% whitened=91.6%]")
    for m in ["centroid", "whitened", "erased"]:
        acc = [1.0 if S[(it["layer"], it["idx"], "gold")][m] > S[(it["layer"], it["idx"], "verbose")][m]
               else 0.0 for it in battery if "verbose" in it["variants"]]
        print(f"  {m:<9} {np.mean(acc)*100:.1f}%")


if __name__ == "__main__":
    main()
