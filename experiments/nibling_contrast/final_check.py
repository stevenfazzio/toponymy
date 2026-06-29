"""
Final check on a representative subset of cells. Two questions:

(1) TUNED CONTRAST: does tight max_dist (0.25) + a concise, anti-enumeration block (v2)
    beat the no-contrast baseline under the blind judge, where the original (v1) lost
    (~27% win-rate)? v1 is re-judged in the same run as an anchor.

(2) DISAMBIGUATIONS: does contrast change how many name-collision groups Toponymy would
    disambiguate? Uses the real trigger (cluster_topic_names_for_renaming: complete-linkage
    agglomerative clustering of name embeddings; groups of size >=2 each need a
    disambiguation). Reported for baseline / v1-contrast / v2-contrast.

Run:  uv run python experiments/nibling_contrast/final_check.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.linalg import norm

HERE = Path("/Users/stevenfazzio/repos/toponymy/experiments/nibling_contrast")
sys.path.insert(0, str(HERE))
from ab_harness import load_dataset, make_namer, make_embedder, name_once  # noqa: E402
from judge import ask  # noqa: E402  (stateless litellm judge call)

from toponymy.clustering import ToponymyClusterer  # noqa: E402
from toponymy.cluster_layer import ClusterLayerText  # noqa: E402
from toponymy.prompt_construction import cluster_topic_names_for_renaming  # noqa: E402

CELLS = [("20ng", "haiku"), ("20ng", "gpt4omini"), ("arxiv", "haiku")]
V2_MAXDIST = 0.25


def reps(cl, objects, emb, L, i, n=5, maxlen=300):
    labels = cl.cluster_layers_[L].cluster_labels
    cent = cl.cluster_layers_[L].centroid_vectors[i]
    members = np.where(labels == i)[0]
    c = cent / (norm(cent) + 1e-12)
    E = emb[members] / (norm(emb[members], axis=1, keepdims=True) + 1e-12)
    order = np.argsort(-(E @ c))[:n]
    return [" ".join(str(objects[members[o]]).split())[:maxlen] for o in order]


def disambig_groups(names_by_layer, embedder):
    """Per layer: (#collision groups >=2, #names involved, #names) via Toponymy's real trigger."""
    out = []
    for names in names_by_layer:
        names = list(names)
        if len(names) < 2:
            out.append((0, 0, len(names)))
            continue
        try:
            _, labels = cluster_topic_names_for_renaming(names, embedding_model=embedder)
            sizes = np.bincount(labels)
            out.append((int((sizes >= 2).sum()), int(sizes[sizes >= 2].sum()), len(names)))
        except Exception:
            dup = len(names) - len(set(names))
            out.append((0 if dup == 0 else 1, dup, len(names)))  # fallback: exact dups
    return out


def judge_changed(cl, objects, emb, baseline, contrast):
    wins = {"contrast": 0, "baseline": 0, "tie": 0}
    for L in range(1, len(baseline)):
        for i, (bn, cn) in enumerate(zip(baseline[L], contrast[L])):
            if bn == cn:
                continue
            docs = reps(cl, objects, emb, L, i)
            w1 = ask(docs, bn, cn)   # A=baseline, B=contrast
            w2 = ask(docs, cn, bn)   # A=contrast, B=baseline
            c1 = {"B": "contrast", "A": "baseline", "tie": "tie"}[w1]
            c2 = {"A": "contrast", "B": "baseline", "tie": "tie"}[w2]
            wins[c1 if c1 == c2 else "tie"] += 1
    dec = wins["contrast"] + wins["baseline"]
    return wins, (100 * wins["contrast"] / dec if dec else float("nan"))


def main():
    t0 = time.time()
    summary = []
    for ds, mdl in CELLS:
        print(f"\n##### {ds} / {mdl} #####", flush=True)
        objects, emb, coords, meta = load_dataset(ds, None)
        embedder = make_embedder(meta["emb_model"])
        cl = ToponymyClusterer(min_clusters=4, base_min_cluster_size=25, verbose=False)
        cl.fit_predict(coords, emb, ClusterLayerText)
        namer = make_namer(mdl)
        common = dict(objects=objects, emb=emb, coords=coords, clusterer=cl,
                      namer=namer, embedder=embedder, meta=meta, k=5, disambig=False)

        print(" naming baseline / v1 / v2 ...", flush=True)
        base, _ = name_once(use_contrast=False, max_dist=None, **common)
        c1, _ = name_once(use_contrast=True, max_dist=None, block_version="v1", **common)
        c2, _ = name_once(use_contrast=True, max_dist=V2_MAXDIST, block_version="v2", **common)

        dz, d1, d2 = (disambig_groups(x, embedder) for x in (base, c1, c2))
        print(" disambiguation groups(>=2) per layer  [baseline | v1 | v2]  (names involved):")
        for L in range(len(base)):
            print(f"   L{L} (n={dz[L][2]:>3}): {dz[L][0]:>2} | {d1[L][0]:>2} | {d2[L][0]:>2}"
                  f"   ({dz[L][1]} | {d1[L][1]} | {d2[L][1]})", flush=True)

        print(" judging v1 (anchor) ...", flush=True)
        w1, wr1 = judge_changed(cl, objects, emb, base, c1)
        print(" judging v2 (tuned) ...", flush=True)
        w2, wr2 = judge_changed(cl, objects, emb, base, c2)
        print(f"  v1: {w1} -> win-rate {wr1:.0f}%")
        print(f"  v2: {w2} -> win-rate {wr2:.0f}%", flush=True)
        summary.append(dict(ds=ds, mdl=mdl, dis=dict(baseline=dz, v1=d1, v2=d2),
                            judge_v1=[w1, wr1], judge_v2=[w2, wr2]))

    (HERE / "data" / "final_check.json").write_text(json.dumps(summary, indent=2))
    print(f"\n================ SUMMARY ({time.time()-t0:.0f}s) ================")
    print("  cell             v1 win%  v2 win%   Σ disambig groups (base/v1/v2)")
    for s in summary:
        g = [sum(x[0] for x in s["dis"][k]) for k in ("baseline", "v1", "v2")]
        print(f"  {s['ds']:>5}/{s['mdl']:<9}  {s['judge_v1'][1]:6.0f}%  {s['judge_v2'][1]:6.0f}%"
              f"     {g[0]} / {g[1]} / {g[2]}")


if __name__ == "__main__":
    main()
