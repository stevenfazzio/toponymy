"""
Embedder robustness: do the metric findings survive a much STRONGER embedder (Cohere embed-v4)
instead of all-MiniLM-L6-v2? We re-embed documents + labels with embed-v4 and recompute the
centroid / whitened-centroid metric on the SAME clusters, labels, and judge ratings (all held fixed
-- no re-clustering, re-naming, or re-judging), then re-run the three decisive comparisons:

  (1) gate-b correlation     Spearman(metric, judge overall)        [MiniLM: centroid 0.740 / whitened 0.708]
  (2) fine discrimination    metric agrees with judge good-vs-good?  [MiniLM: ~chance, whitened 37%]
  (3) ablation detection     metric-Δ tracks judge-Δ?                [MiniLM: ~0 / blind, all features]

If embed-v4 makes the metric sensitive -> our limits are MiniLM-specific; if not -> robust to
embedder quality. Doc/label embeddings are cached so reruns are free.

  uv run --with cohere python experiments/label_quality/embedder_robustness.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))

from metrics import fit_whitening, unit  # noqa: E402

MODEL, DIM, BATCH = "embed-v4.0", 1024, 96


def cohere_embed(texts, co):
    from tqdm import tqdm
    out = []
    for i in tqdm(range(0, len(texts), BATCH), desc="embed-v4"):
        r = co.embed(texts=[t[:8000] for t in texts[i:i + BATCH]], model=MODEL,
                     input_type="clustering", embedding_types=["float"], output_dimension=DIM)
        out.extend(r.embeddings.float_)
    return np.asarray(out, dtype=np.float64)


def main():
    import cohere

    from perturbations import load_fit

    cl, objects, emb, coords, meta = load_fit("20ng", None, 25, 4)
    counts = [int(l.centroid_vectors.shape[0]) for l in cl.cluster_layers_]
    co = cohere.ClientV2(api_key=os.environ["CO_API_KEY"])

    # --- doc embeddings (cached) ---
    dpath = HERE / "data" / "cohere_docs.npy"
    if dpath.exists():
        doc = np.load(dpath)
    else:
        doc = cohere_embed(objects, co)
        np.save(dpath, doc)
    print(f"docs: {doc.shape}")

    # --- collect every label used in the three comparisons, embed (cached) ---
    JR = json.loads((HERE / "data" / "judge_ratings_20ng_sonnet.json").read_text())
    FP = json.loads((HERE / "data" / "finepairs.json").read_text())
    AB = json.loads((HERE / "data" / "ablation.json").read_text())
    labset = {r["label"] for r in JR}
    labset |= {r["base"] for r in FP} | {r["alt"] for r in FP}
    for rows in AB.values():
        labset |= {r["full"] for r in rows} | {r["abl"] for r in rows}
    labs = sorted(labset)
    lpath = HERE / "data" / "cohere_labels.json"
    cache = json.loads(lpath.read_text()) if lpath.exists() else {}
    todo = [l for l in labs if l not in cache]
    if todo:
        for l, v in zip(todo, cohere_embed(todo, co)):
            cache[l] = v.tolist()
        lpath.write_text(json.dumps(cache))
    LE = {l: np.asarray(cache[l], dtype=np.float64) for l in labs}
    print(f"labels: {len(labs)}")

    # --- centroids (embed-v4 space) + whitening, on the FIXED MiniLM clusters ---
    mu, W = fit_whitening(doc)
    def whiten(x):
        return (np.atleast_2d(x) - mu) @ W
    cent, cent_w = {}, {}
    for L in range(len(counts)):
        cls = cl.cluster_layers_[L].cluster_labels
        for i in range(counts[L]):
            members = np.where(cls == i)[0]
            c = doc[members].mean(0) if len(members) else doc.mean(0)
            cent[(L, i)] = unit(c)
            cent_w[(L, i)] = unit(whiten(c))[0]

    def centroid(label, L, i):
        return float(unit(LE[label]) @ cent[(L, i)])
    def whitened(label, L, i):
        return float(unit(whiten(LE[label]))[0] @ cent_w[(L, i)])
    METRICS = {"centroid": centroid, "whitened": whitened}

    print("\n=== (1) GATE-B  Spearman(metric, judge overall)   [MiniLM: centroid 0.740 / whitened 0.708] ===")
    rows = [r for r in JR if r.get("overall") is not None]
    jv = np.array([r["overall"] for r in rows])
    for name, f in METRICS.items():
        mv = np.array([f(r["label"], r["layer"], r["idx"]) for r in rows])
        print(f"  {name:<9} rho = {spearmanr(mv, jv)[0]:+.3f}   (n={len(rows)})")

    print("\n=== (2) FINE-DISCRIMINATION  agreement with judge, good-vs-good   [MiniLM: ~chance] ===")
    dec = [r for r in FP if r["j_alt"] != r["j_base"]]
    for name, f in METRICS.items():
        ok, won, lost = [], [], []
        for r in dec:
            a = (f(r["alt"], r["L"], r["i"]) - f(r["base"], r["L"], r["i"]) > 0) == (r["j_alt"] > r["j_base"])
            ok.append(a)
            (won if r["j_alt"] > r["j_base"] else lost).append(a)
        print(f"  {name:<9} overall {np.mean(ok)*100:4.0f}%  |alt-won {np.mean(won)*100:4.0f}%  "
              f"|base-won {np.mean(lost)*100:4.0f}%   (n={len(dec)})")

    print("\n=== (3) ABLATION-DETECTION  metric-Δ vs judge-Δ   [MiniLM: ~0 / blind] ===")
    for feat, rows in AB.items():
        ch = [r for r in rows if r["full"] != r["abl"]]
        line = f"  {feat:<10}"
        for name, f in METRICS.items():
            md = np.array([f(r["full"], r["L"], r["i"]) - f(r["abl"], r["L"], r["i"]) for r in ch])
            jd = np.array([r["j_full"] - r["j_abl"] for r in ch])
            d2 = [(m, j) for m, j in zip(md, jd) if j != 0]
            ag = np.mean([(m > 0) == (j > 0) for m, j in d2]) * 100 if d2 else float("nan")
            line += f"   {name}: rho {spearmanr(md, jd)[0]:+.2f} agree {ag:.0f}%"
        print(line + f"   (n={len(ch)})")


if __name__ == "__main__":
    main()
