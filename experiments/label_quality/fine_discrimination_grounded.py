"""
Phase 1 -- fine-grained discrimination, RIGOROUS version.

Per cluster, two genuinely-good labels from DIFFERENT namers (haiku vs gpt-4o-mini) are each scored
by a GROUNDED sonnet judge (k=3, grounded in ~15 docs; sonnet is neither namer, so no
self-preference). For each reference point we ask: does sign(metric(gpt4o) - metric(haiku)) agree
with the judge's preference? This is the good-vs-good regime the original negative lived in, now with
strong, calibrated ground truth and decent n.

Reports agreement with a winner-split (both splits must beat 50% for real signal). Caches the judged
+ scored pairs to data/finepairs.json, so re-running only re-does the (free) analysis.

  uv run python experiments/label_quality/fine_discrimination_grounded.py --k 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))

from metrics import METRICS, fit_whitening, medoid_vector, unit  # noqa: E402


def judge_and_score(args):
    from ab_harness import load_dataset, make_embedder
    from judge_fair import sample_docs

    from judge_quality import MODELS, rate
    from perturbations import load_fit

    cl, objects, emb, coords, meta = load_fit("20ng", None, 25, 4)
    for layer in cl.cluster_layers_:
        layer.make_exemplar_texts(objects, emb)
    emb = emb.astype(np.float64)
    mu, W = fit_whitening(emb)
    def whiten(x):
        return (np.atleast_2d(x) - mu) @ W

    base = json.loads((HERE / "data" / f"labels_20ng_{args.base}.json").read_text())
    alt = json.loads((HERE / "data" / f"labels_20ng_{args.alt}.json").read_text())
    counts = [int(l.centroid_vectors.shape[0]) for l in cl.cluster_layers_]
    assert [len(x) for x in base] == counts and [len(x) for x in alt] == counts, "label/cluster mismatch"

    pairs = [(L, i) for L in range(len(counts)) for i in range(counts[L])
             if base[L][i] and alt[L][i] and base[L][i] != "Unlabelled" and alt[L][i] != "Unlabelled"
             and base[L][i] != alt[L][i]]
    embedder = make_embedder(meta["emb_model"])
    uniq = sorted({base[L][i] for L, i in pairs} | {alt[L][i] for L, i in pairs})
    LE = dict(zip(uniq, embedder.encode(uniq, convert_to_numpy=True).astype(np.float64)))
    obj, jm = meta["obj"], MODELS[args.judge]
    print(f"{len(pairs)} clusters with two distinct good labels; judging both with {args.judge} (k={args.k})")

    rows, t0 = [], time.time()
    for n, (L, i) in enumerate(pairs):
        bl, al = base[L][i], alt[L][i]
        docs = sample_docs("20ng", L, i, n_near=args.n_near, n_rand=args.n_rand)
        jb = rate(bl, docs, obj, jm, args.k, args.temp)[0]["overall"]
        ja = rate(al, docs, obj, jm, args.k, args.temp)[0]["overall"]
        if jb is None or ja is None:
            continue
        layer = cl.cluster_layers_[L]
        centroid = layer.centroid_vectors[i].astype(np.float64)
        members = np.where(layer.cluster_labels == i)[0]
        medoid = medoid_vector(emb[members] if len(members) else centroid[None])
        ex = layer.exemplar_indices[i]
        exemplar = emb[ex[0]] if (len(ex) and ex[0] < len(emb)) else medoid
        refs = {"centroid": unit(centroid), "medoid": unit(medoid), "exemplar": unit(exemplar)}
        rw = unit(whiten(centroid))[0]
        eb, ea = unit(LE[bl]), unit(LE[al])
        d = {m: float(ea @ refs[m] - eb @ refs[m]) for m in refs}
        d["whitened"] = float(unit(whiten(LE[al]))[0] @ rw - unit(whiten(LE[bl]))[0] @ rw)
        rows.append(dict(L=L, i=i, base=bl, alt=al, j_base=jb, j_alt=ja,
                         **{f"d_{m}": d[m] for m in METRICS}))
        if (n + 1) % 10 == 0:
            (HERE / "data" / "finepairs.json").write_text(json.dumps(rows, indent=2))
            print(f"  {n+1}/{len(pairs)} ({time.time()-t0:.0f}s)", flush=True)
    (HERE / "data" / "finepairs.json").write_text(json.dumps(rows, indent=2))
    return rows


def analyze(rows):
    dec = [r for r in rows if r["j_alt"] != r["j_base"]]
    print(f"\n{len(rows)} pairs judged; {len(dec)} decided (judge not tied)")
    if not dec:
        return
    altwon = np.mean([r["j_alt"] > r["j_base"] for r in dec])
    print(f"judge preferred the {('gpt4o' )} label {altwon*100:.0f}% (majority line {max(altwon,1-altwon)*100:.0f}%)")
    print("\nagreement with judge on good-vs-good (both splits must beat ~50% for real signal):")
    print(f"{'metric':<10} {'overall':>9}  {'|alt-won':>9}  {'|base-won':>10}")
    for m in METRICS:
        ok = np.array([(r[f"d_{m}"] > 0) == (r["j_alt"] > r["j_base"]) for r in dec], float)
        won = np.array([o for r, o in zip(dec, ok) if r["j_alt"] > r["j_base"]])
        lost = np.array([o for r, o in zip(dec, ok) if r["j_alt"] < r["j_base"]])
        print(f"{m:<10} {ok.mean()*100:8.1f}%  {won.mean()*100:8.1f}%  {lost.mean()*100:9.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="haiku")
    ap.add_argument("--alt", default="gpt4omini")
    ap.add_argument("--judge", default="sonnet")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--n-near", type=int, default=10)
    ap.add_argument("--n-rand", type=int, default=5)
    ap.add_argument("--reanalyze", action="store_true", help="skip judging; re-analyze cached finepairs.json")
    args = ap.parse_args()

    cache = HERE / "data" / "finepairs.json"
    rows = json.loads(cache.read_text()) if (args.reanalyze and cache.exists()) else judge_and_score(args)
    analyze(rows)


if __name__ == "__main__":
    main()
