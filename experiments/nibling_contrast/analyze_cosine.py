"""
Phase 1 (no API): does centroid<->label cosine track the contrast effect, and which
instances move most?

For every contrast-driven rename (injected cluster whose name changed), compute:
    cos_old   = cos(centroid, embed(baseline_label))
    cos_new   = cos(centroid, embed(contrast_label))
    dcent     = cos_new - cos_old            (did contrast move the label toward the centroid?)
    old_new   = cos(embed(baseline), embed(contrast))   (how far the label itself moved)

Centroids come from a deterministic re-fit of the clusterer (same params/data as the
matrix, so cluster indices line up). Labels are embedded with the SAME model the
document vectors use, so cosines are in-space.

Buckets surfaced:
  (1) biggest centrality INCREASE (dcent high)   -- metric says "more representative"
  (2) biggest centrality DECREASE (dcent low)    -- metric says "less representative"
  (3) centrality FLAT but label moved a LOT (|dcent| small, old_new low) -- metric blind spot
  (3b) centrality FLAT and label barely moved (old_new high) -- trivial rewords (sanity)

Run:  uv run python experiments/nibling_contrast/analyze_cosine.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.linalg import norm

HERE = Path("/Users/stevenfazzio/repos/toponymy/experiments/nibling_contrast")
sys.path.insert(0, str(HERE))
from ab_harness import load_dataset  # noqa: E402

from toponymy.clustering import ToponymyClusterer  # noqa: E402
from toponymy.cluster_layer import ClusterLayerText  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

FLAT = 0.01  # |dcent| below this == "centrality unchanged"


def unit(v):
    return v / (norm(v, axis=-1, keepdims=True) + 1e-12)


def jacc(a, b):
    ta, tb = set(a.lower().split()), set(b.lower().split())
    return len(ta & tb) / max(1, len(ta | tb))


def short(s, n=46):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


_CACHE = {}


def fit_for(ds):
    if ds in _CACHE:
        return _CACHE[ds]
    objects, emb, coords, meta = load_dataset(ds, None)
    cl = ToponymyClusterer(min_clusters=4, base_min_cluster_size=25, verbose=False)
    cl.fit_predict(coords, emb, ClusterLayerText)
    embedder = SentenceTransformer(meta["emb_model"], device="cpu")
    _CACHE[ds] = (cl, embedder)
    return _CACHE[ds]


def main():
    rows = []
    for f in sorted((HERE / "data").glob("result_*.json")):
        r = json.loads(f.read_text())
        ds, mdl = r["dataset"], r["model"]
        cl, embedder = fit_for(ds)
        counts = [int(l.centroid_vectors.shape[0]) for l in cl.cluster_layers_]
        assert counts == r["counts"], f"cluster mismatch {ds}: {counts} vs {r['counts']}"
        inj = set(r["injected_names"].keys())
        for row in r["rows"][1:]:  # skip layer 0
            L = row["layer"]
            cents = cl.cluster_layers_[L].centroid_vectors
            ren = [(i, o, n) for (i, o, n) in row["renames"] if f"{L},{i}" in inj]
            if not ren:
                continue
            Eo = embedder.encode([o for _, o, _ in ren], convert_to_numpy=True)
            En = embedder.encode([n for _, _, n in ren], convert_to_numpy=True)
            for k, (i, o, n) in enumerate(ren):
                c, eo, en = unit(cents[i]), unit(Eo[k]), unit(En[k])
                co, cn = float(c @ eo), float(c @ en)
                rows.append(dict(ds=ds, mdl=mdl, L=L, i=i, old=o, new=n,
                                 cos_old=co, cos_new=cn, dcent=cn - co,
                                 old_new=float(eo @ en),
                                 contrast=r["injected_names"].get(f"{L},{i}", [])))

    print(f"contrast-driven renames analysed: {len(rows)}\n")

    # ---- direction summary: does contrast move labels toward or away from centroid? ----
    print("Δcentroid direction (per dataset/model):  mean Δ | %toward(+) | %away(-)")
    keys = sorted({(x["ds"], x["mdl"]) for x in rows})
    for ds, mdl in keys:
        sub = [x["dcent"] for x in rows if x["ds"] == ds and x["mdl"] == mdl]
        if not sub:
            continue
        pos = 100 * sum(d > 0 for d in sub) / len(sub)
        print(f"  {ds:>6}/{mdl:<10} n={len(sub):>3}  mean Δ={np.mean(sub):+.3f}  "
              f"+{pos:.0f}% / -{100-pos:.0f}%")
    alld = [x["dcent"] for x in rows]
    print(f"  {'OVERALL':>17} n={len(alld):>3}  mean Δ={np.mean(alld):+.3f}  "
          f"+{100*sum(d>0 for d in alld)/len(alld):.0f}% / "
          f"-{100*sum(d<0 for d in alld)/len(alld):.0f}%\n")

    def dump(title, items):
        print(f"\n### {title}")
        for x in items:
            tg = "SUBST" if jacc(x["old"], x["new"]) < 0.5 else "minor"
            print(f"  Δ={x['dcent']:+.3f} (cos {x['cos_old']:.2f}->{x['cos_new']:.2f}) "
                  f"old/new={x['old_new']:.2f} [{tg}] {x['ds']}/{x['mdl']} L{x['L']}")
            print(f"      {short(x['old'])!r} -> {short(x['new'])!r}")
            if x["contrast"]:
                print(f"      vs: {', '.join(short(c, 22) for c in x['contrast'][:4])}")

    dump("(1) biggest centrality INCREASE (metric: 'more representative')",
         sorted(rows, key=lambda x: -x["dcent"])[:8])
    dump("(2) biggest centrality DECREASE (metric: 'less representative')",
         sorted(rows, key=lambda x: x["dcent"])[:8])
    flat = [x for x in rows if abs(x["dcent"]) < FLAT]
    dump(f"(3) centrality FLAT (|Δ|<{FLAT}) but label moved MOST (low old/new cos) — metric blind spot",
         sorted(flat, key=lambda x: x["old_new"])[:8])
    dump(f"(3b) centrality FLAT and label barely moved (high old/new cos) — trivial rewords",
         sorted(flat, key=lambda x: -x["old_new"])[:5])

    (HERE / "data" / "cosine_instances.json").write_text(json.dumps(rows, indent=2))
    print(f"\nsaved {len(rows)} instances -> data/cosine_instances.json")


if __name__ == "__main__":
    main()
