"""
Feature ablation: does the whitened metric (and the grounded judge) detect the effect of dropping
each Toponymy naming feature -- exemplars / keyphrases / subtopics?

For each feature we re-name 20NG (haiku) with that feature DROPPED from the naming prompt (monkeypatch
`topic_name_prompt`, the same hook the nibling harness uses), then judge the ablated labels (grounded
sonnet, k=3) and metric-score them (whitened centroid). Baseline = the full-haiku labels and their
gold judge ratings (reused). We report, per feature:
  - fraction of labels the ablation actually CHANGED (did the feature reach the output?)
  - judge-Δ = quality(full) - quality(ablated), the feature's contribution (an INDEPENDENT yardstick)
  - whether metric-Δ tracks judge-Δ (Spearman + sign agreement) -- the actual question about the metric.

The judge column is what disambiguates a null metric result: metric-blind (judge sees a drop, metric
doesn't) vs feature-didn't-matter (neither sees a drop).

Caveats: clustering is the SAME deterministic fit (only the prompt changes); naming is one haiku draw
per condition (haiku is ~mostly deterministic but not fully, so per-cluster Δ carries some draw noise
-- read the aggregate). Subtopics apply only at coarse layers and dropping them also disables the
single-child name-propagation skip.

  uv run python experiments/label_quality/ablation.py --verify           # cheap sanity (no judging)
  uv run python experiments/label_quality/ablation.py --features exemplars keyphrases subtopics --k 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))

import toponymy.cluster_layer as cl_mod  # noqa: E402

from metrics import fit_whitening, unit  # noqa: E402

_ORIG_TNP = cl_mod.topic_name_prompt


class _Empty:
    """Indexable empty: exemplar_texts[i] / keyphrases[i] -> [] for any i."""
    def __getitem__(self, i):
        return []

    def __len__(self):
        return 10 ** 9


@contextmanager
def ablate(feature):
    def patched(topic_index, layer_id, all_topic_names, **kw):
        if feature == "exemplars":
            kw["exemplar_texts"] = _Empty()
        elif feature == "keyphrases":
            kw["keyphrases"] = _Empty()
        elif feature == "subtopics":
            kw["subtopics"] = None
        return _ORIG_TNP(topic_index, layer_id, all_topic_names, **kw)

    cl_mod.topic_name_prompt = patched
    try:
        yield
    finally:
        cl_mod.topic_name_prompt = _ORIG_TNP


def name_ablated(feature, objects, emb, coords, meta, subsample=None):
    from ab_harness import make_embedder, make_namer

    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import ToponymyClusterer
    from toponymy.toponymy import Toponymy

    o, e, c = (objects, emb, coords) if subsample is None else (objects[:subsample], emb[:subsample], coords[:subsample])
    clusterer = ToponymyClusterer(min_clusters=4, base_min_cluster_size=25, verbose=False)
    clusterer.fit_predict(c, e, ClusterLayerText)
    model = Toponymy(make_namer("haiku"), make_embedder(meta["emb_model"]), clusterer=clusterer,
                     object_description=meta["obj"], corpus_description=meta["corpus"], verbose=False)
    with (nullcontext() if feature == "full" else ablate(feature)):
        model.fit(o, e, c)
    return [list(l) for l in model.topic_names_]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=["exemplars", "keyphrases", "subtopics"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--verify", action="store_true", help="cheap: name full vs ablate-exemplars on a subsample, show diffs")
    args = ap.parse_args()

    from ab_harness import load_dataset, make_embedder
    from judge_fair import sample_docs

    from judge_quality import MODELS, rate

    objects, emb, coords, meta = load_dataset("20ng", None)

    if args.verify:
        full = name_ablated("full", objects, emb, coords, meta, subsample=1500)
        abl = name_ablated("exemplars", objects, emb, coords, meta, subsample=1500)
        diff = [(full[0][i], abl[0][i]) for i in range(len(full[0])) if full[0][i] != abl[0][i]]
        print(f"layer 0: {len(diff)}/{len(full[0])} finest labels changed when exemplars dropped")
        for f, a in diff[:6]:
            print(f"  full:    {f!r}\n  ablated: {a!r}")
        return

    emb64 = emb.astype(np.float64)
    mu, W = fit_whitening(emb64)
    def whiten(x):
        return (np.atleast_2d(x) - mu) @ W
    embedder = make_embedder(meta["emb_model"])
    LE = {}
    def emb_of(s):
        if s not in LE:
            LE[s] = embedder.encode([s], convert_to_numpy=True)[0].astype(np.float64)
        return LE[s]

    # deterministic fit for centroids + cluster indices (matches the gold labeling)
    from perturbations import load_fit
    cl, *_ = load_fit("20ng", None, 25, 4)
    counts = [int(l.centroid_vectors.shape[0]) for l in cl.cluster_layers_]
    rw = {(L, i): unit(whiten(cl.cluster_layers_[L].centroid_vectors[i].astype(np.float64)))[0]
          for L in range(len(counts)) for i in range(counts[L])}
    def wscore(label, L, i):
        return float(unit(whiten(emb_of(label)))[0] @ rw[(L, i)])

    full = json.loads((HERE / "data" / "labels_20ng_haiku.json").read_text())
    J = {(r["layer"], r["idx"]): r["overall"]
         for r in json.loads((HERE / "data" / "judge_ratings_20ng_sonnet.json").read_text())
         if r["type"] == "gold" and r.get("overall") is not None}
    obj, jm, t0 = meta["obj"], MODELS["sonnet"], time.time()

    results = {}
    for feature in args.features:
        print(f"\n=== ablating {feature} ===", flush=True)
        ab = name_ablated(feature, objects, emb, coords, meta)
        rows = []
        for L in range(len(counts)):
            for i in range(counts[L]):
                fl, al = full[L][i], ab[L][i]
                if not fl or not al or fl == "Unlabelled" or al == "Unlabelled" or (L, i) not in J:
                    continue
                ja = rate(al, sample_docs("20ng", L, i, n_near=10, n_rand=5), obj, jm, args.k, 0.7)[0]["overall"]
                if ja is None:
                    continue
                rows.append(dict(L=L, i=i, full=fl, abl=al, j_full=J[(L, i)], j_abl=ja,
                                 m_full=wscore(fl, L, i), m_abl=wscore(al, L, i), changed=fl != al))
        results[feature] = rows
        (HERE / "data" / "ablation.json").write_text(json.dumps(results, indent=2))
        print(f"  judged {len(rows)} clusters ({time.time()-t0:.0f}s)", flush=True)

    print("\n================ ABLATION RESULT ================")
    print("judge-Δ = quality(full) - quality(ablated); >0 means the feature helps")
    for feature, rows in results.items():
        ch = [r for r in rows if r["changed"]]
        print(f"\n{feature}: {len(ch)}/{len(rows)} labels changed by the ablation")
        if len(ch) < 3:
            print("  (too few changed to analyze)")
            continue
        jd = np.array([r["j_full"] - r["j_abl"] for r in ch])
        md = np.array([r["m_full"] - r["m_abl"] for r in ch])
        print(f"  judge-Δ  mean {jd.mean():+.2f}   (feature's contribution to quality)")
        print(f"  metric-Δ mean {md.mean():+.3f}")
        print(f"  Spearman(metric-Δ, judge-Δ) = {spearmanr(md, jd)[0]:+.3f}")
        dec = [(m, j) for m, j in zip(md, jd) if j != 0]
        if dec:
            agree = np.mean([(m > 0) == (j > 0) for m, j in dec])
            print(f"  sign agreement (judge≠tie) = {agree*100:.0f}%  (n={len(dec)}; 50% = metric blind to this)")


if __name__ == "__main__":
    main()
