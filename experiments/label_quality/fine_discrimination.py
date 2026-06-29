"""
Phase 1 -- fine-grained discrimination (the regime the original negative result lived in).

The coarse metric (gate b) ranks good-vs-DEGRADED well. Does ANY reference point rank good-vs-GOOD
the way a judge does? Reuses the nibling A/B pairs for free: baseline vs +contrast labels for the
SAME cluster (both good-faith labels from the same namer) and their per-pair blind-judge verdicts,
and asks whether sign(score(contrast) - score(baseline)) agrees with the judge -- for raw centroid
(the original ~chance result) and, the new question, for WHITENED centroid / medoid / exemplar.

Reports overall agreement AND the split by which label won, because the winners are imbalanced
(contrast usually loses): a metric that just predicts the majority would look good overall but fail
on the minority class. Real signal needs BOTH splits above 50%.

Caveat: ground truth here is the nibling BLIND (haiku) judge, which is itself weak on subtle pairs;
a clean version would re-judge with the grounded sonnet judge (small API cost). NO new API here.

  uv run python experiments/label_quality/fine_discrimination.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))

from metrics import METRICS, fit_whitening, medoid_vector, unit  # noqa: E402  (reuse)


def main():
    from ab_harness import make_embedder

    from perturbations import load_fit

    cl, objects, emb, coords, meta = load_fit("20ng", None, 25, 4)
    for layer in cl.cluster_layers_:
        layer.make_exemplar_texts(objects, emb)
    emb = emb.astype(np.float64)
    mu, W = fit_whitening(emb)
    def whiten(x):
        return (np.atleast_2d(x) - mu) @ W

    verdicts = [v for v in json.loads((NIBLING / "data" / "judge_results.json").read_text())
                if v["ds"] == "20ng" and v["winner"] in ("contrast", "baseline")]
    base = np.mean([v["winner"] == "contrast" for v in verdicts])
    print(f"{len(verdicts)} decided good-vs-good 20ng pairs; contrast won {base*100:.0f}% "
          f"(majority-class baseline => {max(base,1-base)*100:.0f}%)")

    embedder = make_embedder(meta["emb_model"])
    labs = sorted({l for v in verdicts for l in (v["old"], v["new"])})
    LE = dict(zip(labs, embedder.encode(labs, convert_to_numpy=True).astype(np.float64)))

    agree = {m: [] for m in METRICS}
    won = {m: [] for m in METRICS}    # agreement on contrast-won pairs
    lost = {m: [] for m in METRICS}   # agreement on baseline-won pairs
    for v in verdicts:
        layer = cl.cluster_layers_[v["L"]]
        i = v["i"]
        centroid = layer.centroid_vectors[i].astype(np.float64)
        members = np.where(layer.cluster_labels == i)[0]
        medoid = medoid_vector(emb[members] if len(members) else centroid[None])
        ex = layer.exemplar_indices[i]
        exemplar = emb[ex[0]] if (len(ex) and ex[0] < len(emb)) else medoid
        refs = {"centroid": unit(centroid), "medoid": unit(medoid), "exemplar": unit(exemplar)}
        rw = unit(whiten(centroid))[0]
        eo, en = unit(LE[v["old"]]), unit(LE[v["new"]])           # baseline, contrast
        delta = {m: float(en @ refs[m] - eo @ refs[m]) for m in refs}
        delta["whitened"] = float(unit(whiten(LE[v["new"]]))[0] @ rw - unit(whiten(LE[v["old"]]))[0] @ rw)
        contrast_won = v["winner"] == "contrast"
        for m in METRICS:
            ok = 1.0 if (delta[m] > 0) == contrast_won else 0.0
            agree[m].append(ok)
            (won if contrast_won else lost)[m].append(ok)

    print("\nagreement with judge on good-vs-good pairs (50% = no signal; beat the majority line):")
    print(f"{'metric':<10} {'overall':>9}  {'|contrast-won':>14}  {'|baseline-won':>14}")
    for m in METRICS:
        print(f"{m:<10} {np.mean(agree[m])*100:8.1f}%  {np.mean(won[m])*100:13.1f}%  {np.mean(lost[m])*100:13.1f}%")
    print("\n(real fine-grained signal requires BOTH split columns above ~50%, not just overall.)")


if __name__ == "__main__":
    main()
