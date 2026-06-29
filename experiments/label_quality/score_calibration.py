"""
Score the human-calibration ratings against the grounded judge: weighted Cohen's kappa +
Spearman/Pearson between your 0-4 ratings and the judge's mean, on the same blinded items.
Confirms whether the sonnet judge -- which gate (b) trusts as ground truth -- agrees with you
(Krumdick: an un-calibrated judge is not trustworthy).

  uv run python experiments/label_quality/score_calibration.py
(reads data/calibration_human.json {id: rating}  and  data/calibration_key.json)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path("/Users/stevenfazzio/repos/toponymy/experiments/label_quality")


def main():
    key = json.loads((HERE / "data" / "calibration_key.json").read_text())
    H = {str(k): float(v) for k, v in json.loads((HERE / "data" / "calibration_human.json").read_text()).items()}
    ids = [i for i in key if i in H]
    if not ids:
        raise SystemExit("no overlap between human ratings and key ids -- check the pasted JSON")
    if len(ids) < len(key):
        print(f"(note: {len(key)-len(ids)} of {len(key)} items unrated)")

    h = np.array([H[i] for i in ids])
    jm = np.array([float(key[i]["judge_overall"]) for i in ids])      # judge mean (continuous)
    hr, jr = np.rint(h).astype(int), np.rint(jm).astype(int)          # rounded for kappa

    print(f"calibration: {len(ids)} items rated by both")
    try:
        from scipy.stats import pearsonr, spearmanr
        print(f"  Spearman(human, judge) rho = {spearmanr(h, jm)[0]:+.3f}")
        print(f"  Pearson (human, judge)  r  = {pearsonr(h, jm)[0]:+.3f}")
    except ImportError:
        print(f"  Pearson r = {np.corrcoef(h, jm)[0, 1]:+.3f}")
    try:
        from sklearn.metrics import cohen_kappa_score
        kq = cohen_kappa_score(hr, jr, weights="quadratic", labels=[0, 1, 2, 3, 4])
        kl = cohen_kappa_score(hr, jr, weights="linear", labels=[0, 1, 2, 3, 4])
        print(f"  Cohen's kappa quadratic = {kq:+.3f}   (linear = {kl:+.3f})")
    except ImportError:
        pass
    print(f"  mean |human - judge|    = {np.mean(np.abs(h - jm)):.2f}  on 0-4")
    print(f"  mean human {h.mean():.2f}  vs  mean judge {jm.mean():.2f}")

    g = defaultdict(list)
    for i in ids:
        g[key[i]["type"]].append((H[i], key[i]["judge_overall"]))
    print("\n  per candidate type (human vs judge):")
    for t, vs in sorted(g.items()):
        print(f"    {t:<9} human {np.mean([a for a,_ in vs]):.2f}  judge {np.mean([b for _,b in vs]):.2f}  (n={len(vs)})")

    print("\n  biggest human-judge gaps:")
    for i in sorted(ids, key=lambda i: -abs(H[i] - key[i]["judge_overall"]))[:5]:
        print(f"    {key[i]['type']:<9} human {H[i]:.0f}  judge {key[i]['judge_overall']:.2f}   ({i})")


if __name__ == "__main__":
    main()
